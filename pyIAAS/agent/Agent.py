from torch.optim import Adam

from .components import *
from .. import get_logger


class Agent:
    def __init__(self, cfg, input_size, hidden_size, max_layers):
        """
        Agent class to do action on neural networks.
        :param cfg: global configuration
        :param input_size: input size of encoder network, embedding vector length
        :param hidden_size: output of encoder networks
        :param max_layers: maximum networks layers
        """
        super().__init__()
        self.cfg = cfg
        self.logger = get_logger('AgentREINFORCE', cfg.LOG_FILE)
        self.encoder_net = EncoderNet(input_size, hidden_size, cfg)
        self.wider_net = WinderActorNet(self.cfg, hidden_size * 2)
        self.deeper_net = DeeperActorNet(self.cfg, hidden_size * 2, max_layers)
        self.selector = SelectorActorNet(self.cfg, hidden_size * 2)
        self.selector_optimizer = Adam(self.selector.parameters())
        self.wider_optimizer = Adam(self.wider_net.parameters())
        self.deeper_optimizer = Adam(self.deeper_net.parameters())
        self.Categorical = torch.distributions.Categorical
        self.entropy_weight = 0.0001
        self.discount = 0.99
        self.rho_max = 10
        if self.cfg.NASConfig['GPU']:
            self.to_cuda()

    def to_cuda(self):
        self.wider_net.cuda()
        self.deeper_net.cuda()
        self.selector.cuda()
        self.encoder_net.cuda()

    def to_cpu(self):
        self.wider_net.cpu()
        self.deeper_net.cpu()
        self.selector.cpu()
        self.encoder_net.cpu()

    def get_action(self, states):
        """
        make decision on each state.
        :param states: state list
        :return: action of each state
        """
        actions, policys, Qs, Vs = [], [], [], []
        for state in states:
            token_list, insert_length, widenable_list, index = state
            output, (h_n, c_n) = self.encoder_net.forward(token_list)
            select_action, select_prob, select_Q, select_V = self.selector.get_action(h_n)
            deeper_action, deeper_prob, deeper_Q, deeper_V = self.deeper_net.get_action(h_n, insert_length)
            wider_action, wider_prob, wider_Q, wider_V = self.wider_net.get_action(output, widenable_list)
            actions.append({
                'select': select_action,
                'wider': wider_action,
                'deeper': deeper_action,
                'net_index': index
            })
            policys.append({
                'select': select_prob,
                'wider': wider_prob,
                'deeper': deeper_prob,
            })
            Qs.append({
                'select': select_Q,
                'wider': wider_Q,
                'deeper': deeper_Q,
            })
            Vs.append({
                'select': select_V,
                'wider': wider_V,
                'deeper': deeper_V,
            })
        return {'action': actions, 'policy': policys, 'Q': Qs, 'V': Vs}

    def _get_log_prob_entropy(self, a_int, a_prob):
        if self.cfg.NASConfig['GPU']:
            a_int = a_int.cuda()
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)

    def update(self, replay_memory):
        """
        update agent actor networks by each reward comes from previous action todo change update method
        :param replay_memory: trajectories and replay memory
        """
        trajectories = replay_memory.get_update_memory(self.cfg.NASConfig['MaxReplayEpisode'])

        if self.cfg.NASConfig['GPU']:
            self.to_cuda()
        for trajectory in trajectories:
            # calculate immediate net output todo finish new agent update algorithm
            states = [i.state for i in trajectory]
            actions = [i.action for i in trajectory]
            old_policies = [i.policy for i in trajectory]
            rewards = [i.reward for i in trajectory]
            new_actions = self.get_action(states)

            # update selector net
            # (do nothing, wider, deeper, prune)
            self._update_selector(self.selector_optimizer,
                                  [i['select'] for i in actions],
                                  [i['select'] for i in new_actions['policy']],
                                  [i['select'] for i in old_policies],
                                  [i['select'] for i in new_actions['Q']],
                                  [i['select'] for i in new_actions['V']],
                                  rewards)

            # update wider net select == 1
            self._update_wider(self.wider_optimizer,
                               [i['wider'] for i in actions],
                               [i['wider'] for i in new_actions['policy']],
                               [i['wider'] for i in old_policies],
                               [i['wider'] for i in new_actions['Q']],
                               [i['wider'] for i in new_actions['V']],
                               rewards,
                               states,
                               [i['select'] for i in actions])

            # update deeper net select == 2
            self._update_deeper(self.wider_optimizer,
                                [i['deeper'] for i in actions],
                                [i['deeper'] for i in new_actions['policy']],
                                [i['deeper'] for i in old_policies],
                                [i['deeper'] for i in new_actions['Q']],
                                [i['deeper'] for i in new_actions['V']],
                                rewards,
                                states,
                                [i['select'] for i in actions])

    def _calculate_loss(self, actions, policies, old_policies, Qs, Vs, rewards):
        """
        update one actor, all these actors can be updated in same scheme
        :param old_policies: behavior policy
        :param actions: one hot form of action
        :param policies: action probability
        :param rewards: reward information to update
        :return loss value
        """
        action_size = policies[0].size(0)
        policy_loss, value_loss = torch.zeros(1, device=Vs[0].device), torch.zeros(1, device=Vs[0].device)
        t = len(rewards)
        if t == 0:  # return if no trajectory
            return
        Qret = torch.zeros(1, device=Vs[0].device)
        for i in reversed(range(t)):
            # Importance sampling weights ρ ← π(∙|s_i) / µ(∙|s_i)  todo stop here
            rho = policies[i].detach() / old_policies[i]
            # Qret ← r_i + γQret
            Qret = rewards[i] + self.discount * Qret
            # Advantage A ← Qret - V(s_i; θ)
            A = Qret - Vs[i]
            # Log policy log(π(a_i|s_i; θ))
            log_prob = policies[i][actions[i]].log()
            # g ← min(c, ρ_a_i)∙∇θ∙log(π(a_i|s_i; θ))∙A
            single_step_policy_loss = - rho[actions[i]].clamp(max=self.rho_max) * log_prob * A.detach()
            # bias correction
            # g ← g + Σ_a [1 - c/ρ_a]_+∙π(a|s_i; θ)∙∇θ∙log(π(a|s_i; θ))∙(Q(s_i, a; θ) - V(s_i; θ)
            single_step_policy_loss -= ((1 - self.rho_max / rho).clamp(min=0) * policies[i].log() * (
                    Qs[i].detach() - Vs[i].expand_as(Qs[i]).detach())).sum()
            # Policy update dθ ← dθ + ∂θ/∂θ∙g
            policy_loss += single_step_policy_loss
            # Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ))
            policy_loss -= self.entropy_weight * -(policies[i].log() * policies[i]).sum()  # Sum over probabilities

            # Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2
            Q = Qs[i][actions[i]]
            value_loss += ((Qret - Q) ** 2 / 2)  # Least squares loss

            # Truncated importance weight ρ¯_a_i = min(1, ρ_a_i)
            truncated_rho = rho[actions[i]].clamp(max=1)
            # Qret ← ρ¯_a_i∙(Qret - Q(s_i, a_i; θ)) + V(s_i; θ)
            Qret = truncated_rho * (Qret - Q.detach()) + Vs[i].detach()
        loss = policy_loss + value_loss
        return loss

    def _step_optimizer(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 100)
        optimizer.step()

    def _update_selector(self, optimizer, actions, policies, old_policies, Qs, Vs, rewards):
        loss = self._calculate_loss(actions, policies, old_policies, Qs, Vs, rewards)
        self._step_optimizer(optimizer, loss)

    def _update_wider(self, optimizer, actions, policies, old_policies, Qs, Vs, rewards, states, selected_ops):
        wider_ops = torch.tensor(selected_ops) == SelectorActorNet.WIDER
        if wider_ops.any():
            actions_, policies_, old_policies_, Qs_, Vs_, rewards_ = [], [], [], [], [], []
            for i in range(len(wider_ops)):
                if wider_ops[i]:
                    editable_list = states[i][2]
                    actions_.append(actions[i])
                    policies_.append(policies[i][editable_list])
                    old_policies_.append(old_policies[i][editable_list])
                    Qs_.append(Qs[i][editable_list])
                    Vs_.append(Vs[i])
                    rewards_.append(rewards[i])
            loss = self._calculate_loss(actions_, policies_, old_policies_, Qs_, Vs_, rewards_)
            self._step_optimizer(optimizer, loss)

    def _update_deeper(self, optimizer, actions, policies, old_policies, Qs, Vs, rewards, states, selected_ops):
        deeper_ops = torch.tensor(selected_ops) == SelectorActorNet.DEEPER
        if deeper_ops.any():
            loss_type = self._calculate_loss(  # type
                [i[0] for i in actions],
                [i[0] for i in policies],
                [i[0] for i in old_policies],
                [i[0] for i in Qs],
                [i[0] for i in Vs],
                rewards)
            loss_index = self._calculate_loss(  # index
                [i[1] for i in actions],
                [i[1] for i in policies],
                [i[1] for i in old_policies],
                [i[1] for i in Qs],
                [i[1] for i in Vs],
                rewards)
            loss = loss_type + loss_index
            self._step_optimizer(optimizer, loss)


def action_describe(cfg, action):
    """
    return a readable action description string
    :param cfg: global configuration
    :param action: action detail information
    :return: readable action description string
    """
    describe = '\n' + 'Action'.center(50, '-') + '\n'
    select = action['select'].item()
    if select == 0:  # do nothing
        describe += f'action:\t0 do noting\n'
    if select == 1:  # wider the net
        describe += f'action:\t1 wider\n'
        describe += f'wider place:\t1 wider at layer {action["wider"].item()}\n'
    if select == 2:  # deeper the net
        describe += f'action:\t2 deeper\n'
        insert_type, insert_index = action["deeper"]
        insert_type = cfg.NASConfig['editable'][insert_type]
        describe += f'deeper place:\t{insert_index.item()}\n'
        describe += f'deeper insert type:\t{insert_type}\n'
    if select == 3:  # deeper the net
        describe += f'action:\t3 prune\n'
        describe += f'prune ratio:\t{cfg.NASConfig["PruningRatio"]}\n'

    describe += f'net index:\t{action["net_index"]}\n'
    describe += 'End'.center(50, '-') + '\n'
    return describe
