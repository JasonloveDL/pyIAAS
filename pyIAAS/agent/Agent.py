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

    def get_action(self, net_pool):
        """
        make decision on each network in net-pool.
        :param net_pool: net-pool list, each network is an instance of NasModel
        :return: action of each model
        """
        action, a_prob = [], []
        if self.cfg.NASConfig['GPU']:
            self.to_cuda()
        for net in net_pool:
            output, (h_n, c_n) = self.encoder_net.forward(net.model_config.token_list)
            select_action, select_prob = self.selector.get_action(h_n, net.model_config.can_widen)
            deeper_action, deeper_prob = self.deeper_net.get_action(h_n, net.model_config.insert_length)
            if net.model_config.can_widen:
                wider_action, wider_prob = self.wider_net.get_action(output, net.model_config.widenable_list)
            else:
                wider_action, wider_prob = None, None
            action.append({
                'select': select_action,
                'wider': wider_action,
                'deeper': deeper_action,
                'net_index': net.index
            })

            a_prob.append({
                'select': select_prob,
                'wider': wider_prob,
                'deeper': deeper_prob,
            })
        if self.cfg.NASConfig['GPU']:
            self.to_cpu()

        return {'action': action, 'prob': a_prob}

    def get_log_prob_entropy(self, a_int, a_prob):
        if self.cfg.NASConfig['GPU']:
            a_int = a_int.cuda()
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)

    def update(self, reward, action, net_pool):
        """
        update agent actor networks by each reward comes from previous action
        :param reward: reward information, create by environment
        :param action: previous action
        :param net_pool: net-pool list
        """
        length = len(reward)
        if self.cfg.NASConfig['GPU']:
            self.to_cuda()
        for i in range(length):
            # calculate immediate net output
            net = net_pool[i]
            action_index = None
            for j in range(len(action['action'])):
                if action['action'][j]['net_index'] == net.index:
                    action_index = j
            if action_index is None or action_index >= length:
                continue
            self.logger.info(action_describe(self.cfg, action["action"][action_index]))
            output, (h_n, c_n) = self.encoder_net.forward(net.model_config.token_list)
            select_action, select_prob = self.selector.get_action(h_n, net.model_config.can_widen)
            deeper_action, deeper_prob = self.deeper_net.get_action(h_n, net.model_config.insert_length)

            # update selector net
            action_onehot = torch.zeros(3)
            action_onehot[action['action'][action_index]['select']] = 1
            action_prob = select_prob
            self.update_one_subnet(self.selector_optimizer, action_onehot, action_prob, reward[action_index])

            # update wider net
            if action['prob'][action_index]['wider'] is not None and net.model_config.can_widen:
                wider_action, wider_prob = self.wider_net.get_action(output, net.model_config.widenable_list)
                action_onehot = torch.zeros(action['prob'][action_index]['wider'].shape[0])
                action_onehot[action['action'][action_index]['wider']] = 1
                action_prob = wider_prob
                self.update_one_subnet(self.wider_optimizer, action_onehot, action_prob, reward[action_index])

            # update deeper net type module
            action_onehot = torch.zeros(len(self.cfg.NASConfig['editable']))
            action_onehot[action['action'][action_index]['deeper'][0]] = 1
            action_prob = deeper_prob[0]
            prob = self.get_log_prob_entropy(action_onehot, action_prob)
            type_loss = (- prob * reward[action_index]).sum()

            # update deeper net index module
            action_onehot = torch.zeros(self.cfg.NASConfig['MaxLayers'])
            action_onehot[action['action'][action_index]['deeper'][1]] = 1
            action_prob = deeper_prob[1]
            prob = self.get_log_prob_entropy(action_onehot, action_prob)
            index_loss = (- prob * reward[action_index]).sum()

            # update deeper net by type and index loss
            deeper_loss = type_loss + index_loss
            self.deeper_optimizer.zero_grad()
            deeper_loss.backward()
            self.deeper_optimizer.step()
        if self.cfg.NASConfig['GPU']:
            self.to_cpu()

    def update_one_subnet(self, optimizer, onehot_action, action_prob, reward):
        """
        update one actor, all these actors can be updated in same scheme
        :param optimizer: optimizer of this actor
        :param onehot_action: one hot form of action
        :param action_prob: action probability
        :param reward: reward information to update
        """
        prob = self.get_log_prob_entropy(onehot_action, action_prob)
        policy_loss = - prob * reward
        policy_loss = policy_loss.mean()
        optimizer.zero_grad()
        policy_loss.backward(create_graph=True)
        optimizer.step()


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

    describe += f'net index:\t{action["net_index"]}\n'
    describe += 'End'.center(50, '-') + '\n'
    return describe
