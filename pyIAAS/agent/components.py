import torch
from torch.distributions import Categorical
from torch.nn import *

_vocabulary = None  # save vocabulary for the whole NAS architecture, used for embedding
_hidden_size = 256

class Vocabulary:

    def __init__(self, token_list):
        """
        store embedding code of each NAS module
        """
        token_list = token_list
        self.vocab = {}
        for idx, token in enumerate(token_list):
            self.vocab[token] = idx
            self.vocab[idx] = token

    @property
    def size(self):
        return len(self.vocab) // 2

    def get_code(self, token_list):
        return [self.vocab[token] for token in token_list]

    def get_token(self, code_list):
        return [self.vocab[code] for code in code_list]

    def __str__(self):
        return str(self.vocab)


def get_vocabulary(modules_config):
    """
    create vocabulary for all search module.
    :param modules_config: modules information, dict
    :return: vocabulary of search modules.
    """
    global _vocabulary
    if _vocabulary is not None:
        return _vocabulary
    token_expand = 10  # give large enough token space
    token_list = []
    for k in modules_config.keys():
        token_list += [f'{k}-%d' % i for i in range(1, token_expand * max(modules_config[k]['out_range']) + 1)]
    _vocabulary = Vocabulary(token_list)
    return _vocabulary


class EncoderNet(Module):
    """
    Encoder network similar to EAS
    """

    def __init__(self, input_size, hidden_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.lstm_unit = torch.nn.LSTM(input_size, hidden_size, 1, True, True, bidirectional=True)
        self.embedding_unit = torch.nn.Embedding(get_vocabulary(self.cfg.modulesConfig).size, input_size)
        if self.cfg.NASConfig['GPU']:
            self.embedding_unit = self.embedding_unit.cuda()
            self.lstm_unit = self.lstm_unit.cuda()

    def embedding(self, token_list):
        vocab = get_vocabulary(self.cfg.modulesConfig)
        codes = vocab.get_code(token_list)
        input_tensor = torch.tensor(codes)
        if self.cfg.NASConfig['GPU']:
            input_tensor = input_tensor.cuda()
        out = self.embedding_unit(input_tensor)

        return out.unsqueeze(0)

    def forward(self, token_list):
        """
        get encoder output
        :param token_list: list of token: list of str
        :return: output, (h_n, c_n) as described in lstm
        """
        embedding_tensor = self.embedding(token_list)
        output, (h_n, c_n) = self.lstm_unit(embedding_tensor)
        return output, (h_n, c_n)


class WinderActorNet(Module):
    def __init__(self, cfg, input_size: int):
        """
        wider actor to do widening action
        :param cfg: global configuration
        :param input_size: input data size, determined by encoder output size
        """
        super().__init__()
        self.cfg = cfg
        self.actor = torch.nn.Sequential(
            Linear(input_size, _hidden_size),
            LeakyReLU(),
            Linear(_hidden_size, 1),
        )
        self.critic = Sequential(
            Linear(input_size, _hidden_size),
            LeakyReLU(),
            Linear(_hidden_size, 1),
        )

    def forward(self, features, editable):
        policy = self.actor(features.squeeze(0)).squeeze()
        # filter layers which cannot be widened
        mask = torch.tensor(list(map(lambda x: 1 if x else 0, editable)), device=features.device)
        policy = torch.softmax(policy, 0) * mask
        policy = policy / (policy.sum() + 1e-9)
        Q = torch.softmax(self.critic(features.squeeze()).squeeze(), 0) * mask
        V = (Q * policy).sum()
        return policy, Q, V

    def get_action(self, features, editable):
        """
        get output of wider actor network
        :param features: hidden states of each network module
        :param editable: indicating if a layer can be widened
        :return: index of layer to be widened
        """
        policy, Q, V = self.forward(features, editable)
        action = Categorical(policy).sample()
        return action, policy, Q, V


class DeeperActorNet(Module):
    def __init__(self, cfg, input_size: int, max_layers):
        """
        deeper net actor net for make decision on deeper the original neural network
        :param cfg: global configuration
        :param input_size: hidden size of encoder network (including bidirectional enlargement of size)
        :param max_layers: maximum layer number of neural network
        """
        super().__init__()
        self.cfg = cfg
        self.max_layers = max_layers
        self.decision_num = 2
        self.deeperNet = RNN(input_size, input_size, batch_first=True)
        self.insert_type_layer = Sequential(
            Linear(input_size, _hidden_size),
            LeakyReLU(),
            Linear(_hidden_size, len(self.cfg.NASConfig['editable'])),
        )
        self.insert_type_critic = Sequential(
            Linear(input_size, _hidden_size),
            LeakyReLU(),
            Linear(_hidden_size, len(self.cfg.NASConfig['editable'])),
        )
        self.insert_index_layer = Sequential(
            Linear(input_size, _hidden_size),
            LeakyReLU(),
            Linear(_hidden_size, max_layers),
        )
        self.insert_index_critic = Sequential(
            Linear(input_size, _hidden_size),
            LeakyReLU(),
            Linear(_hidden_size, max_layers),
        )


    def get_action(self, hn_feature: torch.Tensor, insert_length: int):
        """
        get output of deeper actor network
        :param hn_feature: last hidden states of encoder network
        :param insert_length: max index of the inserted neural layer
        :return: (insert_type, insert_index), (insert_type_policy, insert_index_policy),(insert_type_Q, insert_index_Q), (insert_type_V, insert_index_V)
        """
        hn_feature = hn_feature.reshape(1, -1)
        hn_feature = torch.stack([hn_feature] * self.decision_num, 1)
        output, _ = self.deeperNet(hn_feature)
        insert_type_policy = torch.softmax(self.insert_type_layer(output[-1, 0, :]), dim=0)
        insert_type = Categorical(insert_type_policy).sample()
        insert_index_policy = self.insert_index_layer(output[:, 1, :])[0][:insert_length]
        insert_index_policy = torch.softmax(insert_index_policy, 0)
        insert_index = Categorical(insert_index_policy).sample()
        insert_type_Q, insert_index_Q = self.insert_type_critic(output[-1, 0, :]), \
                                        self.insert_index_critic(output[:, 1, :])[0][:insert_length]
        insert_type_V, insert_index_V = (insert_type_Q * insert_type_policy).sum(), \
                                        (insert_index_Q * insert_index_policy).sum()
        return (insert_type, insert_index), (insert_type_policy, insert_index_policy), (
        insert_type_Q, insert_index_Q), (insert_type_V, insert_index_V)


class SelectorActorNet(Module):
    option_number = 4
    UNCHANGE = 0
    WIDER = 1
    DEEPER = 2
    PRUNE = 3

    def __init__(self, cfg, input_size):
        """
        selector network, output is 4 dimension:
        4 dim:
        (do nothing, wider, deeper, prune)
        :param cfg: global configuration
        :param input_size: input data size, determined by encoder output size
        """
        super().__init__()
        self.cfg = cfg
        self.actor = Sequential(
            Linear(input_size, _hidden_size),
            LeakyReLU(),
            Linear(_hidden_size, self.option_number),
        )
        self.critic = Sequential(
            Linear(input_size, _hidden_size),
            LeakyReLU(),
            Linear(_hidden_size, self.option_number),
        )

    def forward(self, x):
        x = torch.flatten(x)
        policy = self.actor(x)
        policy = torch.softmax(policy, 0)
        Q = self.critic(x)
        V = (Q * policy).sum()  # V is expectation of Q under policy π
        return policy, Q, V

    def get_action(self, x):
        policy, Q, V = self.forward(x)
        action = Categorical(policy).sample()
        return action, policy, Q, V
