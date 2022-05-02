import abc
import random

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init, Module

from ..utils.logger import get_logger


class NasModule:
    def __init__(self, cfg, name, input_shape):
        """
        basic module composing neural networks.
        :param cfg: global configuration
        :param name: module name
        :param input_shape: input data shape
        """
        self.logger = get_logger('module', cfg.LOG_FILE)
        self.cfg = cfg
        self.input_shape = input_shape  # (channel, feature)
        self.name = name
        self.output_shape = None
        self.params = None
        self.widenable = self.cfg.modulesConfig[name]['editable']
        self._module_instance = None  # buffer the instance
        self.current_level = None  # current width level
        self.widen_sample_fn = self.default_sample_strategy

    def __call__(self, x, *args, **kwargs):
        return self._module_instance(x)

    @property
    @abc.abstractmethod
    def is_max_level(self):
        """
        :return: True if this module is max width level,False otherwise
        """
        pass

    @property
    @abc.abstractmethod
    def next_level(self) -> int:
        """
        :return: width of next level
        """
        pass

    def get_level(self, level_list) -> int:
        """
        determine init level, return small level
        :param level_list: all available level list
        :return: level
        """
        return random.sample(level_list[:4], 1)[0]

    @abc.abstractmethod
    def init_param(self, input_shape):
        """
        initialize parameters of this module
        :return: None
        """
        pass

    def on_param_end(self, input_shape):
        """
        common operation after parameter setting ending, should call at
        the end of each subclass's init_param method
        :param input_shape: input shape of this module
        """
        # get output shape
        m = self.get_module_instance()
        input_data = torch.zeros([1, *input_shape])
        output = m(input_data)
        self.output_shape = output.shape[1:]
        self.input_shape = tuple(self.input_shape)
        self.output_shape = tuple(self.output_shape)

    @abc.abstractmethod
    def identity_module(self, cfg, name, input_shape: tuple):
        """
        generate an identity mapping module
        :rtype: NasModule
        :param name: type of identity module:[dense, conv, rnn]
        :return: identity module (of class Modules)
        """
        pass

    @abc.abstractmethod
    def get_module_instance(self) -> Module:
        """
        generate a model instance once and use it for the rest of procedure
        :return: self._module_instance
        """
        pass

    @property
    @abc.abstractmethod
    def token(self) -> str:
        """
        :return: string type toke of this module
        """
        pass

    def __str__(self):
        return f'{self.name}-{self.input_shape}-{self.output_shape}-{self.params}'

    @abc.abstractmethod
    def perform_wider_transformation_current(self) -> (list, list):
        """
        generate a new wider module by the wider FPT(function preserving transformation)
        :return: mapping_g, scale_g
        """
        pass

    @abc.abstractmethod
    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        """
        generate a new wider module by the wider FPT(function preserving transformation)
        :return: module of next level
        """
        pass

    @staticmethod
    def default_sample_strategy(original_size, new_size):
        """
        default sample strategy described in paper
        :param original_size: original layer width
        :param new_size: new layer width, we have new_size >original_size
        :return: sampling sequence
        """
        seq = list(range(original_size))
        num_to_sample = new_size - original_size
        while num_to_sample != 0:
            sample_number = min(original_size, num_to_sample)
            seq += random.sample(seq, sample_number)
            num_to_sample = num_to_sample - sample_number
        return seq


class RNNModule(NasModule):

    @property
    def is_max_level(self):
        out_range = self.cfg.modulesConfig['rnn']['out_range']
        return self.current_level >= max(out_range)

    @property
    def next_level(self):
        out_range = self.cfg.modulesConfig['rnn']['out_range']
        out_range = list(out_range)
        valid_range = []
        for i in out_range:
            if i > self.current_level:
                valid_range.append(i)
        return min(valid_range)

    def init_param(self, input_shape):
        assert len(input_shape) == 2
        self.params = dict()
        out_range = self.cfg.modulesConfig['rnn']['out_range']
        self.current_level = self.get_level(out_range)
        self.params['input_size'] = input_shape[0]
        self.params['output_size'] = self.current_level
        # get output shape
        m = self.get_module_instance()
        input_data = torch.zeros([1, *input_shape])
        output = m(input_data)
        self.output_shape = output.shape[1:]
        self.input_shape = tuple(self.input_shape)
        self.output_shape = tuple(self.output_shape)
        self.on_param_end(input_shape)

    def identity_module(self, cfg, name, input_shape: tuple):
        if type(name) != str:
            name = cfg.NASConfig['editable'][name]
        module = RNNModule(cfg, name, input_shape)
        output_size = input_shape[0]
        input_size = input_shape[0]
        module.params = {
            "output_size": output_size,
            "input_size": input_size,
        }
        module.current_level = output_size
        module.output_shape = input_shape
        rnn = NAS_RNN(
            input_size=input_size,
            hidden_size=output_size,
            nonlinearity='relu',
            batch_first=True,
        )
        bias_ih_l0 = torch.zeros(output_size)
        rnn.rnn_unit.bias_ih_l0 = nn.Parameter(bias_ih_l0, True)
        bias_hh_l0 = torch.zeros(output_size)
        rnn.rnn_unit.bias_hh_l0 = nn.Parameter(bias_hh_l0, True)
        weight_ih_l0 = torch.eye(output_size)
        rnn.rnn_unit.weight_ih_l0 = nn.Parameter(weight_ih_l0, True)
        weight_hh_l0 = torch.zeros((output_size, output_size))
        rnn.rnn_unit.weight_hh_l0 = nn.Parameter(weight_hh_l0, True)
        module._module_instance = rnn
        return module

    def get_module_instance(self):
        if self._module_instance is not None:
            return self._module_instance
        if self.params is None and self.output_shape is None:
            raise ValueError("parameter must be initialized before generate module instance!!!")
        self._module_instance = NAS_RNN(
            input_size=self.params['input_size'],
            hidden_size=self.params['output_size'],
            nonlinearity='relu',
            batch_first=True,
        )
        return self._module_instance

    @property
    def token(self):
        return 'rnn-%d' % (self.params['output_size'])

    def perform_wider_transformation_current(self):
        next_level = self.next_level
        new_module_instance = NAS_RNN(
            input_size=self.params['input_size'],
            hidden_size=next_level,
            nonlinearity='relu',
            batch_first=True,
        )
        # keep previous parameters
        mapping_g = self.widen_sample_fn(self.current_level, next_level)
        scale_g = [1 / mapping_g.count(i) for i in mapping_g]
        scale_g = torch.tensor(scale_g)
        rnn = new_module_instance.rnn_unit
        rnn.bias_ih_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_ih_l0[mapping_g], True)
        rnn.bias_hh_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_hh_l0[mapping_g], True)
        rnn.weight_ih_l0 = nn.Parameter(self._module_instance.rnn_unit.weight_ih_l0[mapping_g], True)
        rnn.weight_hh_l0 = nn.Parameter(
            (self._module_instance.rnn_unit.weight_hh_l0[:, mapping_g] * scale_g)[mapping_g], True)
        self.current_level = next_level
        self._module_instance = new_module_instance
        self.output_shape = (self.current_level, self.output_shape[1])
        self.params['output_size'] = self.current_level
        return mapping_g, scale_g

    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        next_level = len(mapping_g)
        scale_g = torch.tensor(scale_g)
        new_module_instance = NAS_RNN(
            input_size=next_level,
            hidden_size=self.params['output_size'],
            nonlinearity='relu',
            batch_first=True,
        )
        rnn = new_module_instance.rnn_unit
        rnn.bias_ih_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_ih_l0, True)
        rnn.bias_hh_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_hh_l0, True)
        rnn.weight_ih_l0 = \
            torch.nn.Parameter(self._module_instance.rnn_unit.weight_ih_l0[:, mapping_g] *
                               scale_g.unsqueeze(0),
                               requires_grad=True)
        rnn.weight_hh_l0 = \
            torch.nn.Parameter(self._module_instance.rnn_unit.weight_hh_l0,
                               requires_grad=True)
        self._module_instance = new_module_instance
        self.input_shape = (next_level, self.input_shape[1])
        self.params['input_size'] = next_level


class DenseModule(NasModule):
    @property
    def is_max_level(self):
        out_range = self.cfg.modulesConfig['dense']['out_range']
        return self.current_level >= max(out_range)

    @property
    def next_level(self):
        out_range = self.cfg.modulesConfig['dense']['out_range']
        out_range = list(out_range)
        valid_range = []
        for i in out_range:
            if i > self.current_level:
                valid_range.append(i)
        return min(valid_range)

    def init_param(self, input_shape):
        assert len(input_shape) == 2
        out_range = self.cfg.modulesConfig['dense']['out_range']
        in_features = input_shape[1]
        self.current_level = self.get_level(out_range)
        self.params = {'in_features': in_features,
                       "out_features": self.current_level}
        self.on_param_end(input_shape)

    def identity_module(self, cfg, name, input_shape: tuple):
        if type(name) != str:
            name = cfg.NASConfig['editable'][name]
        module = DenseModule(cfg, name, input_shape)
        dense = nn.Linear(input_shape[-1], input_shape[-1], bias=True)
        dense.weight = nn.Parameter(torch.eye(input_shape[-1]))
        dense.bias = nn.Parameter(torch.zeros(input_shape[-1]))
        module.current_level = input_shape[1]
        module.output_shape = input_shape
        module.params = {'in_features': input_shape[1], 'out_features': input_shape[1]}
        module._module_instance = dense
        return module

    def get_module_instance(self):
        if self._module_instance is not None:
            return self._module_instance
        self._module_instance = nn.Linear(self.params['in_features'], self.params['out_features'], bias=True)
        return self._module_instance

    @property
    def token(self):
        return 'dense-%d' % (self.params['out_features'])

    def perform_wider_transformation_current(self):
        next_level = self.next_level
        new_module_instance = nn.Linear(self.params['in_features'], next_level)
        # keep previous parameters
        mapping_g = self.widen_sample_fn(self.current_level, next_level)
        scale_g = [1 / mapping_g.count(i) for i in mapping_g]
        new_module_instance.bias = torch.nn.Parameter(self._module_instance.bias[mapping_g], requires_grad=True)
        new_module_instance.weight = torch.nn.Parameter(self._module_instance.weight[mapping_g], requires_grad=True)
        self.current_level = next_level
        self._module_instance = new_module_instance
        self.output_shape = (self.output_shape[0], self.current_level)
        self.params['out_features'] = self.current_level
        return mapping_g, scale_g

    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        next_level = len(mapping_g)
        scale_g = torch.tensor(scale_g)
        new_module_instance = nn.Linear(next_level, self.params['out_features'])
        # keep previous parameters
        new_module_instance.weight = torch.nn.Parameter(
            self._module_instance.weight[:, mapping_g] * scale_g.unsqueeze(0), requires_grad=True)
        self._module_instance = new_module_instance
        self.input_shape = (self.input_shape[0], next_level)
        self.params['in_features'] = next_level


class ConvModule(NasModule):
    @property
    def is_max_level(self):
        out_range = self.cfg.modulesConfig['conv']['out_range']
        return self.current_level >= max(out_range)

    @property
    def next_level(self):
        out_range = self.cfg.modulesConfig['conv']['out_range']
        out_range = list(out_range)
        valid_range = []
        for i in out_range:
            if i > self.current_level:
                valid_range.append(i)
        return min(valid_range)

    def init_param(self, input_shape):
        assert len(input_shape) == 2
        self.params = {'in_channels': input_shape[0]}
        out_range = self.cfg.modulesConfig['conv']['out_range']
        self.current_level = self.get_level(out_range)
        self.params['out_channels'] = self.current_level
        self.params['kernel_size'] = 3
        self.params['stride'] = 1
        self.params['padding'] = 1

        self.on_param_end(input_shape)

    def identity_module(self, cfg, name, input_shape: tuple):
        if type(name) != str:
            name = cfg.NASConfig['editable'][name]
        module = ConvModule(cfg, name, input_shape)
        out_channel = input_shape[0]
        in_channel = input_shape[0]
        kernel_size = 3
        module.current_level = input_shape[0]
        module.output_shape = input_shape
        module.params = {
            "in_channels": in_channel,
            "out_channels": out_channel,
            "kernel_size": kernel_size,
            "stride": 1,
            "padding": 1,
        }
        conv = nn.Conv1d(in_channels=module.params['in_channels'],
                         out_channels=module.params['out_channels'],
                         kernel_size=module.params['kernel_size'],
                         stride=module.params['stride'],
                         padding=module.params['padding'])
        weight = torch.zeros((out_channel, in_channel, kernel_size))
        for i in range(in_channel):
            weight[i, i, 1] = 1
        conv.weight = nn.Parameter(weight)
        bias = torch.zeros(out_channel)
        conv.bias = nn.Parameter(bias)
        module._module_instance = conv
        return module

    def get_module_instance(self):
        if self._module_instance is not None:
            return self._module_instance
        self._module_instance = nn.Conv1d(
            in_channels=self.params['in_channels'],
            out_channels=self.params['out_channels'],
            kernel_size=self.params['kernel_size'],
            stride=self.params['stride'],
            padding=self.params['padding'])
        return self._module_instance

    @property
    def token(self):
        return 'conv-%d' % (self.params['out_channels'])

    def perform_wider_transformation_current(self):
        next_level = self.next_level
        new_module_instance = nn.Conv1d(in_channels=self.params['in_channels'],
                                        out_channels=self.next_level,
                                        kernel_size=self.params['kernel_size'],
                                        stride=self.params['stride'], padding=self.params['padding'])
        # keep previous parameters
        mapping_g = self.widen_sample_fn(self.current_level, next_level)
        scale_g = [1 / mapping_g.count(i) for i in mapping_g]
        new_module_instance.bias = torch.nn.Parameter(self._module_instance.bias[mapping_g], requires_grad=True)
        new_module_instance.weight = torch.nn.Parameter(self._module_instance.weight[mapping_g], requires_grad=True)
        self.current_level = next_level
        self._module_instance = new_module_instance
        self.output_shape = (self.current_level, self.output_shape[1])
        self.params['out_channels'] = self.current_level
        return mapping_g, scale_g

    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        next_level = len(mapping_g)
        scale_g = torch.tensor(scale_g)
        new_module_instance = nn.Conv1d(in_channels=next_level,
                                        out_channels=self.params['out_channels'],
                                        kernel_size=self.params['kernel_size'],
                                        stride=self.params['stride'], padding=self.params['padding'])
        new_module_instance.weight = \
            torch.nn.Parameter(self._module_instance.weight[:, mapping_g] *
                               scale_g.unsqueeze(0).unsqueeze(2),
                               requires_grad=True)
        self._module_instance = new_module_instance
        self.input_shape = (next_level, self.input_shape[1])
        self.params['in_channels'] = next_level


class LSTMModule(NasModule):
    @property
    def is_max_level(self):
        out_range = self.cfg.modulesConfig['lstm']['out_range']
        return self.current_level >= max(out_range)

    @property
    def next_level(self):
        out_range = self.cfg.modulesConfig['lstm']['out_range']
        out_range = list(out_range)
        valid_range = []
        for i in out_range:
            if i > self.current_level:
                valid_range.append(i)
        return min(valid_range)

    def init_param(self, input_shape):
        assert len(input_shape) == 2
        self.params = dict()
        out_range = self.cfg.modulesConfig['lstm']['out_range']
        self.current_level = self.get_level(out_range)
        self.params['input_size'] = input_shape[0]
        self.params['output_size'] = self.current_level
        self.on_param_end(input_shape)

    def identity_module(self, cfg, name, input_shape: tuple):
        if type(name) != str:
            name = cfg.NASConfig['editable'][name]
        module = LSTMModule(cfg, name, input_shape)
        hidden_size = input_shape[0]
        input_size = input_shape[0]
        module.params = {
            "hidden_size": hidden_size,
            "input_size": input_size,
        }
        module.current_level = hidden_size
        module.output_shape = input_shape
        identity_lstm = NAS_SLSTM.generate_identity(input_size)
        module._module_instance = identity_lstm
        return module

    def get_module_instance(self):
        if self._module_instance is not None:
            return self._module_instance
        self._module_instance = NAS_SLSTM(
            input_size=self.params['input_size'],
            hidden_size=self.params['output_size'],
        )
        return self._module_instance

    @property
    def token(self):
        return 'lstm-%d' % (self.params['output_size'])

    def perform_wider_transformation_current(self):
        next_level = self.next_level
        new_module_instance = NAS_SLSTM(self.params['input_size'], next_level)
        # keep previous parameters
        mapping_g = self.widen_sample_fn(self.current_level, next_level)
        scale_g = [1 / mapping_g.count(i) for i in mapping_g]
        scale_g = torch.tensor(scale_g)
        lstm = new_module_instance
        lstm.Wf = nn.Parameter(self._module_instance.Wf[:, mapping_g])
        lstm.Wi = nn.Parameter(self._module_instance.Wi[:, mapping_g])
        lstm.Wo = nn.Parameter(self._module_instance.Wo[:, mapping_g])
        lstm.Wc = nn.Parameter(self._module_instance.Wc[:, mapping_g])
        lstm.bf = nn.Parameter(self._module_instance.bf[mapping_g])
        lstm.bi = nn.Parameter(self._module_instance.bi[mapping_g])
        lstm.bo = nn.Parameter(self._module_instance.bo[mapping_g])
        lstm.bc = nn.Parameter(self._module_instance.bc[mapping_g])
        lstm.Uf = nn.Parameter((self._module_instance.Uf.T[:, mapping_g] * scale_g)[mapping_g].T)
        lstm.Ui = nn.Parameter((self._module_instance.Ui.T[:, mapping_g] * scale_g)[mapping_g].T)
        lstm.Uo = nn.Parameter((self._module_instance.Uo.T[:, mapping_g] * scale_g)[mapping_g].T)
        lstm.Uc = nn.Parameter((self._module_instance.Uc.T[:, mapping_g] * scale_g)[mapping_g].T)
        self.current_level = next_level
        self._module_instance = new_module_instance
        self.output_shape = (self.current_level, self.output_shape[1])
        self.params['output_size'] = self.current_level
        return mapping_g, scale_g

    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        next_level = len(mapping_g)
        scale_g = torch.tensor(scale_g)
        new_module_instance = NAS_SLSTM(next_level, self.params['output_size'])
        lstm = new_module_instance
        lstm.Wf = nn.Parameter(self._module_instance.Wf[mapping_g] * scale_g.unsqueeze(1))
        lstm.Wi = nn.Parameter(self._module_instance.Wi[mapping_g] * scale_g.unsqueeze(1))
        lstm.Wo = nn.Parameter(self._module_instance.Wo[mapping_g] * scale_g.unsqueeze(1))
        lstm.Wc = nn.Parameter(self._module_instance.Wc[mapping_g] * scale_g.unsqueeze(1))
        lstm.bf = nn.Parameter(self._module_instance.bf)
        lstm.bi = nn.Parameter(self._module_instance.bi)
        lstm.bo = nn.Parameter(self._module_instance.bo)
        lstm.bc = nn.Parameter(self._module_instance.bc)
        lstm.Uf = nn.Parameter(self._module_instance.Uf)
        lstm.Ui = nn.Parameter(self._module_instance.Ui)
        lstm.Uo = nn.Parameter(self._module_instance.Uo)
        lstm.Uc = nn.Parameter(self._module_instance.Uc)
        self._module_instance = new_module_instance
        self.input_shape = (next_level, self.input_shape[1])
        self.params['input_size'] = next_level


class NAS_RNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # kwargs['bias'] = False
        self.rnn_unit = nn.RNN(*args, **kwargs)

    def forward(self, x: Tensor):
        x = x.permute(0, 2, 1)
        output, hn = self.rnn_unit(x)
        y = output
        y = y.permute(0, 2, 1)
        return y


class NAS_SLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.Wf = nn.Parameter(torch.empty(input_size, hidden_size))
        self.Wi = nn.Parameter(torch.empty(input_size, hidden_size))
        self.Wc = nn.Parameter(torch.empty(input_size, hidden_size))
        self.Wo = nn.Parameter(torch.empty(input_size, hidden_size))
        self.Uf = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Ui = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Uc = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Uo = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.zeros(hidden_size))
        self.bi = nn.Parameter(torch.zeros(hidden_size))
        self.bc = nn.Parameter(torch.zeros(hidden_size))
        self.bo = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.Wf)
        init.kaiming_uniform_(self.Wi)
        init.kaiming_uniform_(self.Wc)
        init.kaiming_uniform_(self.Wo)
        init.kaiming_uniform_(self.Uf)
        init.kaiming_uniform_(self.Ui)
        init.kaiming_uniform_(self.Uc)
        init.kaiming_uniform_(self.Uo)

    def forward_step(self, x, h, c):
        """
        do one step LSTM forward calculation
        @param x: input shape [batch, feature]
        """
        ft = torch.sigmoid(torch.matmul(x, self.Wf) + torch.matmul(h, self.Uf) + self.bf)
        it = torch.sigmoid(torch.matmul(x, self.Wi) + torch.matmul(h, self.Ui) + self.bi)
        ot = torch.matmul(x, self.Wo) + torch.matmul(h, self.Uo) + self.bo
        c_ = torch.tanh(torch.matmul(x, self.Wc) + torch.matmul(h, self.Uc) + self.bc)
        c = ft * c + it * c_
        h = ot * torch.tanh(c) + ot
        return h, c

    def forward(self, x):
        '''
        forward propagation
        @param x:  input of size [batch, feature, seqlen]
        @return: hidden state (batch, feature,seqlen)
        '''
        h = torch.zeros((x.shape[0], self.hidden_size), device=x.device)
        c = torch.zeros((x.shape[0], self.hidden_size), device=x.device)
        output = []
        for i in range(x.shape[2]):
            h, c = self.forward_step(x[:, :, i], h, c)
            output.append((h, c))
        output = [torch.stack([i[0] for i in output]), torch.stack([i[1] for i in output])]
        return output[0].permute(1, 2, 0)

    @staticmethod
    def generate_identity(input_size):
        lstm = NAS_SLSTM(input_size, input_size)
        lstm.Wo = nn.Parameter(torch.eye(input_size))
        lstm.Wf = nn.Parameter(torch.zeros_like(lstm.Wf))
        lstm.Wi = nn.Parameter(torch.zeros_like(lstm.Wi))
        lstm.Wc = nn.Parameter(torch.zeros_like(lstm.Wc))
        lstm.Uf = nn.Parameter(torch.zeros_like(lstm.Uf))
        lstm.Uc = nn.Parameter(torch.zeros_like(lstm.Uc))
        lstm.Ui = nn.Parameter(torch.zeros_like(lstm.Ui))
        lstm.Uo = nn.Parameter(torch.zeros_like(lstm.Uo))
        lstm.bf = nn.Parameter(torch.zeros_like(lstm.bf))
        lstm.bc = nn.Parameter(torch.zeros_like(lstm.bc))
        lstm.bi = nn.Parameter(torch.zeros_like(lstm.bi))
        lstm.bo = nn.Parameter(torch.zeros_like(lstm.bo))
        return lstm


def generate_module(cfg, name, input_shape) -> NasModule:
    """
    generate module instance, this function binding module name to exact module class
    :param cfg: global configuration
    :param name: name of module
    :param input_shape: input tensor shape
    :return: subclass of NAS_Module corresponding to name
    """
    if name in cfg.modulesCls:
        return cfg.modulesCls[name](cfg, name, input_shape)
    else:
        raise RuntimeError(f'no such module {name}, please if the module is not registered')


def generate_from_skeleton(cfg, skeleton: list, input_shape):
    """
    generate a list of module instance from skeleton list
    :param cfg: global configuration
    :param skeleton: skeleton list of string specifying each layer's type
    :param input_shape: input data shape
    :return: list of modules
    """
    modules = []
    for name in skeleton + ['dense']:
        module = generate_module(cfg, name, input_shape)
        try:
            module.init_param(input_shape)
            input_shape = module.output_shape
            modules.append(module)
        except Exception as e:
            logger = get_logger('module init', cfg.LOG_FILE)
            logger.error(f'fail to init module {module},\n'
                         f' {"message".center(50, "=")}\n'
                         f'{e}'
                         f'{"end".center(50, "=")}')
    return modules
