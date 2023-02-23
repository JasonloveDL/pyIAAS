import abc
import math
import random
from typing import Iterator

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init, Module, Parameter

from ..utils.logger import get_logger


class NasModule(nn.Module):
    def __init__(self, cfg, name: str, input_shape):
        """
        basic module composing neural networks.
        :param cfg: global configuration
        :param name: module name
        :param input_shape: input data shape
        """
        super().__init__()
        self.logger = get_logger('module', cfg.LOG_FILE)
        self.cfg = cfg
        self.input_shape = input_shape  # (channel, feature)
        self.name = name
        self.output_shape = None
        self.params = None
        self.editable = self.cfg.modulesConfig[name]['editable']
        self._module_instance = None  # buffer the instance
        self.current_level = None  # current width level
        self.widen_sample_fn = self.default_sample_strategy
        # add pruning field
        if cfg.NASConfig['Pruning']:
            self.importance_score = None  # importance score matrix

    @property
    @abc.abstractmethod
    def is_max_level(self):
        """
        :return: True if this module is max width level,False otherwise
        """
        pass

    @property
    def next_level(self) -> int:
        """
        :return: width of next level
        """
        out_range = self.cfg.modulesConfig[self.name]['out_range']
        out_range = list(out_range)
        valid_range = []
        for i in out_range:
            if i > self.current_level:
                valid_range.append(i)
        if len(valid_range) == 0:
            return self.current_level
        else:
            return min(valid_range)

    def get_level(self, level_list, max_width) -> int:
        """
        determine init level, return random level
        :param max_width:
        :param level_list: all available level list
        :return: level
        """
        if max_width is not None:
            level_list = [i for i in level_list if i <= max_width]
        return random.sample(level_list, 1)[0]

    @abc.abstractmethod
    def init_param(self, input_shape, max_width):
        """
        initialize parameters of this module
        :param max_width: max width of neural layer
        :return: None
        """
        pass

    def forward(self, x):
        return self._module_instance(x)

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

    @staticmethod
    @abc.abstractmethod
    def identity_module(cfg, name, input_shape: tuple):
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

    @abc.abstractmethod
    def perform_prune_current(self, mask):
        """
        prune module in current layer, implement this method in each submodule
        :return: module of next level
        """
        pass

    @abc.abstractmethod
    def perform_prune_next(self, mask):
        """
        prune module follow previous layer, implement this method in each submodule
        :return: module of next level
        """
        pass

    @staticmethod
    def mask_sanity_check(mask):
        """
        check if mask makes all neuron deleted, if so, randomly keep 1 neuron to keep network
        work properely
        """
        if mask.any():
            return
        mask[torch.randint(mask.shape[0], (1,))] = True
        assert mask.any()  # keep at least one neuron

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

    def init_param(self, input_shape, max_width):
        assert len(input_shape) == 2
        self.params = dict()
        out_range = self.cfg.modulesConfig['rnn']['out_range']
        self.current_level = self.get_level(out_range, max_width)
        self.params['input_size'] = input_shape[1]
        self.params['output_size'] = self.current_level
        # get output shape
        self.on_param_end(input_shape)
        # add pruning field
        if self.cfg.NASConfig['Pruning']:
            def score_accumulate_hook(module, grad_input, grad_output):
                with torch.enable_grad():
                    parameter_dict = dict(self._module_instance.rnn_unit.named_parameters())
                    weight_mask = ScoreAccumulator.apply(self.importance_score)
                    #
                    # # manually calculate importance score
                    # weight_ih_l0 = \
                    #     parameter_dict['weight_ih_l0'] * torch.unsqueeze(weight_mask, 1)
                    # weight_hh_l0 = \
                    #     parameter_dict['weight_hh_l0'] * torch.unsqueeze(weight_mask, 0) * torch.unsqueeze(weight_mask, 1)
                    # bias_ih_l0 = \
                    #     parameter_dict['bias_ih_l0'] * weight_mask
                    # bias_hh_l0 = \
                    #     parameter_dict['bias_hh_l0'] * weight_mask
                    #
                    # # backward
                    # weight_ih_l0.backward(parameter_dict['weight_ih_l0'].grad)
                    # weight_hh_l0.backward(parameter_dict['weight_hh_l0'].grad)
                    # bias_ih_l0.backward(parameter_dict['bias_ih_l0'].grad)
                    # bias_hh_l0.backward(parameter_dict['bias_hh_l0'].grad)

            def accumulate_importance(score):
                if self.importance_score.grad is None:
                    self.importance_score.grad = score
                else:
                    self.importance_score.grad += score

            def weight_ih_hook(grad):
                grad_calculated = torch.sum(self._module_instance.rnn_unit.weight_ih_l0 * grad, 1)
                accumulate_importance(grad_calculated)
                return grad

            def weight_hh_hook(grad):
                grad_calculated = self._module_instance.rnn_unit.weight_hh_l0 * grad
                grad_calculated = torch.sum(grad_calculated, 0) + torch.sum(grad_calculated, 1)
                accumulate_importance(grad_calculated)
                return grad

            def bias_ih_hook(grad):
                grad_calculated = self._module_instance.rnn_unit.bias_ih_l0 * grad
                accumulate_importance(grad_calculated)
                return grad

            def bias_hh_hook(grad):
                grad_calculated = self._module_instance.rnn_unit.bias_hh_l0 * grad
                accumulate_importance(grad_calculated)
                return grad

            self.importance_score = nn.Parameter(torch.zeros(self.current_level))
            # self._module_instance.rnn_unit.register_full_backward_hook(score_accumulate_hook)
            self._module_instance.rnn_unit.weight_ih_l0.register_hook(weight_ih_hook)
            self._module_instance.rnn_unit.weight_hh_l0.register_hook(weight_hh_hook)
            self._module_instance.rnn_unit.bias_ih_l0.register_hook(bias_ih_hook)
            self._module_instance.rnn_unit.bias_hh_l0.register_hook(bias_hh_hook)

    def forward(self, x):
        return self._module_instance(x)

    @staticmethod
    def identity_module(cfg, name, input_shape: tuple):
        if type(name) != str:
            name = cfg.NASConfig['editable'][name]
        module = RNNModule(cfg, name, input_shape)
        output_size = input_shape[1]
        input_size = input_shape[1]
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
        if cfg.NASConfig['Pruning']:
            module.importance_score = nn.Parameter(torch.zeros(module.current_level))
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
        self.output_shape = (self.output_shape[0], self.current_level)
        self.params['output_size'] = self.current_level
        if self.cfg.NASConfig['Pruning']:
            self.importance_score = Parameter(self.importance_score[mapping_g])
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
            nn.Parameter(self._module_instance.rnn_unit.weight_ih_l0[:, mapping_g] *
                         scale_g.unsqueeze(0),
                         requires_grad=True)
        rnn.weight_hh_l0 = \
            nn.Parameter(self._module_instance.rnn_unit.weight_hh_l0,
                         requires_grad=True)
        self._module_instance = new_module_instance
        self.input_shape = (self.input_shape[0], next_level)
        self.params['input_size'] = next_level

    def perform_prune_current(self, mask):
        """
        prune rnn layer by output neuron
        @param mask:
        """
        self.mask_sanity_check(mask)  # ensure mask legal
        self.current_level = mask.sum().item()
        new_module_instance = NAS_RNN(
            input_size=self.params['input_size'],
            hidden_size=self.current_level,
            nonlinearity='relu',
            batch_first=True,
        )

        # keep previous parameters
        rnn = new_module_instance.rnn_unit
        rnn.bias_ih_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_ih_l0[mask])
        rnn.bias_hh_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_hh_l0[mask])
        rnn.weight_ih_l0 = nn.Parameter(self._module_instance.rnn_unit.weight_ih_l0[mask])
        rnn.weight_hh_l0 = nn.Parameter(
            (self._module_instance.rnn_unit.weight_hh_l0[:, mask])[mask])
        # make all parameters compact
        rnn.flatten_parameters()

        self._module_instance = new_module_instance
        self.output_shape = (self.output_shape[0], self.current_level)
        self.params['output_size'] = self.current_level
        self.importance_score = nn.Parameter(self.importance_score[mask])

    def perform_prune_next(self, mask):
        input_size = mask.sum().item()
        new_module_instance = NAS_RNN(
            input_size=input_size,
            hidden_size=self.params['output_size'],
            nonlinearity='relu',
            batch_first=True,
        )
        rnn = new_module_instance.rnn_unit
        rnn.bias_ih_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_ih_l0, True)
        rnn.bias_hh_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_hh_l0, True)
        rnn.weight_ih_l0 = \
            nn.Parameter(self._module_instance.rnn_unit.weight_ih_l0[:, mask])
        rnn.weight_hh_l0 = \
            nn.Parameter(self._module_instance.rnn_unit.weight_hh_l0)
        # make all parameters compact
        rnn.flatten_parameters()
        self._module_instance = new_module_instance
        self.input_shape = (self.input_shape[0], input_size)
        self.params['input_size'] = input_size


class DenseModule(NasModule):
    def perform_prune_current(self, mask):
        """
        prune dense layer by row, inplace operation
        :param mask: mask of pruning, should be equal to row of weight
        """
        self.mask_sanity_check(mask)  # ensure mask legal
        self.current_level = mask.sum().item()
        new_module_instance = nn.Linear(self.params['in_features'], self.current_level)

        # assigning new attribute
        new_module_instance.bias = nn.Parameter(self._module_instance.bias[mask])
        new_module_instance.weight = nn.Parameter(self._module_instance.weight[mask])
        self._module_instance = new_module_instance
        self.output_shape = (self.output_shape[0], self.current_level)
        self.params['out_features'] = self.current_level
        self.importance_score = nn.Parameter(self.importance_score[mask])

    def perform_prune_next(self, mask):
        weight = nn.Parameter(self._module_instance.weight[:, mask])
        self.params['in_features'] = mask.sum().item()
        new_module_instance = nn.Linear(self.params['in_features'], self.current_level)
        new_module_instance.weight = weight
        new_module_instance.bias = nn.Parameter(self._module_instance.bias)

        # assigning new attribute
        self._module_instance = new_module_instance
        self.input_shape = (self.input_shape[0], self.params['in_features'])

    @property
    def is_max_level(self):
        out_range = self.cfg.modulesConfig['dense']['out_range']
        return self.current_level >= max(out_range)

    def init_param(self, input_shape, max_width):
        assert len(input_shape) == 2
        out_range = self.cfg.modulesConfig['dense']['out_range']
        in_features = input_shape[1]
        self.current_level = self.get_level(out_range, max_width)
        self.params = {'in_features': in_features,
                       "out_features": self.current_level}
        self.on_param_end(input_shape)
        # add pruning field
        if self.cfg.NASConfig['Pruning']:
            self.importance_score = nn.Parameter(torch.zeros(self.current_level))

    def forward(self, x):
        if self.cfg.NASConfig['Pruning']:
            # accumulate importance score of whole row, we use this as importance estimation of output neuron
            score = self.importance_score
            weight_mask = ScoreAccumulator.apply(score)  # no change
            # weight_mask = TopKBinarizer.apply(score,1 - self.cfg.NASConfig['PruningRatio'])  # pruning at running
            bias_mask = weight_mask
            weight_mask = torch.unsqueeze(weight_mask, 1) * torch.ones(self._module_instance.weight.shape[1],
                                                                       device=weight_mask.device)

            # weight and bias is not changed but score tensor is attached to the computational graph
            masked_weight = self._module_instance.weight * weight_mask
            masked_bias = self._module_instance.bias * bias_mask
            return F.linear(x, masked_weight, masked_bias)
        else:
            return self._module_instance(x)

    @staticmethod
    def identity_module(cfg, name, input_shape: tuple):
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
        # add pruning field
        if cfg.NASConfig['Pruning']:
            module.importance_score = nn.Parameter(torch.zeros(module.current_level))
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
        new_module_instance.bias = nn.Parameter(self._module_instance.bias[mapping_g])
        new_module_instance.weight = nn.Parameter(self._module_instance.weight[mapping_g])
        self.current_level = next_level
        self._module_instance = new_module_instance
        self.output_shape = (self.output_shape[0], self.current_level)
        self.params['out_features'] = self.current_level
        if self.cfg.NASConfig['Pruning']:
            self.importance_score = Parameter(self.importance_score[mapping_g])
        return mapping_g, scale_g

    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        next_level = len(mapping_g)
        scale_g = torch.tensor(scale_g)
        new_module_instance = nn.Linear(next_level, self.params['out_features'])
        # keep previous parameters
        new_module_instance.weight = nn.Parameter(
            self._module_instance.weight[:, mapping_g] * scale_g.unsqueeze(0))
        new_module_instance.bias = nn.Parameter(
            self._module_instance.bias)
        self._module_instance = new_module_instance
        self.input_shape = (self.input_shape[0], next_level)
        self.params['in_features'] = next_level


class ConvModule(NasModule):
    def perform_prune_current(self, mask):
        """
        prune conv layer by feature maps, inplace operation
        :param mask: mask of pruning, should be equal to out channels of module
        """
        self.mask_sanity_check(mask)  # ensure mask legal
        self.current_level = mask.sum().item()
        new_module_instance = nn.Conv1d(in_channels=self.params['in_channels'],
                                        out_channels=self.current_level,
                                        kernel_size=self.params['kernel_size'],
                                        stride=self.params['stride'],
                                        padding=self.params['padding'])

        # assigning new attribute
        new_module_instance.bias = nn.Parameter(self._module_instance.bias[mask])
        new_module_instance.weight = nn.Parameter(self._module_instance.weight[mask])
        self._module_instance = new_module_instance
        self.output_shape = (self.output_shape[0], self.current_level)
        self.params['out_channels'] = self.current_level
        self.importance_score = nn.Parameter(self.importance_score[mask])

    def perform_prune_next(self, mask):
        self.params['in_channels'] = mask.sum().item()
        new_module_instance = nn.Conv1d(in_channels=self.params['in_channels'],
                                        out_channels=self.params['out_channels'],
                                        kernel_size=self.params['kernel_size'],
                                        stride=self.params['stride'], padding=self.params['padding'])
        new_module_instance.weight = nn.Parameter(self._module_instance.weight[:, mask])
        new_module_instance.bias = self._module_instance.bias

        # assigning new attribute
        self._module_instance = new_module_instance
        self.input_shape = (self.input_shape[0], self.params['in_channels'])

    @property
    def is_max_level(self):
        out_range = self.cfg.modulesConfig['conv']['out_range']
        return self.current_level >= max(out_range)

    def init_param(self, input_shape, max_width):
        assert len(input_shape) == 2
        self.params = {'in_channels': input_shape[1]}
        out_range = self.cfg.modulesConfig['conv']['out_range']
        self.current_level = self.get_level(out_range, max_width)
        self.params['out_channels'] = self.current_level
        self.params['kernel_size'] = 3
        self.params['stride'] = 1
        self.params['padding'] = 1
        self.on_param_end(input_shape[::-1])
        self.output_shape = self.output_shape[::-1]
        # add pruning field
        if self.cfg.NASConfig['Pruning']:
            self.importance_score = nn.Parameter(torch.zeros(self.current_level))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.cfg.NASConfig['Pruning']:
            assert self.importance_score is not None
            # accumulate importance score of whole row, we use this as importance estimation of output neuron
            score = self.importance_score
            weight_mask = ScoreAccumulator.apply(score)  # no change
            # weight_mask = TopKBinarizer.apply(score,1 - self.cfg.NASConfig['PruningRatio'])  # pruning at running
            bias_mask = weight_mask

            # weight and bias is not changed but score tensor is attached to the computational graph
            masked_weight = self._module_instance.weight * torch.unsqueeze(torch.unsqueeze(weight_mask, 1), 1)
            masked_bias = self._module_instance.bias * bias_mask
            y = F.conv1d(x, masked_weight, masked_bias, self._module_instance.stride,
                         self._module_instance.padding)
        else:
            y = self._module_instance(x)
        y = y.permute(0, 2, 1)
        return y

    @staticmethod
    def identity_module(cfg, name, input_shape: tuple):
        if type(name) != str:
            name = cfg.NASConfig['editable'][name]
        module = ConvModule(cfg, name, input_shape)
        out_channel = input_shape[1]
        in_channel = input_shape[1]
        kernel_size = 3
        module.current_level = out_channel
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
        if cfg.NASConfig['Pruning']:
            module.importance_score = nn.Parameter(torch.zeros(module.current_level))
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
                                        stride=self.params['stride'],
                                        padding=self.params['padding'])
        # keep previous parameters
        mapping_g = self.widen_sample_fn(self.current_level, next_level)
        scale_g = [1 / mapping_g.count(i) for i in mapping_g]
        new_module_instance.bias = nn.Parameter(self._module_instance.bias[mapping_g])
        new_module_instance.weight = nn.Parameter(self._module_instance.weight[mapping_g])
        self.current_level = next_level
        self._module_instance = new_module_instance
        self.output_shape = (self.output_shape[0], self.current_level)
        self.params['out_channels'] = self.current_level
        if self.cfg.NASConfig['Pruning']:
            self.importance_score = Parameter(self.importance_score[mapping_g])
        return mapping_g, scale_g

    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        next_level = len(mapping_g)
        scale_g = torch.tensor(scale_g)
        new_module_instance = nn.Conv1d(in_channels=next_level,
                                        out_channels=self.params['out_channels'],
                                        kernel_size=self.params['kernel_size'],
                                        stride=self.params['stride'], padding=self.params['padding'])
        new_module_instance.weight = \
            nn.Parameter(self._module_instance.weight[:, mapping_g] *
                         scale_g.unsqueeze(0).unsqueeze(2), )
        new_module_instance.bias = self._module_instance.bias
        self._module_instance = new_module_instance
        self.params['in_channels'] = next_level
        self.input_shape = (self.input_shape[0], self.params['in_channels'])


class LSTMModule(NasModule):
    def perform_prune_current(self, mask):
        """
        prune rnn layer by output neuron
        @param mask:
        """
        self.mask_sanity_check(mask)  # ensure mask legal
        self.current_level = mask.sum().item()
        new_module_instance = NAS_SLSTM(self.params['input_size'], self.current_level)

        # keep previous parameters
        new_module_instance.cell.weight_ih = nn.Parameter(
            torch.cat(tuple(map(lambda x: x[:, mask], self._module_instance.cell.weight_ih.chunk(4, 1))), dim=1))
        new_module_instance.cell.weight_hh = nn.Parameter(
            torch.cat(tuple(map(lambda x: x[mask][:, mask], self._module_instance.cell.weight_hh.chunk(4, 1))), dim=1))
        new_module_instance.cell.bias_ih = nn.Parameter(
            torch.cat(tuple(map(lambda x: x[mask], self._module_instance.cell.bias_ih.chunk(4)))))

        self._module_instance = new_module_instance
        self.output_shape = (self.output_shape[0], self.current_level)
        self.params['output_size'] = self.current_level
        self.importance_score = nn.Parameter(self.importance_score[mask])

    def perform_prune_next(self, mask):
        input_size = mask.sum().item()
        new_module_instance = NAS_SLSTM(input_size, self.params['output_size'])
        new_module_instance.cell.weight_ih = nn.Parameter(self._module_instance.cell.weight_ih[mask])
        new_module_instance.cell.weight_hh = nn.Parameter(self._module_instance.cell.weight_hh)
        new_module_instance.cell.bias_ih = nn.Parameter(self._module_instance.cell.bias_ih)

        self._module_instance = new_module_instance

        self.input_shape = (self.input_shape[0], input_size)
        self.params['input_size'] = input_size

    @property
    def is_max_level(self):
        out_range = self.cfg.modulesConfig['lstm']['out_range']
        return self.current_level >= max(out_range)


    def init_param(self, input_shape, max_width):
        assert len(input_shape) == 2
        self.params = dict()
        out_range = self.cfg.modulesConfig['lstm']['out_range']
        self.current_level = self.get_level(out_range, max_width)
        self.params['input_size'] = input_shape[1]
        self.params['output_size'] = self.current_level
        self.on_param_end(input_shape)
        self.output_shape = self.output_shape
        # add pruning field
        if self.cfg.NASConfig['Pruning']:
            self.importance_score = nn.Parameter(torch.zeros(self.current_level))
            clip_threshold = 1e6

            def accumulate_importance(score):
                if self.importance_score.grad is None:
                    self.importance_score.grad = score
                else:
                    self.importance_score.grad += score

            def weight_ih_accumulate(grad):
                # add gradient clip to avoid gradient exploding
                grad = torch.clip(grad, -clip_threshold, clip_threshold)
                weight = self._module_instance.cell.weight_ih
                grad_calculated = torch.sum(weight * grad, 0)
                grad_calculated = sum(grad_calculated.chunk(4))
                accumulate_importance(grad_calculated)
                return grad

            def weight_hh_accumulate(grad):
                grad = torch.clip(grad, -clip_threshold, clip_threshold)
                weight = self._module_instance.cell.weight_hh
                grad_calculated = weight * grad
                gates = grad_calculated.chunk(4, 1)
                grad_calculated = sum(map(lambda x: torch.sum(x, 0), gates)) + sum(
                    map(lambda x: torch.sum(x, 1), gates))
                accumulate_importance(grad_calculated)
                return grad

            def bias_ih_accumulate(grad):
                grad = torch.clip(grad, -clip_threshold, clip_threshold)
                weight = self._module_instance.cell.bias_ih
                grad_calculated = weight * grad
                grad_calculated = sum(grad_calculated.chunk(4))
                accumulate_importance(grad_calculated)
                return grad

            self._module_instance.cell.weight_ih.register_hook(weight_ih_accumulate)
            self._module_instance.cell.weight_hh.register_hook(weight_hh_accumulate)
            self._module_instance.cell.bias_ih.register_hook(bias_ih_accumulate)

    def forward(self, x):
        return self.get_module_instance()(x)

    @staticmethod
    def identity_module(cfg, name, input_shape: tuple):
        if type(name) != str:
            name = cfg.NASConfig['editable'][name]
        module = LSTMModule(cfg, name, input_shape)
        hidden_size = input_shape[1]
        input_size = input_shape[1]
        module.params = {
            "output_size": hidden_size,
            "input_size": input_size,
        }
        module.current_level = hidden_size
        module.output_shape = input_shape
        identity_lstm = NAS_SLSTM.generate_identity(input_size)
        module._module_instance = identity_lstm
        if cfg.NASConfig['Pruning']:
            module.importance_score = nn.Parameter(torch.zeros(module.current_level))
        return module

    def get_module_instance(self):
        if self._module_instance is not None:
            return self._module_instance
        self._module_instance = NAS_SLSTM(
            input_size=self.params['input_size'],
            hidden_size=self.params['output_size'],
        )
        # if not isinstance(self._module_instance,torch.jit.RecursiveScriptModule):
        #     self._module_instance = torch.jit.script(self._module_instance)
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
        new_module_instance.cell.weight_ih = nn.Parameter(
            torch.cat(tuple(map(lambda x: x[:, mapping_g], self._module_instance.cell.weight_ih.chunk(4, 1))), dim=1))
        new_module_instance.cell.weight_hh = nn.Parameter(torch.cat(tuple(
            map(lambda x: x[mapping_g][:, mapping_g] * torch.unsqueeze(scale_g, 1),
                self._module_instance.cell.weight_hh.chunk(4, 1))), dim=1))
        new_module_instance.cell.bias_ih = nn.Parameter(
            torch.cat(tuple(map(lambda x: x[mapping_g], self._module_instance.cell.bias_ih.chunk(4)))))
        self.current_level = next_level
        self._module_instance = new_module_instance
        self.output_shape = (self.output_shape[0], self.current_level)
        self.params['output_size'] = self.current_level
        if self.cfg.NASConfig['Pruning']:
            self.importance_score = Parameter(self.importance_score[mapping_g])
        return mapping_g, scale_g

    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        next_level = len(mapping_g)
        scale_g = torch.tensor(scale_g)
        new_module_instance = NAS_SLSTM(next_level, self.params['output_size'])
        new_module_instance.cell.weight_ih = nn.Parameter(
            self._module_instance.cell.weight_ih[mapping_g] * torch.unsqueeze(scale_g, 1))
        new_module_instance.cell.weight_hh = nn.Parameter(self._module_instance.cell.weight_hh)
        new_module_instance.cell.bias_ih = nn.Parameter(self._module_instance.cell.bias_ih)
        self._module_instance = new_module_instance
        self.input_shape = (self.input_shape[0], next_level)
        self.params['input_size'] = next_level


# this class comes from [Movement Pruning](https://github.com/huggingface/nn_pruning)
class TopK(autograd.Function):
    """
    Top-k Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of S.

    Implementation is inspired from:
        https://github.com/allenai/hidden-networks
        What's hidden in a randomly weighted neural network?
        Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold %
        mask = inputs.clone()
        _, idx = inputs.flatten().sort(descending=True)
        j = int(threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        return mask == 1

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


# Implementation inspired by TopKBinarizer
class ScoreAccumulator(autograd.Function):
    """
    Accumulate importance score while training. Do not change forward data
    """

    @staticmethod
    def forward(ctx, score):
        return torch.ones_like(score, device=score.device)

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class NAS_RNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # kwargs['bias'] = False
        self.rnn_unit = nn.RNN(*args, **kwargs)

    def forward(self, x: Tensor):
        output, hn = self.rnn_unit(x)
        y = output
        return y


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(input_size, 4 * hidden_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, 4 * hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))

    def forward(self, input: Tensor, h, c):
        hx, cx = h, c
        gates = (torch.mm(input, self.weight_ih) + self.bias_ih +
                 torch.mm(hx, self.weight_hh))
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = outgate

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy) + outgate

        return hy, cy


class NAS_SLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cell = LSTMCell(input_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        follow implementation with pytorch
        """
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        """
        forward propagation
        @param x:  input of size [batch, seqlen, feature]
        @return: hidden state [batch, seqlen, feature]
        """
        h = torch.zeros((x.shape[0], self.hidden_size), device=x.device)
        c = torch.zeros((x.shape[0], self.hidden_size), device=x.device)
        input = x.unbind(1)
        outputs = []
        for i in range(len(input)):
            h, c = self.cell(input[i], h, c)
            outputs += [h]
        output = torch.stack(outputs, 1)
        return output

    @staticmethod
    def generate_identity(input_size):
        lstm = NAS_SLSTM(input_size, input_size)
        weight_ih = torch.zeros_like(lstm.cell.weight_ih)
        weight_ih[:, 3 * input_size:] = torch.eye(input_size)
        lstm.cell.weight_ih = nn.Parameter(weight_ih)
        lstm.cell.weight_hh = nn.Parameter(torch.zeros_like(lstm.cell.weight_hh))
        lstm.cell.bias_ih = nn.Parameter(torch.zeros_like(lstm.cell.bias_ih))
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


def generate_from_skeleton(cfg, skeleton: list, input_shape, max_width):
    """
    generate a list of module instance from skeleton list
    :param cfg: global configuration
    :param skeleton: skeleton list of string specifying each layer's type
    :param input_shape: input data shape
    :param max_width: max width of neural layer
    :return: list of modules
    """
    modules = []
    for name in skeleton + ['dense']:
        module = generate_module(cfg, name, input_shape)
        try:
            module.init_param(input_shape, max_width)
            input_shape = module.output_shape
            modules.append(module)
        except Exception as e:
            logger = get_logger('module init', cfg.LOG_FILE)
            logger.error(f'fail to init module {module},\n'
                         f' {"message".center(50, "=")}\n'
                         f'{e}'
                         f'{"end".center(50, "=")}')
    return modules
