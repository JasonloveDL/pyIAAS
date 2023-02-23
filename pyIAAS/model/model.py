import copy
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import torch.nn
from torch.utils.data.dataloader import *
from torch.utils.data.dataset import TensorDataset

from .module import NasModule, generate_from_skeleton, TopK
from ..utils.logger import get_logger
from ..utils.sql_connector import get_total_model_count, insert_new_model_config, update_new_train_result, \
    update_new_pool_state

total_model_count = None
activate = torch.nn.ReLU
plt.rcParams['figure.figsize'] = [16, 4]


def reset_model_count():
    global total_model_count
    total_model_count = None


class ModelConfig:

    def __init__(self, cfg, modules: list, feature_shape, target_shape):
        """
        model configuration class, containing model structure information and another static information
        :param cfg: global configuration
        :param modules: list of modules
        :param feature_shape: input feature shape
        :param target_shape: target tensor shape
        """
        self.cfg = cfg
        self.target_shape = target_shape
        self.feature_shape = feature_shape
        self.modules = modules
        self.tail_layers = []

    @property
    def insert_length(self):
        """
        insert new layer length avalible, used in deeper actor
        :return: int
        """
        return len(self.modules) + 1

    @property
    def editable_list(self):
        widenable = [m.editable for m in self.modules]
        widenable[-1] = False  # The last layer is not allowed to be widened
        return widenable

    def generate_model(self):
        """
        generate a new NasModel from scratch, all NasModel trainable parameters is randomly generated
        :return: NAS_Module
        """
        module_instances = []
        for m in self.modules:
            module_instances.append(m)
            module_instances.append(activate())
        output_shape = self.modules[-1].output_shape
        if len(output_shape) > 1:
            s = 1
            for i in output_shape:
                s *= i
            output_shape = s
            self.tail_layers.append(torch.nn.Flatten())
        self.tail_layers.append(torch.nn.Linear(output_shape, self.target_shape))
        module_instances = [*module_instances, *self.tail_layers]
        model_instance = torch.nn.Sequential(*module_instances)
        m = NasModel(self.cfg, model_instance, self)
        return m

    @property
    def token_list(self):
        token_list = []
        for m in self.modules:
            assert isinstance(m, NasModule)  # m should be subclass of NAS_Module
            token_list.append(m.token)
        return token_list

    def __str__(self):
        return str.join('->', self.token_list)


class NasModel:
    def __init__(self, cfg, model_instance: torch.nn.Module, model_config: ModelConfig, train_times=0, prev_index=-1, prev_flops = None):
        """
        NAS model class, this class representing running instance of a neural network and we can do wider and deeper transformation
        in this class
        :param cfg: global configuration file
        :param model_instance: model instance of type torch.nn.Module
        :param model_config: model configuration class
        :param prev_index: previous netowrk index
        """
        self.cfg = cfg
        self.logger = get_logger('NasModel', cfg.LOG_FILE)
        self.model_config = model_config
        self.model_instance = model_instance
        self.train_times = train_times  # record total train times
        self.loss_list = []
        self.test_loss_best = None
        self.test_loss_best_iteration = 0
        self.test_loss = None
        self.next_save = False
        self.prev_index = prev_index
        self.update_global_information()
        self.transformation_record = pd.DataFrame(
            {'prev': -1, 'current': self.index, 'train_times': 0, 'structure': str(self.model_config)}, index=[0])
        self.transitions = []
        self.optimizer = None
        self.activate = activate
        self.flops = None
        # input_data = torch.zeros((1, *model_config.feature_shape))
        # with torch.profiler.profile(
        #     with_flops=True
        # ) as p:
        #     model_instance(input_data)
        # self.flops = int(p.profiler.function_events.total_average().flops)
        # if prev_flops is None:
        #     self.flop_change_ratio = 0.0
        # else:
        #     assert isinstance(prev_flops, int)
        #     assert self.flops > 0 and prev_flops > 0
        #     self.flop_change_ratio = float(self.flops / prev_flops) - 1


    @property
    def state(self):
        return self.model_config.token_list, \
               self.model_config.insert_length, \
               self.model_config.editable_list, \
               self.index

    def __eq__(self, other):
        return self.index == other.index

    def __hash__(self):
        return hash(self.index)

    def predict(self, input):
        """
        predict target value based on input
        :param input: input tensor , features
        :return: preidction
        """
        return self.__call__(input)

    def to_cuda(self):
        self.model_instance.cuda()

    def to_cpu(self):
        self.model_instance.cpu()

    def add_transformation_record(self, prev, current, train_times, loss_list):
        """
        add a transformation record.
        :param prev: index of previous model
        :param current: index of current model
        :param train_times: train times of prevous model
        :param loss_list: loss list information
        """
        # self.transformation_record = self.transformation_record.append(
        #     {'prev': prev, 'current': current, 'train_times': train_times}, ignore_index=True)
        self.transformation_record = pandas.concat([self.transformation_record,
                                                    pd.DataFrame(
                                                        {'prev': prev,
                                                         'current': current,
                                                         'train_times': train_times,
                                                         'structure': str(self.model_config)},
                                                        index=[self.transformation_record.shape[0]])])
        self.loss_list = copy.deepcopy(loss_list)

    def update_global_information(self):
        """
        update information in SQLite.
        """
        global total_model_count
        if total_model_count is None:
            total_model_count = get_total_model_count(self.cfg)
        # try:
        total_model_count += 1
        self.index = total_model_count
        insert_new_model_config(self.cfg, self.index, str(self.model_config), self.prev_index)
        # except:
        #     id, structure, train_time, loss, prev_index = get_prev_record(self.cfg, str(self.model_config))
        #     self.index = id
        #     self.prev_index = -1
        #     total_model_count -= 1
        #     self.train_times = 0
        self.logger.info(f'create model {self.index} from {self.prev_index}, structure: {self.model_config}')

    def update_pool_state(self, in_pool:bool):
        if in_pool:
            update_new_pool_state(self.cfg, self.index, 1)
        else:
            update_new_pool_state(self.cfg, self.index, 0)

    def __call__(self, x):
        return self.model_instance(x)

    def __str__(self):
        return str(self.index) + " " + str(self.model_config)

    def train(self, X_train, y_train):
        """
        train NasModel directly(without batch), all feature make forward computation at once
        :param X_train: train hn_feature
        :param y_train: train targets
        :return: None
        """
        optimizer = self._get_optimizer()
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.995, self.train_times)
        loss_fn = self._get_loss_function()
        dataloader = DataLoader(TensorDataset(X_train, y_train),
                                self.cfg.NASConfig['BATCH_SIZE'],
                                shuffle=True)
        epochs = self.cfg.NASConfig['IterationEachTime']
        # if self.warm_start:
        #     self.warm_start = False
        #     epochs = self.cfg.NASConfig['WarmUpEpoch']

        for i in range(epochs):
            # Compute prediction and loss
            st = time.time()
            loss_list = []
            # model_save = self.model_instance todo may use torchscript in the future
            # self.model_instance = torch.jit.script(self.model_instance)
            target_shape = y_train.shape
            for step, (batch_x, batch_y) in enumerate(dataloader):
                pred = self.model_instance(batch_x)
                pred = pred.view(batch_y.shape)
                loss = loss_fn(pred, batch_y)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                loss_list.append(loss.item())
            # self.model_instance = model_save
            self.train_times += 1
            self.loss_list.append(np.mean(loss_list))
            if self.train_times % self.cfg.NASConfig['MonitorIterations'] == 0:
                try:
                    self.logger.info(f'NasModel {self.index} '
                                     f'train {self.train_times} times '
                                     f'loss {loss.item()} batch: {len(loss_list)} * {self.cfg.NASConfig["BATCH_SIZE"]} '
                                     f'{self.cfg.NASConfig["IterationEachTime"]} epoch time: {(time.time() - st) * self.cfg.NASConfig["IterationEachTime"]} sec.')
                except:
                    pass  # ignore logger exception

    def test(self, X_test, y_test):
        """
        test this network in test data. all result stored in member variable
        :param X_test: input features
        :param y_test: target value
        """
        with torch.no_grad():
            loss_fn = self._get_loss_function()
            pred = self.model_instance(X_test).view(-1)
            pred = pred.view(y_test.shape)
            loss = loss_fn(pred, y_test)
            self.test_loss = loss.item()
            self.next_save = False
            if np.isnan(self.test_loss):
                self.test_loss = 1e5
            if self.test_loss_best is None:
                self.test_loss_best = self.test_loss
                self.next_save = True
            if self.test_loss_best > self.test_loss:
                self.test_loss_best = self.test_loss
                self.next_save = True
            if self.next_save:
                self.test_loss_best_iteration = len(self.loss_list)
            return self.test_loss

    def perform_wider_transformation(self, wider_index):
        """
        generate a new NasModel by widen the specific layer
        the widen method follows Net2Net(arXiv:1511.05641)
        :param wider_index: index of layer to widen
        :return: new NasModel with the widen layer
        """
        self.to_cpu()
        model_config = copy.deepcopy(self.model_config)
        module_instances = []
        i = 0
        while True:
            if i == len(model_config.modules):
                break
            if i == wider_index:
                # wider the specific network
                module = model_config.modules[i]
                mapping_g, scale_g = module.perform_wider_transformation_current()
                module_instances.append(model_config.modules[i])
                module_instances.append(self.activate())
                i += 1
                # modify next module to match current module
                module = model_config.modules[i]
                module.perform_wider_transformation_next(mapping_g, scale_g)
                module_instances.append(model_config.modules[i])
                module_instances.append(self.activate())

            else:
                # inherent from father network
                module_instances.append(model_config.modules[i])
                module_instances.append(self.activate())
            i += 1
        module_instances = [*module_instances, *model_config.tail_layers]
        model_instance = torch.nn.Sequential(*module_instances)
        m = NasModel(self.cfg, model_instance, model_config, self.train_times, self.index, self.flops)
        m.transformation_record = self.transformation_record.copy()
        m.add_transformation_record(self.index, m.index, self.train_times, self.loss_list)
        m.transitions = self.transitions.copy()
        return m

    def perform_deeper_transformation(self, deeper_action):
        """
        generate a new NasModel by insert a new layer
        the deeper method follows Net2Net(arXiv:1511.05641)
        :param deeper_action: (insert layer type, insert layer place)
        :return: new NasModel with deeper layers
        """
        self.to_cpu()
        insert_type, insert_index = deeper_action
        if type(insert_type) != str:
            insert_type = self.cfg.NASConfig['editable'][insert_type]
        model_config = copy.deepcopy(self.model_config)
        module_instances = []
        module_length = len(model_config.modules)
        identity_module = None
        for i in range(module_length):
            if i == insert_index:
                # insert layer inside the network
                input_shape = self.model_config.modules[i].input_shape
                identity_module = self.cfg.modulesCls[insert_type].identity_module(self.cfg, insert_type, input_shape)
                module_instances.append(identity_module)
                module_instances.append(self.activate())

            module_instances.append(model_config.modules[i])
            module_instances.append(self.activate())

        # insert in tail
        if insert_index == module_length:
            input_shape = self.model_config.modules[-1].output_shape
            identity_module = self.cfg.modulesCls[insert_type].identity_module(self.cfg, insert_type, input_shape)
            module_instances.append(identity_module)
            module_instances.append(self.activate())
        model_config.modules.insert(insert_index, identity_module)

        module_instances = [*module_instances, *model_config.tail_layers]
        model_instance = torch.nn.Sequential(*module_instances)
        m = NasModel(self.cfg, model_instance, model_config, self.train_times, self.index, self.flops)
        m.transformation_record = self.transformation_record.copy()
        m.add_transformation_record(self.index, m.index, self.train_times, self.loss_list)
        m.transitions = self.transitions.copy()
        return m

    def save_pred_result(self, X_test, y_test, model_dir=None):
        """
        save prediction result to disk
        :param X_test: input features
        :param y_test: target value
        """
        if model_dir is None:
            model_dir = os.path.join(self.cfg.NASConfig['OUT_DIR'], f'{self.index}')
        pred_result_path = os.path.join(model_dir, 'best_pred.csv')
        pred_figure_path = os.path.join(model_dir, 'pred.png')
        # detail_pred_figure_path = os.path.join(model_dir, 'detail_pred.png')
        best_pred_result_path = os.path.join(model_dir, 'best_pred.csv')
        best_pred_figure_path = os.path.join(model_dir, 'best_pred.png')
        # best_detail_pred_figure_path = os.path.join(model_dir, 'best_detail_pred.png')
        with torch.no_grad():
            pred = self.model_instance(X_test)
        y_test = y_test.cpu()
        pred = pred.cpu()
        plt.plot(y_test, label='y')
        plt.plot(pred, label='pred')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(pred_figure_path)
        if self.next_save:
            plt.savefig(best_pred_figure_path)
        # plt.clf()
        # plt.plot(y_test[:200], label='y')
        # plt.plot(pred[:200], label='pred')
        # plt.legend(loc='upper left')
        # plt.tight_layout()
        # plt.savefig(detail_pred_figure_path)
        if self.next_save:
            # plt.savefig(best_detail_pred_figure_path)
            # save prediction result
            pd.DataFrame({'truth': y_test.view(-1), 'pred': pred.view(-1)}).to_csv(pred_result_path)
        plt.clf()
        plt.close()

    def save_model(self, model_dir=None):
        """
        save model parameters and additional information of this netowrk to disk
        """
        if model_dir is None:
            model_dir = os.path.join(self.cfg.NASConfig['OUT_DIR'], f'{self.index}')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'NasModel.pth')
        loss_figure_path = os.path.join(model_dir, 'loss.png')
        model_str_path = os.path.join(model_dir, 'NasModel.txt')
        model_transformation_path = os.path.join(model_dir, 'transformation.csv')
        loss_path = os.path.join(model_dir, 'loss.csv')
        if self.next_save:
            torch.save(self.model_instance, model_path)
        plt.plot(self.loss_list)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        # x0, y0 = self.test_loss_best_iteration, self.loss_list[self.train_times - 1]
        # plt.plot(x0, y0, 'om')
        # plt.annotate('save model', xy=(x0, y0), xytext=(x0, y0))
        plt.savefig(loss_figure_path)
        plt.close()
        with open(model_str_path, 'w') as f:
            f.write(str(self.model_config))
        pd.DataFrame(self.loss_list).to_csv(loss_path)
        self.transformation_record.to_csv(model_transformation_path)
        update_new_train_result(self.cfg, self.index, self.train_times, self.test_loss_best)

    def prune(self):
        """
        Prune structure by importance score.
        :return: new NasModel with deeper layers
        """
        model_config = copy.deepcopy(self.model_config)
        module_instances = []
        module_length = len(model_config.modules)

        # get all modules' importance score
        modules = model_config.modules
        prune_list = model_config.editable_list
        score_list = []
        for i in range(len(prune_list)):
            if prune_list[i]:
                score_list.append(modules[i].importance_score)
        neuron_scores = torch.cat(score_list)
        # globally prune neurons
        mask = TopK.apply(neuron_scores, 1 - self.cfg.NASConfig['PruningRatio'])
        pointer = 0
        mask_list = []
        for i in range(len(score_list)):
            mask_list.append(mask[pointer:pointer + score_list[i].shape[0]])
            pointer += score_list[i].shape[0]

        # use prune mask to prune each layer
        pointer = 0
        for i in range(len(prune_list)):
            if prune_list[i]:
                modules[i].perform_prune_current(mask_list[pointer])
                modules[i + 1].perform_prune_next(mask_list[pointer])
                pointer += 1

        # construct new model instance
        for i in range(module_length):
            module_instances.append(model_config.modules[i])
            module_instances.append(self.activate())
        module_instances = [*module_instances, *model_config.tail_layers]
        model_instance = torch.nn.Sequential(*module_instances)
        m = NasModel(self.cfg, model_instance, model_config, self.train_times, self.index, self.flops)
        m.transformation_record = self.transformation_record.copy()
        m.add_transformation_record(self.index, m.index, self.train_times, self.loss_list)
        m.transitions = self.transitions.copy()
        return m

    def _get_loss_function(self):
        # default rmse loss function
        loss_fn = self.rmse
        if 'LossFunction' in self.cfg.NASConfig:
            if self.cfg.NASConfig['LossFunction'] == "rmse":
                loss_fn = self.rmse
            elif self.cfg.NASConfig['LossFunction'] == "max_rmse":
                loss_fn = self.max_rmse
            elif self.cfg.NASConfig['LossFunction'] == "mix_rmse":
                loss_fn = self.mix_rmse
        return loss_fn

    def _get_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # self.optimizer = torch.optim.Adam([{'params': self.model_instance.parameters(),
        #                                     'initial_lr': 1e-3}], weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(self.model_instance.parameters())
        return self.optimizer

    @staticmethod
    def mape(pred, truth):
        return torch.mean(torch.abs((pred - truth) / (truth + 1e-5)))

    @staticmethod
    def mae(pred, truth):
        return torch.mean(torch.abs((pred - truth)))

    @staticmethod
    def rmse(pred, truth):
        import torch.nn.functional as F
        return torch.sqrt(F.mse_loss(pred, truth))

    @staticmethod
    def max_rmse(pred, truth):
        import torch.nn.functional as F
        target_shape = pred.shape[1]
        rmse = [torch.sqrt(F.mse_loss(pred[:,i], truth[:,i])) for i in range(target_shape)]
        max_rmse = max(rmse)
        return max_rmse

    @staticmethod
    def mix_rmse(pred, truth):
        return NasModel.rmse(pred, truth) + NasModel.max_rmse(pred, truth)


def generate_new_model_config(cfg, feature_shape, targe_shape, skeleton=None, max_width=None) -> ModelConfig:
    """
    generate random NasModel from scratch
    :return: ModelConfig of new NasModel
    """
    # generate NasModel configuration
    if skeleton is None:
        maxLayers = cfg.NASConfig['MaxInitLayers']
        layers = np.random.randint(1, maxLayers + 1)
        skeleton = [j[0] for j in [random.sample(cfg.modulesList, 1) for i in range(layers)]]
    modules = generate_from_skeleton(cfg, skeleton, feature_shape, max_width)
    # generate NasModel train operations
    config = ModelConfig(cfg, modules, feature_shape, targe_shape)
    return config
