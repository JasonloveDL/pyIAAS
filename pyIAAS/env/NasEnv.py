import copy
import os
import pickle
import random
from typing import Any, List

import gym
import numpy as np
import torch

from ..agent import SelectorActorNet, Transition, recursive_tensor_detach
from ..model import generate_new_model_config, reset_model_count, NasModel
from ..utils.logger import get_logger


class NasEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cfg, pool_size, train_test_data):
        """
        NAS environment , the NAS search action and evaluation is in this class.
        @param pool_size:
        @param train_test_data:  (X_train, y_train, X_test, y_test), for input X, the dimension should be
        (batch, feature, time)
        """
        self.logger = get_logger('NAS_ENV', cfg.LOG_FILE)
        self.cfg = cfg
        self.train_test_data = train_test_data
        self.net_pool = None
        self.pool_size = pool_size
        self.best_performance = 1e9  # record best performance
        self.global_train_times = 0
        X_train, y_train, X_test, y_test = self.train_test_data
        self.feature_shape = X_train.shape[1:]
        assert len(y_train.shape) ==1 or len(y_train.shape) == 2, 'target data shape should be 1 dim or 2 dim'
        self.target_shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]

        if self.cfg.NASConfig['GPU']:
            self.train_test_data = []
            for i in range(len(train_test_data)):
                self.train_test_data.append(train_test_data[i].cuda())

    def step(self, action: Any):
        """
        do the actions given by agent and train the networks in pool, then drop low performance networks.
        :param replay_memory: memory to store trajectories
        :param action: action given by agent
        :return: observation, reward, done, info
        """
        net_pool: List[NasModel] = []
        transition: List[dict] = []

        # remove all model from pool
        for net in self.net_pool:
            net.update_pool_state(False)

        for i in range(len(self.net_pool)):
            net = self.net_pool[i]
            # if net.train_times >= self.cfg.NASConfig['MaxTrainTimes']:
            #     # done
            #     # record nothing
            #     continue
            action_i = action['action'][i]
            select = action_i['select']
            prev_net = net

            # add prev net
            if self.cfg.NASConfig['KeepPrevNet']:
                net_pool.append(prev_net)
                transition_item = {}
                transition_item['prev net'] = prev_net
                transition_item['next net'] = prev_net
                transition_item['action'] = copy.deepcopy(action['action'][i])
                transition_item['action']['select'] = SelectorActorNet.UNCHANGE
                transition_item['policy'] = action['policy'][i]
                transition.append(transition_item)

            # select representations
            if select == SelectorActorNet.UNCHANGE:  # do nothing
                self.logger.info(f"net index {i}-{net.index} :do not change network {net}")
                if self.cfg.NASConfig['KeepPrevNet']:
                    continue
            elif select == SelectorActorNet.WIDER:  # wider the net
                self.logger.info(f"net index {i}-{net.index} :wider the net {net}")
                net = net.perform_wider_transformation(action_i['wider'])
            elif select == SelectorActorNet.DEEPER:  # deeper the net
                if len(net.model_config.modules) < self.cfg.NASConfig['MaxLayers']:  # constrain the network's depth
                    self.logger.info(f"net index {i}-{net.index} :deeper the net {net}")
                    net = net.perform_deeper_transformation(action_i['deeper'])
                else:
                    if self.cfg.NASConfig['KeepPrevNet']:
                        continue
            elif select == SelectorActorNet.PRUNE:  # prune the net
                self.logger.info(f"net index {i}-{net.index} :prune the net {net}")
                net = net.prune()
            else:
                raise RuntimeError(f'no such action type: {select}')
            net_pool.append(net)
            transition_item = {}
            transition_item['prev net'] = prev_net
            transition_item['next net'] = net
            transition_item['action'] = action['action'][i]
            transition_item['policy'] = action['policy'][i]
            transition.append(transition_item)
        self.net_pool = net_pool
        for net in self.net_pool:
            net.update_pool_state(True)
        random_nets = self.generate_random_net()
        self.net_pool.extend(random_nets)
        self.net_pool = list(set(self.net_pool))
        self._train_and_test()
        self.net_pool = sorted(self.net_pool, key=lambda x: x.test_loss)
        reward_dict = self.get_reward()
        for i in transition:
            i['reward'] = reward_dict[i['next net']]
            t = i['prev net'].state, i['action'], i['reward'], i['policy']
            t = recursive_tensor_detach(t)
            i['next net'].transitions.append(Transition(*t))
        in_pool_trajectory, finished_trajectory = [], []
        for i in range(len(self.net_pool)):
            if i < self.pool_size:
                in_pool_trajectory.append(self.net_pool[i].transitions)
                self.net_pool[i].update_pool_state(True)
            else:
                finished_trajectory.append(self.net_pool[i].transitions)
                self.net_pool[i].update_pool_state(False)
        self.net_pool = self.net_pool[:self.pool_size]
        state = self.get_state()
        return state, in_pool_trajectory, finished_trajectory

    def performance(self):
        """
        :return: average performance in the net pool
        """
        return np.mean([net.test_loss_best for net in self.net_pool])

    def top_performance(self):
        """
        :return: best performance in the net pool
        """
        return np.min([net.test_loss_best for net in self.net_pool])

    def get_reward(self):
        """
        calculate reward for each network
        :return:
        """
        reward_dict = {}
        for net in self.net_pool:
            regularize_layer_number = self.cfg.NASConfig['RewardRegularisationLayers'] * max(0, len(net.model_config.modules) -
                                                                           self.cfg.NASConfig['LayerNumberStartRegularize'])
            regularize_layer_width = self.cfg.NASConfig['RewardRegularisationWidth'] * max(0, max(i.current_level for i in net.model_config.modules) -
                                                                           self.cfg.NASConfig['LayerWidthStartRegularize'])
            reward_dict[net] = (1 / net.test_loss) - regularize_layer_number - regularize_layer_width
        return reward_dict

    def reset(self):
        """
        reset all net to random init, this should call once in whole program
        train one round before return netpool
        :return:  net pool containing all network under searching
        """
        reset_model_count()
        X_train, y_train, X_test, y_test = self.train_test_data
        self.net_pool = [generate_new_model_config(self.cfg, self.feature_shape, self.target_shape, [i]).generate_model() for i in self.cfg.modulesConfig.keys()]
        self._train_and_test()
        self.render()
        return self.get_state()

    def get_state(self):
        return [i.state for i in self.net_pool]


    def render(self, mode='human'):
        """
        save network to dist
        :param mode: inherent from super, not used here
        """
        X_train, y_train, X_test, y_test = self.train_test_data
        for net in self.net_pool:
            if self.cfg.NASConfig['GPU']:
                net.to_cuda()
            net.save_model()
            net.save_pred_result(X_test, y_test)
            if self.cfg.NASConfig['GPU']:
                net.to_cpu()
        model_dir = os.path.join(self.cfg.NASConfig['OUT_DIR'], 'best')
        best_model = min(self.net_pool, key=lambda x: x.test_loss)
        if self.cfg.NASConfig['GPU']:
            best_model.to_cuda()
        best_model.save_model(model_dir)
        best_model.save_pred_result(X_test, y_test, model_dir)
        if self.cfg.NASConfig['GPU']:
            best_model.to_cpu()

    def _train_and_test(self):
        """
        train and test the networks in net pool, then save networks in to disk
        """
        X_train, y_train, X_test, y_test = self.train_test_data
        self.global_train_times += self.cfg.NASConfig['IterationEachTime']
        for net in self.net_pool:
            if self.cfg.NASConfig['GPU']:
                net.to_cuda()
            net.train(X_train, y_train)
            net.test(X_test, y_test)
        self.render()  # save train result

    def save(self, path=None):
        path = os.path.join(self.cfg.NASConfig['OUT_DIR'], 'env.pkl') if path is None else path
        with open(path,'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def try_load(cfg, logger, path=None):
        path = os.path.join(cfg.NASConfig['OUT_DIR'], 'env.pkl') if path is None else path
        if not os.path.exists(path):
            return None
        with open(path,'rb') as f:
            env = pickle.load(f)
            env.cfg = cfg
            for n in env.net_pool:
                n.cfg = cfg
                n.model_config.cfg = cfg
            logger.critical('load env checkpoint')
            return env

    def generate_random_net(self):
        net_number = self.cfg.NASConfig['RandomAddNumber']
        random_nets = []
        # max_length = max(len(i.model_config.modules) for i in self.net_pool)
        # max_width = max(max(j.current_level for j in i.model_config.modules) for i in self.net_pool)
        for i in range(net_number):
            # net_layers = min(max(1, int(random.expovariate(10/self.cfg.NASConfig['MaxLayers']))), max_length)
            # skeleton = [random.sample(self.cfg.modulesCls.keys(), 1)[0] for i in range(net_layers)]
            # random_model = generate_new_model_config(self.cfg, self.feature_shape, self.target_shape,
            #                                          skeleton, max_width).generate_model()
            # random_nets.append(random_model)
            net_layers = min(max(1, int(random.expovariate(0.2))), self.cfg.NASConfig['MaxLayers'])
            skeleton = [random.sample(self.cfg.modulesCls.keys(),1)[0] for i in range(net_layers)]
            random_model = generate_new_model_config(self.cfg, self.feature_shape, self.target_shape, skeleton).generate_model()
            random_nets.append(random_model)
        return random_nets

