import os
import random
from typing import Any

import gym
import numpy as np

from ..utils.logger import get_logger
from ..model import generate_new_model_config, reset_model_count


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
        self.no_enhancement_episodes = 0  # record episodes that no enhancement of  prediction performance
        if self.cfg.NASConfig['GPU']:
            self.train_test_data = []
            for i in range(len(train_test_data)):
                self.train_test_data.append(train_test_data[i].cuda())

    def step(self, action: Any):
        """
        do the actions given by agent and train the networks in pool, then drop low performance networks.
        :param action: action given by agent
        :return: observation, reward, done, info
        """
        net_pool = []
        for i in range(len(self.net_pool)):
            net = self.net_pool[i]
            action_i = action['action'][i]
            select = action_i['select']
            if net.train_times < 1000:
                net_pool.append(net)
                self.logger.info(f"net index {i}-{net.index} : continue training({net.train_times})")
            # select representations
            if select == 0:  # do nothing
                self.logger.info(f"net index {i}-{net.index} :do not change network")
                continue
            if select == 1:  # wider the net
                self.logger.info(f"net index {i}-{net.index} :wider the net")
                net = net.perform_wider_transformation(action_i['wider'])
                net_pool.append(net)
                continue
            if select == 2:  # deeper the net
                self.logger.info(f"net index {i}-{net.index} :deeper the net")
                net = net.perform_deeper_transformation(action_i['deeper'])
                if len(net.model_config.modules) <= self.cfg.NASConfig['MaxLayers']:  # constrain the network's depth
                    net_pool.append(net)
                continue
        self.net_pool = net_pool
        X_train, y_train, X_test, y_test = self.train_test_data
        feature_shape = X_train.shape[1:]
        base_structure = [  # manual design
            ['dense'],
        ]
        skeleton = random.sample(base_structure, 1)[0]
        self.net_pool.append(generate_new_model_config(self.cfg, feature_shape, 1, skeleton).generate_model())
        self.net_pool = list(set(self.net_pool))
        self._train_and_test()
        self.net_pool = sorted(self.net_pool, key=lambda x: x.test_loss)
        self.net_pool = self.net_pool[:self.pool_size]
        observation = self.net_pool
        reward = self.get_reward()
        done = True
        info = {}
        return observation, reward, done, info

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
        reward = []
        for net in self.net_pool:
            reward.append(1 / net.test_loss)
        return reward

    def reset(self):
        """
        reset all net to random init, this should call once in whole program
        train one round before return netpool
        :return:  net pool containing all network under searching
        """
        reset_model_count()
        X_train, y_train, X_test, y_test = self.train_test_data
        feature_shape = X_train.shape[1:]
        self.net_pool = [generate_new_model_config(self.cfg, feature_shape,1, [i]).generate_model() for i in  self.cfg.modulesConfig.keys()]
        self._train_and_test()
        self.render()
        return self.net_pool

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
