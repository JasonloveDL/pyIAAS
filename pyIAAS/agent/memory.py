# -*- coding: utf-8 -*-
import os.path
import pickle
import random
from collections import deque, namedtuple

import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy'))


class Trajectory():
    def __init__(self):
        self.trajectory = []

    def append(self, state, action, reward, policy):
        self.trajectory.append(Transition(state, action, reward, policy))  # Save s_i, a_i, r_i+1, µ(·|s_i)


# based on https://github.com/Kaixhin/ACER
class ReplayMemory():
    def __init__(self):
        # Max number of transitions possible will be the memory capacity, could be much less
        self.memory = deque()
        self.trajectories = []

    # Samples random trajectory
    def get_update_memory(self, max_len=0):
        memory = []
        # add running instances
        memory.extend(self.trajectories)
        # sample from memory
        if len(self.memory) <= max_len:
            memory.extend(self.memory)
        else:
            memory.extend(random.sample(self.memory, max_len))
        memory_ = []
        for i in memory:
            if len(i) > 0:
                memory_.append(i)
        return memory_

    def record_trajectory(self, in_pool_trajectory, finished_trajectory):
        self.trajectories = in_pool_trajectory
        self.memory.extend(finished_trajectory)

    def save_memories(self, save_dir):
        path = os.path.join(save_dir, 'replay_memory.pkl')
        with open(path, 'wb') as f:
            pickle.dump([self.memory, self.trajectories], f)

    def load_memories(self, save_dir):
        path = os.path.join(save_dir, 'replay_memory.pkl')
        if not os.path.exists(path):
            return
        with open(path, 'rb') as f:
            self.memory, self.trajectories = pickle.load(f)
        print(f'load memory :{len(self.memory)}')


def recursive_tensor_detach(data):
    if isinstance(data, dict):
        for k in data.keys():
            data[k] = recursive_tensor_detach(data[k])
        return data
    elif isinstance(data, torch.Tensor):
        if data.numel() == 1:
            return data.item()
        return data.detach()
    elif isinstance(data, list) or isinstance(data, tuple):
        return [recursive_tensor_detach(i) for i in data]
    else:
        return data
