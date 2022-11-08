# -*- coding: utf-8 -*-
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
        self.trajectories = {}

    # Samples random trajectory
    def get_update_memory(self, max_len=0):
        memory = []
        # add running instances
        memory.extend(self.trajectories.values())
        # sample from memory
        if len(self.memory) <= max_len:
            memory.extend(self.memory)
        else:
            memory.extend(random.sample(self.memory, max_len))
        return memory

    def record_trajectory(self, transition):  # todo wait for test
        existing_keys = set(self.trajectories.keys())
        for k in transition.keys():
            t = k.state, transition[k]['action'], transition[k]['reward'], transition[k]['policy']
            t = recursive_tensor_detach(t)
            t = Transition(*t)
            if k in self.trajectories.keys():
                # recorded network, add transition
                existing_keys.remove(k)
                # add new transition to previous trajectory
                trajectory = self.trajectories.pop(k)
                trajectory.append(t)
                self.trajectories[transition[k]['next net']] = trajectory
            else:
                # new network, add new trajectory
                self.trajectories[transition[k]['next net']] = [t]
        # keep finished trajectory to memory
        if len(existing_keys) > 0:
            for k in existing_keys:
                trajectory = self.trajectories.pop(k)
                self.memory.append(trajectory)


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
