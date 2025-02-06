import random
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, NamedTuple, Optional

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium import spaces
import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import VecNormalize


class BernoulliMaskReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    masks: torch.Tensor


def seed_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def convert_to_tensor(data, device):

    if isinstance(data, torch.Tensor):
        data.to(device=device)
        return data
    else:
        tensor = torch.tensor(data, device=device)
        return tensor

def get_kwargs(dict):
    """
    Gets the kwargs provided in the config files
    Args:
        dict:

    Returns: kwargs

    """
    return {k: v for k, v in dict.items() if k != 'type'}


def inject_weight_into_state(state, weight):
    if len(state.shape) == 1:
        weight_array = np.array([weight], dtype=np.float32)
        array = np.append(state.copy(), weight_array)
    else:
        weight_array = np.array([[weight]], dtype=np.float32)
        array = np.append(state.copy(), weight_array, axis=1)

    return array

def inject_ctrl_action_into_state(state, action):
    if len(state.shape) == 1:
        array = np.append(state.copy(), action)
    else:
        if len(action.shape) == 1:
            action = action[np.newaxis, :]
        array = np.append(state.copy(), action, axis=1)

    return array


def compute_clipped_cv(std, mean):
    clipped_cv = np.where(np.abs(mean)>1, std/np.abs(mean), std)
    return clipped_cv


def compute_distance_between_vectors(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape
    with torch.no_grad():
        return (tensor1 - tensor2).pow(2).sum(dim=-1).sqrt()


def make_env(env_id, seed):
    """
    Returns a thunk that creates and initializes a gym environment with the given ID and seed
    Args:
        env_id: string identifying the gym environment to create
        seed: integer specifying the random seed to use for the environment
    Returns:
        callable thunk that creates and returns a gym environment with a seeded initial state, action space, and observation spaces
    """

    def thunk():
        env = gym.make(env_id)
        # env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


class SimpleRingBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.cur_index = 0
        self.buffer = []

    def add_element(self, element):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(element)
        else:
            self.buffer[self.cur_index % self.buffer_size] = element
            self.cur_index += 1

    def get_buffer_mean(self):
        return np.mean(self.buffer)

    def clear_buffer(self):
        self.buffer = []
        self.cur_index = 0


class WeightScheduler(ABC):

    def __init__(self, start_weight):
        self.current_weight = start_weight

    @abstractmethod
    def adapt_weight(self, uncertainty, step, force=False):
        pass

    @abstractmethod
    def episode_weight_reset(self):
        pass

    def get_weight(self):
        return self.current_weight


class FixedWeightScheduler(WeightScheduler):

    def __init__(self, weight):
        super().__init__(weight)

    def episode_weight_reset(self):
        pass

    def adapt_weight(self, uncertainty, step, force=False):
        pass





