from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch

import gymnasium as gym


class Algorithm(Enum):
    RL = 'rl'
    CTRL = 'ctrl'
    MIXED = 'mixed'


class ObservationManager(ABC):

    @abstractmethod
    def __init__(self, augmented_env: bool, single_env: gym.Env, ctrl_action_included: bool = False):
        pass

    @abstractmethod
    def get_rl_state(self, state: torch.Tensor | np.ndarray):
        pass

    @abstractmethod
    def get_ctrl_state(self, state: torch.Tensor | np.ndarray):
        pass

    @property
    @abstractmethod
    def obs_shape_rl(self):
        pass

    @property
    @abstractmethod
    def obs_shape_ctrl(self):
        pass

    @property
    @abstractmethod
    def obs_space_rl(self):
        pass

    @property
    @abstractmethod
    def augmented_env(self):
        pass

    @property
    @abstractmethod
    def ctrl_action_included(self,):
        pass
