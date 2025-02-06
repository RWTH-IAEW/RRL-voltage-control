from abc import ABC, abstractmethod
from collections import deque

import numpy
import scipy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

from .observation_manager import ObservationManager


class Controller(nn.Module, ABC):

    def __init__(self, obs_man: ObservationManager):
        self.obs_man = obs_man
        super(Controller, self).__init__()

    @abstractmethod
    def get_action(self, state, greedy=True):
        pass


class AttenuatedController(Controller):
    def __init__(self, controller: Controller, attenuation_factor):
        super(AttenuatedController, self).__init__(controller.obs_man)
        self.attenuation_factor = attenuation_factor
        self.controller = controller

    def get_action(self, state, greedy=True):
        return self.attenuation_factor * self.controller.get_action(state, greedy)