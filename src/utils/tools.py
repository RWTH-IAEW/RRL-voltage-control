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

class BernoulliMaskReplayBuffer(ReplayBuffer):

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            mask_size : int,
            p_masking: float,
            device: Union[torch.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super(BernoulliMaskReplayBuffer, self).__init__(buffer_size=buffer_size,
                                           observation_space=observation_space,
                                           action_space=action_space,
                                           device=device,
                                           n_envs=n_envs,
                                           optimize_memory_usage=optimize_memory_usage,
                                           handle_timeout_termination=handle_timeout_termination)

        self.mask_size = mask_size
        self.bernoulli_mask = np.zeros((self.buffer_size, self.n_envs, self.mask_size), dtype=np.float32)

        self.p_masking = p_masking

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:

        super(BernoulliMaskReplayBuffer, self).add(obs, next_obs, action, reward, done, infos)

        # sample bernoulli mask
        mask = np.random.choice([1., 0.], p=[self.p_masking, 1-self.p_masking], size=(self.n_envs, self.mask_size)).astype(np.float32)
        self.bernoulli_mask[self.pos-1] = mask

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> BernoulliMaskReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)
            # Do not sample the element with index `self.pos` as the transitions is invalid
            # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)


        return self._get_samples(batch_inds, env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> BernoulliMaskReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self._normalize_reward(self.rewards[batch_inds], env),
            self.bernoulli_mask[batch_inds, 0, :]

        )
        return BernoulliMaskReplayBufferSamples(*tuple(map(self.to_torch, data)))





