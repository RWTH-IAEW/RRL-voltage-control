import copy
from abc import abstractmethod, ABC

import torch
import numpy as np

from custom_envs.observation_manager import ObservationManager, Algorithm
from utils.tools import inject_weight_into_state, inject_ctrl_action_into_state, WeightScheduler

class Evaluator(ABC):

    def __init__(self,
                 env,
                 obs_man: ObservationManager,
                 device=torch.device('cpu'),
                 eval_count=10,
                 seed=42,
                 greedy_eval=True,
                 ):

        self.eval_count = eval_count
        self.seed = seed
        self.greedy_eval = greedy_eval
        self.obs_man = obs_man
        self.device = device

        # setting the eval_env
        self.single_eval_env = env
        self.single_eval_env.unwrapped.render_mode = 'rgb_array'

        # generate the test start states
        self.eval_start_states = self.init_eval_start_states()


    def _convert_state_for_agent(self, state, agent_type):
        if agent_type == Algorithm.RL:
            return self.obs_man.get_rl_state(state)
        elif agent_type == Algorithm.CTRL:
            return self.obs_man.get_ctrl_state(state)
        else:
            return state

    @abstractmethod
    def init_eval_start_states(self):
        pass

    @abstractmethod
    def init_start_state(self, state_no):
        pass

    def evaluate_uncertainty_on_start_states(self, agent_trainer, weight, cv=False):

        states = [inject_weight_into_state(self.eval_start_states[j], weight) for j in range(self.eval_count)]
        states = np.array(states)
        states = torch.tensor(states, device=self.device)
        states = self.obs_man.get_rl_state(states)

        with torch.no_grad():
            actions = agent_trainer.agent.get_action(states)
            mean, std = agent_trainer.get_q_net_std(states, actions)

        if cv:
            return torch.abs(std/mean).mean().item()
        else:
            return std.mean().item()

    def evaluate_static_on_start_states(self, agent_trainer, weight):
        return self.evaluate_static_on_states(agent_trainer, weight, self.eval_start_states)

    def evaluate_static_on_states(self, agent_trainer, weight, states):

        states = [inject_weight_into_state(state, weight) for state in states]
        states = np.array(states)
        states = torch.tensor(states, device=self.device)
        states = self.obs_man.get_rl_state(states)

        with torch.no_grad():
            actions = agent_trainer.agent.get_action(states)
            mean, std = agent_trainer.get_q_net_std(states, actions)

        return mean.mean().item(), std.mean().item()

    def evaluate_agent_on_start_states(self, agent, agent_type, weight=1):

        done = False
        # env = gym.make(env_id)
        total_reward = [0] * len(self.eval_start_states)
        for i in range(len(self.eval_start_states)):
            state = self.init_start_state(i)
            if self.obs_man.augmented_env:
                state = inject_weight_into_state(state, weight)
            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state = self._convert_state_for_agent(state, agent_type)
            while not done:
                with torch.no_grad():
                    if agent_type == Algorithm.MIXED:
                        action = agent.get_action(state.unsqueeze(0), self.greedy_eval, weight=weight)
                    else:
                        action = agent.get_action(state.unsqueeze(0), self.greedy_eval)
                action = action.squeeze().cpu().numpy()
                action = action.clip(self.single_eval_env.action_space.low.reshape(action.shape),
                                     self.single_eval_env.action_space.high.reshape(action.shape))
                next_state, reward, terminated, truncated, _ = self.single_eval_env.step(action)
                done = terminated or truncated
                if self.obs_man.augmented_env:
                    next_state = inject_weight_into_state(next_state, weight)
                # state = torch.tensor(next_state, dtype=torch.float32)
                state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
                state = self._convert_state_for_agent(state, agent_type)
                total_reward[i] += reward
            done = False
        return np.mean(total_reward)

    @DeprecationWarning
    def evaluate_mixed_agent_on_start_states_diff_weights(self, mixed_agent, action_weight, eval_weight):

        done = False
        # env = gym.make(env_id)
        total_reward = [0] * len(self.eval_start_states)
        for i in range(len(self.eval_start_states)):
            state = self.init_start_state(i)
            if self.obs_man.augmented_env:
                state = inject_weight_into_state(state, action_weight)
            state = torch.tensor(state, device=self.device)
            while not done:
                with torch.no_grad():
                    # action_pi = mixed_agent.get_rl_action(state, self.greedy_eval)
                    # action_c = mixed_agent.get_control_action(state)
                    action = mixed_agent.get_action(state, self.greedy_eval)
                # action = action_pi * eval_weight + action_c * (1-eval_weight)
                action = action.squeeze().cpu().numpy()
                action = action.clip(self.single_eval_env.action_space.low.reshape(action.shape),
                                     self.single_eval_env.action_space.high.reshape(action.shape))
                next_state, reward, terminated, truncated, _ = self.single_eval_env.step(action)
                done = terminated or truncated
                if self.obs_man.augmented_env:
                    next_state = inject_weight_into_state(next_state, action_weight)
                state = torch.tensor(next_state, device=self.device)
                total_reward[i] += reward
            done = False

        return np.mean(total_reward)

    def evaluate_actions_over_episode(self, agents, agent_types, weight=0.5):

        return self.evaluate_actions_over_episode_from_state(agents, agent_types, 0, weight=weight)

    def evaluate_actions_over_episode_from_state(self, agents, agent_types, start_state_no, weight=0.5):
        actions = []
        for _ in agents:
            actions.append([])

        states = []

        done = False
        state = self.init_start_state(start_state_no)
        if self.obs_man.augmented_env:
            state = inject_weight_into_state(state, weight)
        states.append(state.copy())
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        while not done:
            with torch.no_grad():
                for idx, agent in enumerate(agents):
                    agent_state = self._convert_state_for_agent(state, agent_types[idx])
                    if agent_types[idx] == Algorithm.MIXED:
                        action = agent.get_action(agent_state, greedy=True, weight=weight)
                    else:
                        action = agent.get_action(agent_state, greedy=True)
                    action = action.squeeze().cpu().numpy()
                    action = action.clip(self.single_eval_env.action_space.low.reshape(action.shape),
                                         self.single_eval_env.action_space.high.reshape(action.shape))
                    actions[idx].append(action)
            next_state, reward, terminated, truncated, _ = self.single_eval_env.step(actions[0][-1])
            done = terminated or truncated
            if self.obs_man.augmented_env:
                next_state = inject_weight_into_state(next_state, weight)
            # state = torch.tensor(next_state, dtype=torch.float32)
            states.append(next_state.copy())
            state = torch.tensor(next_state, device=self.device, dtype=torch.float32).unsqueeze(0)

        return actions, states

    def collect_video_frames(self, agent, agent_type, weight=0.5, random_start_state=True, start_state_no=0):

        frames = []

        done = False
        if random_start_state:
            state, _ = self.single_eval_env.reset()
        else:
            state = self.init_start_state(start_state_no)
        
        # initial frame
        frames.append(self.single_eval_env.render())

        if self.obs_man.augmented_env:
            state = inject_weight_into_state(state, weight)
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        while not done:
            with torch.no_grad():
                state = self._convert_state_for_agent(state, agent_type)
                if agent_type == Algorithm.MIXED:
                    action = agent.get_action(state.unsqueeze(0), greedy=True, weight=weight)
                else:
                    action = agent.get_action(state.unsqueeze(0), greedy=True)

                action = action.squeeze().cpu().numpy()
                action = action.clip(self.single_eval_env.action_space.low.reshape(action.shape),
                                     self.single_eval_env.action_space.high.reshape(action.shape))
            next_state, reward, terminated, truncated, _ = self.single_eval_env.step(action)
            done = terminated or truncated
            if self.obs_man.augmented_env:
                next_state = inject_weight_into_state(next_state, weight)
            state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
            # state = state.unsqueeze(0)
            out = self.single_eval_env.render()
            frames.append(out)

        return frames
