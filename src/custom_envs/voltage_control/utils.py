import gymnasium as gym
import numpy as np
import torch

from custom_envs.observation_manager import Algorithm, ObservationManager
from utils.evaluator import Evaluator
from utils.tools import inject_weight_into_state

# import voltage_control_env as vc
from voltage_control_env.env import FlattenObservationWrapper, MeanRewardWrapper, SumRewardWrapper
from voltage_control_env.dataset import SimbenchCodeDataSet, SimbenchDataSet, LoadedGridDataSet
from voltage_control_env.env_utils import create_PQ_area, StandardObservationGenerator, StandardRewardGenerator, ScenarioManager

# make env function
def make_voltage_env(seed,
                     delta_step_env=False,
                     path_to_net=None,
                     random_setpoint_reset=True,
                     step_delay=0,
                     normalize_obs=True,
                     reward_setting='global',
                     episode_steps=20,
                     delta_step=0.0001,
                     reward_wrapper='mean',
                     switch_reward=True
                     ):
    def thunk():

        if path_to_net is None:
            net_path = "./notebooks/data/1-LV-rural1-upper-augmented-OPF-cleaned.p"
        else:
            net_path = path_to_net
        
        #sim_data = SimbenchCodeDataSet("1-LV-rural1--2-sw")
        sim_data = LoadedGridDataSet(net_path=net_path)
        sgen_indices = list(range(len(sim_data.net.sgen)))
        # pq_areas = [create_PQ_area('cone') for _ in sgen_indices]
        scenario_manager = ScenarioManager(net=sim_data.net,
                                           dataset=sim_data,
                                           ctrl_sgen_idx=sgen_indices,
                                           pq_areas = 'cone90')

        reward_gen = StandardRewardGenerator(sim_data.net, sgen_indices, min_volt=0.95, max_volt=1.05,
                                             global_curtail_norm=True if reward_setting=='global' else False,
                                             switch_reward=switch_reward)
        obs_gen = StandardObservationGenerator(sim_data.net, sgen_indices, normalize=normalize_obs)

        if delta_step_env:

            # delta_step = 0.0035  # ugly to set it manually
            env = gym.make('DeltaStepVoltageControlEnv-v0',
                           scenario_manager=scenario_manager,
                           reward_generator=reward_gen,
                           observation_generator=obs_gen,
                           delta_step=delta_step,
                           random_setpoint_reset=random_setpoint_reset)

            env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_steps)

        else:
            env = gym.make('VoltageControlEnv-v0',
                           scenario_manager=scenario_manager,
                           reward_generator=reward_gen,
                           observation_generator=obs_gen,
                           random_setpoint_reset=random_setpoint_reset,
                           step_delay=step_delay)

            env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_steps)
        
        # create corresponding wrappers
        env = FlattenObservationWrapper(env)
        if reward_wrapper == 'mean':
            env = MeanRewardWrapper(env)
        elif reward_wrapper == 'sum':
            env = SumRewardWrapper(env)
        else:
            raise ValueError('Unknown Reward Wrapper!')
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # apply seeding
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env
        

    return thunk


class VoltageControlEvaluator(Evaluator):

    def __init__(self,
                 env,
                 obs_man: ObservationManager,
                 device=torch.device('cpu'),
                 eval_count=10,
                 seed=42,
                 greedy_eval=True,
                 random_sampling=True
                 ):
        
        self.random_sampling = random_sampling

        if not random_sampling:
            eval_count = min(eval_count, env.sm.dataset.get_length())

        super().__init__(env=env, obs_man=obs_man, device=device, eval_count=eval_count, seed=seed, greedy_eval=greedy_eval)


    def init_eval_start_states(self):
        eval_start_states = []
        for i in range(self.eval_count):
            if self.random_sampling:
                self.single_eval_env.reset(seed=self.seed + i)
                eval_start_states.append(self.single_eval_env.scenario)
            else:
                self.single_eval_env.reset(options={'scenario_id': i})
                eval_start_states.append(self.single_eval_env.scenario)

        return eval_start_states

    def init_start_state(self, state_no):
        state, _ = self.single_eval_env.reset(options={'scenario': self.eval_start_states[state_no], 'random_setpoint_reset': False})
        return state
    
    def evaluate_states_and_voltages_over_episode(self, agent, agent_type, start_state_no=0, weight=1):

        states = []
        voltages = []

        done = False
        state = self.init_start_state(start_state_no)
        if self.obs_man.augmented_env:
            state = inject_weight_into_state(state, weight)
        voltage = np.array(self.single_eval_env.sm.net.res_bus['vm_pu'])
        voltages.append(voltage)
        states.append(state.copy())
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        while not done:
            with torch.no_grad():
                agent_state = self._convert_state_for_agent(state, agent_type)
                if agent_type == Algorithm.MIXED:
                    action = agent.get_action(agent_state, greedy=True, weight=weight)
                else:
                    action = agent.get_action(agent_state, greedy=True)
                action = action.squeeze().cpu().numpy()
                action = action.clip(self.single_eval_env.action_space.low.reshape(action.shape),
                                        self.single_eval_env.action_space.high.reshape(action.shape))
            next_state, reward, terminated, truncated, _ = self.single_eval_env.step(action)
            done = terminated or truncated
            if self.obs_man.augmented_env:
                next_state = inject_weight_into_state(next_state, weight)
            # state = torch.tensor(next_state, dtype=torch.float32)
            voltage = np.array(self.single_eval_env.sm.net.res_bus['vm_pu'])
            voltages.append(voltage)
            states.append(next_state.copy())
            state = torch.tensor(next_state, device=self.device, dtype=torch.float32).unsqueeze(0)

        return states, voltages


class VoltageControlObservationManager(ObservationManager):
    
    def __init__(self, augmented_env: bool, env: gym.Env, ctrl_action_included: bool = False):
        self._augmented_env = augmented_env
        self._ctrl_action_included = ctrl_action_included
        self._raw_obs_dim = env.observation_space.shape[0]
        self._total_action_size = np.prod(env.action_space.shape)
        self._obs_space_rl = self.__construct_rl_obs_space(env)

    def get_rl_state(self, state: torch.Tensor | np.ndarray):
        return state

    def get_ctrl_state(self, state: torch.Tensor | np.ndarray):
        if len(state.shape) > 1:
            return state[:, :self._raw_obs_dim]
        else:
            return state[:self._raw_obs_dim]

    @property
    def obs_shape_rl(self):
        return (self._raw_obs_dim + self.ctrl_action_included * self._total_action_size + self.augmented_env,)

    @property
    def obs_shape_ctrl(self):
        return (self._raw_obs_dim,)

    @property
    def obs_space_rl(self):
        return self._obs_space_rl

    def __construct_rl_obs_space(self, env):
        low = env.observation_space.low
        high = env.observation_space.high

        if self.ctrl_action_included:
            low = np.append(low, np.ones(self._total_action_size) * -1)
            high = np.append(high, np.ones(self._total_action_size) * 1)

        if self.augmented_env:
            low = np.append(low, 0)
            high = np.append(high, 1)

        obs_space = gym.spaces.Box(low=low, high=high, shape=high.shape)

        return obs_space

    @property
    def augmented_env(self):
        return self._augmented_env
    
    @property
    def ctrl_action_included(self,):
        return self._ctrl_action_included

if __name__ == '__main__':
    pass

