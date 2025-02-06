from copy import deepcopy

from utils.tools import FixedWeightScheduler
from custom_envs.voltage_control.utils import make_voltage_env, VoltageControlEvaluator, VoltageControlObservationManager
from custom_envs.voltage_control.controller import CentralizedDroopController
from simple_rl_baseline import train as train_rl
from residual_rl_baseline import train as train_res_rl
from sac_trainer import SACTrainer

LOGGING = {
    'wandb_project_name': 'RRL-Voltage-Control',
    'capture_video': True,
    'model_save_frequency': 50000
}

EVALUATION = {
    'eval_agent': True,
    'eval_count': 30,
    'eval_frequency': 250,
}

TRAIN_CONFIG = {
    'agent_trainer': {'type': SACTrainer,
                      'hidden_layer_size_q': 512, 'hidden_layers_q': 1, 'cnn_filters_q': 32, 'cnn_layers_q': 2,
                      'hidden_layer_size_actor': 32, 'hidden_layers_actor': 4, 'attention_heads_actor': 4, 'transformer_ff_size_actor': 64,
                      'model_type_q': 'cnn', 'model_type_actor': 'transformer'},
    'number_steps': 900000,
    'device': 'auto',
    'seed': 1,
}

VOLTAGE_CTRL_TRAIN_CONFIG = deepcopy(TRAIN_CONFIG)
VOLTAGE_CTRL_TRAIN_CONFIG['agent_trainer']['q_lr'] = 3e-4

VOLTAGE_CTRL = {
    'env_id': 'VoltageControlEnv-v0',
    'make_env_function': {'type': make_voltage_env, 'delta_step_env': False, 'random_setpoint_reset': True,
                          'step_delay': 0.3684, 'normalize_obs': True, 'reward_setting': 'global', 'episode_steps': 20,
                          'path_to_net': "./data/1-LV-rural3-minQmaxP-augmented-curtail_cleaned.p"},
                          #'path_to_net': "./data/1-LV-rural1-upper-augmented-OPF-cleaned.p"},
    'evaluator': {'type': VoltageControlEvaluator},
    'controller': {'type': CentralizedDroopController, 'p_u_thresholds': (1.045, 1.055), 'q_u_thresholds': (0.97, 0.99, 1.01, 1.03)},
    'observation_manager': {'type': VoltageControlObservationManager, 'ctrl_action_included': False}
}

ENTRY_POINT_ABSTRACT = {
    'env': VOLTAGE_CTRL,
    'training': TRAIN_CONFIG,
    'evaluation': EVALUATION,
    'logging': LOGGING,
}

# RL BASELINE CONFIG
RL_BASELINE_ENTRY_POINT = deepcopy(ENTRY_POINT_ABSTRACT)
RL_BASELINE_ENTRY_POINT['training'] = deepcopy(VOLTAGE_CTRL_TRAIN_CONFIG)
RL_BASELINE_ENTRY_POINT['entry_point'] = {'type': train_rl}

# RESIDUAL RL CONFIG
RESIDUAL_RL_ENTRY_POINT = deepcopy(ENTRY_POINT_ABSTRACT)
RESIDUAL_RL_ENTRY_POINT['training'] = deepcopy(VOLTAGE_CTRL_TRAIN_CONFIG)
RESIDUAL_RL_ENTRY_POINT['training']['weight_scheduler'] = {
    'type': FixedWeightScheduler, 'weight': 1
}
RESIDUAL_RL_ENTRY_POINT['entry_point'] = {'type': train_res_rl}

