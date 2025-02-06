import argparse

import warnings
from copy import deepcopy

from utils.tools import get_kwargs

from configs import RL_BASELINE_ENTRY_POINT, RESIDUAL_RL_ENTRY_POINT, VOLTAGE_CTRL

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-algo', '--algorithm', required=True, help='The algorithm you want to run (rl, residual)')
    parser.add_argument('-rname', '--run-name', help='The name of the run. Defaults to the algorithm name.')
    parser.add_argument('-train-start', '--train-start', type=int, default=1000, help='When the training begins.')
    parser.add_argument('-exp', '--exp-type', help='exp-type for better structuring of the experiments. Defaults to the env.')
    parser.add_argument('-env', '--environment', default='voltage', help='The environment you want to run (voltage)')
    parser.add_argument('-steps', '--training-steps', type=int, default=900000, help='The number of training steps.')
    parser.add_argument('-seed', '--seed', type=int, default=0, help='The training run seed.')
    parser.add_argument('-lam-res', '--lambda-residual', type=float, default=1.0, help="The fixed weight for residual RL approach.")
    parser.add_argument('-ctrl-inc', '--control-action-included', action='store_true', help="Whether the control action should be included in the observation space (only residual).")
    parser.add_argument('-qnet-arc', '--qnet-architecture', default='cnn', help="Setting the architecture of the Q-net (cnn = prepending shared linear layers, mlp)")
    parser.add_argument('-pfreq', '--policy-frequency', type=int, default=2, help='The policy update frequency.')

    args = parser.parse_args()

    if args.exp_type is None:
        args.exp_type = args.environment

    if args.run_name is None:
        args.run_name = args.algorithm

    if args.algorithm in ['rl', 'residual']:
        # Choose the right Entry Point
        if args.algorithm == 'rl':
            ENTRY_POINT = deepcopy(RL_BASELINE_ENTRY_POINT)
            entry_point = ENTRY_POINT['entry_point']['type']
            entry_point_kwargs = get_kwargs(ENTRY_POINT['entry_point'])
        elif args.algorithm == 'residual':
            ENTRY_POINT = deepcopy(RESIDUAL_RL_ENTRY_POINT)
            entry_point = ENTRY_POINT['entry_point']['type']
            entry_point_kwargs = get_kwargs(ENTRY_POINT['entry_point'])

        # Update name and log folder
        ENTRY_POINT['run_name'] = args.run_name
        ENTRY_POINT['exp_type'] = args.exp_type

        # Update environment
        if args.environment == 'voltage':
            ENTRY_POINT['env'] = deepcopy(VOLTAGE_CTRL)
        else:
            raise NotImplementedError()

        # Update general hyperparams
        ENTRY_POINT['training']['number_steps'] = args.training_steps
        ENTRY_POINT['training']['seed'] = args.seed
        ENTRY_POINT['training']['agent_trainer']['learning_starts'] = args.train_start
        ENTRY_POINT['training']['agent_trainer']['policy_frequency'] = args.policy_frequency

        # Update network architecture
        if args.qnet_architecture not in ['cnn', 'mlp']:
            raise NotImplementedError()
        ENTRY_POINT['training']['agent_trainer']['model_type_q'] = args.qnet_architecture

        # setting algorithm specific hyperparams
        if args.algorithm == 'residual':
            ENTRY_POINT['training']['weight_scheduler']['weight'] = args.lambda_residual
            ENTRY_POINT['env']['observation_manager']['ctrl_action_included'] = args.control_action_included

        entry_point(config=ENTRY_POINT, **entry_point_kwargs)
    else:
        raise NotImplementedError()