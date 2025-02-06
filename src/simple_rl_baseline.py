import warnings
from datetime import datetime

import numpy as np

from custom_envs.observation_manager import Algorithm

from utils.logger import Logger
from utils.plotter import Plotter
from utils.tools import seed_experiment, get_kwargs

import gymnasium as gym
import wandb
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config):
    # READ RELEVANT CONFIG DATA
    env_config = config['env']
    log_config = config['logging']
    train_config = config['training']
    eval_config = config['evaluation']

    device = train_config['device'] if train_config['device'] != 'auto' else ("cuda" if torch.cuda.is_available() else "cpu")
    seed = train_config['seed']
    number_steps = train_config['number_steps']

    eval_agent = eval_config['eval_agent']
    eval_count = eval_config['eval_count']
    eval_frequency = eval_config['eval_frequency']

    wandb_prj_name = log_config['wandb_project_name']
    capture_video = log_config['capture_video']
    model_save_frequency = log_config['model_save_frequency']

    env_id = env_config['env_id']
    exp_type = config['exp_type']
    run_name = config['run_name']

    augmented_env = False

    seed_experiment(seed)

    envs = gym.vector.SyncVectorEnv(
        [env_config['make_env_function']['type'](seed=seed, **get_kwargs(env_config['make_env_function']))])
    envs.single_observation_space.dtype = np.float32
    
    obs_man_kwargs = get_kwargs(env_config['observation_manager'])
    obs_man_kwargs['ctrl_action_included'] = False
    obs_man = env_config['observation_manager']['type'](augmented_env=augmented_env, env=envs.envs[0],
                                                        **obs_man_kwargs)

    agent_trainer = train_config['agent_trainer']['type'](envs, obs_man, device,
                                                          **get_kwargs(train_config['agent_trainer']))

    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')

    # name prefix of output files
    run_name = f"{run_name}_{env_id}_s={seed}_{timestamp}"

    # init list to track agent's performance throughout training
    last_evaluated_episode = None  # stores the episode_step of when the agent's performance was last evaluated
    eval_max_return = -float('inf')

    # Eval
    eval_env = env_config['make_env_function']['type'](seed=seed, **get_kwargs(env_config['make_env_function']))()
    evaluator = env_config['evaluator']['type'](eval_env, obs_man, device=device, eval_count=eval_count,
                                                greedy_eval=True,
                                                **get_kwargs(env_config['evaluator']))
    logger = Logger(run_name=run_name, exp_name=exp_type)
    plotter = Plotter(logger, augmented_env=augmented_env)

    global_step = 0
    episode_step = 0

    no_fails = 0

    obs, _ = envs.reset(seed=seed)

    total_timesteps = train_config['number_steps']

    logger.init_wandb_logging(wandb_prj_name=wandb_prj_name,
                              config={
                                  "config": config,
                                  "hyperparams": agent_trainer.get_hyperparams_dict(),
                                  "timestamp": timestamp
                              })

    # TRAINING LOOP
    for global_step in range(total_timesteps):

        state = torch.Tensor(obs).to(device)

        with torch.no_grad():
            actions_pi = agent_trainer.get_exploration_action(state, global_step)

        actions = actions_pi

        # execute the game and log data
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        real_next_obs = next_obs.copy()

        # record rewards for plotting purposes at the end of every episode
        if 'final_info' in infos:

            for info in infos['final_info']:

                wandb.log(
                    {"rollout/episodic_return": info["episode"]["r"],
                     "rollout/episodic_length": info["episode"]["l"],
                     "rollout/fails": no_fails,
                     "Charts/episode_step": episode_step}, step=global_step
                )

                episode_step += 1

                print(f"Step:{global_step}, Episode: {episode_step}, Reward: {info['episode']['r']}")

                # generate average performance statistics of current learned agent
                if eval_agent and episode_step % eval_frequency == 0 and last_evaluated_episode != episode_step and global_step >= \
                        agent_trainer.get_learning_starts()[0]:

                    tracked_return_10 = evaluator.evaluate_agent_on_start_states(agent_trainer.agent, agent_type=Algorithm.RL)

                    wandb.log({"evaluation/evaluation_return": tracked_return_10,
                               }, step=global_step)

                    last_evaluated_episode = episode_step

                    print("Performance for eval weight: ", tracked_return_10, eval_max_return)
                    if tracked_return_10 > eval_max_return:
                        eval_max_return = tracked_return_10
                        logger.save_model(agent_trainer.agent)
                        if capture_video:
                            frames = evaluator.collect_video_frames(agent_trainer.agent, agent_type=Algorithm.RL)
                            video_file = plotter.create_video_from_frames(frames, episode_step, fps=30)
                            wandb.log({'video': wandb.Video(video_file, fps=4, format='gif')}, step=global_step)

        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos['final_observation'][idx]

        # save data to replay buffer
        agent_trainer.add_to_replay_buffer(obs, real_next_obs, actions_pi, rewards, terminations, infos)

        obs = next_obs

        # execute agent trainer train method for gradient descends
        agent_trainer.train_and_log(global_step, episode_step)

        if global_step % model_save_frequency == 0:
            logger.save_model(agent_trainer.agent, f'agent_{global_step}')

    envs.close()

    # writer.close()
    if wandb.run is not None:
        wandb.finish(quiet=True)
        wandb.init(mode="disabled")


    logger.save_model(agent_trainer.agent, model_name='final_agent')


