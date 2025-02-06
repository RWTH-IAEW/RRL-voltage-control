from datetime import datetime

import numpy as np
import torch

import wandb
import gymnasium as gym

from agents import MixedAgent
from custom_envs.observation_manager import Algorithm
from utils.logger import Logger
from utils.plotter import Plotter
from utils.tools import inject_ctrl_action_into_state, inject_weight_into_state, seed_experiment, \
    get_kwargs


def train(config):
    # READ RELEVANT CONFIG DATA
    env_config = config['env']
    log_config = config['logging']
    train_config = config['training']
    eval_config = config['evaluation']

    device = train_config['device'] if train_config['device'] != 'auto' else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f'Used device {device}')

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

    obs_man = env_config['observation_manager']['type'](augmented_env=augmented_env, env=envs.envs[0],
                                                        **get_kwargs(env_config['observation_manager']))

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

    controller = env_config['controller']['type'](obs_man=obs_man, **get_kwargs(env_config['controller'])).to(device)
    mixed_agent = MixedAgent(agent_trainer.agent, controller, obs_man=obs_man, mixing_type='residual')

    weight_scheduler = train_config['weight_scheduler']['type'](**get_kwargs(train_config['weight_scheduler']))
    weight = weight_scheduler.get_weight()

    # evaluating controllers
    suboptimal_controller_return = evaluator.evaluate_agent_on_start_states(controller, agent_type=Algorithm.CTRL)

    # last inits
    global_step = 0
    episode_step = 0

    no_fails = 0

    obs, _ = envs.reset(seed=seed)

    mixing_weights = []

    logger.init_wandb_logging(wandb_prj_name=wandb_prj_name,
                              config={
                                  "config": config,
                                  "hyperparams": agent_trainer.get_hyperparams_dict(),
                                  "timestamp": timestamp
                              })

    episode_actions_pi = []
    episode_actions_c = []

    # TRAINING LOOP
    for global_step in range(number_steps):

        mixing_weights.append(weight)
        state = torch.tensor(obs, dtype=torch.float32).to(device)

        # Evaluate control action
        with torch.no_grad():
            actions_c = controller.get_action(obs_man.get_ctrl_state(state))
            actions_c = actions_c.detach().cpu().numpy().clip(envs.single_action_space.low.reshape(actions_c.shape),
                                                              envs.single_action_space.high.reshape(actions_c.shape))
        if obs_man.ctrl_action_included:
            obs = inject_ctrl_action_into_state(obs, actions_c)
        if augmented_env:
            obs = inject_weight_into_state(obs, weight)
        
        state = torch.Tensor(obs).to(device)
        
        # Evaluate RL action
        with torch.no_grad():
            actions_pi = agent_trainer.get_exploration_action(state, global_step)

        mixing_component = weight

        episode_actions_pi.append(actions_pi)
        episode_actions_c.append(actions_c)

        actions = mixing_component * actions_pi + actions_c
        actions = actions.clip(envs.single_action_space.low.reshape(actions_c.shape),
                               envs.single_action_space.high.reshape(actions_c.shape))

        # execute the game and log data
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        real_next_obs = next_obs.copy()

        if obs_man.ctrl_action_included:
            next_state = torch.tensor(next_obs, dtype=torch.float32).to(device)

            # Evaluate control action
            with torch.no_grad():
                actions_c = controller.get_action(obs_man.get_ctrl_state(next_state))
                actions_c = actions_c.detach().cpu().numpy().clip(envs.single_action_space.low.reshape(actions_c.shape),
                                                                  envs.single_action_space.high.reshape(
                                                                      actions_c.shape))
                real_next_obs = inject_ctrl_action_into_state(real_next_obs, actions_c)

        if augmented_env:
            real_next_obs = inject_weight_into_state(next_obs, mixing_component)

        wandb.log({
            "rollout/step_lambda": mixing_component,
            },
            step=global_step)

        episode_actions_pi.append(actions_pi)
        episode_actions_c.append(actions_c)

        # record rewards for plotting purposes at the end of every episode

        if 'final_info' in infos:
            # for idx, info in enumerate(infos):
            for info in infos['final_info']:

                wandb.log(
                    {"rollout/episodic_return": info["episode"]["r"],
                     "rollout/episodic_length": info["episode"]["l"],
                     "rollout/mixing_coeff_mean": np.mean(mixing_weights),
                     "Charts/episode_step": episode_step}, step=global_step
                )

                episode_step += 1

                print(f"Step:{global_step}, Episode: {episode_step}, Reward: {info['episode']['r']}")

                # generate average performance statistics of current learned agent
                if eval_agent and episode_step % eval_frequency == 0 and last_evaluated_episode != episode_step and global_step >= \
                        agent_trainer.get_learning_starts()[0]:

                    # save model
                    logger.save_model(mixed_agent, f'agent_{global_step}')

                    eval_return = evaluator.evaluate_agent_on_start_states(mixed_agent, agent_type=Algorithm.MIXED, weight=weight)

                    if global_step >= agent_trainer.get_learning_starts()[1] or last_evaluated_episode is None:
                        pass

                    wandb.log({
                        "rollout/evaluation_return_agent_controller": suboptimal_controller_return,
                    }, step=global_step)

                    wandb.log({
                        f"evaluation/evaluation_return": eval_return
                    }, step=global_step)

                    last_evaluated_episode = episode_step

                    print("Performance for eval weight: ", eval_return, eval_max_return)
                    if eval_return > eval_max_return:
                        eval_max_return = eval_return
                        logger.save_model(agent_trainer.agent)
                        if capture_video:
                            frames = evaluator.collect_video_frames(mixed_agent, agent_type=Algorithm.MIXED, weight=weight,
                                                                    random_start_state=False)
                            video_file = plotter.create_video_from_frames(frames, episode_step, fps=30)
                            wandb.log({'video': wandb.Video(video_file, fps=4, format='gif')}, step=global_step)

                mixing_weights = []
                episode_actions_c = []
                episode_actions_pi = []

        for idx, trunc in enumerate(truncations):
            if trunc:
                final_obs = infos['final_observation'][idx]

                if obs_man.ctrl_action_included:
                    final_state = torch.tensor(final_obs, dtype=torch.float32).to(device)

                    # Evaluate control action
                    with torch.no_grad():
                        actions_c = controller.get_action(obs_man.get_ctrl_state(final_state))
                        actions_c = actions_c.detach().cpu().numpy().clip(
                            envs.single_action_space.low.reshape(actions_c.shape),
                            envs.single_action_space.high.reshape(actions_c.shape))
                        final_obs = inject_ctrl_action_into_state(final_obs, actions_c)

                if augmented_env:
                    final_obs = inject_weight_into_state(final_obs, mixing_component)

                real_next_obs[idx] = final_obs

        # save data to replay buffer
        agent_trainer.add_to_replay_buffer(obs, real_next_obs, actions_pi, rewards, terminations, infos)

        obs = next_obs

        # execute agent trainer train method for gradient descends
        agent_trainer.train_and_log(global_step, episode_step)

        if global_step % model_save_frequency == 0:
            logger.save_model(mixed_agent, f'agent_{global_step}')

    envs.close()

    # writer.close()
    if wandb.run is not None:
        wandb.finish(quiet=True)
        wandb.init(mode="disabled")

    logger.save_model(mixed_agent, model_name='final_agent')

