import time

import numpy as np
import torch
from agents import EnsembleSACAgent
import torch.optim as optim
import wandb
from easydict import EasyDict as edict

from custom_envs.observation_manager import ObservationManager
from utils.tools import BernoulliMaskReplayBuffer


class SACTrainer:

    def __init__(self, env, obs_man: ObservationManager, device,
                 learning_starts=10000,
                 learning_starts_actor=None,
                 buffer_size=int(1e6),
                 batch_size=256,
                 gamma=0.99,
                 tau=0.005,
                 policy_lr=3e-4,
                 q_lr=1e-3,
                 policy_frequency=2,
                 target_network_frequency=1,
                 alpha=0.2,
                 autotune=True,
                 ensemble_size=2,
                 use_rpf=False,
                 rpf_scale=0,
                 model_type_q='mlp',
                 model_type_actor='mlp',
                 hidden_layer_size_q=64,
                 hidden_layer_size_actor=16,
                 hidden_layers_q = 1,
                 hidden_layers_actor = 1,
                 cnn_filters_q=32,
                 cnn_filters_actor=32,
                 cnn_layers_q=2,
                 cnn_layers_actor=2,
                 attention_heads_q = 4,
                 transformer_ff_size_q = 128,
                 attention_heads_actor=4,
                 transformer_ff_size_actor=64,
                 attenuate_actor=False,
                 bernoulli_mask_coeff=1.,
                 action_scale=1,
                 initial_random_sampling=True,
                 random_target=True,
                 update_steps=1,
                 pi_update_avg_q=False
                 ):
        # agent training specific parameters and hyperparameters
        self.hypp = edict()

        # agent training specific parameters and hyperparameters
        self.hypp.learning_starts = learning_starts  # timestep after which gradient descent updates begins
        if learning_starts_actor is not None:
            self.hypp.learning_starts_actor = learning_starts_actor  # timestep after which actor starts being trained
        else:
            self.hypp.learning_starts_actor = learning_starts
        self.hypp.buffer_size = buffer_size  # size of replay buffer
        self.hypp.batch_size = batch_size  # number of mini-batches for gradient updates

        self.hypp.gamma = gamma  # discount factor over future rewards
        self.hypp.tau = tau  # smoothing coefficient for target q networks parameters
        self.hypp.policy_lr = policy_lr  # learning rate of the policy network optimizer
        self.hypp.q_lr = q_lr  # learning rate of the Q network optimizer
        self.hypp.policy_frequency = policy_frequency  # frequency of training policy (delayed)
        self.hypp.target_network_frequency = target_network_frequency  # frequency of updates for the target networks
        self.hypp.alpha = alpha  # Entropy regularization coefficient
        self.hypp.autotune = autotune  # automatic tuning of the entropy coefficient

        self.hypp.ensemble_size = ensemble_size  # how many q networks should be used during training
        self.hypp.use_rpf = use_rpf  # whether to use random functions to initialize the Q networks
        self.hypp.rpf_scale = rpf_scale  # the scale of the random functions

        self.hypp.model_type_q = model_type_q
        self.hypp.model_type_actor = model_type_actor
        self.hypp.hidden_layer_size_q = hidden_layer_size_q
        self.hypp.hidden_layer_size_actor = hidden_layer_size_actor
        self.hypp.hidden_layers_q = hidden_layers_q
        self.hypp.hidden_layers_actor = hidden_layers_actor
        self.hypp.cnn_filters_q = cnn_filters_q
        self.hypp.cnn_filters_actor = cnn_filters_actor
        self.hypp.cnn_layers_q = cnn_layers_q
        self.hypp.cnn_layers_actor = cnn_layers_actor
        self.hypp.attention_heads_q = attention_heads_q
        self.hypp.transformer_ff_size_q = transformer_ff_size_q
        self.hypp.attention_heads_actor = attention_heads_actor
        self.hypp.transformer_ff_size_actor = transformer_ff_size_actor

        self.hypp.attenuate_actor = attenuate_actor

        self.hypp.bernoulli_mask_coeff = bernoulli_mask_coeff

        self.hypp.action_scale = action_scale
        self.hypp.initial_random_sampling = initial_random_sampling

        self.hypp.random_target = random_target
        self.hypp.update_steps = update_steps
        self.hypp.pi_update_avg_q = pi_update_avg_q

        self.device = device
        self.obs_man = obs_man
        self.env = env

        self.rb = BernoulliMaskReplayBuffer(
            buffer_size=int(self.hypp.buffer_size),
            observation_space=self.obs_man.obs_space_rl,
            action_space=env.single_action_space,
            mask_size=self.hypp.ensemble_size,
            p_masking=self.hypp.bernoulli_mask_coeff,
            device=self.device,
            handle_timeout_termination=False
        )

        self.agent = EnsembleSACAgent(env=env,
                                      obs_man=self.obs_man,
                                      ensemble_size=self.hypp.ensemble_size,
                                      use_rpf=self.hypp.use_rpf,
                                      rpf_scale=self.hypp.rpf_scale,
                                      model_type_q=self.hypp.model_type_q,
                                      model_type_actor=self.hypp.model_type_actor,
                                      hidden_layer_size_q=self.hypp.hidden_layer_size_q,
                                      hidden_layer_size_actor=self.hypp.hidden_layer_size_actor,
                                      hidden_layers_q=self.hypp.hidden_layers_q,
                                      hidden_layers_actor=self.hypp.hidden_layers_actor,
                                      cnn_filters_q=self.hypp.cnn_filters_q,
                                      cnn_filters_actor=self.hypp.cnn_filters_actor,
                                      cnn_layers_q=self.hypp.cnn_layers_q,
                                      cnn_layers_actor=self.hypp.cnn_layers_actor,
                                      attention_heads_q=self.hypp.attention_heads_q,
                                      transformer_ff_size_q=self.hypp.transformer_ff_size_q,
                                      attention_heads_actor=self.hypp.attention_heads_actor,
                                      transformer_ff_size_actor=self.hypp.transformer_ff_size_actor,
                                      attenuate_actor=self.hypp.attenuate_actor,
                                      action_scale=self.hypp.action_scale).to(device)
        self.agent_name = "SAC"

        print('--- Parameters ---')
        print('Q-net: ', sum(p.numel() for p in self.agent.ensemble[0].parameters()))
        print('Actor:', sum(p.numel() for p in self.agent.actor_net.parameters()))
        print('Total:', sum(p.numel() for p in self.agent.parameters()))

        q_params = []
        for j in range(self.hypp.ensemble_size):
            q_params += list(self.agent.ensemble[j].parameters())

        self.q_optimizer = optim.Adam(q_params, lr=self.hypp.q_lr)
        self.actor_optimizer = optim.Adam(self.agent.actor_net.parameters(), lr=self.hypp.policy_lr)

        # Automatic entropy tuning
        if self.hypp.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.single_action_space.shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.hypp.q_lr)
        else:
            self.alpha = self.hypp.alpha

        self.start_time = time.time()

    def get_hyperparams_dict(self):
        return dict(self.hypp)

    def get_learning_starts(self):
        return self.hypp.learning_starts, self.hypp.learning_starts_actor

    def get_exploration_action(self, state, step):
        state = self.obs_man.get_rl_state(state)
        if step >= self.hypp.learning_starts or not self.hypp.initial_random_sampling:
            actions = self.agent.get_action(state, greedy=False)
            actions = actions.detach().cpu().numpy()
        else:
            actions = np.array([self.env.single_action_space.sample()]).reshape(1, -1)

        return actions

    def add_to_replay_buffer(self, obs, real_next_obs, actions, rewards, dones, infos):
        obs = self.obs_man.get_rl_state(obs)
        real_next_obs = self.obs_man.get_rl_state(real_next_obs)
        self.rb.add(obs, real_next_obs, actions, rewards, dones, infos)

    def train_and_log(self, step, episode_step):

        if step > self.hypp.learning_starts:

            for _ in range(self.hypp.update_steps):  # G update steps (REDQ)

                data = self.rb.sample(self.hypp.batch_size)

                # Establish which two Q-networks are used for the target and the policy update
                if self.hypp.random_target:
                    indices = np.random.choice(self.hypp.ensemble_size, 2, replace=False)
                else:
                    indices = [0, 1]

                # compute targets for Q networks
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = self.agent.get_action_and_logprob(data.next_observations)
                    q1_next_target = self.agent.get_ensemble_q_target(data.next_observations, next_state_actions, indices[0])
                    q2_next_target = self.agent.get_ensemble_q_target(data.next_observations, next_state_actions, indices[1])

                    min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi

                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.hypp.gamma * (
                        min_q_next_target).view(
                        -1)

                # compute error loss for Q networks
                q_a_values = [self.agent.get_ensemble_q_value(data.observations, data.actions, j).view(-1) for j in
                              range(self.agent.ensemble_size)]

                q_losses = [((q_a_values[j] - next_q_value)**2 * data.masks[:, j].view(-1)).sum() / torch.sum(data.masks[:, j]) for j in range(self.agent.ensemble_size)]

                q_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                for loss in q_losses:
                    q_loss = q_loss + loss

                # optimize Q network
                self.q_optimizer.zero_grad()
                q_loss.backward()
                self.q_optimizer.step()

                # update the target networks
                if step % self.hypp.target_network_frequency == 0:
                    self.agent.update_target_networks(self.hypp.tau)

            # improve policy using entropy regularized policy update
            if step % self.hypp.policy_frequency == 0 and step >= self.hypp.learning_starts_actor:  # TD 3 Delayed update support
                for _ in range(
                        self.hypp.policy_frequency):  # compensate for the delay by doing 'hypp.policy_frequency' times instead of 1
                    pi, log_pi, _ = self.agent.get_action_and_logprob(data.observations)

                    if self.hypp.pi_update_avg_q:
                        q_vals = [self.agent.get_ensemble_q_value(data.observations, pi, j) for j in range(self.agent.ensemble_size)]
                        mean_q_pi = torch.sum(torch.stack(q_vals), dim=0)/self.agent.ensemble_size
                        policy_loss = ((self.alpha * log_pi) - mean_q_pi).mean()
                    else:
                        q1_pi = self.agent.get_ensemble_q_value(data.observations, pi, indices[0])
                        q2_pi = self.agent.get_ensemble_q_value(data.observations, pi, indices[1])
                        min_q_pi = torch.min(q1_pi, q2_pi).view(-1)
                        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

                    # optimize policy
                    self.actor_optimizer.zero_grad()
                    policy_loss.backward()
                    self.actor_optimizer.step()

                    # update alpha parameter
                    if self.hypp.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self.agent.get_action_and_logprob(data.observations)
                        alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                        # tune entropy temperature
                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

            # log training losses to tensorboard
            if step % 100 == 0:
                wandb.log(
                    {
                        "train/q_loss": q_loss.item() / self.hypp.ensemble_size,
                        "Charts/episode_step": episode_step,
                        "others/SPS": int(step / (time.time() - self.start_time))
                    }, step=step
                )

                if step >= self.hypp.learning_starts_actor:
                    wandb.log(
                        {
                            "train/policy_loss": policy_loss.item(),
                            "hyperparameters/alpha": self.alpha,
                            "hyperparameters/target_entropy": self.target_entropy,
                            "train/entropy": -log_pi.detach().mean().item()
                        }, step=step
                    )
                    if self.hypp.autotune:
                        wandb.log({"losses/alpha_loss": alpha_loss.item()}, step=step)