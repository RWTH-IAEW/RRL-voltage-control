import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn.init as init
import gymnasium as gym

from custom_envs.observation_manager import ObservationManager


class QNetwork(nn.Module):
    '''
    Q-network
    '''
    def __init__(self, env, obs_man: ObservationManager, hidden_layer_size=64, hidden_layers=1):
        super().__init__()
        self.input_layer = nn.Linear(np.array(obs_man.obs_shape_rl).prod() + np.prod(env.single_action_space.shape), hidden_layer_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layer_size, hidden_layer_size) for _ in range(hidden_layers)])
        # self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        # x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x
    
class QNetworkCNN(nn.Module):
    '''
    Q-network designed specifially for the case of multiple controllers
    '''
    def __init__(self, env, obs_man: ObservationManager, hidden_layer_size_mlp=64, hidden_mlp_layers=1, hidden_cnn_layers=1, cnn_filters=32):
        super().__init__()
        augmented = obs_man.augmented_env
        ctrl_action_included = obs_man.ctrl_action_included
        single_agent_obs_size = env.envs[0].single_agent_obs_size
        single_agent_action_size = 2
        no_controllers = env.envs[0].sm.no_agents

        self.input_regrouper = StateControllerGroup(no_controllers=no_controllers,
                                            augmented_env=augmented, ctrl_action_included=ctrl_action_included,
                                            single_agent_action_size=single_agent_action_size, single_agent_obs_size=single_agent_obs_size)

        self.input_cnn = nn.Conv2d(1, cnn_filters, kernel_size=(1, single_agent_obs_size + single_agent_action_size + ctrl_action_included*single_agent_action_size + augmented))
        self.cnn_layers = nn.ModuleList([nn.Conv2d(cnn_filters, cnn_filters, kernel_size=1) for _ in range(hidden_cnn_layers)])

        self.input_mlp = nn.Linear(cnn_filters*no_controllers, hidden_layer_size_mlp)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layer_size_mlp, hidden_layer_size_mlp) for _ in range(hidden_mlp_layers)])
        self.output_layer = nn.Linear(hidden_layer_size_mlp, 1)

    def forward(self, x, a):
        # regroup data into correct form
        x = self.input_regrouper(x, a)

        x = x.unsqueeze(1) # add filter dimensions
        x = F.relu(self.input_cnn(x))
        for cnn_layer in self.cnn_layers:
            x = F.relu(cnn_layer(x))

        # feed the output of CNN into MLP for "communication"
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.input_mlp(x))
        for mlp_layer in self.hidden_layers:
            x = F.relu(mlp_layer(x))
        x = self.output_layer(x)

        return x

class QNetworkTransformer(nn.Module):
    '''
    Q-net based on Transformer-Encoder architecture for multiple actors.
    '''
    def __init__(self, env, obs_man: ObservationManager, d_model=64, n_heads=4, dim_feedforward=128, n_layers=3):
        super().__init__()
        augmented = obs_man.augmented_env
        ctrl_action_included = obs_man.ctrl_action_included
        single_agent_obs_size = env.envs[0].single_agent_obs_size
        single_agent_action_size = 2
        self.no_controllers = env.envs[0].sm.no_agents

        self.input_regrouper = StateControllerGroup(no_controllers=self.no_controllers,
                                                    augmented_env=augmented, ctrl_action_included=ctrl_action_included,
                                                    single_agent_action_size=single_agent_action_size, single_agent_obs_size=single_agent_obs_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.state_to_embed = nn.Linear(single_agent_obs_size + single_agent_action_size + ctrl_action_included*single_agent_action_size + augmented, d_model)
        self.positional_embedding = nn.Embedding(self.no_controllers + 1, d_model)  # no controllers + cls_token

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.fc_out = nn.Linear(d_model, 1)  # head to retrieve the Q-value

    def forward(self, x, a):
        B = x.size(0) # batch size

         # regroup data into correct form
        x = self.input_regrouper(x, a)

        x = self.state_to_embed(x)  # embedding the (s,a) pair to model dimension
        
        # prepending the CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # bringing the cls token to batch dimension
        x = torch.cat([cls_tokens, x], dim=1)

        # positional encoding
        pos = self.positional_embedding(torch.arange(0, self.no_controllers + 1, device=x.device)) # size: n_controllers + 1, d_model
        x = x + pos

        # feed to transformer
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1) # size: B, n_controllers + 1, d_model

        cls_output = x[:, 0, :]  # retrieve value at the CLS token

        return self.fc_out(cls_output)


class StateControllerGroup(nn.Module):
    def __init__(self, no_controllers, augmented_env, ctrl_action_included, single_agent_obs_size, single_agent_action_size):
        super().__init__()
        self.no_controllers = no_controllers
        self.augmented_env = augmented_env
        self.ctrl_action_included = ctrl_action_included
        self.single_agent_obs_size = single_agent_obs_size
        self.single_agent_action_size = single_agent_action_size

    def forward(self, x, a=None):
        # Expecting input of shape (B, no_controllers*controller_obs_size) (+ no_controllers*single_agent_action_size if ctrl_action_included) (+1 if augmented_env)
        # Output (B, no_controllers, controller_obs_size (+ single_agent_action_size if ctrl_action_included) (+1 if augmented))
        B = x.size(0)

        if self.augmented_env:
            weight = x[:, -1]
            x = x[:, :-1]

        if self.ctrl_action_included:
            ctrl_actions = x[:, -self.no_controllers*self.single_agent_action_size:]
            x = x[:, :-self.no_controllers*self.single_agent_action_size]

        state_data = x
        
        # reshape state
        state_data = state_data.view(B, self.no_controllers, -1)
        
        # append the action part if necessary
        if a is not None:
            a = a.view(B, self.no_controllers, -1)
            state_data = torch.cat([state_data, a], dim=2)

        # if necessary append the ctrl action to each block
        if self.ctrl_action_included:
            ctrl_actions = ctrl_actions.view(B, self.no_controllers, -1)
            state_data = torch.cat([state_data, ctrl_actions], dim=2)

        # if necessary append the weight information to each block
        if self.augmented_env:
            weight = weight.unsqueeze(-1).unsqueeze(-1)
            weight = weight.expand(B, self.no_controllers, 1)
            state_data = torch.cat([state_data, weight], dim=2)

        return state_data

class SACActorTemplate(nn.Module):
    """Actor Template for SAC"""
    def __init__(self, env, action_scale):
        super().__init__()

        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_scale * env.action_space.high.flatten() - action_scale * env.action_space.low.flatten()) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_scale * env.action_space.high.flatten() + action_scale * env.action_space.low.flatten()) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        pass

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
        


class SACActor(SACActorTemplate):
    """Actor Network for SAC"""
    def __init__(self, env, obs_man: ObservationManager, hidden_layer_size=16, hidden_layers=1, attenuate_actor=False, action_scale=1):
        super().__init__(env=env, action_scale=action_scale)
        
        self.fc1 = nn.Linear(np.array(obs_man.obs_shape_rl).prod(), hidden_layer_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layer_size, hidden_layer_size) for _ in range(hidden_layers)])
        self.fc_mean = nn.Linear(hidden_layer_size, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(hidden_layer_size, np.prod(env.single_action_space.shape))

        # because Residual Learning attenuating weights of final layer
        if attenuate_actor:
            with torch.no_grad():
                self.fc_mean.weight *= 1e-3
                self.fc_mean.bias *= 1e-3

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((action_scale * env.action_space.high.flatten() - action_scale * env.action_space.low.flatten()) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_scale * env.action_space.high.flatten() + action_scale * env.action_space.low.flatten()) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

class SACActorCNN(SACActorTemplate):
    "Actor Network using CNN architecture for multiple actors."
    def __init__(self, env, obs_man: ObservationManager, hidden_layer_size_mlp=16, hidden_mlp_layers=1,
                 hidden_cnn_layers=1, cnn_filters=32, attenuate_actor=False, action_scale=1):
        super().__init__(env=env, action_scale=action_scale) 

        augmented = obs_man.augmented_env
        ctrl_action_included = obs_man.ctrl_action_included
        single_agent_action_size = 2
        single_agent_obs_size = env.envs[0].single_agent_obs_size
        no_controllers = env.envs[0].sm.no_agents

        self.input_regrouper = StateControllerGroup(no_controllers=no_controllers,
                                                    augmented_env=augmented, ctrl_action_included=ctrl_action_included,
                                                    single_agent_action_size=single_agent_action_size, single_agent_obs_size=single_agent_obs_size)

        self.input_cnn = nn.Conv2d(1, cnn_filters, kernel_size=(1, single_agent_obs_size + ctrl_action_included*single_agent_action_size + augmented))
        self.cnn_layers = nn.ModuleList([nn.Conv2d(cnn_filters, cnn_filters, kernel_size=1) for _ in range(hidden_cnn_layers)])

        self.fc1 = nn.Linear(cnn_filters*no_controllers, hidden_layer_size_mlp)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layer_size_mlp, hidden_layer_size_mlp) for _ in range(hidden_mlp_layers)])

        self.fc_mean = nn.Linear(hidden_layer_size_mlp, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(hidden_layer_size_mlp, np.prod(env.single_action_space.shape))

    def forward(self, x):
        # regroup data into correct form
        x = self.input_regrouper(x)

        x = x.unsqueeze(1) # add filter dimensions
        x = F.relu(self.input_cnn(x))
        for cnn_layer in self.cnn_layers:
            x = F.relu(cnn_layer(x))

        # feed the output of CNN into MLP for "communication"
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        for mlp_layer in self.hidden_layers:
            x = F.relu(mlp_layer(x))
        
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

class SACActorTransformer(SACActorTemplate):
    "Actor Network using Transformer Encoder architecture for multiple actors."
    def __init__(self, env, obs_man: ObservationManager, d_model, n_heads, dim_feedforward, n_layers, attenuate_actor=False, action_scale=1):
        super().__init__(env=env, action_scale=action_scale)

        augmented = obs_man.augmented_env
        ctrl_action_included = obs_man.ctrl_action_included
        single_agent_action_size = 2
        single_agent_obs_size = env.envs[0].single_agent_obs_size
        self.no_controllers = env.envs[0].sm.no_agents

        self.input_regrouper = StateControllerGroup(no_controllers=self.no_controllers,
                                                    augmented_env=augmented, ctrl_action_included=ctrl_action_included,
                                                    single_agent_action_size=single_agent_action_size, single_agent_obs_size=single_agent_obs_size)

        self.state_to_embed = nn.Linear(single_agent_obs_size + ctrl_action_included * single_agent_action_size + augmented, d_model)
        self.positional_embedding = nn.Embedding(self.no_controllers, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.fc_mean = nn.Linear(d_model, 2)
        self.fc_log_std = nn.Linear(d_model, 2)

    def forward(self, x):
        # regroup data into correct form
        x = self.input_regrouper(x) # size: B, n_controllers, n_obs_size

        x = self.state_to_embed(x) # perform embedding to correct dimension - size: B, n_controllers, d_model
        pos = self.positional_embedding(torch.arange(0, self.no_controllers, device=x.device)) # size: n_controllers, d_model

        # add positional encoding to embedding
        x = x + pos

        # feed to transformer
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1) # size: B, n_controllers, d_model

        # extract mean and log_std
        mean = torch.flatten(self.fc_mean(x), start_dim=1)
        log_std = torch.flatten(self.fc_log_std(x), start_dim=1)

        # transform log_std
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

class SACActorCNNTransformer(SACActorTemplate):
    """
    CNN feature extractor with following Transformer Encoder architecture for token communication
    """
    def __init__(self, env, obs_man: ObservationManager, n_heads, dim_feedforward, n_layers, attenuate_actor=False, action_scale=1,
                 hidden_cnn_layers=1, cnn_filters=32):
        super().__init__(env=env, action_scale=action_scale)

        augmented = obs_man.augmented_env
        ctrl_action_included = obs_man.ctrl_action_included
        single_agent_action_size = 2
        single_agent_obs_size = env.envs[0].single_agent_obs_size
        self.no_controllers = env.envs[0].sm.no_agents

        self.input_regrouper = StateControllerGroup(no_controllers=self.no_controllers,
                                                    augmented_env=augmented, ctrl_action_included=ctrl_action_included,
                                                    single_agent_action_size=single_agent_action_size, single_agent_obs_size=single_agent_obs_size)

        # CNN feature extractor
        self.input_cnn = nn.Conv2d(1, cnn_filters, kernel_size=(1, single_agent_obs_size + ctrl_action_included*single_agent_action_size + augmented))
        self.cnn_layers = nn.ModuleList([nn.Conv2d(cnn_filters, cnn_filters, kernel_size=1) for _ in range(hidden_cnn_layers)])

        # Transformer Actor
        self.positional_embedding = nn.Embedding(self.no_controllers, cnn_filters)

        encoder_layer = nn.TransformerEncoderLayer(
        d_model=cnn_filters,
        nhead=n_heads,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # mean and log_std head
        self.fc_mean = nn.Linear(cnn_filters, 2)
        self.fc_log_std = nn.Linear(cnn_filters, 2)

    def forward(self, x):
        # regroup data into correct form
        x = self.input_regrouper(x) # size: B, n_controllers, n_obs_size

        x = x.unsqueeze(1) # add filter dimensions
        x = F.relu(self.input_cnn(x))
        for cnn_layer in self.cnn_layers:
            x = F.relu(cnn_layer(x)) # size: B, cnn_filters, n_controllers, 1

        # prepare input to feed into the transformer
        x = x.squeeze(-1).transpose(1, 2) # size: B, n_controllers, cnn_filters

        # positional encoding
        pos = self.positional_embedding(torch.arange(0, self.no_controllers, device=x.device)) # size: n_controllers, d_model
        x = x + pos

        # feed to transformer
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1) # size: B, n_controllers, cnn_filters

        # extract mean and log_std
        mean = torch.flatten(self.fc_mean(x), start_dim=1)
        log_std = torch.flatten(self.fc_log_std(x), start_dim=1)

        # transform log_std
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std


class EnsembleSACAgent(nn.Module):

    def __init__(self, env, obs_man: ObservationManager,
                 ensemble_size=2,
                 use_rpf=False,
                 rpf_scale=1,
                 model_type_q='mlp',
                 model_type_actor='mlp',
                 hidden_layer_size_q=64,
                 hidden_layer_size_actor=16,
                 hidden_layers_q=1,
                 hidden_layers_actor=1,
                 cnn_filters_q=32,
                 cnn_filters_actor=32,
                 cnn_layers_q=2,
                 cnn_layers_actor=2,
                 attention_heads_q = 4,
                 transformer_ff_size_q = 128,
                 attention_heads_actor=4,
                 transformer_ff_size_actor=64,
                 attenuate_actor=False,
                 action_scale=1):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.use_rpf = use_rpf
        self.rpf_scale = rpf_scale

        self.model_type_q = model_type_q
        self.model_type_actor = model_type_actor

        if model_type_q == 'mlp':
            q_model = QNetwork
            q_params = {
                'env': env,
                'obs_man': obs_man,
                'hidden_layer_size': hidden_layer_size_q,
                'hidden_layers': hidden_layers_q
            }
        elif model_type_q == 'cnn':
            q_model = QNetworkCNN
            q_params = {
                'env': env,
                'obs_man': obs_man,
                'hidden_layer_size_mlp': hidden_layer_size_q,
                'hidden_mlp_layers': hidden_layers_q,
                'hidden_cnn_layers': cnn_layers_q,
                'cnn_filters': cnn_filters_q
            }
        elif model_type_q == 'transformer':
            q_model = QNetworkTransformer
            q_params = {
                'env': env,
                'obs_man': obs_man,
                'd_model': hidden_layer_size_q,
                'n_layers': hidden_layers_q,
                'n_heads': attention_heads_q,
                'dim_feedforward': transformer_ff_size_q,
            }
        else:
            raise ValueError(f'Model type {model_type_q} is unknown for Q-networks!')
        
        if model_type_actor == 'mlp':
            actor_model = SACActor
            actor_params = {
                'env': env,
                'obs_man': obs_man,
                'hidden_layer_size': hidden_layer_size_actor,
                'hidden_layers': hidden_layers_actor,
                'attenuate_actor': attenuate_actor,
                'action_scale': action_scale
            }
        elif model_type_actor == 'cnn':
            actor_model = SACActorCNN
            actor_params = {
                'env': env,
                'obs_man': obs_man,
                'hidden_layer_size_mlp': hidden_layer_size_actor,
                'hidden_mlp_layers': hidden_layers_actor,
                'hidden_cnn_layers': cnn_layers_actor,
                'cnn_filters': cnn_filters_actor,
                'attenuate_actor': attenuate_actor,
                'action_scale': action_scale
            }
        elif model_type_actor == 'transformer':
            actor_model = SACActorTransformer
            actor_params = {
                'env': env,
                'obs_man': obs_man,
                'd_model': hidden_layer_size_actor,
                'n_layers': hidden_layers_actor,
                'n_heads': attention_heads_actor,
                'dim_feedforward': transformer_ff_size_actor,
                'attenuate_actor': attenuate_actor,
                'action_scale': action_scale
            }
        elif model_type_actor == 'cnn_transformer':
            actor_model = SACActorCNNTransformer
            actor_params = {
                'env': env,
                'obs_man': obs_man,
                'hidden_cnn_layers': cnn_layers_actor,
                'cnn_filters': cnn_filters_actor,
                'n_layers': hidden_layers_actor,
                'n_heads': attention_heads_actor,
                'dim_feedforward': transformer_ff_size_actor,
                'attenuate_actor': attenuate_actor,
                'action_scale': action_scale
            }
        else:
            raise ValueError(f"Unknown model type for actor: {model_type_actor}")

        # self.ensemble = nn.ModuleList([QNetwork(env, obs_man, hidden_layer_size_q, hidden_layers_q) for _ in range(self.ensemble_size)])
        # self.ensemble_target = nn.ModuleList([QNetwork(env, obs_man, hidden_layer_size_q, hidden_layers_q) for _ in range(self.ensemble_size)])

        self.ensemble = nn.ModuleList([q_model(**q_params) for _ in range(self.ensemble_size)])
        self.ensemble_target = nn.ModuleList([q_model(**q_params) for _ in range(self.ensemble_size)])

        if self.use_rpf:
            self.rpf_ensemble = nn.ModuleList([q_model(**q_params) for _ in range(self.ensemble_size)])

        # self.actor_net = SACActor(env, obs_man, hidden_layer_size_actor, hidden_layers_actor, attenuate_actor, action_scale)
        self.actor_net = actor_model(**actor_params)

        for j in range(self.ensemble_size):
            self.ensemble_target[j].load_state_dict(self.ensemble[j].state_dict())

    def get_ensemble_std(self, state, action):
        q_values = []

        for j in range(self.ensemble_size):
            q_values.append(self.get_ensemble_q_value(state, action, j).unsqueeze(-1))

        q_values = torch.cat(q_values, dim=-1)
        mean = torch.mean(q_values, dim=-1)
        std = torch.std(q_values, dim=-1)

        return mean, std

    def get_ensemble_q_value(self, x, a, j):
        if self.use_rpf:
            return self.ensemble[j](x, a) + self.rpf_scale * self.rpf_ensemble[j](x, a)
        else:
            return self.ensemble[j](x, a)

    def get_q1_value(self, x, a):
        return self.get_ensemble_q_value(x, a, 0)

    def get_q2_value(self, x, a):
        return self.get_ensemble_q_value(x, a, 1)

    def get_ensemble_q_target(self, x, a, j):
        if self.use_rpf:
            return self.ensemble_target[j](x, a) + self.rpf_scale * self.rpf_ensemble[j](x, a)
        else:
            return self.ensemble_target[j](x, a)

    def get_target_q1_value(self, x, a):
        return self.get_ensemble_q_target(x, a, 0)

    def get_target_q2_value(self, x, a):
        return self.get_ensemble_q_target(x, a, 1)

    def get_action(self, x, greedy=False):
        mean, log_std = self.actor_net(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if not greedy:
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        else:
            x_t = mean  # greedy action

        y_t = torch.tanh(x_t)
        action = y_t * self.actor_net.action_scale + self.actor_net.action_bias

        return action

    def get_action_and_logprob(self, x):
        mean, log_std = self.actor_net(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.actor_net.action_scale + self.actor_net.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.actor_net.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.actor_net.action_scale + self.actor_net.action_bias
        return action, log_prob, mean

    def update_target_networks(self, tau):
        for j in range(self.ensemble_size):
           for param, target_param in zip(self.ensemble[j].parameters(), self.ensemble_target[j].parameters()):
               target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class MixedAgent(nn.Module):
    def __init__(self, rl_agent, controller, obs_man: ObservationManager, mixing_type='regularized'):
        super(MixedAgent, self).__init__()
        self.controller = controller
        self.rl_agent = rl_agent
        self.obs_man = obs_man

        assert mixing_type in ['regularized', 'residual'], 'The mixing type has either to be "regularized" or "residual!"'

        self.mixing_type = mixing_type

    def get_action(self, state, greedy=True, ret_sep_actions=False, weight=None):
        # should do the mixing later on
        if self.obs_man.augmented_env:
            # retrieve weight from state
            if len(state.shape) == 1:
                weight = state[-1]
                pure_state = state[:-1]
            else:
                # batch of states
                weight = state[:, -1].view(-1, 1)
                pure_state = state[:, :-1]
        else:
            assert weight is not None, "In an unaugmented environment the weight has to be provided."
            pure_state = state

        # TODO: there is currently no clipping at this point. If controller outputs action out of action space, this is unnoticed.
        control_action = self.controller.get_action(self.obs_man.get_ctrl_state(state))

        if self.obs_man.ctrl_action_included:
            if self.obs_man.augmented_env:
                state = torch.cat([pure_state, control_action, weight], dim=-1)
            else:
                state = torch.cat([pure_state, control_action], dim=-1)

        rl_agent_action = self.rl_agent.get_action(self.obs_man.get_rl_state(state), greedy)
        
        if self.mixing_type == 'regularized':
            total_action =  weight * rl_agent_action + (1-weight) * control_action
        elif self.mixing_type == 'residual':
            total_action =  weight * rl_agent_action + control_action
        else:
            raise ValueError("Undefined mixing type.")

        if ret_sep_actions:
            return total_action, rl_agent_action, control_action
        else:
            return total_action

    def get_rl_action(self, state, greedy=True):
        return self.rl_agent.get_action(self.obs_man.get_rl_state(state), greedy)

    def get_control_action(self, state):
        return self.controller.get_action(self.obs_man.get_ctrl_state(state))
