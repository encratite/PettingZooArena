from typing import Optional, Union, Any, cast
import torch as th
from torch import nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.dqn.dqn import get_parameters_by_name, get_linear_fn
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from gymnasium import spaces
from .wrapper import PZEnvWrapper

class MaskableDQN(DQN):
	def predict(
			self,
			observation: Union[np.ndarray, dict[str, np.ndarray]],
			state: Optional[tuple[np.ndarray, ...]] = None,
			episode_start: Optional[np.ndarray] = None,
			deterministic: bool = False,
	) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
		if not deterministic and np.random.rand() < self.exploration_rate:
			if self.policy.is_vectorized_observation(observation):
				if isinstance(observation, dict):
					n_batch = observation[next(iter(observation.keys()))].shape[0]
				else:
					n_batch = observation.shape[0]
				action = np.array([self._sample_action_space() for _ in range(n_batch)])
			else:
				action = np.array(self._sample_action_space())
		else:
			action, state = self.policy.predict(observation, state, episode_start, deterministic)
		return action, state

	def _sample_action(
			self,
			learning_starts: int,
			action_noise: Optional[ActionNoise] = None,
			n_envs: int = 1,
	) -> tuple[np.ndarray, np.ndarray]:
		if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
			unscaled_action = np.array([self._sample_action_space() for _ in range(n_envs)])
		else:
			assert self._last_obs is not None, "self._last_obs was not set"
			unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

		if isinstance(self.action_space, spaces.Box):
			scaled_action = self.policy.scale_action(unscaled_action)

			if action_noise is not None:
				scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

			buffer_action = scaled_action
			action = self.policy.unscale_action(scaled_action)
		else:
			buffer_action = unscaled_action
			action = buffer_action
		return action, buffer_action

	def _setup_model(self) -> None:
		self._setup_lr_schedule()
		self.set_random_seed(self.seed)

		if self.replay_buffer_class is None:
			if isinstance(self.observation_space, spaces.Dict):
				self.replay_buffer_class = DictReplayBuffer
			else:
				self.replay_buffer_class = ReplayBuffer

		if self.replay_buffer is None:
			replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
			if issubclass(self.replay_buffer_class, HerReplayBuffer):
				assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
				replay_buffer_kwargs["env"] = self.env
			self.replay_buffer = self.replay_buffer_class(
				self.buffer_size,
				self.observation_space,
				self.action_space,
				device=self.device,
				n_envs=self.n_envs,
				optimize_memory_usage=self.optimize_memory_usage,
				**replay_buffer_kwargs,
			)

		env = self._get_env()
		self.policy = MaskableDQNPolicy(
			env,
			self.observation_space,
			cast(spaces.Discrete, self.action_space),
			self.lr_schedule,
			**self.policy_kwargs,
		)
		self.policy = self.policy.to(self.device)

		self._convert_train_freq()

		self._create_aliases()
		self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
		self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
		self.exploration_schedule = get_linear_fn(
			self.exploration_initial_eps,
			self.exploration_final_eps,
			self.exploration_fraction,
		)

		assert not (self.n_envs > 1 and self.n_envs > self.target_update_interval)

	def _action_mask(self):
		env = self._get_env()
		action_mask = env.action_masks()
		return action_mask

	def _get_env(self) -> PZEnvWrapper:
		return cast(Any, self.env).envs[0].env

	def _sample_action_space(self):
		action_mask = self._action_mask()
		while True:
			sample = self.action_space.sample()
			if action_mask[sample]:
				# A valid action has been sampled
				return sample

class MaskableQNetwork(QNetwork):
	env: PZEnvWrapper

	def __init__(
		self,
		env: PZEnvWrapper,
		observation_space: spaces.Space,
		action_space: spaces.Discrete,
		features_extractor: BaseFeaturesExtractor,
		features_dim: int,
		net_arch: Optional[list[int]] = None,
		activation_fn: type[nn.Module] = nn.ReLU,
		normalize_images: bool = True,
	) -> None:
		self.env = env
		super().__init__(
			observation_space=observation_space,
			action_space=action_space,
			features_extractor=features_extractor,
			features_dim=features_dim,
			net_arch=net_arch,
			activation_fn=activation_fn,
			normalize_images=normalize_images
		)

	def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
		assert self.env is not None
		q_values = self(observation)
		action_mask = self.env.action_masks()
		indices = []
		i = 0
		for x in action_mask:
			if not x:
				indices.append(i)
			i += 1
		q_values[0, indices] = float("-inf")
		action = q_values.argmax(dim=1).reshape(-1)
		return action

class MaskableDQNPolicy(DQNPolicy):
	env: PZEnvWrapper

	def __init__(
			self,
			env: PZEnvWrapper,
			observation_space: spaces.Space,
			action_space: spaces.Discrete,
			lr_schedule: Schedule,
			net_arch: Optional[list[int]] = None,
			activation_fn: type[nn.Module] = nn.ReLU,
			features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
			features_extractor_kwargs: Optional[dict[str, Any]] = None,
			normalize_images: bool = True,
			optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
			optimizer_kwargs: Optional[dict[str, Any]] = None
	) -> None:
		self.env = env
		super().__init__(
			observation_space=observation_space,
			action_space=action_space,
			lr_schedule=lr_schedule,
			net_arch=net_arch,
			activation_fn=activation_fn,
			features_extractor_class=features_extractor_class,
			features_extractor_kwargs=features_extractor_kwargs,
			normalize_images=normalize_images,
			optimizer_class=optimizer_class,
			optimizer_kwargs=optimizer_kwargs
		)

	def make_q_net(self) -> MaskableQNetwork:
		assert self.env is not None
		net_args = self._update_features_extractor(self.net_args, features_extractor=None)
		network = MaskableQNetwork(self.env, **net_args).to(self.device)
		return network