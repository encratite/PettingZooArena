import pathlib
import io
from typing import Optional, Union, Any, cast
import torch as th
from torch import nn
import numpy as np
from collections.abc import Iterable
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule, GymEnv
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

	def save(
			self,
			path: Union[str, pathlib.Path, io.BufferedIOBase],
			exclude: Optional[Iterable[str]] = None,
			include: Optional[Iterable[str]] = None,
	) -> None:
		del self.policy.env
		del self.policy.q_net.env
		super().save(path, exclude, include)
		self._apply_env()

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

		self.policy = MaskableDQNPolicy(
			self.observation_space,
			cast(spaces.Discrete, self.action_space),
			self.lr_schedule,
			**self.policy_kwargs,
		)
		self.policy = self.policy.to(self.device)
		self._apply_env()

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

	def _apply_env(self) -> None:
		env = self._get_env()
		self.policy.env = env
		self.policy.q_net.env = env

	def _sample_action_space(self):
		action_mask = self._action_mask()
		while True:
			sample = self.action_space.sample()
			if action_mask[sample]:
				# A valid action has been sampled
				return sample

class MaskableQNetwork(QNetwork):
	env: PZEnvWrapper | None

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
	env: PZEnvWrapper | None

	def make_q_net(self) -> MaskableQNetwork:
		net_args = self._update_features_extractor(self.net_args, features_extractor=None)
		network = MaskableQNetwork(**net_args).to(self.device)
		return network