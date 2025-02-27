from typing import Optional, Union
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.noise import ActionNoise
from gymnasium import spaces

class MaskableDQN(DQN):
	def train(self, gradient_steps: int, batch_size: int = 100) -> None:
		self.policy.set_training_mode(True)
		self._update_learning_rate(self.policy.optimizer)

		losses = []
		for _ in range(gradient_steps):
			replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

			with th.no_grad():
				next_q_values = self.q_net_target(replay_data.next_observations)
				next_q_values, _ = next_q_values.max(dim=1)
				next_q_values = next_q_values.reshape(-1, 1)
				target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

			current_q_values = self.q_net(replay_data.observations)

			current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

			loss = F.smooth_l1_loss(current_q_values, target_q_values)
			losses.append(loss.item())

			self.policy.optimizer.zero_grad()
			loss.backward()

			th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
			self.policy.optimizer.step()

		self._n_updates += gradient_steps

		self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
		self.logger.record("train/loss", np.mean(losses))

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

	def _action_mask(self):
		env = self.env.envs[0].env
		action_mask = env.action_masks()
		return action_mask

	def _sample_action_space(self):
		action_mask = self._action_mask()
		while True:
			sample = self.action_space.sample()
			if action_mask[sample]:
				# A valid action has been sampled
				break
		return sample