import os
from glob import glob
import time
from datetime import datetime
from typing import Union, Optional, Callable, TypeAlias, Final, cast
from abc import ABC, abstractmethod
import numpy as np
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import Logger
from sb3_contrib import MaskablePPO
from .model import PZArenaModel, ReloadModelsCallback
from .config import Configuration
from .sb3_algorithm import MaskableDQN, MaskableDQNPolicy, MaskableQNetwork

OnStepCallback: TypeAlias = Callable[[Logger], None]

class SB3Model(PZArenaModel, ABC):
	EXTENSION: Final[str] = ".zip"

	on_step: OnStepCallback | None
	_model: BaseAlgorithm

	def __init__(self, name: str, env: Env, on_step: OnStepCallback | None = None, **kwargs):
		super().__init__(name)
		self.on_step = on_step
		self._model = self.create_model(env, **kwargs)

	@abstractmethod
	def create_model(self, env: Env, **kwargs) -> BaseAlgorithm:
		pass

	@abstractmethod
	def save(self) -> None:
		pass

	@abstractmethod
	def load(self) -> None:
		pass

	def learn(self, reload_models: ReloadModelsCallback) -> None:
		callback = RolloutTimerCallback(self.on_step, reload_models, Configuration.MODEL_UPDATE_FREQUENCY)
		self._model.learn(
			total_timesteps=Configuration.TOTAL_TIME_STEPS,
			tb_log_name=self.name,
			callback=callback
		)

	def predict(
			self,
			observation: Union[np.ndarray, dict[str, np.ndarray]],
			action_masks: Optional[np.ndarray] = None
	) -> np.ndarray:
		action, _observation = self._model.predict(observation)
		return action

	def _get_file_name(self) -> str:
		timestamp = datetime.now().strftime("%y%m%d%H%M%S")
		file_name = f"{self.name} {timestamp}{self.EXTENSION}"
		return file_name

	def _get_most_recent_model_path(self) -> str | None:
		pattern = os.path.join(Configuration.MODEL_DIRECTORY, f"{self.name}*.zip")
		files = glob(pattern)
		files = sorted(files, reverse=True)
		if len(files) > 0:
			target = files[0]
			return target
		else:
			return None

class PPOModel(SB3Model):
	def create_model(self, env: Env, **kwargs) -> BaseAlgorithm:
		return MaskablePPO(
			"MlpPolicy",
			env,
			device="cpu",
			tensorboard_log=Configuration.TENSORBOARD_LOG,
			**kwargs
		)

	def save(self) -> None:
		file_name = self._get_file_name()
		path = os.path.join(Configuration.MODEL_TEMP_DIRECTORY, file_name)
		print(f"Saving PPO model: {path}")
		self._model.save(path)
		source = path
		destination = os.path.join(Configuration.MODEL_DIRECTORY, file_name)
		os.rename(source, destination)

	def load(self) -> None:
		path = self._get_most_recent_model_path()
		if path is not None:
			print(f"Loading PPO model: {path}")
			self._model = MaskablePPO.load(path)

	def predict(
			self,
			observation: Union[np.ndarray, dict[str, np.ndarray]],
			action_masks: Optional[np.ndarray] = None
	) -> np.ndarray:
		ppo_model = cast(PPOModel, self._model)
		action, _observation = ppo_model.predict(observation, action_masks=action_masks)
		return action

class DQNModel(SB3Model):
	def create_model(self, env: Env, **kwargs) -> BaseAlgorithm:
		return MaskableDQN(
			MaskableDQNPolicy,
			env,
			device="cpu",
			tensorboard_log=Configuration.TENSORBOARD_LOG,
			**kwargs
		)

	def load(self) -> None:
		model = cast(MaskableDQN, self._model)
		path = self._get_most_recent_model_path()
		if path is not None:
			print(f"Loading DQN model: {path}")
			model.policy.q_net = MaskableQNetwork.load(path)
			model.apply_env()

	def save(self) -> None:
		model = cast(MaskableDQN, self._model)
		del model.policy.q_net.env
		file_name = self._get_file_name()
		path = os.path.join(Configuration.MODEL_TEMP_DIRECTORY, file_name)
		print(f"Saving DQN model: {path}")
		model.policy.q_net.save(path)
		model.apply_env()
		source = path
		destination = os.path.join(Configuration.MODEL_DIRECTORY, file_name)
		os.rename(source, destination)

class RolloutTimerCallback(BaseCallback):
	def __init__(self, on_step: OnStepCallback | None, reload_callback: ReloadModelsCallback, update_frequency: float):
		super().__init__()
		self._on_step_callback = on_step
		self._reload_callback = reload_callback
		self._update_frequency = update_frequency
		self._last_update = time.perf_counter()

	def _on_step(self) -> bool:
		if self._on_step_callback is not None:
			self._on_step_callback(self.logger)
		return True

	def _on_rollout_end(self) -> None:
		now = time.perf_counter()
		if now - self._last_update > self._update_frequency:
			# Enough time has passed, perform callback to reload models
			self._reload_callback()
			self._last_update = now