import os
import time
from typing import Union, Optional, Callable, TypeAlias
import numpy as np
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from .model import PZArenaModel, ReloadModelsCallback
from .config import Configuration
from stable_baselines3.common.logger import Logger

OnStepCallback: TypeAlias = Callable[[Logger], None]

class PPOModel(PZArenaModel):
	on_step: OnStepCallback | None
	_model: MaskablePPO
	_path: str

	def __init__(self, name: str, env: Env, on_step: OnStepCallback | None = None, **kwargs):
		super().__init__(name)
		self.on_step = on_step
		self._model = MaskablePPO(
			"MlpPolicy",
			env,
			device="cpu",
			tensorboard_log=Configuration.TENSORBOARD_LOG,
			**kwargs
		)
		self._path = os.path.join(Configuration.MODEL_PATH, name)

	def save(self) -> None:
		self._model.save(self._path)

	def load(self) -> None:
		try:
			self._model = MaskablePPO.load(self._path)
		except FileNotFoundError:
			pass

	def learn(self, reload_models: ReloadModelsCallback) -> None:
		callback = RolloutTimerCallback(self.on_step, reload_models, Configuration.MODEL_UPDATE_FREQUENCY)
		self._model.learn(
			total_timesteps=1_000_0000,
			tb_log_name=self.name,
			callback=callback
		)

	def predict(
			self,
			observation: Union[np.ndarray, dict[str, np.ndarray]],
			action_masks: Optional[np.ndarray] = None
	) -> np.ndarray:
		action, _observation = self._model.predict(observation, action_masks=action_masks)
		return action

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