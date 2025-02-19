import os
import time
from typing import Union, Optional, Callable, Final
import numpy as np
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from .model import PZArenaModel, ReloadModelsCallback

TENSORBOARD_LOG: Final[str] = "./tensorboard"

class SB3PPOModel(PZArenaModel):
	def __init__(self, name: str, env: Env, **kwargs):
		super().__init__(name)
		self._kwargs = kwargs
		self._model = MaskablePPO(
			"MlpPolicy",
			env,
			device="cpu",
			tensorboard_log=TENSORBOARD_LOG,
			**kwargs
		)
		self._path = os.path.join(PZArenaModel.MODEL_PATH, name)

	def save(self):
		print(f"Saving model {self._path}")
		self._model.save(self._path)

	def load(self):
		print(f"Loading model {self._path}")
		self._model = MaskablePPO.load(self._path)

	def learn(self, reload_models: ReloadModelsCallback):
		callback = RolloutTimerCallback(reload_models, update_frequency=15)
		self._model.learn(
			total_timesteps=1_000_0000,
			progress_bar=True,
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
	def __init__(self, callback: ReloadModelsCallback, update_frequency: float):
		super().__init__()
		self._callback = callback
		self._update_frequency = update_frequency
		self._last_update = time.perf_counter()

	def _on_step(self) -> bool:
		return True

	def _on_rollout_end(self) -> None:
		now = time.perf_counter()
		if now - self._last_update > self._update_frequency:
			# Enough time has passed, perform callback to reload models
			self._callback()
			self._last_update = now