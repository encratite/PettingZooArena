import os
import time
from typing import Union, Optional
import numpy as np
from pettingzoo import AECEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from .model import PZArenaModel
from .wrapper import PZEnvWrapper

class SB3PPOModel(PZArenaModel):
	def __init__(self, name: str, env: AECEnv, **kwargs):
		super().__init__(name, env)
		self._model = MaskablePPO(
			"MlpPolicy",
			self.env,
			device="cpu",
			**kwargs
		)
		self._path = os.path.join(PZArenaModel.MODEL_PATH, name)

	def save(self):
		print(f"Saving model {self._path}")
		self._model.save(self._path)

	def load(self):
		print(f"Loading model {self._path}")
		self._model = MaskablePPO.load(self._path)

	def learn(self):
		callback = ModelUpdateCallback(self.env, update_frequency=15)
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

class ModelUpdateCallback(BaseCallback):
	def __init__(self, model: PZArenaModel, env: PZEnvWrapper, update_frequency: float):
		super().__init__()
		self._model = model
		self._env = env
		self._update_frequency = update_frequency
		self._last_update = time.perf_counter()

	def _on_step(self) -> bool:
		return True

	def _on_rollout_end(self) -> None:
		now = time.perf_counter()
		if now - self._last_update > self._update_frequency:
			# Some time has passed, store our own model and reload the others to increase the difficulty
			self._model.save()
			self._env.reload_models()
			self._last_update = now
