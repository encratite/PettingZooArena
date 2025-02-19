from typing import Union, Optional, Final
from abc import ABC, abstractmethod
import numpy as np
from pettingzoo import AECEnv
from .wrapper import PZEnvWrapper

class PZArenaModel(ABC):
	MODEL_PATH: Final[str] = "model"

	name: str
	env: PZEnvWrapper

	def __init__(self, name: str, env: AECEnv):
		self.name = name
		self.env = PZEnvWrapper(env)

	@abstractmethod
	def save(self):
		pass

	@abstractmethod
	def load(self):
		pass

	@abstractmethod
	def learn(self):
		pass

	@abstractmethod
	def predict(
			self,
			observation: Union[np.ndarray, dict[str, np.ndarray]],
			action_masks: Optional[np.ndarray] = None
	) -> np.ndarray:
		pass