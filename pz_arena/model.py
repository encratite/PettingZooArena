from typing import Union, Optional, Final, Callable, TypeAlias
from abc import ABC, abstractmethod
import numpy as np

ReloadModelsCallback: TypeAlias = Callable[[], None]

class PZArenaModel(ABC):
	name: str

	def __init__(self, name: str):
		self.name = name

	@abstractmethod
	def save(self):
		pass

	@abstractmethod
	def load(self):
		pass

	@abstractmethod
	def learn(self, reload_models: ReloadModelsCallback):
		pass

	@abstractmethod
	def predict(
			self,
			observation: Union[np.ndarray, dict[str, np.ndarray]],
			action_masks: Optional[np.ndarray] = None
	) -> np.ndarray:
		pass