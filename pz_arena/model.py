from typing import Union, Optional
from abc import ABC, abstractmethod
import numpy as np

class PZArenaModel(ABC):
	def __init__(self, path: str):
		self._path = path

	@abstractmethod
	def save(self):
		pass

	@abstractmethod
	def load(self):
		pass

	@abstractmethod
	def predict(
			self,
			observation: Union[np.ndarray, dict[str, np.ndarray]],
			action_masks: Optional[np.ndarray] = None
	) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
		pass