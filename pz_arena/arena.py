import multiprocessing
from threading import Lock
from .model import PZArenaModel, ReloadModelsCallback
from .wrapper import PZEnvWrapper

class ModelLock:
	_model: PZArenaModel
	_lock: Lock

	def __init__(self, model: PZArenaModel, lock: Lock):
		self._model = model
		self._lock = lock

	@property
	def name(self) -> str:
		return self._model.name

	def save(self) -> None:
		print(f"[{multiprocessing.current_process().name}] Saving model: {self.name}")
		self._lock.acquire()
		try:
			self._model.save()
		finally:
			self._lock.release()
		print(f"[{multiprocessing.current_process().name}] Finished saving model: {self.name}")

	def load(self) -> None:
		print(f"[{multiprocessing.current_process().name}] Loading model: {self.name}")
		self._lock.acquire()
		try:
			self._model.load()
		finally:
			self._lock.release()
		print(f"[{multiprocessing.current_process().name}] Finished loading model: {self.name}")

	def learn(self, reload_models: ReloadModelsCallback):
		self._model.learn(reload_models)

class PZArena:
	_env: PZEnvWrapper
	_model_locks: list[ModelLock]
	_model_index: int | None

	def __init__(self, env: PZEnvWrapper, model_locks: list[ModelLock]):
		self._env = env
		self._model_locks = model_locks
		self._model_index = None

	def run(self, model_index: int) -> None:
		self._model_index = model_index
		model_lock = self._model_locks[model_index]
		print(f"Running model {model_lock.name}")
		model_lock.learn(lambda: self._on_reload_models(model_lock))

	def _on_reload_models(self, model_lock: ModelLock) -> None:
		model_lock.save()
		for i in range(len(self._model_locks)):
			if i != self._model_index:
				model_lock = self._model_locks[i]
				model_lock.load()