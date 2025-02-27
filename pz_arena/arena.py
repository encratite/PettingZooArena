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
		self._lock.acquire()
		try:
			self._model.save()
		finally:
			self._lock.release()

	def load(self) -> None:
		self._lock.acquire()
		try:
			self._model.load()
		finally:
			self._lock.release()

	def learn(self, reload_models: ReloadModelsCallback):
		self._model.learn(reload_models)

class PZArena:
	_env: PZEnvWrapper
	_model_locks: list[ModelLock]

	def __init__(self, env: PZEnvWrapper, model_locks: list[ModelLock]):
		self._env = env
		self._model_locks = model_locks

	def run(self, model_index: int) -> None:
		model_lock = self._model_locks[model_index]
		print(f"Running model {model_lock.name}")
		model_lock.learn(lambda: self._on_reload_models(model_lock))

	def _on_reload_models(self, model_lock: ModelLock) -> None:
		model_lock.save()
		self._env.reload_models()