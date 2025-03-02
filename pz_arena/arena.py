from .model import PZArenaModel
from .wrapper import PZEnvWrapper

class PZArena:
	_env: PZEnvWrapper
	_models: list[PZArenaModel]
	_model_index: int | None

	def __init__(self, env: PZEnvWrapper, models: list[PZArenaModel]):
		self._env = env
		self._models = models
		self._model_index = None

	def run(self, model_index: int) -> None:
		self._model_index = model_index
		model = self._models[model_index]
		print(f"Running model {model.name}")
		model.learn(lambda: self._on_reload_models(model))

	def _on_reload_models(self, model: PZArenaModel) -> None:
		model.save()
		for i in range(len(self._models)):
			if i != self._model_index:
				model = self._models[i]
				model.load()