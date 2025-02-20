import os
from glob import glob
from .model import PZArenaModel
from .wrapper import PZEnvWrapper

class PZArena:
	_env: PZEnvWrapper
	_models: list[PZArenaModel]
	_resume: bool

	def __init__(self, env: PZEnvWrapper, models: list[PZArenaModel], resume=False):
		self._env = env
		self._models = models
		self._resume = resume

	def run(self, model_index: int) -> None:
		model = self._models[model_index]
		print(f"Running model {model.name}")
		self._process_models()
		model.learn(lambda: self._on_reload_models(model))

	def _on_reload_models(self, model: PZArenaModel) -> None:
		model.save()
		self._env.reload_models()

	def _process_models(self) -> None:
		if self._resume:
			# Load models from filesystem
			for model in self._models:
				model.load()
		else:
			# Delete all previously trained models
			path = os.path.join(PZArenaModel.MODEL_PATH, "*.zip")
			zip_paths = glob(path)
			for path in zip_paths:
				os.remove(path)