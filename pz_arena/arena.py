import os
from glob import glob
from multiprocessing import Pool
from .model import PZArenaModel

class PZArena:
	def __init__(self, models: list[PZArenaModel], resume=False):
		self._models = models
		self._resume = resume

	def run(self):
		if __name__ == "__main__":
			pool_size = len(self._models)
			with Pool(pool_size) as p:
				p.map(self._worker, self._models)

	def _worker(self, model: PZArenaModel):
		print(f"Running model {model.name}")
		self._process_models()
		model.learn()

	def _process_models(self):
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