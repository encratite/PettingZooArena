import os
import shutil
from glob import glob
from typing import Final
from functools import partial
from threading import Lock
from multiprocessing import Pool, Manager
import threading
import time
import thumper
from thumper.stats import ThumperStats
from stable_baselines3.common.logger import Logger
from pz_arena.config import Configuration
from pz_arena.arena import PZArena, ModelLock
from pz_arena.model import PZArenaModel
from pz_arena.wrapper import PZEnvWrapper
from pz_arena.sb3_model import PPOModel, DQNModel

# Disable for easier debugging
ENABLE_MULTIPROCESSING: Final[bool] = True

def model_on_step(raw_env: thumper.raw_env, env: PZEnvWrapper, stats: ThumperStats, logger: Logger) -> None:
	stats.logger = logger
	index = env.agent_index
	stats.on_step(raw_env, index)

def get_env_models(index: int) -> tuple[PZEnvWrapper, list[PZArenaModel]]:
	raw_env = thumper.raw_env()
	wrapped_env = thumper.wrap_env(raw_env)
	env = PZEnvWrapper(wrapped_env)
	stats = ThumperStats()
	models = [
		DQNModel("DQN1", env, learning_rate=1e-4),
		DQNModel("DQN2", env, learning_rate=1e-3),
		DQNModel("DQN3", env, learning_rate=1e-4, gamma=0.997),
		PPOModel("PPO1", env, learning_rate=1e-4),
		PPOModel("PPO2", env, learning_rate=1e-3),
		PPOModel("PPO3", env, learning_rate=1e-4, gamma=0.997, gae_lambda=0.97),
	]
	models[index].on_step = partial(model_on_step, raw_env, env, stats)
	return env, models

def run_arena(index: int, locks: list[Lock]) -> None:
	env, models = get_env_models(index)
	env.set_opponent_models(models, index)
	assert len(models) == len(locks)
	model_locks = [ModelLock(models[i], locks[i]) for i in range(len(models))]
	arena = PZArena(env, model_locks)
	arena.run(index)

def cleanup() -> None:
	# Delete all previously trained models
	path = os.path.join(Configuration.MODEL_PATH, "*.zip")
	zip_paths = glob(path)
	for path in zip_paths:
		os.remove(path)
	# Delete TensorBoard data
	try:
		shutil.rmtree(Configuration.TENSORBOARD_LOG)
	except FileNotFoundError:
		pass

def main() -> None:
	cleanup()
	# Retrieve models to determine the number of processes
	env, models = get_env_models(0)
	process_count = len(models)
	# Delete references to enable pickling with multiprocessing
	del env
	del models
	if ENABLE_MULTIPROCESSING:
		with Manager() as manager:
			pool = Pool(process_count)
			locks = [manager.Lock() for _ in range(process_count)]
			arguments = [(i, locks) for i in range(process_count)]
			try:
				pool.starmap_async(run_arena, arguments)
				# Hack to enable shutting down the entire pool by pressing Ctrl + C (without this it just hangs)
				while True:
					time.sleep(0.1)
			except KeyboardInterrupt:
				print("Shutting down pool")
				pool.terminate()
				pool.join()
	else:
		locks = [threading.Lock() for _ in range(process_count)]
		run_arena(0, locks)

if __name__ == "__main__":
	main()