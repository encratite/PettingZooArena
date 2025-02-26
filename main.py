from typing import Final
from functools import partial
from threading import Lock
from multiprocessing import Pool, Manager
import threading
import time
import thumper
from thumper.stats import ThumperStats
from stable_baselines3.common.logger import Logger
from pz_arena.arena import PZArena, ModelLock
from pz_arena.model import PZArenaModel
from pz_arena.wrapper import PZEnvWrapper
from pz_arena.sb3 import PPOModel

# Disable for easier debugging
ENABLE_MULTIPROCESSING: Final[bool] = False

def model_on_step(env: thumper.raw_env, stats: ThumperStats, logger: Logger) -> None:
	stats.logger = logger
	stats.on_step(env.last_action)

def get_ppo_model(name, **kwargs) -> PPOModel:
	raw_env = thumper.raw_env()
	wrapped_env = thumper.wrap_env(raw_env)
	env = PZEnvWrapper(wrapped_env)
	stats = ThumperStats()
	on_step = partial(model_on_step, raw_env, stats)
	model = PPOModel(name, env, raw_env, on_step, **kwargs)
	return model

def get_models() -> list[PZArenaModel]:
	models = [
		get_ppo_model("PPO1", learning_rate=1e-4),
		get_ppo_model("PPO2", learning_rate=1e-3),
		get_ppo_model("PPO3", learning_rate=1e-4, gamma=0.997),
		get_ppo_model("PPO4", learning_rate=3e-4, gamma=0.997, gae_lambda=0.97),
	]
	return models

def run_arena(index: int, locks: list[Lock]) -> None:
	models = get_models()
	for model in models:
		model.env.set_opponent_models(models, index)
	assert len(models) == len(locks)
	model_locks = [ModelLock(models[i], locks[i]) for i in range(len(models))]
	arena = PZArena(env, model_locks)
	arena.run(index)

def main() -> None:
	# Retrieve models to determine the number of processes
	env, models = get_env_models()
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