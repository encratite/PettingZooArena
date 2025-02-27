from typing import Final, cast
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
from pz_arena.sb3_model import SB3Model, PPOModel, DQNModel

# Disable for easier debugging
ENABLE_MULTIPROCESSING: Final[bool] = False

def model_on_step(raw_env: thumper.raw_env, env: PZEnvWrapper, stats: ThumperStats, logger: Logger) -> None:
	stats.logger = logger
	index = env.agent_index
	stats.on_step(raw_env, index)

def get_env_model(name: str, constructor, **kwargs) -> tuple[PZEnvWrapper, PZArenaModel]:
	raw_env = thumper.raw_env()
	wrapped_env = thumper.wrap_env(raw_env)
	env = PZEnvWrapper(wrapped_env)
	stats = ThumperStats()
	on_step = partial(model_on_step, raw_env, env, stats)
	model = cast(PZArenaModel, constructor(name, env, on_step=on_step, **kwargs))
	return env, model

def get_env_models() -> list[tuple[PZEnvWrapper, PZArenaModel]]:
	env_models = [
		get_env_model("DQN1", DQNModel),
		get_env_model("PPO1", PPOModel, learning_rate=1e-4),
		get_env_model("PPO2", PPOModel, learning_rate=1e-3),
		get_env_model("PPO3", PPOModel, learning_rate=1e-4, gamma=0.997),
		get_env_model("PPO4", PPOModel, learning_rate=1e-4, gamma=0.997, gae_lambda=0.97),
	]
	return env_models

def run_arena(index: int, locks: list[Lock]) -> None:
	env_models = get_env_models()
	envs = [x[0] for x in env_models]
	models = [x[1] for x in env_models]
	for x in envs:
		x.set_opponent_models(models, index)
	assert len(models) == len(locks)
	env = envs[index]
	model_locks = [ModelLock(models[i], locks[i]) for i in range(len(models))]
	arena = PZArena(env, model_locks)
	arena.run(index)

def main() -> None:
	# Retrieve models to determine the number of processes
	env_models = get_env_models()
	process_count = len(env_models)
	# Delete references to enable pickling with multiprocessing
	del env_models
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