import os
import shutil
from typing import Final
from functools import partial
from multiprocessing import Pool
import time
import thumper
from thumper.stats import ThumperStats
from stable_baselines3.common.logger import Logger
from pz_arena.config import Configuration
from pz_arena.arena import PZArena
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
	end_of_game_rewards = index == 2 or index == 5
	raw_env = thumper.raw_env(end_of_game_rewards=end_of_game_rewards)
	wrapped_env = thumper.wrap_env(raw_env)
	env = PZEnvWrapper(wrapped_env)
	stats = ThumperStats()
	models = [
		DQNModel("DQN1", env, learning_rate=1e-4),
		DQNModel("DQN2", env, learning_rate=1e-3),
		DQNModel("DQN3", env, learning_rate=1e-4),
		PPOModel("PPO1", env, learning_rate=1e-4),
		PPOModel("PPO2", env, learning_rate=1e-3),
		PPOModel("PPO3", env, learning_rate=1e-4),
	]
	models[index].on_step = partial(model_on_step, raw_env, env, stats)
	return env, models

def run_arena(index: int) -> None:
	env, models = get_env_models(index)
	env.set_opponent_models(models, index)
	arena = PZArena(env, models)
	arena.run(index)

def cleanup() -> None:
	# Delete all previously trained models
	try:
		shutil.rmtree(Configuration.MODEL_PATH)
		os.mkdir(Configuration.MODEL_PATH)
	except FileNotFoundError:
		pass
	# Delete TensorBoard data
	try:
		shutil.rmtree(Configuration.TENSORBOARD_LOG)
		os.mkdir(Configuration.TENSORBOARD_LOG)
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
		pool = Pool(process_count)
		try:
			pool.map_async(run_arena, range(process_count))
			# Hack to enable shutting down the entire pool by pressing Ctrl + C (without this it just hangs)
			while True:
				time.sleep(0.1)
		except KeyboardInterrupt:
			print("Shutting down pool")
			pool.terminate()
			pool.join()
	else:
		run_arena(0)

if __name__ == "__main__":
	main()