import os
import shutil
import re
from glob import glob
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
	end_of_game_rewards = index == 1 or index == 3
	raw_env = thumper.raw_env(end_of_game_rewards=end_of_game_rewards)
	wrapped_env = thumper.wrap_env(raw_env)
	env = PZEnvWrapper(wrapped_env)
	stats = ThumperStats()
	models = [
		DQNModel("DQN1", env, learning_rate=1e-4),
		DQNModel("DQN2", env, learning_rate=5e-3),
		PPOModel("PPO1", env, learning_rate=1e-4),
		PPOModel("PPO2", env, learning_rate=5e-3),
	]
	models[index].on_step = partial(model_on_step, raw_env, env, stats)
	return env, models

def run_arena(index: int) -> None:
	env, models = get_env_models(index)
	env.set_opponent_models(models, index)
	arena = PZArena(env, models)
	arena.run(index)

def remove_directories() -> None:
	remove_directory(Configuration.MODEL_DIRECTORY)
	remove_directory(Configuration.MODEL_TEMP_DIRECTORY)
	remove_directory(Configuration.TENSORBOARD_LOG)

def remove_directory(directory: str) -> None:
	try:
		shutil.rmtree(directory)
		os.mkdir(directory)
	except FileNotFoundError:
		pass

def remove_old_models() -> None:
	pattern = os.path.join(Configuration.MODEL_DIRECTORY, "*.zip")
	files = glob(pattern)
	groups = {}
	group_pattern = re.compile("^[^ ]+")
	for file in files:
		file_name = os.path.basename(file)
		match = group_pattern.match(file_name)
		group = match[0]
		if group not in groups:
			groups[group] = []
		groups[group].append(file)
	for group in groups:
		files = sorted(groups[group])
		files_to_delete = max(len(files) - Configuration.MODEL_FILE_LIMIT, 0)
		if files_to_delete > 0:
			targets = files[0 : files_to_delete]
			for file in targets:
				os.remove(file)

def main() -> None:
	remove_directories()
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
				remove_old_models()
				time.sleep(0.25)
		except KeyboardInterrupt:
			print("Shutting down pool")
			pool.terminate()
			pool.join()
	else:
		run_arena(0)

if __name__ == "__main__":
	main()