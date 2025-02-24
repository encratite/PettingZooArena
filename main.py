from multiprocessing import Pool, Lock
import time
import thumper
from typing import Final
from pz_arena.arena import PZArena, ModelLock
from pz_arena.model import PZArenaModel
from pz_arena.wrapper import PZEnvWrapper
from pz_arena.sb3 import PPOModel

# Disable for easier debugging
ENABLE_MULTIPROCESSING: Final[bool] = True

def get_env_models() -> tuple[PZEnvWrapper, list[PZArenaModel]]:
	pz_env = thumper.env()
	env = PZEnvWrapper(pz_env)
	models = [
		PPOModel("PPO1", env),
		PPOModel("PPO2", env),
		PPOModel("PPO3", env),
		PPOModel("PPO4", env),
	]
	return env, models

def run_arena(index: int, locks: list[Lock]) -> None:
	env, models = get_env_models()
	env.set_opponent_models(models, index)
	assert len(models) == len(locks)
	model_locks = [ModelLock(models[i], locks[i]) for i in range(len(models))]
	arena = PZArena(env, model_locks, locks)
	arena.run(index)

def main() -> None:
	# Retrieve models to determine the number of processes
	env, models = get_env_models()
	process_count = len(models)
	# Delete references to enable pickling with multiprocessing
	del env
	del models
	if ENABLE_MULTIPROCESSING:
		pool = Pool(process_count)
		locks = [Lock() for _ in range(process_count)]
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
		run_arena(0)

if __name__ == "__main__":
	main()