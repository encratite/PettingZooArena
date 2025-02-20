from typing import Final
from multiprocessing import Pool
import thumper
from pz_arena.arena import PZArena
from pz_arena.wrapper import PZEnvWrapper
from pz_arena.sb3 import PPOModel

MODEL_COUNT: Final[int] = 4

def run_arena(index: int) -> None:
	pz_env = thumper.env()
	env = PZEnvWrapper(pz_env)
	models = [
		PPOModel("PPO1", env),
		PPOModel("PPO2", env),
		PPOModel("PPO3", env),
		PPOModel("PPO4", env),
	]
	assert len(models) == MODEL_COUNT
	env.set_opponent_models(models, index)
	arena = PZArena(env, models)
	arena.run(index)

if __name__ == "__main__":
	with Pool(MODEL_COUNT) as p:
		p.map(run_arena, range(MODEL_COUNT))