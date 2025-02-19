import thumper
from pz_arena.arena import PZArena
from pz_arena.wrapper import PZEnvWrapper
from pz_arena.sb3_model import SB3PPOModel

pz_env = thumper.env()
env = PZEnvWrapper(pz_env)
models = [
	SB3PPOModel("PPO1", env),
	SB3PPOModel("PPO2", env),
	SB3PPOModel("PPO3", env),
	SB3PPOModel("PPO4", env),
]
arena = PZArena(env, models)
if __name__ == "__main__":
	arena.run()