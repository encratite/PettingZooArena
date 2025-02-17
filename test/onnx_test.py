import time
import gymnasium as gym
import pickle
import torch as th
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.save_util import recursive_getattr, data_to_json

class OnnxableSB3Policy(th.nn.Module):
	def __init__(self, policy: BasePolicy):
		super().__init__()
		self.policy = policy

	def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
		return self.policy(observation, deterministic=True)

def time_operation(description, operation):
	start = time.perf_counter()
	operation()
	end = time.perf_counter()
	duration = (end - start) * 1000
	print(f"{description}: {duration} ms")

def save_onnx(model):
	onnx_policy = OnnxableSB3Policy(model.policy)

	observation_size = model.observation_space.shape
	dummy_input = th.randn(1, *observation_size)
	th.onnx.export(
		onnx_policy,
		dummy_input,
		"model/ppo_cartpole.onnx",
		opset_version=17,
		input_names=["input"],
	)

def pickle_model(model):
	data = model.__dict__.copy()
	exclude = []
	exclude = set(exclude).union(model._excluded_save_params())
	state_dicts_names, torch_variable_names = model._get_torch_save_params()
	all_pytorch_variables = state_dicts_names + torch_variable_names
	for torch_var in all_pytorch_variables:
		# We need to get only the name of the top most module as we'll remove that
		var_name = torch_var.split(".")[0]
		# Any params that are in the save vars must not be saved by data
		exclude.add(var_name)

	# Remove parameter entries of parameters which are to be excluded
	for param_name in exclude:
		data.pop(param_name, None)

	# Build dict of torch variables
	pytorch_variables = None
	if torch_variable_names is not None:
		pytorch_variables = {}
		for name in torch_variable_names:
			attr = recursive_getattr(model, name)
			pytorch_variables[name] = attr

	# Build dict of state_dicts
	params_to_save = model.get_parameters()

	# del data["clip_range"]
	# del data["lr_schedule"]

	output = {
		"data": data,
		"params": params_to_save,
		"pytorch_variables": pytorch_variables
	}
	# data = pickle.dumps(output)
	# print(f"Size of pickled data: {len(data)} ({data[:20]})")
	json1 = data_to_json(data)
	json2 = data_to_json(params_to_save)
	print(f"Size of data: {len(json1)}")
	print(f"Size of params: {len(json2)}")

if __name__ == "__main__":
	# env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)
	env = gym.make("CartPole-v1")
	model = PPO("MlpPolicy", env, verbose=1, device="cpu")
	model.learn(total_timesteps=1000)
	time_operation(
		"zip",
		lambda: model.save("R:/model/ppo_cartpole")
	)
	time_operation(
		"ONNX",
		lambda: save_onnx(model)
	)
	time_operation(
		"JSON",
		lambda: pickle_model(model)
	)
	time_operation(
		"zip",
		lambda: model.save("R:/model/ppo_cartpole2")
	)