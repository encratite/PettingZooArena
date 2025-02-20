import random
import numpy as np
from typing import Any, Final, cast
from gymnasium import Env
from gymnasium.core import ObsType, ActType, SupportsFloat
from gymnasium.spaces import Discrete, Dict
from pettingzoo import AECEnv
from pettingzoo.utils.env import AgentID
from .error import PZArenaError
from .model import PZArenaModel

class PZEnvWrapper(Env):
	OBSERVATION_KEY: Final[str] = "observation"
	ACTION_MASK_KEY: Final[str] = "action_mask"

	# The underlying PettingZoo AEC environment that is being simulated
	_env: AECEnv
	# The pre-trained models used to select opponent actions (initially None to select actions randomly)
	# Only a randomized subset of these is used in each run
	_opponent_models: list[PZArenaModel] | None
	# The agent ID that represents the agent that is being trained through this environment
	# It is chosen randomly after every reset
	_agent: AgentID | None
	# This dict maps the agent IDs of opponents to models to be used to select actions
	# If no pre-trained models are available, it is initially set to None
	_opponent_model_map: dict[AgentID, PZArenaModel] | None
	# The reward cumulated for this agent ID, both from own moves as well as opponents' moves
	# This is necessary because PettingZoo permits returning rewards for more than one agent at a time
	_reward: float

	def __init__(self, env: AECEnv):
		self._env = env
		self._opponent_models = None
		self._agent = None
		self._opponent_model_map = None
		self._reward = 0
		# Reset necessary to access metadata required by the Gymnasium API
		self._env.reset()
		first_agent = self._get_first_agent()
		action_space = env.action_space(first_agent)
		if not isinstance(action_space, Discrete):
			raise PZArenaError("Only Discrete action spaces are supported")
		self.action_space = action_space
		pz_observation_space = env.observation_space(first_agent)
		if isinstance(pz_observation_space, Dict):
			if self.OBSERVATION_KEY not in pz_observation_space.spaces:
				raise PZArenaError(f"Expected \"{self.OBSERVATION_KEY}\" key in PettingZoo observation_space dictionary")
			self.observation_space = pz_observation_space[self.OBSERVATION_KEY]
		else:
			self.observation_space = pz_observation_space

	def set_opponent_models(self, opponent_models: list[PZArenaModel], skip_index: int) -> None:
		self._opponent_models = []
		for i in range(len(opponent_models)):
			if i == skip_index:
				continue
			model = opponent_models[i]
			model.load()
			self._opponent_models.append(model)

	def reset(
			self,
			*,
			seed: int | None = None,
			options: dict[str, Any] | None = None,
	) -> tuple[ObsType, dict[str, Any]]:
		self._env.reset()
		env_agents = self._env.agents
		if len(env_agents) == 0:
			raise PZArenaError("Underlying env must have at least one agent")
		# Choose a random agent ID for the model that is being trained through this env
		self._agent = random.choice(env_agents)
		# Randomly select a subset of the opponent models for the other agent IDs and store the mapping
		opponent_ids = list(filter(lambda agent_id: agent_id != self._agent, env_agents))
		opponent_model_count = len(env_agents) - 1
		if opponent_model_count > len(self._opponent_models):
			raise PZArenaError("There aren't enough opponent models to train using the underlying PettingZoo environment")
		opponent_models = random.sample(self._opponent_models, k=opponent_model_count)
		assert len(opponent_ids) == len(opponent_models)
		self._opponent_model_map = {}
		for opponent_id, opponent_model in zip(opponent_ids, opponent_models):
			self._opponent_model_map[opponent_id] = opponent_model
		self._perform_opponent_moves()
		observation = self._get_observation()
		info = {}
		return observation, info

	def step(
			self,
			action: ActType
	) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
		assert self._env.agent_selection == self._agent
		self._perform_action(action)
		# Perform all opponent moves until it's either our turn again or the env has been terminated
		# This is convenient for collecting rewards outside our turn and then returning them right away
		self._perform_opponent_moves()
		observation = self._get_observation()
		reward = self._reward
		self._reward = 0
		terminated = self._terminated()
		truncated = self._truncated()
		info = {}
		return observation, reward, terminated, truncated, info

	# Not part of the Gymnasium API but is used by MaskablePPO from the Stable Baselines3 contributions
	def action_masks(self) -> list[bool]:
		observation = self._env.observe(self._env.agent_selection)
		# PettingZoo uses 0/1 int8 values while Gymnasium expects bool
		if self.ACTION_MASK_KEY in observation:
			action_mask = [x == 1 for x in observation[self.ACTION_MASK_KEY]]
		else:
			# The PettingZoo env doesn't have an action mask, so just enable all actions
			discrete_action_space = cast(Discrete, self.action_space)
			action_mask = [True] * discrete_action_space.n
		return action_mask

	def reload_models(self) -> None:
		for model in self._opponent_models:
			model.load()

	def _get_first_agent(self) -> AgentID:
		if len(self._env.agents) == 0:
			raise PZArenaError("No agents in environment")
		first_agent = self._env.agents[0]
		return first_agent

	def _terminated(self) -> bool:
		return self._env.terminations[self._agent]

	def _truncated(self) -> bool:
		return self._env.truncations[self._agent]

	def _get_observation(self):
		assert self._env.agent_selection == self._agent
		observation = self._env.observe(self._agent)
		if isinstance(observation, dict):
			if self.OBSERVATION_KEY not in observation:
				raise PZArenaError(f"Observation is a dict but lacks \"{self.OBSERVATION_KEY}\" key")
			observation = observation[self.OBSERVATION_KEY]
		return observation

	def _perform_opponent_moves(self):
		while self._is_opponent_move():
			self._single_opponent_move()

	def _is_opponent_move(self) -> bool:
		terminated = self._terminated()
		is_opponent_move = self._env.agent_selection != self._agent
		return not terminated and is_opponent_move

	def _single_opponent_move(self) -> None:
		current_agent = self._env.agent_selection
		assert self._agent != current_agent
		action_mask = self.action_masks()
		discrete_action_space = cast(Discrete, self.action_space)
		assert len(action_mask) == discrete_action_space.n
		if self._opponent_models is not None:
			# Use pre-trained opponent models to select actions
			opponent_model = self._opponent_model_map[current_agent]
			observation = self._env.observe(current_agent)
			translated_action_mask = np.array([1 if enabled else 0 for enabled in action_mask])
			action = opponent_model.predict(observation, translated_action_mask)
		else:
			# No pre-trained models are available yet, perform a random move for each opponent
			all_pairs = zip(range(discrete_action_space.n), action_mask)
			enabled_pairs = filter(lambda pair: pair[1], all_pairs)
			enabled_actions = [cast(ActType, i) for i, _enabled in enabled_pairs]
			action = random.choice(enabled_actions)
		self._perform_action(action)

	def _perform_action(self, action: ActType) -> None:
		self._env.step(action)
		self._reward += self._env.rewards[self._agent]