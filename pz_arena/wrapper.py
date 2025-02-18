import random
import numpy as np
from typing import Any, Final, cast
from gymnasium import Env
from gymnasium.core import ObsType, ActType, SupportsFloat
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils.env import AgentID
from .error import PZArenaError
from .model import PZArenaModel

class PettingZooWrapper(Env):
	OBSERVATION_KEY: Final[str] = "observation"
	ACTION_MASK_KEY: Final[str] = "action_mask"

	_env: AECEnv
	_opponent_models: list[PZArenaModel] | None
	_agent_id: AgentID | None
	_opponent_model_map: dict[AgentID, PZArenaModel] | None

	def __init__(self, env: AECEnv, opponent_models: list[PZArenaModel] | None = None):
		self._env = env
		self._opponent_models = opponent_models
		self._agent_id = None
		self._opponent_model_map = None
		first_agent = self._get_first_agent()
		action_space = env.action_space(first_agent)
		if not isinstance(action_space, Discrete):
			raise PZArenaError("Only Discrete action spaces are supported")
		self.action_space = action_space
		pz_observation_space = env.observation_space(first_agent)
		if isinstance(pz_observation_space, dict):
			if self.OBSERVATION_KEY not in pz_observation_space:
				raise PZArenaError(f"Expected \"{self.OBSERVATION_KEY}\" key in PettingZoo observation_space dictionary")
			self.observation_space = pz_observation_space[self.OBSERVATION_KEY]
		else:
			self.observation_space = pz_observation_space
		if opponent_models is not None:
			for model in opponent_models:
				model.load()

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
		self._agent_id = random.choice(env_agents)
		# Randomly select a subset of the opponent models for the other agent IDs and store the mapping
		opponent_ids = [agent_id != self._agent_id for agent_id in env_agents]
		opponent_model_count = len(env_agents) - 1
		if opponent_model_count > len(self._opponent_models):
			raise PZArenaError("There aren't enough opponent models to train using the underlying PettingZoo environment")
		opponent_models = random.sample(self._opponent_models, k=opponent_model_count)
		assert len(opponent_ids) == len(opponent_models)
		for opponent_id, opponent_model in zip(opponent_ids, opponent_models):
			self._opponent_model_map[opponent_id] = opponent_model
		self._perform_opponent_moves()
		observation = self._env.observe(self._env.agent_selection)
		info = {}
		return observation, info

	def step(
			self,
			action: ActType
	) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
		raise NotImplementedError()

	# Not part of the Gymnasium API but is used by MaskablePPO from the Stable Baselines3 contributions
	def action_mask(self) -> list[bool]:
		observation = self._env.observe(self._env.agent_selection)
		# PettingZoo uses 0/1 int8 values while Gymnasium expects bool
		if self.ACTION_MASK_KEY in observation:
			action_mask = [x == 1 for x in observation[self.ACTION_MASK_KEY]]
		else:
			# The PettingZoo env doesn't have an action mask, so just enable all actions
			discrete_action_space = cast(Discrete, self.action_space)
			action_mask = [True] * discrete_action_space.n
		return action_mask

	def _get_first_agent(self) -> AgentID:
		if len(self._env.agents) == 0:
			raise PZArenaError("No agents in environment")
		first_agent = self._env.agents[0]
		return first_agent

	def _perform_opponent_moves(self):
		raise NotImplementedError()

	def _perform_opponent_move(self):
		action_mask = self.action_mask()
		discrete_action_space = cast(Discrete, self.action_space)
		assert len(action_mask) == discrete_action_space.n
		if self._opponent_models is not None:
			opponent_model = self._opponent_model_map[self._env.agent_selection]
			observation = self._env.observe(self._env.agent_selection)
			translated_action_mask = np.array([1 if enabled else 0 for enabled in action_mask])
			action = opponent_model.predict(observation, translated_action_mask)
		else:
			# No pre-trained models are available yet, perform a random move
			all_pairs = zip(range(discrete_action_space.n), action_mask)
			enabled_pairs = filter(lambda pair: pair[1], all_pairs)
			enabled_actions = [cast(ActType, i) for i, _enabled in enabled_pairs]
			action = random.choice(enabled_actions)
		self._env.step(action)