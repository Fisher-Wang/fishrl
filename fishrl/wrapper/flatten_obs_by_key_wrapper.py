from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces


class FlattenObsByKeyWrapper(gym.ObservationWrapper):
    """Assume the original observation space is a dict, and flatten the observation space for the specified keys.

    Reference: https://gymnasium.farama.org/v0.29.0/api/wrappers/observation_wrappers/#gymnasium.wrappers.FlattenObservation
    """

    def __init__(self, env: gym.Env, keys: list[str]):
        super().__init__(env)
        self.env = env
        self.keys = keys

        space_dict = {}
        for key in env.observation_space.keys():
            if key in keys:
                space_dict[key] = spaces.flatten_space(env.observation_space[key])
            else:
                space_dict[key] = env.observation_space[key]
        self.observation_space = spaces.Dict(space_dict)

    def observation(self, observation: dict):
        for key in self.keys:
            observation[key] = spaces.flatten(self.env.observation_space[key], observation[key])
        return observation
