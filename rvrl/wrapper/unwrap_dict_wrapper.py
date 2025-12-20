import warnings

import gymnasium as gym
import numpy as np

from rvrl.envs.base import BaseVecEnv


class UnwrapDictWrapper(gym.Wrapper):
    """
    Assume the observation space is a dict with single key, and unwrap the observation space to expose the value.
    """

    env: BaseVecEnv

    def __init__(self, env):
        super().__init__(env)

    def _get_obs(self, obs):
        return next(iter(obs.values()))

    def _get_final_obs(self, final_obs):
        new_final_obs = np.full(self.num_envs, None)
        new_final_obs[:] = [next(iter(o.values())) if o is not None else None for o in final_obs]
        return new_final_obs

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if "final_observation" in info:
            try:
                info["final_observation"] = self._get_final_obs(info["final_observation"])
            except Exception as e:
                warnings.warn(f"Failed to unwrap final observation: {e}")

        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "final_observation" in info:
            try:
                info["final_observation"] = self._get_final_obs(info["final_observation"])
            except Exception as e:
                warnings.warn(f"Failed to unwrap final observation: {e}")

        return self._get_obs(obs), reward, terminated, truncated, info

    @property
    def single_observation_space(self):
        return next(iter(self.env.single_observation_space.values()))

    @property
    def observation_space(self):
        return next(iter(self.env.observation_space.values()))
