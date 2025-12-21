from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import gymnasium as gym
import numpy as np
from dm_control import suite
from dm_env.specs import Array, BoundedArray, DiscreteArray
from gymnasium import spaces


def dm_spec2gym_space(spec) -> spaces.Space[Any]:
    """Converts a dm_env spec to a gymnasium space.
    Reference: https://github.com/Farama-Foundation/Shimmy/blob/main/shimmy/utils/dm_env.py
    """
    if isinstance(spec, (OrderedDict, dict)):
        return spaces.Dict({key: dm_spec2gym_space(value) for key, value in copy.copy(spec).items()})
    elif type(spec) is BoundedArray:
        low = np.broadcast_to(spec.minimum, spec.shape)
        high = np.broadcast_to(spec.maximum, spec.shape)
        return spaces.Box(
            low=low,
            high=high,
            shape=spec.shape,
            dtype=spec.dtype,
        )
    elif type(spec) is Array:
        if np.issubdtype(spec.dtype, np.integer):
            low = np.iinfo(spec.dtype).min
            high = np.iinfo(spec.dtype).max
        elif np.issubdtype(spec.dtype, np.inexact):
            low = float("-inf")
            high = float("inf")
        elif spec.dtype == "bool":
            low = 0
            high = 1
        else:
            raise TypeError(f"Unknown dtype {spec.dtype} for spec {spec}.")

        return spaces.Box(
            low=low,
            high=high,
            shape=spec.shape,
            dtype=spec.dtype,
        )
    elif type(spec) is DiscreteArray:
        return spaces.Discrete(spec.num_values)
    else:
        raise NotImplementedError(f"Cannot convert dm_spec to gymnasium space, unknown spec: {spec}, please report.")


_DEFAULT_CAMERA_ID = {
    "quadruped": 2,  # same as dreamerv3
}


class DMControlEnv(gym.Env):
    def __init__(
        self,
        env_id: str,
        seed: int,
        image_size: tuple[int, int],
        obs_mode: str,
        action_repeat: int = 1,
    ):
        domain_name = env_id.split("-")[0]
        task_name = env_id.split("-")[1]
        self.env = suite.load(domain_name, task_name, task_kwargs={"random": seed})
        self.width = image_size[0]
        self.height = image_size[1]
        self.action_repeat = action_repeat
        self.obs_mode = obs_mode
        self._obs_space = self._get_obs_space(dm_spec2gym_space(self.env.observation_spec()))
        self._true_action_space = dm_spec2gym_space(self.env.action_spec())
        self._norm_action_space = spaces.Box(low=-1, high=1, shape=self._true_action_space.shape, dtype=np.float32)

        self._obs_space.seed(seed)
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._camera_id = _DEFAULT_CAMERA_ID.get(domain_name, 0)

    def _convert_action(self, action) -> np.ndarray:
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    def _get_obs_space(self, original_obs_space: spaces.Space) -> spaces.Space:
        space_dict = {}
        if "prop" in self.obs_mode.split(","):
            space_dict["prop"] = original_obs_space
        if "rgb" in self.obs_mode.split(","):
            space_dict["rgb"] = spaces.Box(low=0, high=1, shape=(3, self.width, self.height), dtype=np.float32)
        if "state" in self.obs_mode.split(","):
            space_dict["state"] = spaces.Box(low=-1, high=1, shape=self.env.physics.get_state().shape, dtype=np.float32)
        return spaces.Dict(space_dict)

    def _get_obs(self, timestep) -> dict[str, np.ndarray]:
        obs_dict = {}
        if "state" in self.obs_mode.split(","):
            obs_dict["state"] = self.env.physics.get_state()
        if "prop" in self.obs_mode.split(","):
            obs_dict["prop"] = timestep.observation
        if "rgb" in self.obs_mode.split(","):
            rgb = self.env.physics.render(width=self.width, height=self.height, camera_id=self._camera_id)
            rgb = np.transpose(rgb, (2, 0, 1)).copy() / 255.0  # (H, W, 3) -> (3, H, W)
            obs_dict["rgb"] = rgb
        return obs_dict

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        timestep = self.env.reset()
        return self._get_obs(timestep), {}

    def set_state(self, state: np.ndarray):
        self.env.physics.set_state(state)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = self._convert_action(action)
        reward = 0
        for _ in range(self.action_repeat):
            timestep = self.env.step(action)
            reward += timestep.reward
            done = timestep.last()
            if done:
                break
        truncated = timestep.last() and timestep.discount == 1.0
        terminated = timestep.last() and timestep.discount == 0.0
        return self._get_obs(timestep), reward, terminated, truncated, {}

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def metadata(self):
        return {}

    @property
    def episode_length(self) -> int:
        return int(self.env._step_limit / self.action_repeat)
