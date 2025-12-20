# Refer to https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo/ppo_rgb.py and https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo/ppo.py

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from torch import Tensor

from rvrl.envs import BaseVecEnv


class ManiskillVecEnvWithoutRender(BaseVecEnv):
    def __init__(
        self,
        task_name: str,
        num_envs: int,
        seed: int,
        device: str | torch.device,
        obs_mode: str,
        image_size: tuple[int, int] | None = None,
    ):
        assert obs_mode == "prop", "Only prop mode is supported"
        self.env = gym.make(
            task_name,
            num_envs=num_envs,
            obs_mode="state",
            render_mode="rgb_array",
            sim_backend="physx_cuda",
            control_mode="pd_ee_delta_pose",  # XXX: make this configurable, currently same as tdmpc2
            robot_uids="panda",  # XXX: make this configurable; this will affect agent_sensor
            sensor_configs={},
            # reward_mode="sparse",  # XXX: make this configurable
        )
        if isinstance(self.env.action_space, gym.spaces.Dict):
            self.env = FlattenActionSpaceWrapper(self.env)
        self.env = ManiSkillVectorEnv(self.env, num_envs)
        self._obs_mode = obs_mode

        self._action_space = self.env.action_space
        self._single_action_space = self.env.single_action_space
        self._obs_space = self._get_obs_space(self.env.observation_space)
        self._single_obs_space = self._get_obs_space(self.env.single_observation_space)
        self._obs_space.seed(seed)
        self._single_obs_space.seed(seed)
        self._action_space.seed(seed)
        self._single_action_space.seed(seed)

    def _get_obs_space(self, original_obs_space):
        space_dict = {}
        if "prop" in self._obs_mode.split(","):
            space_dict["prop"] = original_obs_space
        return gym.spaces.Dict(space_dict)

    def _get_obs(self, obs):
        obs_dict = {}
        if "prop" in self._obs_mode.split(","):
            obs_dict["prop"] = obs
        return obs_dict

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options={} if options is None else options)
        return self._get_obs(obs), info

    def step(self, action: Tensor):
        obs, reward, terminations, truncations, info = self.env.step(action)
        if "final_observation" in info:
            info["final_observation"] = self._get_obs(info["final_observation"])
        return self._get_obs(obs), reward, terminations, truncations, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def single_action_space(self):
        return self._single_action_space

    @property
    def single_observation_space(self):
        return self._single_obs_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def num_envs(self) -> int:
        return self.env.num_envs


class ManiskillVecEnvWithRender(BaseVecEnv):
    def __init__(
        self,
        task_name: str,
        num_envs: int,
        seed: int,
        device: str | torch.device,
        obs_mode: str,
        image_size: tuple[int, int] | None = None,
    ):
        self.env = gym.make(
            task_name,
            num_envs=num_envs,
            obs_mode="rgb+state",
            render_mode="all",
            sim_backend="physx_cuda",
            control_mode="pd_ee_delta_pose",  # XXX: make this configurable, currently same as tdmpc2
            robot_uids="panda",  # XXX: make this configurable; this will affect agent_sensor
            sensor_configs={"width": image_size[0], "height": image_size[1]} if "rgb" in obs_mode.split(",") else {},
        )
        self.env = FlattenRGBDObservationWrapper(
            self.env,
            state="prop" in obs_mode.split(","),
            rgb="rgb" in obs_mode.split(","),
            depth=False,
        )
        self.width = image_size[0]
        self.height = image_size[1]
        if isinstance(self.env.action_space, gym.spaces.Dict):
            self.env = FlattenActionSpaceWrapper(self.env)
        self.env = ManiSkillVectorEnv(self.env, num_envs)
        self._obs_mode = obs_mode

        self._action_space = self.env.action_space
        self._single_action_space = self.env.single_action_space
        self._obs_space = self._get_obs_space(self.env.observation_space)
        self._single_obs_space = self._get_obs_space(self.env.single_observation_space)
        self._obs_space.seed(seed)
        self._single_obs_space.seed(seed)
        self._action_space.seed(seed)
        self._single_action_space.seed(seed)

    def _get_obs_space(self, original_obs_space):
        space_dict = {}
        if "rgb" in self._obs_mode.split(","):
            space_dict["rgb"] = gym.spaces.Box(low=0, high=1, shape=(3, self.width, self.height), dtype=np.float32)
        if "prop" in self._obs_mode.split(","):
            space_dict["prop"] = original_obs_space["state"]
        return gym.spaces.Dict(space_dict)

    def _get_obs(self, obs):
        obs_dict = {}
        if "rgb" in self._obs_mode.split(","):
            obs_dict["rgb"] = obs["rgb"].permute(0, 3, 1, 2) / 255.0
        if "prop" in self._obs_mode.split(","):
            obs_dict["prop"] = obs["state"]
        return obs_dict

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options={} if options is None else options)
        return self._get_obs(obs), info

    def step(self, action: Tensor):
        obs, reward, terminations, truncations, info = self.env.step(action)
        if "final_observation" in info:
            info["final_observation"] = self._get_obs(info["final_observation"])
        return self._get_obs(obs), reward, terminations, truncations, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def single_action_space(self):
        return self._single_action_space

    @property
    def single_observation_space(self):
        return self._single_obs_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def num_envs(self) -> int:
        return self.env.num_envs
