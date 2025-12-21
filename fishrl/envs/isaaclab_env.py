from typing import Any

import gymnasium as gym
from isaaclab.app import AppLauncher

from fishrl.envs import BaseVecEnv

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class IsaacLabEnv(BaseVecEnv):
    def __init__(self, task_name: str, num_envs: int = 1, seed: int = 0):
        env_cfg = parse_env_cfg(task_name, device="cuda", num_envs=num_envs)
        env_cfg.seed = seed
        self.envs = gym.make(task_name, cfg=env_cfg, render_mode=None)

    def reset(self, seed: int | None = None, options: Any | None = None):
        obs, extra = self.envs.reset(seed=seed, options=options)
        return obs["policy"], extra

    def step(self, action):
        obs, reward, terminations, truncations, info = self.envs.step(action)
        return obs["policy"], reward, terminations, truncations, info

    def render(self):
        raise NotImplementedError

    @property
    def single_observation_space(self):
        return self.envs.unwrapped.single_observation_space["policy"]

    @property
    def observation_space(self):
        return self.envs.unwrapped.observation_space["policy"]

    @property
    def single_action_space(self):
        return self.envs.unwrapped.single_action_space

    @property
    def action_space(self):
        return self.envs.unwrapped.action_space

    @property
    def num_envs(self) -> int:
        return self.envs.unwrapped.num_envs
