"""DeepMind Lab environment wrapper for fishrl."""

from __future__ import annotations

from typing import Any

import deepmind_lab
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Action sets from DreamerV3 / NeDreamer references
IMPALA_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),  # Forward
    (0, 0, 0, -1, 0, 0, 0),  # Backward
    (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),  # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),  # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),  # Fire
)

POPART_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),  # Forward
    (0, 0, 0, -1, 0, 0, 0),  # Backward
    (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
    (-10, 0, 0, 0, 0, 0, 0),  # Small Look Left
    (10, 0, 0, 0, 0, 0, 0),  # Small Look Right
    (-60, 0, 0, 0, 0, 0, 0),  # Large Look Left
    (60, 0, 0, 0, 0, 0, 0),  # Large Look Right
    (0, 10, 0, 0, 0, 0, 0),  # Look Down
    (0, -10, 0, 0, 0, 0, 0),  # Look Up
    (-10, 0, 0, 1, 0, 0, 0),  # Forward + Small Look Left
    (10, 0, 0, 1, 0, 0, 0),  # Forward + Small Look Right
    (-60, 0, 0, 1, 0, 0, 0),  # Forward + Large Look Left
    (60, 0, 0, 1, 0, 0, 0),  # Forward + Large Look Right
    (0, 0, 0, 0, 1, 0, 0),  # Fire
)

ACTION_SETS = {
    "impala": IMPALA_ACTION_SET,
    "popart": POPART_ACTION_SET,
}


class DMLab(gym.Env):
    """DeepMind Lab environment wrapped as a gymnasium Env.

    Args:
        level: DMLab30 level name (e.g. "rooms_collect_good_objects_train")
        seed: Random seed
        image_size: (width, height) for rendered images
        action_repeat: Number of times to repeat each action
        action_set: Name of action set ("impala" or "popart")
        mode: "train" or "test"

    References:
    - https://github.com/corl-team/nedreamer/blob/main/envs/dmlab.py
    - https://github.com/danijar/dreamerv3/blob/main/embodied/envs/dmlab.py
    """

    metadata = {}

    def __init__(
        self,
        level: str,
        seed: int,
        image_size: tuple[int, int] = (64, 64),
        action_repeat: int = 4,
        action_set: str = "impala",
        mode: str = "train",
    ):

        self._size = image_size  # (width, height)
        self._action_repeat = action_repeat
        self._actions = ACTION_SETS[action_set]
        self._random = np.random.RandomState(seed)

        config = {
            "width": str(image_size[0]),
            "height": str(image_size[1]),
            "logLevel": "WARN",
        }
        if mode == "test":
            config["allowHoldOutLevels"] = "true"
            config["mixerSeed"] = str(0x600D5EED)

        # Handle level name: if it doesn't have a prefix, add contributed/dmlab30/
        if "/" not in level:
            level = f"contributed/dmlab30/{level}"

        self._env = deepmind_lab.Lab(
            level=level,
            observations=["RGB_INTERLEAVED"],
            config=config,
        )

        self._done = True
        self._last_image = np.zeros((*image_size, 3), dtype=np.uint8)

        # Spaces
        self._observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0.0, high=1.0, shape=(3, image_size[0], image_size[1]), dtype=np.float32),
        })
        self._action_space = spaces.Discrete(len(self._actions))
        self._action_space.seed(seed)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _get_obs(self) -> dict[str, np.ndarray]:
        if self._done:
            image = np.zeros_like(self._last_image)
        else:
            image = self._env.observations()["RGB_INTERLEAVED"]
        self._last_image = image
        # Convert (H, W, 3) uint8 -> (3, H, W) float32 in [0, 1]
        rgb = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        return {"rgb": rgb}

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        self._done = False
        self._env.reset(seed=self._random.randint(0, 2**31 - 1))
        return self._get_obs(), {}

    def step(self, action: int | np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        if not isinstance(action, int):
            action = int(action)
        raw_action = np.array(self._actions[action], dtype=np.intc)
        reward = self._env.step(raw_action, num_steps=self._action_repeat)
        self._done = not self._env.is_running()
        obs = self._get_obs()
        # DMLab episodes are always truncated (time limit), never truly terminated
        terminated = False
        truncated = self._done
        return obs, float(reward), terminated, truncated, {}

    def render(self):
        return self._last_image

    def close(self):
        self._env.close()
