from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


class Info(TypedDict):
    final_observation: NDArray[np.object_]  # array of arrays


class BaseVecEnv(ABC):
    """
    Base class for all the environments used in the algorithm implementation. This class is designed to be as close as possible to the ``gymnasium.vector.VectorEnv`` class. All the methods and properties of this class should has exactly the same definition as the ``gymnasium.vector.VectorEnv`` class. Some uncommon methods and properties are not implemented.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, device: str | torch.device):
        pass

    @abstractmethod
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Tensor], Info]:
        pass

    @abstractmethod
    def step(self, action: Tensor) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Info]:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @property
    @abstractmethod
    def single_observation_space(self) -> gym.spaces.Space:
        pass

    @property
    @abstractmethod
    def single_action_space(self) -> gym.spaces.Space:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space:
        pass

    @property
    @abstractmethod
    def num_envs(self) -> int:
        pass
