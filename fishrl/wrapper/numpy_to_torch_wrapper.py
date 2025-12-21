from __future__ import annotations

from typing import Any, Union

import gymnasium as gym
import numpy as np
import torch

Device = Union[str, torch.device]


def numpy_to_torch(x: np.ndarray | dict[str, np.ndarray], device: Device | None = None) -> torch.Tensor:
    if isinstance(x, dict):
        return {k: numpy_to_torch(v, device) for k, v in x.items()}
    else:
        if device is None:
            return torch.from_numpy(x).float()
        else:
            return torch.from_numpy(x).float().to(device)


def torch_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class NumpyToTorch:
    """Wraps a NumPy-based environment such that it can be interacted with PyTorch Tensors.

    Actions must be provided as PyTorch Tensors and observations will be returned as PyTorch Tensors.
    A vector version of the wrapper exists, :class:`gymnasium.wrappers.vector.NumpyToTorch`.

    Note:
        For ``rendered`` this is returned as a NumPy array not a pytorch Tensor.

    Example:
        >>> import torch
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> env = NumpyToTorch(env)
        >>> obs, _ = env.reset(seed=123)
        >>> type(obs)
        <class 'torch.Tensor'>
        >>> action = torch.tensor(env.action_space.sample())
        >>> obs, reward, terminated, truncated, info = env.step(action)
        >>> type(obs)
        <class 'torch.Tensor'>
        >>> type(reward)
        <class 'float'>
        >>> type(terminated)
        <class 'bool'>
        >>> type(truncated)
        <class 'bool'>
    """

    def __init__(self, env: gym.Env, device: Device | None = None):
        """Wrapper class to change inputs and outputs of environment to PyTorch tensors.

        Args:
            env: The NumPy-based environment to wrap
            device: The device the torch Tensors should be moved to
        """
        self.device: Device | None = device
        self.env = env

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        """Using a PyTorch based action that is converted to NumPy to be used by the environment.

        Args:
            action: A PyTorch-based action

        Returns:
            The PyTorch-based Tensor next observation, reward, termination, truncation, and extra info
        """
        jax_action = torch_to_numpy(action)
        obs, reward, terminated, truncated, info = self.env.step(jax_action)

        return (
            numpy_to_torch(obs, self.device),
            numpy_to_torch(reward, self.device),
            numpy_to_torch(terminated, self.device),
            numpy_to_torch(truncated, self.device),
            info,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Resets the environment returning PyTorch-based observation and info.

        Args:
            seed: The seed for resetting the environment
            options: The options for resetting the environment, these are converted to jax arrays.

        Returns:
            PyTorch-based observations and info
        """
        if options:
            options = torch_to_numpy(options)

        obs, extra = self.env.reset(seed=seed, options=options)
        return numpy_to_torch(obs, self.device), extra

    def set_state(self, state: torch.Tensor, index: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sets the state of the environment and returns the observation and info."""
        if index is not None:
            index = torch_to_numpy(index)
        state = torch_to_numpy(state)
        self.env.set_state(state, index)

    def render(self) -> None:
        raise NotImplementedError

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def single_observation_space(self):
        return self.env.single_observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def single_action_space(self):
        return self.env.single_action_space

    @property
    def num_envs(self) -> int:
        return self.env.num_envs
