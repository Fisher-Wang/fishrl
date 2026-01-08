from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule
from torch import Tensor, nn
from torch.distributions import (
    Bernoulli,
    Distribution,
    Independent,
    Normal,
    OneHotCategoricalStraightThrough,
    kl_divergence,
)
from torch.distributions.utils import probs_to_logits
from torchvision.utils import make_grid
from tqdm.rich import tqdm

from fishrl.envs import BaseVecEnv, create_vector_env
from fishrl.utils.logger import JsonlOutput, Logger, TensorboardOutput, WandbOutput
from fishrl.utils.metrics import MetricAggregator
from fishrl.utils.reproducibility import enable_deterministic_run, seed_everything
from fishrl.utils.timer import timer
from fishrl.utils.utils import Ratio
from fishrl.wrapper import UnwrapDictWrapper


class ObsShiftWrapper(gym.Wrapper):
    env: BaseVecEnv

    # change observation space from [0, 1] to [-0.5, 0.5]
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = obs - 0.5
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs - 0.5
        return obs, reward, terminated, truncated, info


########################################################
## Standalone utils
########################################################
def make_logger(
    output_names: list[Literal["wandb", "tensorboard", "jsonl"]],
    logdir: str,
    config: dict[str, Any],
    init_step: int = 0,
    wandb_entity: str | None = None,
    wandb_project: str = "fishrl",
) -> Logger:
    run_name = logdir.split("/")[-1]
    domain_name = logdir.split("/")[-2]
    outputs = []
    for output in output_names:
        if output == "wandb":
            wandb_kargs = dict(dir=logdir, project=wandb_project, config=config, entity=wandb_entity)
            outputs.append(WandbOutput(name=f"{domain_name}/{run_name}", **wandb_kargs))
        elif output == "tensorboard":
            outputs.append(TensorboardOutput(logdir=logdir, config=config))
        elif output == "jsonl":
            outputs.append(JsonlOutput(logdir=logdir, filename="metrics.jsonl"))
    logger = Logger(outputs, init_step=init_step)
    return logger


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int] | dict,
        action_size: int,
        device: str | torch.device,
        num_envs: int = 1,
        capacity: int = 5000000,
    ):
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity

        self.observation = self.build_observation_buffer_recursively(observation_shape)
        self.next_observation = self.build_observation_buffer_recursively(observation_shape)
        self.action = np.empty((self.capacity, self.num_envs, action_size), dtype=np.float32)
        self.reward = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.terminated = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def build_observation_buffer_recursively(self, observation_shape: tuple[int] | dict) -> np.ndarray | dict:
        if isinstance(observation_shape, tuple):
            return np.empty((self.capacity, self.num_envs, *observation_shape), dtype=np.float32)
        elif isinstance(observation_shape, dict):
            return {
                key: self.build_observation_buffer_recursively(observation_shape[key])
                for key in observation_shape.keys()
            }

    def add(
        self,
        observation: Tensor,
        action: Tensor,
        reward: Tensor,
        next_observation: Tensor,
        done: Tensor,
        terminated: Tensor,
    ):
        self.observation[self.buffer_index] = observation.detach().cpu().numpy()
        self.next_observation[self.buffer_index] = next_observation.detach().cpu().numpy()
        self.action[self.buffer_index] = action.detach().cpu().numpy()
        self.reward[self.buffer_index] = reward.unsqueeze(-1).detach().cpu().numpy()
        self.done[self.buffer_index] = done.unsqueeze(-1).detach().cpu().numpy()
        self.terminated[self.buffer_index] = terminated.unsqueeze(-1).detach().cpu().numpy()

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, chunk_size) -> dict[str, Tensor]:
        """
        Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.
        """
        last_filled_index = self.buffer_index - chunk_size + 1
        assert self.full or (last_filled_index > batch_size), "too short dataset or too long chunk_size"
        sample_index = np.random.randint(0, self.capacity if self.full else last_filled_index, batch_size).reshape(
            -1, 1
        )
        chunk_length = np.arange(chunk_size).reshape(1, -1)

        sample_index = (sample_index + chunk_length) % self.capacity
        env_index = np.random.randint(0, self.num_envs, batch_size)
        flattened_index = sample_index * self.num_envs + env_index[:, None]

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        observation = torch.as_tensor(flatten(self.observation)[flattened_index], device=self.device).float()
        next_observation = torch.as_tensor(flatten(self.next_observation)[flattened_index], device=self.device).float()
        action = torch.as_tensor(flatten(self.action)[flattened_index], device=self.device)
        reward = torch.as_tensor(flatten(self.reward)[flattened_index], device=self.device)
        done = torch.as_tensor(flatten(self.done)[flattened_index], device=self.device)
        terminated = torch.as_tensor(flatten(self.terminated)[flattened_index], device=self.device)

        sample = TensorDict(
            obs=observation,
            action=action,
            reward=reward,
            next_obs=next_observation,
            done=done,
            terminated=terminated,
            batch_size=action.shape[0],
            device=self.device,
        )
        return sample


def compute_lambda_values(
    rewards: Tensor, values: Tensor, continues: Tensor, horizon: int, gae_lambda: float
) -> Tensor:
    """
    Compute lambda returns (λ-returns) for Generalized Advantage Estimation (GAE).

    The lambda return is computed recursively as:
    R_t^λ = r_t + γ * [(1 - λ) * V(s_{t+1}) + λ * R_{t+1}^λ]

    Args:
        rewards: (batch_size, time_step) - r_t is the immediate reward received after taking action at time t
        values: (batch_size, time_step) - V(s_t) is the value estimate of the state s_t
        continues: (batch_size, time_step) - c_t is the continue flag after taking action at time t. It is already multiplied by gamma (γ).
        horizon: int - T is the length of the planning horizon
        gae_lambda: float - lambda parameter for GAE (λ, typically 0.95)

    Returns:
        Tensor: (batch_size, horizon-1) - R_t^λ is the lambda return at time t = 0, ..., T-2.
    """
    # Given the following diagram, with horizon=4
    # Actions:            a'0      a'1      a'2
    #                     ^ \      ^ \      ^ \
    #                    /   \    /   \    /   \
    #                   /     \  /     \  /     \
    # States:         z0  ->  z'1  ->  z'2  ->  z'3
    # Values:         v'0    [v'1]    [v'2]    [v'3]      <-- input
    # Rewards:       [r'0]   [r'1]    [r'2]     r'3       <-- input
    # Continues:     [c'0]   [c'1]    [c'2]     c'3       <-- input
    # Lambda-values: [l'0]   [l'1]    [l'2]     l'3       <-- output

    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]

    # Compute the base term: r_t + γ * (1 - λ) * V(s_{t+1})
    inputs = rewards + continues * next_values * (1 - gae_lambda)

    # Compute lambda returns backward in time
    outputs = torch.zeros_like(values)
    outputs[:, -1] = next_values[:, -1]  # initialize with the last value
    for t in range(horizon - 2, -1, -1):  # t = T-2, ..., 0
        # R_t^λ = [r_t + γ * (1 - λ) * V(s_{t+1})] + γ * λ * R_{t+1}^λ
        outputs[:, t] = inputs[:, t] + continues[:, t] * gae_lambda * outputs[:, t + 1]

    return outputs[:, :-1]


class LayerNormChannelLast(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class MSEDistribution:
    """
    Copied from https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/utils/distribution.py#L196
    """

    def __init__(self, mode: Tensor, dims: int, agg: str = "sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims :]

    @property
    def mode(self) -> Tensor:
        return self._mode

    @property
    def mean(self) -> Tensor:
        return self._mode

    def log_prob(self, value: Tensor) -> Tensor:
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class LayerNormGRUCell(nn.Module):
    """A GRU cell with a LayerNorm
    copied from https://github.com/Eclectic-Sheep/sheeprl/blob/4441dbf4bcd7ae0daee47d35fb0660bc1fe8bd4b/sheeprl/models/models.py#L331
    which was taken from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py#L317.

    This particular GRU cell accepts 3-D inputs, with a sequence of length 1, and applies
    a LayerNorm after the projection of the inputs.

    Args:
        input_size (int): the input size.
        hidden_size (int): the hidden state size
        bias (bool, optional): whether to apply a bias to the input projection.
            Defaults to True.
        batch_first (bool, optional): whether the first dimension represent the batch dimension or not.
            Defaults to False.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to nn.Identiy.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {}.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = False,
        layer_norm_cls: Callable[..., nn.Module] = nn.Identity,
        layer_norm_kw: dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=self.bias)
        # Avoid multiple values for the `normalized_shape` argument
        layer_norm_kw.pop("normalized_shape", None)
        self.layer_norm = layer_norm_cls(3 * hidden_size, **layer_norm_kw)

    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        is_3d = input.dim() == 3
        if is_3d:
            if input.shape[int(self.batch_first)] == 1:
                input = input.squeeze(int(self.batch_first))
            else:
                raise AssertionError(
                    "LayerNormGRUCell: Expected input to be 3-D with sequence length equal to 1 but received "
                    f"a sequence of length {input.shape[int(self.batch_first)]}"
                )
        if hx.dim() == 3:
            hx = hx.squeeze(0)
        assert input.dim() in (
            1,
            2,
        ), f"LayerNormGRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        hx = hx.unsqueeze(0) if not is_batched else hx

        input = torch.cat((hx, input), -1)
        x = self.linear(input)
        x = self.layer_norm(x)
        reset, cand, update = torch.chunk(x, 3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        hx = update * cand + (1 - update) * hx

        if not is_batched:
            hx = hx.squeeze(0)
        elif is_3d:
            hx = hx.unsqueeze(0)

        return hx


# From https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/jaxutils.py
@torch.jit.script
def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class TwoHotEncodingDistribution:
    """
    Copied from https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/utils/distribution.py#L224
    """

    def __init__(
        self,
        logits: Tensor,
        dims: int = 0,
        low: int = -20,
        high: int = 20,
        transfwd: Callable[[Tensor], Tensor] = symlog,
        transbwd: Callable[[Tensor], Tensor] = symexp,
    ):
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self._batch_shape = logits.shape[: len(logits.shape) - dims]
        self._event_shape = logits.shape[len(logits.shape) - dims : -1] + (1,)

    @property
    def mean(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    @property
    def mode(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    def log_prob(self, x: Tensor) -> Tensor:
        x = self.transfwd(x)
        # below in [-1, len(self.bins) - 1]
        below = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        # above in [0, len(self.bins)]
        above = below + 1

        # above in [0, len(self.bins) - 1]
        above = torch.minimum(above, torch.full_like(above, len(self.bins) - 1))
        # below in [0, len(self.bins) - 1]
        below = torch.maximum(below, torch.zeros_like(below))

        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, len(self.bins)) * weight_below[..., None]
            + F.one_hot(above, len(self.bins)) * weight_above[..., None]
        ).squeeze(-2)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (target * log_pred).sum(dim=self.dims)


class SafeBernoulli(Bernoulli):
    @property
    def mode(self) -> Tensor:
        mode = (self.probs >= 0.5).to(self.probs)
        return mode


class Moments(nn.Module):
    """
    Copied from https://github.com/Eclectic-Sheep/sheeprl/blob/419c7ce05b67b0fd89b62ae0b73b71b3f7a96514/sheeprl/algos/dreamer_v3/utils.py#L40
    """

    def __init__(
        self, decay: float = 0.99, max_: float = 1.0, percentile_low: float = 0.05, percentile_high: float = 0.95
    ) -> None:
        super().__init__()
        self._decay = decay
        self._max = torch.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: Tensor) -> Any:
        low = torch.quantile(x, self._percentile_low)
        high = torch.quantile(x, self._percentile_high)
        with torch.no_grad():  # ! stop tracing gradient, otherwise will cause memory leak
            self.low = self._decay * self.low + (1 - self._decay) * low
            self.high = self._decay * self.high + (1 - self._decay) * high
        invscale = torch.max(1 / self._max, self.high - self.low)
        return self.low.detach(), invscale.detach()


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L929
def init_weights(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L957
def uniform_init_weights(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


########################################################
## Configs
########################################################
@dataclass
class Dm3Cfg:
    ratio: float = 0.5
    batch_size: int = 16
    chunk_size: int = 64

    ## Training
    batch_length: int = 64
    horizon: int = 15
    bins: int = 255

    ## World Model
    model_lr: float = 1e-4
    model_eps: float = 1e-8
    model_clip: float = 1000.0
    free_nats: float = 1.0
    stochastic_length: int = 32
    stochastic_classes: int = 32
    deterministic_size: int = 512
    embedded_obs_size: int = 4096

    ## Actor Critic
    actor_grad: Literal["dynamics", "reinforce"] = "dynamics"
    actor_lr: float = 8e-5
    actor_eps: float = 1e-5
    actor_clip: float = 100.0
    actor_ent_coef: float = 0.0003
    critic_lr: float = 8e-5
    critic_eps: float = 1e-5
    critic_clip: float = 100.0
    critic_tau: float = 0.02
    ac_batch_size: int = -1  # if -1, ac_batch_size = batch_size * batch_length
    gae_lambda: float = 0.95
    gamma: float = 0.997

    @property
    def stochastic_size(self):
        return self.stochastic_length * self.stochastic_classes

    def __post_init__(self):
        if self.ac_batch_size == -1:
            self.ac_batch_size = self.batch_size * self.batch_length


########################################################
## Networks
########################################################
class Encoder(nn.Module):
    ## HACK: the output size is 4096, which should be equal to embedded_obs_size
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(32, eps=1e-3),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(64, eps=1e-3),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(128, eps=1e-3),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(256, eps=1e-3),
            nn.SiLU(),
        )
        self.encoder.apply(init_weights)

    def forward(self, obs: Tensor) -> Tensor:
        B = obs.shape[0]
        embedded_obs = self.encoder(obs)
        return embedded_obs.reshape(B, -1)  # flatten the last 3 dimensions C, H, W


class Decoder(nn.Module):
    def __init__(self, cfg: Dm3Cfg):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(cfg.deterministic_size + cfg.stochastic_size, 4096),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(128, eps=1e-3),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(64, eps=1e-3),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNormChannelLast(32, eps=1e-3),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )
        [m.apply(init_weights) for m in self.decoder[:-1]]
        self.decoder[-1].apply(uniform_init_weights(1.0))

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Tensor:
        x = torch.cat([posterior, deterministic], dim=-1)
        input_shape = x.shape
        x = x.flatten(0, 1)
        reconstructed_obs = self.decoder(x)
        reconstructed_obs = reconstructed_obs.unflatten(0, input_shape[:2])
        return reconstructed_obs


class RecurrentModel(nn.Module):
    def __init__(self, cfg: Dm3Cfg, envs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.stochastic_size + envs.single_action_space.shape[0], 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
        )
        self.recurrent = LayerNormGRUCell(
            512, cfg.deterministic_size, bias=False, layer_norm_cls=nn.LayerNorm, layer_norm_kw={"eps": 1e-3}
        )
        self.mlp.apply(init_weights)
        self.recurrent.apply(init_weights)

    def forward(self, state: Tensor, action: Tensor, deterministic: Tensor) -> Tensor:
        x = torch.cat([state, action], dim=1)
        x = self.mlp(x)
        x = self.recurrent(x, deterministic)
        return x


def _unimix(logits: Tensor, num_classes: int) -> Tensor:
    probs = logits.softmax(dim=-1)
    uniform = torch.ones_like(probs) / num_classes
    probs = 0.99 * probs + 0.01 * uniform
    logits = probs_to_logits(probs)
    return logits


class TransitionModel(nn.Module):
    def __init__(self, cfg: Dm3Cfg):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.deterministic_size, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, cfg.stochastic_size),
        )
        [m.apply(init_weights) for m in self.net[:-1]]
        self.net[-1].apply(uniform_init_weights(1.0))

    def forward(self, deterministic: Tensor) -> tuple[Distribution, Tensor]:
        logits = self.net(deterministic).view(-1, self.cfg.stochastic_length, self.cfg.stochastic_classes)
        logits = _unimix(logits, self.cfg.stochastic_classes)
        dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        return dist, logits.view(-1, self.cfg.stochastic_size)


class RepresentationModel(nn.Module):
    def __init__(self, cfg: Dm3Cfg):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.embedded_obs_size + cfg.deterministic_size, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, cfg.stochastic_size),
        )
        [m.apply(init_weights) for m in self.net[:-1]]
        self.net[-1].apply(uniform_init_weights(1.0))

    def forward(self, embedded_obs: Tensor, deterministic: Tensor) -> tuple[Distribution, Tensor]:
        x = torch.cat([embedded_obs, deterministic], dim=1)
        logits = self.net(x).view(-1, self.cfg.stochastic_length, self.cfg.stochastic_classes)
        logits = _unimix(logits, self.cfg.stochastic_classes)
        dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        return dist, logits.view(-1, self.cfg.stochastic_size)


class RewardPredictor(nn.Module):
    def __init__(self, cfg: Dm3Cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.deterministic_size + cfg.stochastic_size, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, cfg.bins),
        )
        [m.apply(init_weights) for m in self.net[:-1]]
        self.net[-1].apply(uniform_init_weights(0.0))

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Tensor:
        input_shape = posterior.shape
        posterior = posterior.flatten(0, 1)
        deterministic = deterministic.flatten(0, 1)
        x = torch.cat([posterior, deterministic], dim=1)
        predicted_reward_bins = self.net(x)
        predicted_reward_bins = predicted_reward_bins.unflatten(0, input_shape[:2])
        return predicted_reward_bins


class ContinueModel(nn.Module):
    def __init__(self, cfg: Dm3Cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.deterministic_size + cfg.stochastic_size, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, 1),
        )
        [m.apply(init_weights) for m in self.net[:-1]]
        self.net[-1].apply(uniform_init_weights(1.0))

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Tensor:
        input_shape = posterior.shape
        posterior = posterior.flatten(0, 1)
        deterministic = deterministic.flatten(0, 1)
        x = torch.cat([posterior, deterministic], dim=1)
        logits = self.net(x)
        return logits.unflatten(0, input_shape[:2])


class Actor(nn.Module):
    def __init__(self, cfg: Dm3Cfg, envs: BaseVecEnv):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(cfg.deterministic_size + cfg.stochastic_size, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, envs.single_action_space.shape[0] * 2),
        )
        [m.apply(init_weights) for m in self.actor[:-1]]
        self.actor[-1].apply(uniform_init_weights(1.0))

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Distribution:
        x = torch.cat([posterior, deterministic], dim=-1)
        mean, std = self.actor(x).chunk(2, dim=-1)
        std_min, std_max = 0.1, 1
        mean = F.tanh(mean)
        std = std_min + (std_max - std_min) * F.sigmoid(std + 2.0)
        action_dist = Independent(Normal(mean, std), 1)
        return action_dist


class Critic(nn.Module):
    def __init__(self, cfg: Dm3Cfg):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(cfg.stochastic_size + cfg.deterministic_size, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, cfg.bins),
        )
        [m.apply(init_weights) for m in self.critic[:-1]]
        self.critic[-1].apply(uniform_init_weights(0.0))

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Tensor:
        x = torch.cat([posterior, deterministic], dim=-1)
        predicted_value_bins = self.critic(x)
        return predicted_value_bins


########################################################
## Agent
########################################################
class Dm3Agent:
    def __init__(self, cfg: Dm3Cfg, envs: BaseVecEnv, device: torch.device, amp: bool = False):
        self.cfg = cfg
        self.device = device
        self.num_envs = envs.num_envs
        self.amp = amp
        self.use_kl_curiosity = False

        self.encoder = Encoder().to(device)
        self.decoder = Decoder(cfg).to(device)
        self.recurrent_model = RecurrentModel(cfg, envs).to(device)
        self.transition_model = TransitionModel(cfg).to(device)
        self.representation_model = RepresentationModel(cfg).to(device)
        self.reward_predictor = RewardPredictor(cfg).to(device)
        self.continue_model = ContinueModel(cfg).to(device)
        self.actor = Actor(cfg, envs).to(device)
        self.critic = Critic(cfg).to(device)
        self.target_critic = Critic(cfg).to(device)
        self.critic_params = from_module(self.critic).data
        self.target_critic_params = from_module(self.target_critic).data
        self.moments = Moments().to(device)

        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.recurrent_model.parameters())
            + list(self.transition_model.parameters())
            + list(self.representation_model.parameters())
            + list(self.reward_predictor.parameters())
            + list(self.continue_model.parameters())
        )
        self.model_optimizer = torch.optim.Adam(self.model_params, lr=cfg.model_lr, eps=cfg.model_eps)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, eps=cfg.actor_eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, eps=cfg.critic_eps)
        self.model_scaler = torch.amp.GradScaler(enabled=amp)
        self.actor_scaler = torch.amp.GradScaler(enabled=amp)
        self.critic_scaler = torch.amp.GradScaler(enabled=amp)

    def get_action(self, data: TensorDict, explore: bool = False) -> Tensor:
        if explore:
            action = self.actor(data["posterior"], data["deterministic"]).sample()
        else:
            action = self.actor(data["posterior"], data["deterministic"]).mode
        return action

    def dynamic_learning(self, data: TensorDict) -> tuple[TensorDict, Tensor, Tensor]:
        # TODO: utilize "next_obs" to update the model
        # TODO: since the replay buffer may contain termination/truncation in the middle of a rollout, we need to handle this case by resetting posterior, deterministic, and action to initial state (zero)

        # Given the following diagram, with batch_length=4
        # Actions:           [a'0]    [a'1]    [a'2]    a'3  <-- input
        #                       \        \        \
        #                        \        \        \
        #                         \        \        \
        # States:          0  ->  z'1  ->  z'2  ->  z'3      <-- output
        # Observations:   o'0    [o'1]    [o'2]    [o'3]     <-- input
        # Rewards:                r'1      r'2      r'3      <-- output
        # Continues:              c'1      c'2      c'3      <-- output

        cfg = self.cfg

        with torch.autocast(self.device.type, enabled=self.amp):
            posterior = torch.zeros(cfg.batch_size, cfg.stochastic_size, device=self.device)
            deterministic = torch.zeros(cfg.batch_size, cfg.deterministic_size, device=self.device)
            embeded_obs = self.encoder(data["obs"].flatten(0, 1)).unflatten(0, (cfg.batch_size, cfg.batch_length))

            deterministics = []
            priors_logits = []
            posteriors = []
            posteriors_logits = []
            for t in range(1, cfg.batch_length):
                deterministic = self.recurrent_model(posterior, data["action"][:, t - 1], deterministic)
                prior_dist, prior_logits = self.transition_model(deterministic)
                posterior_dist, posterior_logits = self.representation_model(embeded_obs[:, t], deterministic)
                posterior = posterior_dist.rsample().view(-1, cfg.stochastic_size)

                deterministics.append(deterministic)
                priors_logits.append(prior_logits)
                posteriors.append(posterior)
                posteriors_logits.append(posterior_logits)

            deterministics = torch.stack(deterministics, dim=1).to(self.device)
            prior_logits = torch.stack(priors_logits, dim=1).to(self.device)
            posteriors = torch.stack(posteriors, dim=1).to(self.device)
            posteriors_logits = torch.stack(posteriors_logits, dim=1).to(self.device)

            if self.use_kl_curiosity:
                # calculate real kl curio reward with the wm itself
                # TODO: check if this reward is really correct
                kl_reward1 = kl_divergence(
                    Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 0),
                    Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 0),
                )
                kl_reward2 = kl_divergence(
                    Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 0),
                    Independent(OneHotCategoricalStraightThrough(logits=prior_logits.detach()), 0),
                )
                kl_reward = 0.5 * kl_reward1 + 0.1 * kl_reward2
                kl_reward = kl_reward.detach()
                kl_reward = kl_reward.unsqueeze(-1)
                # the reward is only used for the next step,
                # so we only need to update the reward for the next step
                data["reward"][:, 1:] = kl_reward.to(self.device)

            reconstructed_obs = self.decoder(posteriors, deterministics)
            reconstructed_obs_dist = MSEDistribution(
                reconstructed_obs, 3
            )  # 3 is number of dimensions for observation space, shape is (3, H, W)
            reconstructed_obs_loss = -reconstructed_obs_dist.log_prob(data["obs"][:, 1:]).mean()

            predicted_reward_bins = self.reward_predictor(posteriors, deterministics)
            predicted_reward_dist = TwoHotEncodingDistribution(predicted_reward_bins, dims=1)
            reward_loss = -predicted_reward_dist.log_prob(data["reward"][:, 1:]).mean()

            predicted_continue = self.continue_model(posteriors, deterministics)
            predicted_continue_dist = SafeBernoulli(logits=predicted_continue)
            true_continue = 1 - data["terminated"][:, 1:]
            continue_loss = -predicted_continue_dist.log_prob(true_continue).mean()

            # KL balancing, Eq. 3 in the paper
            kl = kl_loss1 = kl_divergence(
                Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
                Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1),
            )
            kl_loss1 = torch.max(kl_loss1, torch.tensor(cfg.free_nats, device=self.device))
            kl_loss2 = kl_divergence(
                Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
                Independent(OneHotCategoricalStraightThrough(logits=prior_logits.detach()), 1),
            )
            kl_loss2 = torch.max(kl_loss2, torch.tensor(cfg.free_nats, device=self.device))
            kl_loss = (0.5 * kl_loss1 + 0.1 * kl_loss2).mean()

            model_loss = reconstructed_obs_loss + reward_loss + continue_loss + kl_loss

        self.model_optimizer.zero_grad()
        self.model_scaler.scale(model_loss).backward()
        self.model_scaler.unscale_(self.model_optimizer)
        model_grad_norm = nn.utils.clip_grad_norm_(self.model_params, cfg.model_clip)
        self.model_scaler.step(self.model_optimizer)
        self.model_scaler.update()

        metrics = TensorDict.from_dict(
            {
                "loss/reconstruction_loss": reconstructed_obs_loss.detach(),
                "loss/reward_loss": reward_loss.detach(),
                "loss/continue_loss": continue_loss.detach(),
                "loss/kl_loss": kl_loss.detach(),
                "loss/model_loss": model_loss.detach(),
                "state/kl": kl.detach().mean(),
                "state/prior_entropy": prior_dist.entropy().detach().mean(),
                "state/posterior_entropy": posterior_dist.entropy().detach().mean(),
                "grad_norm/model": model_grad_norm.detach().mean(),
            },
            batch_size=torch.Size([]),
        )

        return metrics, posteriors.detach(), deterministics.detach()

    def behavior_learning(self, posteriors_: Tensor, deterministics_: Tensor):
        cfg = self.cfg

        ## reuse the `posteriors` and `deterministics` from model learning, important to detach them!
        state = posteriors_.detach().view(-1, cfg.stochastic_size)
        deterministic = deterministics_.detach().view(-1, cfg.deterministic_size)

        idx = torch.randperm(state.shape[0])[: cfg.ac_batch_size]
        state = state[idx]
        deterministic = deterministic[idx]

        # Given the following diagram, with horizon=4
        # Actions:            a'0      a'1      a'2       a'3
        #                    ^  \     ^  \     ^  \      ^  \
        #                   /    \   /    \   /    \    /    \
        #                  /      \ /      \ /      \  /      \
        # States:        z'0  ->  z'1  ->  z'2  ->  z'3  ->  z'4    <-- input is z'0, output is z'1~z'4
        # Rewards:                r'1      r'2      r'3      r'4    <-- output
        # Continues:              c'1      c'2      c'3      c'4    <-- output
        # Values:                 v'1      v'2      v'3      v'4    <-- output
        # Lambda-values:          l'1      l'2      l'3             <-- output

        with torch.autocast(self.device.type, enabled=self.amp):
            actions = []
            states = []
            deterministics = []
            for t in range(cfg.horizon):
                action = self.actor(state.detach(), deterministic.detach()).rsample()  # detach help speed up about 10%
                deterministic = self.recurrent_model(state, action, deterministic)
                state_dist, state_logits = self.transition_model(deterministic)
                state = state_dist.rsample().view(-1, cfg.stochastic_size)
                actions.append(action)
                states.append(state)
                deterministics.append(deterministic)

            actions = torch.stack(actions, dim=1)
            states = torch.stack(states, dim=1)
            deterministics = torch.stack(deterministics, dim=1)

            predicted_rewards = TwoHotEncodingDistribution(self.reward_predictor(states, deterministics), dims=1).mean
            predicted_values = TwoHotEncodingDistribution(self.critic(states, deterministics), dims=1).mean

            continues_logits = self.continue_model(states, deterministics)
            continues = SafeBernoulli(logits=continues_logits).mode
            lambda_values = compute_lambda_values(
                predicted_rewards, predicted_values, continues * cfg.gamma, cfg.horizon, cfg.gae_lambda
            )

            ## Normalize return, Eq. 7 in the paper
            baselines = predicted_values[:, :-1]
            offset, invscale = self.moments(lambda_values)
            normalized_lambda_values = (lambda_values - offset) / invscale
            normalized_baselines = (baselines - offset) / invscale

            advantages = normalized_lambda_values - normalized_baselines

            # TODO: what would happen if we don't use discount factor?
            with torch.no_grad():
                discount = torch.cumprod(continues[:, :-1] * cfg.gamma, dim=1) / cfg.gamma

            actor_dist = self.actor(states[:, :-1], deterministics[:, :-1])
            actor_entropy = actor_dist.entropy().unsqueeze(-1)
            if cfg.actor_grad == "dynamics":
                # Below directly computes the gradient through dynamics.
                actor_target = advantages
            elif cfg.actor_grad == "reinforce":
                actor_target = advantages.detach() * actor_dist.log_prob(actions[:, :-1]).unsqueeze(-1)
            # For discount factor, see https://ai.stackexchange.com/q/7680
            actor_loss = -((actor_target + cfg.actor_ent_coef * actor_entropy) * discount).mean()
        self.actor_optimizer.zero_grad()
        self.actor_scaler.scale(actor_loss).backward()
        self.actor_scaler.unscale_(self.actor_optimizer)
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.actor_clip)
        self.actor_scaler.step(self.actor_optimizer)
        self.actor_scaler.update()

        # TODO: implement target critic
        with torch.autocast(self.device.type, enabled=self.amp):
            predicted_value_bins = self.critic(states[:, :-1].detach(), deterministics[:, :-1].detach())
            predicted_value_dist = TwoHotEncodingDistribution(predicted_value_bins, dims=1)
            target_values = TwoHotEncodingDistribution(
                self.target_critic(states[:, :-1].detach(), deterministics[:, :-1].detach()), dims=1
            ).mean
            value_loss = -predicted_value_dist.log_prob(lambda_values.detach())
            value_loss -= predicted_value_dist.log_prob(target_values.detach())
            value_loss = (value_loss * discount.squeeze(-1)).mean()
        self.critic_optimizer.zero_grad()
        self.critic_scaler.scale(value_loss).backward()
        self.critic_scaler.unscale_(self.critic_optimizer)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_clip)
        self.critic_scaler.step(self.critic_optimizer)
        self.critic_scaler.update()

        self.target_critic_params.lerp_(self.critic_params, cfg.critic_tau)

        metrics = TensorDict.from_dict(
            {
                "loss/actor_loss": actor_loss.detach(),
                "loss/value_loss": value_loss.detach(),
                "state/actor_entropy": actor_entropy.detach().mean(),
                "state/value_mean": predicted_values.detach().mean(),
                "state/value_std": predicted_values.detach().std(),
                "grad_norm/actor": actor_grad_norm.detach().mean(),
                "grad_norm/critic": critic_grad_norm.detach().mean(),
            },
            batch_size=torch.Size([]),
        )
        return metrics

    def update(self, data: TensorDict) -> TensorDict:
        metrics, posteriors, deterministics = self.dynamic_learning(data)
        metrics.update(self.behavior_learning(posteriors, deterministics))
        return metrics

    @torch.inference_mode()
    def evaluate(self, episodes: int, eval_envs: list[BaseVecEnv]) -> TensorDict:
        cfg = self.cfg
        num_envs = 1
        episodic_returns = []
        videos = []
        for i in range(episodes):
            envs = eval_envs[i]
            obs, _ = envs.reset()
            posterior = torch.zeros(num_envs, cfg.stochastic_size, device=self.device)
            deterministic = torch.zeros(num_envs, cfg.deterministic_size, device=self.device)
            action = torch.zeros(num_envs, envs.single_action_space.shape[0], device=self.device)
            episodic_return = torch.zeros(num_envs, device=self.device)
            imgs = [obs.cpu()]
            while True:
                embeded_obs = self.encoder(obs)
                deterministic = self.recurrent_model(posterior, action, deterministic)
                posterior_dist, _ = self.representation_model(embeded_obs.view(num_envs, -1), deterministic)
                posterior = posterior_dist.mode.view(-1, cfg.stochastic_size)
                action = self.actor(posterior, deterministic).mode
                obs, reward, terminated, truncated, info = envs.step(action)
                done = torch.logical_or(terminated, truncated)
                episodic_return += reward
                if done.any():
                    break
                imgs.append(obs.cpu())
            video = torch.cat(imgs, dim=0)  # (T, C, H, W)
            videos.append(video)
            episodic_returns.append(episodic_return.item())
            envs.close()

        videos = torch.stack(videos)  # (N, T, C, H, W)
        videos = videos + 0.5
        videos = (videos.clamp(0, 1) * 255).to(torch.uint8)
        videos = torch.stack([
            make_grid(videos[:, t], nrow=4, padding=0) for t in range(videos.shape[1])
        ])  # (T, C, H, W)

        return TensorDict.from_dict(
            {
                "reward/eval_episodic_return": torch.tensor(episodic_returns).mean(),
                "eval/video": videos,
            },
            batch_size=torch.Size([]),
        )


@dataclass
class Args:
    # Agent, Environment and Device
    agent: Dm3Cfg
    obs_mode: str = "rgb"
    env_id: str = "dmc/walker-walk-v0"
    device: str = "cuda"

    # Experiment, Logging and Checkpoint
    logger: list[str] = field(default_factory=lambda: ["wandb"])
    exp_name: str = "dm3"
    log_every: int = 500
    eval_every: int = 2000
    eval_episodes: int = 8
    ckpt_path: str = ""
    wandb_entity: str = ""
    wandb_project: str = "fishrl"
    tqdm: Literal["enabled", "disabled", "rich"] = "enabled"

    # Training
    seed: int = 0
    num_envs: int = 4
    action_repeat: int = 2
    deterministic: bool = False

    buffer_size: int = 100_000
    prefill: int = 1000  # same as urlb
    total_steps: int = 500_000

    # Acceleration
    compile: bool = False
    cudagraph: bool = False
    amp: bool = False


def main():
    args = tyro.cli(Args)

    device = torch.device(args.device)
    seed_everything(args.seed)
    if args.deterministic:
        enable_deterministic_run()

    ## env and replay buffer
    envs = create_vector_env(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        num_envs=args.num_envs,
        seed=args.seed,
        action_repeat=args.action_repeat,
    )
    envs = UnwrapDictWrapper(envs)
    envs = ObsShiftWrapper(envs)

    eval_envs = []
    for i in range(args.eval_episodes):
        seed = args.seed + 6666 + i  # ensure different seeds for different episodes
        eval_env = create_vector_env(
            env_id=args.env_id, obs_mode=args.obs_mode, num_envs=1, seed=seed, action_repeat=args.action_repeat
        )
        eval_env = UnwrapDictWrapper(eval_env)
        eval_env = ObsShiftWrapper(eval_env)
        eval_envs.append(eval_env)

    buffer = ReplayBuffer(
        observation_shape=envs.single_observation_space.shape,
        action_size=envs.single_action_space.shape[0],
        device=device,
        num_envs=args.num_envs,
        capacity=args.buffer_size,
    )

    ## logger
    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.env_id}__{args.exp_name}__env={args.num_envs}__seed={args.seed}__{_timestamp}"
    logdir = f"logdir/{run_name}"
    os.makedirs(logdir, exist_ok=True)
    logger = make_logger(args.logger, logdir, asdict(args), wandb_entity=args.wandb_entity)
    aggregator = MetricAggregator(device=device)

    agent = Dm3Agent(args.agent, envs, device, args.amp)
    if args.ckpt_path:
        agent.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    if args.compile:
        agent.update = torch.compile(agent.update, backend="cudagraphs")
    if args.cudagraph:
        agent.update = CudaGraphModule(agent.update)

    episodic_return = torch.zeros(args.num_envs, device=device)
    episodic_length = torch.zeros(args.num_envs, device=device)

    posterior = torch.zeros(args.num_envs, args.agent.stochastic_size, device=device)
    deterministic = torch.zeros(args.num_envs, args.agent.deterministic_size, device=device)
    action = torch.zeros(args.num_envs, envs.single_action_space.shape[0], device=device)

    obs, _ = envs.reset()
    ratio = Ratio(ratio=args.agent.ratio)
    pbar = tqdm(total=args.total_steps, desc="Training", disable=args.tqdm == "disabled")
    for global_step in range(0, args.total_steps, args.num_envs):
        ## Get action
        with torch.inference_mode(), timer("time/get_action"):
            if global_step < args.prefill:
                action = torch.as_tensor(envs.action_space.sample(), device=device)
            else:
                embeded_obs = agent.encoder(obs)
                deterministic = agent.recurrent_model(posterior, action, deterministic)
                posterior_dist, _ = agent.representation_model(embeded_obs.view(args.num_envs, -1), deterministic)
                posterior = posterior_dist.sample().view(-1, args.agent.stochastic_size)
                action = agent.get_action(TensorDict(posterior=posterior, deterministic=deterministic), explore=True)

        ## Step the environment
        with torch.inference_mode(), timer("time/step"):
            next_obs, reward, terminated, truncated, info = envs.step(action)

        ## Add to buffer
        with torch.inference_mode(), timer("time/add_to_buffer"):
            done = torch.logical_or(terminated, truncated)
            real_next_obs = next_obs.clone()

            if truncated.any():
                try:
                    real_next_obs[truncated.bool()] = torch.as_tensor(
                        np.stack(info["final_observation"][truncated.bool().numpy(force=True)]),
                        device=device,
                        dtype=torch.float32,
                    )
                except Exception as e:
                    warnings.warn(
                        f'Error when accessing info["final_observation"], using next_obs instead. Note that this may cause a bias. Error: {e}'
                    )
                    real_next_obs = next_obs.clone()

            buffer.add(obs, action, reward, real_next_obs, done, terminated)
            obs = next_obs

            episodic_return += reward
            episodic_length += 1
            if done.any():
                logger.add({
                    "reward/episodic_return": episodic_return[done].mean().item(),
                    "reward/episodic_length": episodic_length[done].mean().item(),
                })
                print(f"global_step={global_step}, episodic_return={episodic_return[done].mean().item():.1f}")

                # setting the state of the done envs to some sampled states
                episodic_return[done] = 0
                episodic_length[done] = 0
                posterior[done] = 0
                deterministic[done] = 0
                action[done] = 0

        ## Update the model
        if global_step > args.prefill:
            with timer("time/train"):
                gradient_steps = ratio(global_step - args.prefill)
                for _ in range(gradient_steps):
                    with timer("time/data_sample"):
                        data = buffer.sample(args.agent.batch_size, args.agent.chunk_size)
                    with timer("time/update_agent"):
                        metrics = agent.update(data)
                    with torch.no_grad(), timer("time/aggregate_metrics"):
                        for k, v in metrics.items():
                            aggregator.update(k, v)

        # ## Evaluation
        # if (
        #     args.eval_every > 0
        #     and global_step > args.prefill
        #     and (global_step - args.prefill) % args.eval_every < args.num_envs
        # ):
        #     with timer("time/eval"), torch.inference_mode():
        #         metrics = agent.evaluate(args.eval_episodes, eval_envs)
        #     with timer("time/logger_add"):
        #         # Convert TensorDict to regular dict if needed
        #         if hasattr(metrics, "to_dict"):
        #             metrics_dict = metrics.to_dict()
        #         else:
        #             metrics_dict = metrics
        #         logger.add(metrics_dict)

        # ## Logging
        if (
            args.log_every > 0
            and global_step > args.prefill
            and (global_step - args.prefill) % args.log_every < args.num_envs
        ):
            with timer("time/logger_add"):
                logger.add(aggregator.compute())
                aggregator.reset()
            if not timer.disabled:
                logger.add(timer.compute())
                timer.reset()

        with timer("time/logger_write"):
            logger.write()  # NOTE: This should be called only once per step
        pbar.update(args.num_envs)
        logger.step += args.num_envs

    logger.close()


if __name__ == "__main__":
    main()
