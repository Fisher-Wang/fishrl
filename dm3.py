from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from typing import Any, Callable, Literal, Sequence

os.environ["MUJOCO_GL"] = "egl"  # significantly faster rendering compared to glfw and osmesa

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from loguru import logger as log
from rich.logging import RichHandler
from torch import Tensor
from torch.distributions import (
    Bernoulli,
    Distribution,
    Independent,
    Normal,
    OneHotCategoricalStraightThrough,
    kl_divergence,
)
from torch.distributions.utils import probs_to_logits
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import MeanMetric
from tqdm import tqdm

from fishrl.envs.env_factory import create_vector_env
from fishrl.utils.metrics import MetricAggregator
from fishrl.utils.reproducibility import enable_deterministic_run, seed_everything
from fishrl.utils.timer import timer
from fishrl.utils.utils import Ratio

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


########################################################
## Standalone utils
########################################################
class ObsShiftWrapper(gym.Wrapper):
    # change observation space from [0, 1] to [-0.5, 0.5]
    # TODO: also change observation space
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs - 0.5, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs - 0.5, reward, terminated, truncated, info


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        device: str | torch.device,
        num_envs: int = 1,
        capacity: int = 5000000,
    ):
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity

        state_type = np.uint8 if len(observation_shape) < 3 else np.float32

        self.observation = np.empty((self.capacity, self.num_envs, *observation_shape), dtype=state_type)
        self.next_observation = np.empty((self.capacity, self.num_envs, *observation_shape), dtype=state_type)
        self.action = np.empty((self.capacity, self.num_envs, action_size), dtype=np.float32)
        self.reward = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.terminated = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

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
        self.action[self.buffer_index] = action.detach().cpu().numpy()
        self.reward[self.buffer_index] = reward.unsqueeze(-1).detach().cpu().numpy()
        self.next_observation[self.buffer_index] = next_observation.detach().cpu().numpy()
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

        sample = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
            "terminated": terminated,
        }
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


########################################################
## Args
########################################################
@dataclass
class Args:
    exp_name: str = "dreamerv3"
    seed: int = 0
    device: str = "cuda"
    debug: bool = False
    log_every: int = 500
    eval_every: int = 2000
    checkpoint_every: int = 10_000
    eval_episodes: int = 8
    amp: bool = False
    deterministic: bool = False
    compile: bool = False

    ## Environment
    env_id: str = "dmc/walker-walk-v0"
    num_envs: int = 4

    ## Training
    batch_size: int = 16
    batch_length: int = 64
    horizon: int = 15
    total_steps: int = 500000
    prefill: int = 1000

    ## All models
    bins: int = 255

    ## World Model
    model_lr: float = 1e-4
    model_eps: float = 1e-8
    model_clip: float = 1000.0
    free_nats: float = 1.0
    stochastic_length: int = 32
    stochastic_classes: int = 32
    deterministic_size: int = 512
    embedded_obs_size: int = 4096  # = 256 * 4 * 4

    ## Actor Critic
    actor_grad: Literal["dynamics", "reinforce"] = "dynamics"
    actor_lr: float = 8e-5
    actor_eps: float = 1e-5
    actor_clip: float = 100.0
    actor_ent_coef: float = 0.0003
    critic_lr: float = 8e-5
    critic_eps: float = 1e-5
    critic_clip: float = 100.0
    gae_lambda: float = 0.95
    gamma: float = 0.997

    @property
    def stochastic_size(self):
        return self.stochastic_length * self.stochastic_classes

    def __post_init__(self):
        if self.debug:
            self.batch_size = 2
            self.batch_length = 3
            self.prefill = self.num_envs * self.batch_size * self.batch_length
            self.train_per_rollout = 1


args = tyro.cli(Args)


########################################################
## Networks
########################################################
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


class Encoder(nn.Module):
    ## HACK: the output size is 4096, which should be equal to args.embedded_obs_size
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
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(args.deterministic_size + args.stochastic_size, 4096),
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
    def __init__(self, envs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(args.stochastic_size + envs.single_action_space.shape[0], 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
        )
        self.recurrent = LayerNormGRUCell(
            512, args.deterministic_size, bias=False, layer_norm_cls=nn.LayerNorm, layer_norm_kw={"eps": 1e-3}
        )
        self.mlp.apply(init_weights)
        self.recurrent.apply(init_weights)

    def forward(self, state: Tensor, action: Tensor, deterministic: Tensor) -> Tensor:
        x = torch.cat([state, action], dim=1)
        x = self.mlp(x)
        x = self.recurrent(x, deterministic)
        return x


def _unimix(logits: Tensor) -> Tensor:
    probs = logits.softmax(dim=-1)
    uniform = torch.ones_like(probs) / args.stochastic_classes
    probs = 0.99 * probs + 0.01 * uniform
    logits = probs_to_logits(probs)
    return logits


class TransitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.deterministic_size, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, args.stochastic_size),
        )
        [m.apply(init_weights) for m in self.net[:-1]]
        self.net[-1].apply(uniform_init_weights(1.0))

    def forward(self, deterministic: Tensor) -> tuple[Distribution, Tensor]:
        logits = self.net(deterministic).view(-1, args.stochastic_length, args.stochastic_classes)
        logits = _unimix(logits)
        dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        return dist, logits.view(-1, args.stochastic_size)


class RepresentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.embedded_obs_size + args.deterministic_size, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, args.stochastic_size),
        )
        [m.apply(init_weights) for m in self.net[:-1]]
        self.net[-1].apply(uniform_init_weights(1.0))

    def forward(self, embedded_obs: Tensor, deterministic: Tensor) -> tuple[Distribution, Tensor]:
        x = torch.cat([embedded_obs, deterministic], dim=1)
        logits = self.net(x).view(-1, args.stochastic_length, args.stochastic_classes)
        logits = _unimix(logits)
        dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        return dist, logits.view(-1, args.stochastic_size)


class RewardPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.deterministic_size + args.stochastic_size, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, args.bins),
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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.deterministic_size + args.stochastic_size, 512, bias=False),
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
    def __init__(self, envs):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(args.deterministic_size + args.stochastic_size, 512, bias=False),
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
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(args.stochastic_size + args.deterministic_size, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, 512, bias=False),
            nn.LayerNorm(512, eps=1e-3),
            nn.SiLU(),
            nn.Linear(512, args.bins),
        )
        [m.apply(init_weights) for m in self.critic[:-1]]
        self.critic[-1].apply(uniform_init_weights(0.0))

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Tensor:
        x = torch.cat([posterior, deterministic], dim=-1)
        predicted_value_bins = self.critic(x)
        return predicted_value_bins


########################################################
## Main
########################################################

## setup
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}" + (f" (GPU {torch.cuda.current_device()})" if torch.cuda.is_available() else ""))
seed_everything(args.seed)
if args.deterministic:
    enable_deterministic_run()

## logger
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{args.env_id}__{args.exp_name}__env={args.num_envs}__seed={args.seed}__{_timestamp}"
logdir = f"logdir/{run_name}"
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

## env and replay buffer
envs = create_vector_env(args.env_id, "rgb", args.num_envs, args.seed, action_repeat=2, image_size=(64, 64))
envs = ObsShiftWrapper(envs)
buffer = ReplayBuffer(
    envs.single_observation_space.shape,
    envs.single_action_space.shape[0],
    device,
    num_envs=args.num_envs,
    capacity=500_000,
)

## networks
encoder = Encoder().to(device)
decoder = Decoder().to(device)
recurrent_model = RecurrentModel(envs).to(device)
transition_model = TransitionModel().to(device)
representation_model = RepresentationModel().to(device)
reward_predictor = RewardPredictor().to(device)
continue_model = ContinueModel().to(device)
actor = Actor(envs).to(device)
critic = Critic().to(device)
moments = Moments().to(device)
if args.compile:
    encoder = torch.compile(encoder)
    decoder = torch.compile(decoder)
    recurrent_model = torch.compile(recurrent_model)
    transition_model = torch.compile(transition_model)
    representation_model = torch.compile(representation_model)
    reward_predictor = torch.compile(reward_predictor)
    continue_model = torch.compile(continue_model)
    actor = torch.compile(actor)
    critic = torch.compile(critic)
    moments = torch.compile(moments)

model_params = chain(
    encoder.parameters(),
    decoder.parameters(),
    recurrent_model.parameters(),
    transition_model.parameters(),
    representation_model.parameters(),
    reward_predictor.parameters(),
    continue_model.parameters(),
)
model_optimizer = torch.optim.Adam(model_params, lr=args.model_lr, eps=args.model_eps)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=args.actor_eps)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, eps=args.critic_eps)
model_scaler = torch.amp.GradScaler(enabled=args.amp)
actor_scaler = torch.amp.GradScaler(enabled=args.amp)
critic_scaler = torch.amp.GradScaler(enabled=args.amp)


## logging
global_step = 0
ratio = Ratio(ratio=0.5)
aggregator = MetricAggregator({
    "loss/reconstruction_loss": MeanMetric(sync_on_compute=False),
    "loss/reward_loss": MeanMetric(sync_on_compute=False),
    "loss/continue_loss": MeanMetric(sync_on_compute=False),
    "loss/kl_loss": MeanMetric(sync_on_compute=False),
    "loss/model_loss": MeanMetric(sync_on_compute=False),
    "loss/actor_loss": MeanMetric(sync_on_compute=False),
    "loss/value_loss": MeanMetric(sync_on_compute=False),
    "state/kl": MeanMetric(sync_on_compute=False),
    "state/prior_entropy": MeanMetric(sync_on_compute=False),
    "state/posterior_entropy": MeanMetric(sync_on_compute=False),
    "state/actor_entropy": MeanMetric(sync_on_compute=False),
    "grad_norm/model": MeanMetric(sync_on_compute=False),
    "grad_norm/actor": MeanMetric(sync_on_compute=False),
    "grad_norm/critic": MeanMetric(sync_on_compute=False),
})


def dynamic_learning(data: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    # TODO: utilize "next_observation" to update the model
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

    with torch.autocast(args.device, enabled=args.amp):
        posterior = torch.zeros(args.batch_size, args.stochastic_size, device=device)
        deterministic = torch.zeros(args.batch_size, args.deterministic_size, device=device)
        embeded_obs = encoder(data["observation"].flatten(0, 1)).unflatten(0, (args.batch_size, args.batch_length))

        deterministics = []
        priors_logits = []
        posteriors = []
        posteriors_logits = []
        for t in range(1, args.batch_length):
            deterministic = recurrent_model(posterior, data["action"][:, t - 1], deterministic)
            prior_dist, prior_logits = transition_model(deterministic)
            posterior_dist, posterior_logits = representation_model(embeded_obs[:, t], deterministic)
            posterior = posterior_dist.rsample().view(-1, args.stochastic_size)

            deterministics.append(deterministic)
            priors_logits.append(prior_logits)
            posteriors.append(posterior)
            posteriors_logits.append(posterior_logits)

        deterministics = torch.stack(deterministics, dim=1).to(device)
        prior_logits = torch.stack(priors_logits, dim=1).to(device)
        posteriors = torch.stack(posteriors, dim=1).to(device)
        posteriors_logits = torch.stack(posteriors_logits, dim=1).to(device)

        reconstructed_obs = decoder(posteriors, deterministics)
        reconstructed_obs_dist = MSEDistribution(
            reconstructed_obs, 3
        )  # 3 is number of dimensions for observation space, shape is (3, H, W)
        reconstructed_obs_loss = -reconstructed_obs_dist.log_prob(data["observation"][:, 1:]).mean()

        predicted_reward_bins = reward_predictor(posteriors, deterministics)
        predicted_reward_dist = TwoHotEncodingDistribution(predicted_reward_bins, dims=1)
        reward_loss = -predicted_reward_dist.log_prob(data["reward"][:, 1:]).mean()

        predicted_continue = continue_model(posteriors, deterministics)
        predicted_continue_dist = SafeBernoulli(logits=predicted_continue)
        true_continue = 1 - data["terminated"][:, 1:]
        continue_loss = -predicted_continue_dist.log_prob(true_continue).mean()

        # KL balancing, Eq. 3 in the paper
        kl = kl_loss1 = kl_divergence(
            Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
            Independent(OneHotCategoricalStraightThrough(logits=prior_logits), 1),
        )
        kl_loss1 = torch.max(kl_loss1, torch.tensor(args.free_nats, device=device))
        kl_loss2 = kl_divergence(
            Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
            Independent(OneHotCategoricalStraightThrough(logits=prior_logits.detach()), 1),
        )
        kl_loss2 = torch.max(kl_loss2, torch.tensor(args.free_nats, device=device))
        kl_loss = (0.5 * kl_loss1 + 0.1 * kl_loss2).mean()

        model_loss = reconstructed_obs_loss + reward_loss + continue_loss + kl_loss

    model_optimizer.zero_grad()
    model_scaler.scale(model_loss).backward()
    model_scaler.unscale_(model_optimizer)
    model_grad_norm = nn.utils.clip_grad_norm_(model_params, args.model_clip)
    model_scaler.step(model_optimizer)
    model_scaler.update()

    with torch.no_grad():
        aggregator.update("loss/reconstruction_loss", reconstructed_obs_loss.item())
        aggregator.update("loss/reward_loss", reward_loss.item())
        aggregator.update("loss/continue_loss", continue_loss.item())
        aggregator.update("loss/kl_loss", kl_loss.item())
        aggregator.update("loss/model_loss", model_loss.item())
        aggregator.update("state/kl", kl.mean().item())
        aggregator.update("state/prior_entropy", prior_dist.entropy().mean().item())
        aggregator.update("state/posterior_entropy", posterior_dist.entropy().mean().item())
        aggregator.update("grad_norm/model", model_grad_norm.mean().item())

    return posteriors, deterministics


def behavior_learning(posteriors_: Tensor, deterministics_: Tensor):
    ## reuse the `posteriors` and `deterministics` from model learning, important to detach them!
    state = posteriors_.detach().view(-1, args.stochastic_size)
    deterministic = deterministics_.detach().view(-1, args.deterministic_size)

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

    with torch.autocast(args.device, enabled=args.amp):
        actions = []
        states = []
        deterministics = []
        for t in range(args.horizon):
            action = actor(state.detach(), deterministic.detach()).rsample()  # detach help speed up about 10%
            deterministic = recurrent_model(state, action, deterministic)
            state_dist, state_logits = transition_model(deterministic)
            state = state_dist.rsample().view(-1, args.stochastic_size)
            actions.append(action)
            states.append(state)
            deterministics.append(deterministic)

        actions = torch.stack(actions, dim=1)
        states = torch.stack(states, dim=1)
        deterministics = torch.stack(deterministics, dim=1)

        predicted_rewards = TwoHotEncodingDistribution(reward_predictor(states, deterministics), dims=1).mean
        predicted_values = TwoHotEncodingDistribution(critic(states, deterministics), dims=1).mean

        continues_logits = continue_model(states, deterministics)
        continues = SafeBernoulli(logits=continues_logits).mode
        lambda_values = compute_lambda_values(
            predicted_rewards, predicted_values, continues * args.gamma, args.horizon, args.gae_lambda
        )

        ## Normalize return, Eq. 7 in the paper
        baselines = predicted_values[:, :-1]
        offset, invscale = moments(lambda_values)
        normalized_lambda_values = (lambda_values - offset) / invscale
        normalized_baselines = (baselines - offset) / invscale

        advantages = normalized_lambda_values - normalized_baselines

        # TODO: what would happen if we don't use discount factor?
        with torch.no_grad():
            discount = torch.cumprod(continues[:, :-1] * args.gamma, dim=1) / args.gamma

        actor_dist = actor(states[:, :-1], deterministics[:, :-1])
        actor_entropy = actor_dist.entropy().unsqueeze(-1)
        if args.actor_grad == "dynamics":
            # Below directly computes the gradient through dynamics.
            actor_target = advantages
        elif args.actor_grad == "reinforce":
            actor_target = advantages.detach() * actor_dist.log_prob(actions[:, :-1]).unsqueeze(-1)
        # For discount factor, see https://ai.stackexchange.com/q/7680
        actor_loss = -((actor_target + args.actor_ent_coef * actor_entropy) * discount).mean()
    actor_optimizer.zero_grad()
    actor_scaler.scale(actor_loss).backward()
    actor_scaler.unscale_(actor_optimizer)
    actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), args.actor_clip)
    actor_scaler.step(actor_optimizer)
    actor_scaler.update()

    # TODO: implement target critic
    with torch.autocast(args.device, enabled=args.amp):
        predicted_value_bins = critic(states[:, :-1].detach(), deterministics[:, :-1].detach())
        predicted_value_dist = TwoHotEncodingDistribution(predicted_value_bins, dims=1)
        value_loss = -predicted_value_dist.log_prob(lambda_values.detach())
        value_loss = (value_loss * discount.squeeze(-1)).mean()
    critic_optimizer.zero_grad()
    critic_scaler.scale(value_loss).backward()
    critic_scaler.unscale_(critic_optimizer)
    critic_grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), args.critic_clip)
    critic_scaler.step(critic_optimizer)
    critic_scaler.update()

    with torch.no_grad():
        aggregator.update("loss/actor_loss", actor_loss.item())
        aggregator.update("loss/value_loss", value_loss.item())
        aggregator.update("state/actor_entropy", actor_entropy.mean().item())
        aggregator.update("grad_norm/actor", actor_grad_norm.mean().item())
        aggregator.update("grad_norm/critic", critic_grad_norm.mean().item())


@torch.inference_mode()
def evaluation(episodes: int):
    num_envs = 1
    episodic_returns = []
    videos = []
    for i in range(episodes):
        seed = args.seed + 6666 + i  # ensure different seeds for different episodes
        envs = create_vector_env(args.env_id, "rgb", num_envs, seed, action_repeat=2, image_size=(64, 64))
        envs = ObsShiftWrapper(envs)
        obs, _ = envs.reset()
        posterior = torch.zeros(num_envs, args.stochastic_size, device=device)
        deterministic = torch.zeros(num_envs, args.deterministic_size, device=device)
        action = torch.zeros(num_envs, envs.single_action_space.shape[0], device=device)
        episodic_return = torch.zeros(num_envs, device=device)
        imgs = [obs.cpu()]
        while True:
            embeded_obs = encoder(obs)
            deterministic = recurrent_model(posterior, action, deterministic)
            posterior_dist, _ = representation_model(embeded_obs.view(num_envs, -1), deterministic)
            posterior = posterior_dist.mode.view(-1, args.stochastic_size)
            action = actor(posterior, deterministic).mode
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
    writer.add_scalar("reward/eval_episodic_return", np.mean(episodic_returns), global_step)
    writer.add_video("eval/video", videos + 0.5, global_step, fps=15)


def save_checkpoint():
    state = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "recurrent_model": recurrent_model.state_dict(),
        "transition_model": transition_model.state_dict(),
        "representation_model": representation_model.state_dict(),
        "reward_predictor": reward_predictor.state_dict(),
        "continue_model": continue_model.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "moments": moments.state_dict(),
        "model_optimizer": model_optimizer.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "model_scaler": model_scaler.state_dict(),
        "actor_scaler": actor_scaler.state_dict(),
        "critic_scaler": critic_scaler.state_dict(),
        "ratio": ratio.state_dict(),
        "global_step": global_step,
    }
    checkpoint_dir = os.path.join(logdir, "ckpt")
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, f"ckpt_{global_step}.pth"))
    log.info(f"Saved checkpoint to {os.path.join(checkpoint_dir, f'ckpt_{global_step}.pth')}")


def main():
    global global_step
    pbar = tqdm(total=args.total_steps, desc="Training")
    episodic_return = torch.zeros(args.num_envs, device=device)

    posterior = torch.zeros(args.num_envs, args.stochastic_size, device=device)
    deterministic = torch.zeros(args.num_envs, args.deterministic_size, device=device)
    action = torch.zeros(args.num_envs, envs.single_action_space.shape[0], device=device)

    obs, _ = envs.reset()
    while global_step < args.total_steps:
        ## Step the environment and add to buffer
        with torch.inference_mode(), timer("time/step"), timer("time/step_avg_per_env", MeanMetric):
            embeded_obs = encoder(obs)
            deterministic = recurrent_model(posterior, action, deterministic)
            posterior_dist, _ = representation_model(embeded_obs.view(args.num_envs, -1), deterministic)
            posterior = posterior_dist.sample().view(-1, args.stochastic_size)
            if global_step < args.prefill:
                action = torch.as_tensor(envs.action_space.sample(), device=device)
            else:
                action = actor(posterior, deterministic).sample()
            next_obs, reward, terminated, truncated, info = envs.step(action)
            done = torch.logical_or(terminated, truncated)
            buffer.add(obs, action, reward, next_obs, done, terminated)
            obs = next_obs

            episodic_return += reward
            if done.any():
                writer.add_scalar("reward/episodic_return", episodic_return[done].mean().item(), global_step)
                episodic_return[done] = 0
                posterior[done] = 0
                deterministic[done] = 0
                action[done] = 0

        ## Update the model
        if global_step > args.prefill:
            with timer("time/train"), timer("time/train_avg", MeanMetric):
                gradient_steps = ratio(global_step - args.prefill)
                for _ in range(gradient_steps):
                    with timer("time/data_sample"):
                        data = buffer.sample(args.batch_size, args.batch_length)
                    with timer("time/dynamic_learning"):
                        posteriors, deterministics = dynamic_learning(data)
                    with timer("time/behavior_learning"):
                        behavior_learning(posteriors, deterministics)

        ## Evaluation
        if global_step > args.prefill and (global_step - args.prefill) % args.eval_every < args.num_envs:
            with timer("time/eval"):
                evaluation(args.eval_episodes)

        ## Logging
        if global_step > args.prefill and (global_step - args.prefill) % args.log_every < args.num_envs:
            metrics_dict = aggregator.compute()
            for k, v in metrics_dict.items():
                writer.add_scalar(k, v, global_step)
            aggregator.reset()

            if not timer.disabled:
                metrics_dict = timer.compute()
                for k, v in metrics_dict.items():
                    if k == "time/step_avg_per_env":
                        v = v / args.num_envs
                    writer.add_scalar(k, v, global_step)
                timer.reset()

        ## Save checkpoint
        if global_step > args.prefill and (global_step - args.prefill) % args.checkpoint_every < args.num_envs:
            with timer("time/save_checkpoint"):
                save_checkpoint()

        global_step += args.num_envs
        pbar.update(args.num_envs)


if __name__ == "__main__":
    main()
