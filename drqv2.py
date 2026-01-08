from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
from dataclasses import dataclass
from datetime import datetime

os.environ["MUJOCO_GL"] = "egl"  # significantly faster rendering compared to glfw and osmesa


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from loguru import logger as log
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.distributions.utils import _standard_normal
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import MeanMetric
from tqdm import tqdm

from fishrl.envs.env_factory import create_vector_env
from fishrl.utils.metrics import MetricAggregator
from fishrl.utils.reproducibility import enable_deterministic_run, seed_everything
from fishrl.utils.timer import timer


########################################################
## Standalone utils
########################################################
class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int],
        action_size: int,
        device: str | torch.device,
        num_envs: int = 1,
        nstep: int = 3,
        discount: float = 0.99,
        capacity: int = 5000000,
    ):
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity
        self.nstep = nstep
        self.discount = discount

        self.obs = np.empty((self.capacity, self.num_envs, *observation_shape), dtype=np.float32)
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
        done: Tensor,
        terminated: Tensor,
    ):
        self.obs[self.buffer_index] = observation.detach().cpu().numpy()
        self.action[self.buffer_index] = action.detach().cpu().numpy()
        self.reward[self.buffer_index] = reward.unsqueeze(-1).detach().cpu().numpy()
        self.done[self.buffer_index] = done.unsqueeze(-1).detach().cpu().numpy()
        self.terminated[self.buffer_index] = terminated.unsqueeze(-1).detach().cpu().numpy()

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size) -> dict[str, Tensor]:
        """
        Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.
        """
        assert self.full or (self.buffer_index > batch_size), "too short dataset or too long chunk_size"
        max_index = self.capacity if self.full else self.buffer_index
        max_index = max_index - self.nstep
        sample_index = np.random.randint(0, max_index + 1, batch_size)
        env_index = np.random.randint(0, self.num_envs, batch_size)
        flattened_index = sample_index * self.num_envs + env_index

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        obs = torch.as_tensor(flatten(self.obs)[flattened_index], device=self.device).float()
        next_obs = torch.as_tensor(flatten(self.obs)[flattened_index + self.nstep], device=self.device).float()
        action = torch.as_tensor(flatten(self.action)[flattened_index], device=self.device)
        done = torch.as_tensor(flatten(self.done)[flattened_index], device=self.device)
        terminated = torch.as_tensor(flatten(self.terminated)[flattened_index], device=self.device)
        reward = torch.zeros_like(done)
        discount = torch.ones_like(done)
        for i in range(self.nstep):
            step_reward = torch.as_tensor(flatten(self.reward)[flattened_index + i], device=self.device)
            reward += discount * step_reward
            discount *= self.discount  # TODO: consider termination when envs are not dmc

        sample = TensorDict(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
            terminated=terminated,
            batch_size=obs.shape[0],
            device=self.device,
        )
        return sample


class TruncatedNormal(Normal):
    ## Copied from https://github.com/facebookresearch/drqv2/blob/7ad7e05fa44378c64998dc89586a9703b74531ab/utils.py#L105
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


########################################################
## Args
########################################################
@dataclass
class Args:
    exp_name: str = "drqv2"
    seed: int = 0
    device: str = "cuda"
    deterministic: bool = False
    env_id: str = "dmc/walker-walk-v0"
    num_envs: int = 1
    buffer_size: int = 1_000_000
    total_timesteps: int = 1_000_000
    prefill: int = 4_000
    log_every: int = 100
    compile: bool = False
    cudagraph: bool = False

    ## train
    batch_size: int = 256
    embedded_obs_size: int = 4096
    actor_lr: float = 1e-4
    q_lr: float = 1e-4
    encoder_lr: float = 1e-4
    gamma: float = 0.99  # discount factor
    tau: float = 0.005
    expl_noise_std: float = 0.1
    nstep: int = 3


args = tyro.cli(Args)


########################################################
## Networks
########################################################
def _weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    # HACK: the output size is 4096, which should be equal to args.embedded_obs_size
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.apply(_weight_init)

    def forward(self, obs):
        B = obs.shape[0]
        embedded_obs = self.encoder(obs)
        return embedded_obs.reshape(B, -1)  # flatten the last 3 dimensions C, H, W


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(args.embedded_obs_size, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, env.single_action_space.shape[0]),
        )
        self.apply(_weight_init)

    def forward(self, obs: Tensor, std: float) -> Distribution:
        h = self.trunk(obs)
        mean = self.policy(h)
        mean = F.tanh(mean)
        std = torch.ones_like(mean) * std
        dist = TruncatedNormal(mean, std)
        return dist


class QNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(args.embedded_obs_size, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
        )
        self.q1 = nn.Sequential(
            nn.Linear(50 + env.single_action_space.shape[0], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(50 + env.single_action_space.shape[0], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        self.apply(_weight_init)

    def forward(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(obs)
        x = torch.cat([h, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2


########################################################
## Main
########################################################
def main():
    ## setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}" + (f" (GPU {torch.cuda.current_device()})" if torch.cuda.is_available() else ""))
    seed_everything(args.seed)
    if args.deterministic:
        enable_deterministic_run()

    ## env and replay buffer
    envs = create_vector_env(
        args.env_id,
        "prop,rgb",
        args.num_envs,
        args.seed,
        action_repeat=2,
        # image_size=(84, 84),  # official code uses 84x84
        image_size=(64, 64),  # our code uses 64x64
    )
    buffer = ReplayBuffer(
        envs.single_observation_space["rgb"].shape,
        envs.single_action_space.shape[0],
        device,
        num_envs=args.num_envs,
        capacity=args.buffer_size,
        nstep=args.nstep,
        discount=args.gamma,
    )

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

    ## networks
    aug = RandomShiftsAug(pad=4)
    encoder = Encoder().to(device)
    qf = QNet(envs).to(device)
    qf_target = QNet(envs).to(device)
    qf_target.load_state_dict(qf.state_dict())
    qf_params = from_module(qf).data
    qf_target_params = from_module(qf_target).data
    actor = Actor(envs).to(device)

    qf_optimizer = torch.optim.Adam(qf.parameters(), lr=args.q_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr)

    ## logging
    aggregator = MetricAggregator({
        "loss/q_loss": MeanMetric(),
        "loss/actor_loss": MeanMetric(),
        "state/q_value": MeanMetric(),
    })

    def update_model(data: dict[str, Tensor]) -> TensorDict:
        obs = data["observation"]
        action = data["action"]
        reward = data["reward"]
        next_obs = data["next_observation"]
        terminated = data["terminated"]

        obs = encoder(aug(obs))
        with torch.no_grad():
            next_obs = encoder(aug(next_obs))

        ## Update Q-function
        with torch.no_grad():
            std = 0.3  # TODO: use schedule
            next_action_dist = actor(next_obs, std)
            next_action = next_action_dist.sample(clip=0.3)
            q1_target, q2_target = qf_target(next_obs, next_action)
            q_target = reward + (1 - terminated) * args.gamma * torch.min(q1_target, q2_target)

        q1, q2 = qf(obs, action)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        encoder_optimizer.zero_grad()
        qf_optimizer.zero_grad()
        q_loss.backward()
        qf_optimizer.step()
        encoder_optimizer.step()

        ## Update Actor
        obs = obs.detach()
        std = 0.3  # TODO: use schedule
        action_dist = actor(obs, std)
        action = action_dist.sample(clip=0.3)
        q1, q2 = qf(obs, action)
        q = torch.min(q1, q2)

        actor_loss = -q.mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        qf_target_params.lerp_(qf_params.data, args.tau)

        return TensorDict(q_loss=q_loss.detach(), q_value=q.detach().mean(), actor_loss=actor_loss.detach())

    if args.compile:
        update_model = torch.compile(update_model)
    if args.cudagraph:
        update_model = CudaGraphModule(update_model, in_keys=[], out_keys=[], warmup=5)

    ## main loop
    global_step = 0
    pbar = tqdm(total=args.total_timesteps, desc="Training")
    episodic_return = torch.zeros(args.num_envs, device=device)
    episodic_length = torch.zeros(args.num_envs, device=device)

    obs, _ = envs.reset(seed=args.seed)
    obs = obs["rgb"]
    while global_step < args.total_timesteps:
        ## Step the environment and add to buffer
        with torch.inference_mode(), timer("time/step"):
            if global_step < args.prefill:
                action = torch.as_tensor(envs.action_space.sample(), device=device)
            else:
                embeded_obs = encoder(obs)
                action = actor(embeded_obs, 0.3).sample(clip=None)
            next_obs, reward, terminated, truncated, info = envs.step(action)
            next_obs = next_obs["rgb"]
            done = torch.logical_or(terminated, truncated)
            buffer.add(obs, action, reward, done, terminated)
            obs = next_obs
            episodic_return += reward
            episodic_length += 1
            if done.any():
                writer.add_scalar("reward/episodic_return", episodic_return[done].mean().item(), global_step)
                writer.add_scalar("reward/episodic_length", episodic_length[done].mean().item(), global_step)
                tqdm.write(f"global_step={global_step}, episodic_return={episodic_return[done].mean().item():.1f}")
                episodic_return[done] = 0
                episodic_length[done] = 0

        ## Update the model
        if global_step >= args.prefill:
            with timer("time/train"):
                with timer("time/data_sample"):
                    data = buffer.sample(args.batch_size)
                with timer("time/update_model"):
                    metrics = update_model(data)

            with torch.no_grad(), timer("time/update_metrics"):
                aggregator.update("loss/q_loss", metrics["q_loss"].item())
                aggregator.update("loss/actor_loss", metrics["actor_loss"].item())
                aggregator.update("state/q_value", metrics["q_value"].item())

        ## Logging
        if global_step > args.prefill and global_step % args.log_every < args.num_envs:
            for k, v in aggregator.compute().items():
                writer.add_scalar(k, v, global_step)
            aggregator.reset()

            if not timer.disabled:
                for k, v in timer.compute().items():
                    writer.add_scalar(k, v, global_step)
                timer.reset()

        global_step += args.num_envs
        pbar.update(args.num_envs)


if __name__ == "__main__":
    main()
