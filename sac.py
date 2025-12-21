from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
from dataclasses import dataclass
from datetime import datetime
from itertools import chain

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
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution
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
        capacity: int = 5000000,
    ):
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity

        self.observation = np.empty((self.capacity, self.num_envs, *observation_shape), dtype=np.float32)
        self.next_observation = np.empty((self.capacity, self.num_envs, *observation_shape), dtype=np.float32)
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

    def sample(self, batch_size) -> dict[str, Tensor]:
        """
        Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.
        """
        assert self.full or (self.buffer_index > batch_size), "too short dataset or too long chunk_size"
        sample_index = np.random.randint(0, self.capacity if self.full else self.buffer_index, batch_size)
        env_index = np.random.randint(0, self.num_envs, batch_size)
        flattened_index = sample_index * self.num_envs + env_index

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        observation = torch.as_tensor(flatten(self.observation)[flattened_index], device=self.device).float()
        next_observation = torch.as_tensor(flatten(self.next_observation)[flattened_index], device=self.device).float()
        action = torch.as_tensor(flatten(self.action)[flattened_index], device=self.device)
        reward = torch.as_tensor(flatten(self.reward)[flattened_index], device=self.device)
        done = torch.as_tensor(flatten(self.done)[flattened_index], device=self.device)
        terminated = torch.as_tensor(flatten(self.terminated)[flattened_index], device=self.device)

        sample = TensorDict(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            terminated=terminated,
            batch_size=observation.shape[0],
            device=self.device,
        )
        return sample


########################################################
## Args
########################################################
@dataclass
class Args:
    exp_name: str = "sac"
    seed: int = 0
    device: str = "cuda"
    deterministic: bool = False
    env_id: str = "gym/Hopper-v4"
    num_envs: int = 1
    buffer_size: int = 1_000_000
    total_timesteps: int = 1000_000
    prefill: int = 5000
    log_every: int = 100
    compile: bool = False
    cudagraph: bool = False

    ## train
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99  # discount factor
    policy_frequency: int = 2
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    alpha: float = 0.2
    tau: float = 0.005


args = tyro.cli(Args)


########################################################
## Networks
########################################################
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(np.prod(env.single_observation_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * np.prod(env.single_action_space.shape)),
        )

    def forward(self, obs):
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        log_std_min, log_std_max = -5, 2
        log_std = log_std_min + (log_std_max - log_std_min) * (F.tanh(log_std) + 1) / 2
        return mean, log_std

    def get_action(self, obs):
        # assume action is bounded in [-1, 1], which is the value range of tanh. Otherwise we need to add an affine transform

        ## Option 1
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        action_dist = TransformedDistribution(
            Normal(mean, std), TanhTransform(cache_size=1)
        )  # ! use cache_size=1 to avoid atanh which could cause nan
        action_dist = Independent(action_dist, 1)
        action = action_dist.rsample()
        return action, action_dist.log_prob(action).unsqueeze(-1)

        ## Option 2
        # mean, log_std = self(obs)
        # std = log_std.exp()
        # normal = Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # action = torch.tanh(x_t)
        # log_prob = normal.log_prob(x_t)
        # log_prob -= torch.log((1 - action**2) + 1e-6)  # ! 1e-6 to avoid nan
        # log_prob = log_prob.sum(1, keepdim=True)
        # return action, log_prob


class QNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.qnet = nn.Sequential(
            nn.Linear(np.prod(env.single_observation_space.shape) + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.qnet(x)


########################################################
## Main
########################################################

## setup
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}" + (f" (GPU {torch.cuda.current_device()})" if torch.cuda.is_available() else ""))
seed_everything(args.seed)
if args.deterministic:
    enable_deterministic_run()

## env and replay buffer
envs = create_vector_env(args.env_id, "state", args.num_envs, args.seed)
buffer = ReplayBuffer(
    envs.single_observation_space.shape, envs.single_action_space.shape[0], device, args.num_envs, args.buffer_size
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
actor = Actor(envs).to(device)
qf1 = QNet(envs).to(device)
qf2 = QNet(envs).to(device)
qf1_target = QNet(envs).to(device)
qf2_target = QNet(envs).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
qf1_params = from_module(qf1).data
qf2_params = from_module(qf2).data
qf1_target_params = from_module(qf1_target).data
qf2_target_params = from_module(qf2_target).data

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
critic_optimizer = torch.optim.Adam(chain(qf1.parameters(), qf2.parameters()), lr=args.critic_lr)

alpha = args.alpha  # TODO: implement automatic alpha tuning

## Logging
global_step = 0
aggregator = MetricAggregator({
    "loss/q1_loss": MeanMetric(),
    "loss/q2_loss": MeanMetric(),
    "loss/critic_loss": MeanMetric(),
    "loss/actor_loss": MeanMetric(),
    "state/q1_value": MeanMetric(),
    "state/q2_value": MeanMetric(),
})


def update_q(data: dict[str, Tensor]) -> TensorDict:
    obs = data["observation"]
    action = data["action"]
    next_obs = data["next_observation"]
    reward = data["reward"]
    terminated = data["terminated"]

    with torch.no_grad():
        next_action, next_log_prob = actor.get_action(next_obs)
        qf1_next_target = qf1_target(next_obs, next_action)
        qf2_next_target = qf2_target(next_obs, next_action)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_log_prob
        next_q = reward + (1 - terminated) * args.gamma * min_qf_next_target

    q1 = qf1(obs, action)
    q2 = qf2(obs, action)
    q1_loss = F.mse_loss(q1, next_q)
    q2_loss = F.mse_loss(q2, next_q)
    critic_loss = q1_loss + q2_loss
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return TensorDict(
        q1_loss=q1_loss.detach(),
        q2_loss=q2_loss.detach(),
        critic_loss=critic_loss.detach(),
        q1_value=q1.detach().mean(),
        q2_value=q2.detach().mean(),
    )


def update_actor(data: dict[str, Tensor]) -> TensorDict:
    obs = data["observation"]
    action, log_prob = actor.get_action(obs)
    q1 = qf1(obs, action)
    q2 = qf2(obs, action)
    min_q = torch.min(q1, q2)
    actor_loss = (alpha * log_prob - min_q).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return TensorDict(actor_loss=actor_loss.detach())


if args.compile:
    update_q = torch.compile(update_q)
    update_actor = torch.compile(update_actor)

if args.cudagraph:
    update_q = CudaGraphModule(update_q, in_keys=[], out_keys=[], warmup=5)
    update_actor = CudaGraphModule(update_actor, in_keys=[], out_keys=[], warmup=5)


def main():
    global global_step
    pbar = tqdm(total=args.total_timesteps, desc="Training")
    episodic_return = torch.zeros(args.num_envs, device=device)
    episodic_length = torch.zeros(args.num_envs, device=device)

    obs, _ = envs.reset(seed=args.seed)
    while global_step < args.total_timesteps:
        ## Step the environment and add to buffer
        with torch.inference_mode(), timer("time/step"):
            if global_step < args.prefill:
                action = torch.as_tensor(envs.action_space.sample(), device=device)
            else:
                action, _ = actor.get_action(obs)
            next_obs, reward, terminated, truncated, info = envs.step(action)
            done = torch.logical_or(terminated, truncated)
            real_next_obs = next_obs.clone()
            if truncated.any():
                real_next_obs[truncated.bool()] = torch.as_tensor(
                    np.stack(info["final_observation"][truncated.bool().numpy(force=True)]),
                    device=device,
                    dtype=torch.float32,
                )
            buffer.add(obs, action, reward, real_next_obs, done, terminated)
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
                    metrics = update_q(data)
                    if global_step % args.policy_frequency == 0:
                        metrics.update(update_actor(data))
                    if global_step % args.target_network_frequency == 0:
                        qf1_target_params.lerp_(qf1_params.data, args.tau)
                        qf2_target_params.lerp_(qf2_params.data, args.tau)

            with torch.no_grad(), timer("time/update_metrics"):
                aggregator.update("loss/q1_loss", metrics["q1_loss"].item())
                aggregator.update("loss/q2_loss", metrics["q2_loss"].item())
                aggregator.update("loss/critic_loss", metrics["critic_loss"].item())
                if "actor_loss" in metrics:
                    aggregator.update("loss/actor_loss", metrics["actor_loss"].item())
                aggregator.update("state/q1_value", metrics["q1_value"].item())
                aggregator.update("state/q2_value", metrics["q2_value"].item())

        # Logging
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
