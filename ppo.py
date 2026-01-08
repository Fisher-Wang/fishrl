from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from loguru import logger as log
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fishrl.envs import create_vector_env
from fishrl.wrapper import UnwrapDictWrapper


########################################################
## Standalone utils
########################################################
class RollingMeter:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)

    def update(self, rewards: torch.Tensor):
        self.deque.extend(rewards.cpu().numpy().tolist())

    @property
    def mean(self) -> float:
        return np.mean(self.deque).item()

    @property
    def std(self) -> float:
        return np.std(self.deque).item()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


########################################################
## Args
########################################################
@dataclass
class Args:
    env_id: str = "dmc/cartpole-balance-v0"
    exp_name: str = "ppo"
    seed: int = 0
    num_envs: int = 1
    num_steps: int = 2048
    clip_coef: float = 0.2
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    epochs: int = 10
    total_timesteps: int = 1000000
    anneal_lr: bool = True
    device: str = "cuda"  # Device for IsaacLab environments
    window_size: int = 100
    rv_sim: str = "mujoco"

    # network
    num_layers: int = 2
    hidden_dim: int = 64
    init_actor_logstd: float = 0.0

    # evaluation
    eval_every: int = -1
    """evaluate every N iterations. -1 means no evaluation"""
    eval_num_envs: int = -1
    """number of evaluation environments. -1 means same as num_envs"""
    eval_num_steps: int = -1
    """number of evaluation steps. -1 means same as num_steps"""
    eval_ignore_done: bool = False
    """continue evaluation even if all environments are done"""
    eval_env_id: str | None = None
    """evaluation environment id. if None, use the same as env_id"""

    def __post_init__(self):
        if self.eval_num_envs == -1:
            self.eval_num_envs = self.num_envs
        if self.eval_num_steps == -1:
            self.eval_num_steps = self.num_steps
        if self.eval_env_id is None:
            self.eval_env_id = self.env_id


########################################################
## Networks
########################################################
def init_weights(std=np.sqrt(2), bias_const=0.0):
    def f(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, std)
            nn.init.constant_(m.bias, bias_const)
        return m

    return f


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, act: nn.Module):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(input_dim, hidden_dim), act()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Agent(nn.Module):
    def __init__(self, envs, hidden_dim: int, num_layers: int, init_actor_logstd: float):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        obs_dim = np.prod(envs.single_observation_space.shape)
        self.critic = MLP(obs_dim, hidden_dim, 1, num_layers, nn.Tanh)
        self.actor_mean = MLP(obs_dim, hidden_dim, action_dim, num_layers, nn.Tanh)
        # Orthogonal Initialization of Weights and Constant Initialization of biases (core-2)
        [m.apply(init_weights()) for m in self.critic.net[:-1]]
        self.critic.net[-1].apply(init_weights(std=1.0))
        [m.apply(init_weights()) for m in self.actor_mean.net[:-1]]
        self.actor_mean.net[-1].apply(init_weights(std=0.01))

        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * init_actor_logstd)

    def get_action(self, obs: torch.Tensor, action: torch.Tensor | None = None, deterministic: bool = False):
        action_mean = self.actor_mean(obs)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(dim=1), probs.entropy().sum(dim=1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)


########################################################
## Main
########################################################
def main():
    args = tyro.cli(Args)
    batch_size = args.num_envs * args.num_steps
    mini_batch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}" + (f" (GPU {torch.cuda.current_device()})" if torch.cuda.is_available() else ""))
    seed_everything(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.env_id}__{args.exp_name}__env={args.num_envs}__lr={args.lr:.0e}__annealLr={args.anneal_lr}__seed={args.seed}__{timestamp}"
    writer = SummaryWriter(f"logdir/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Use the unified environment interface
    envs = create_vector_env(
        args.env_id,
        "prop",
        args.num_envs,
        args.seed,
        device=args.device,
        scenario_cfg=dict(simulator=args.rv_sim),
    )
    envs = UnwrapDictWrapper(envs)

    if args.eval_every > 0:
        eval_envs = create_vector_env(
            args.eval_env_id,
            "prop",
            args.eval_num_envs,
            args.seed,
            device=args.device,
        )
        eval_envs = UnwrapDictWrapper(eval_envs)

    log.info(f"{envs.single_action_space.shape=}")
    log.info(f"{envs.single_observation_space.shape=}")

    obs, _ = envs.reset()
    assert obs.ndim == 2
    agent = Agent(envs, args.hidden_dim, args.num_layers, args.init_actor_logstd).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    obss = torch.zeros((args.num_steps + 1, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps + 1, args.num_envs) + envs.single_action_space.shape, device=device)
    log_probs = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    values = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    advantages = torch.zeros((args.num_steps + 1, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps + 1, args.num_envs), device=device)

    episode_rewards = RollingMeter(args.window_size)
    episode_lengths = RollingMeter(args.window_size)
    cur_rewards_sum = torch.zeros(args.num_envs, device=device)
    cur_episode_length = torch.zeros(args.num_envs, device=device)

    obss[0] = obs
    dones[0] = torch.zeros(args.num_envs).to(device)

    start_time = time.time()
    global_step = 0

    for iteration in tqdm(range(num_iterations)):
        ## evaluation
        if args.eval_every > 0 and iteration % args.eval_every == 1:
            eval_obs, _ = eval_envs.reset()
            eval_episodic_return = torch.zeros(args.eval_num_envs, device=device)
            eval_done_once = torch.zeros(args.eval_num_envs, device=device)
            for _ in range(args.eval_num_steps):
                with torch.inference_mode():
                    eval_action = agent.get_action(eval_obs, deterministic=True)
                eval_obs, eval_reward, eval_terminated, eval_truncated, eval_infos = eval_envs.step(eval_action)
                eval_done = torch.logical_or(eval_terminated, eval_truncated)
                eval_done_once = torch.logical_or(eval_done_once, eval_done)
                if not args.eval_ignore_done:
                    eval_reward[eval_done_once] = 0
                eval_episodic_return += eval_reward
                if not args.eval_ignore_done and eval_done_once.all():
                    break

            writer.add_scalar("charts/eval_episodic_return", eval_episodic_return.mean().item(), global_step)

        ## anneal lr (core-4)
        if args.anneal_lr:
            lr = args.lr * (1 - iteration / num_iterations)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        ## collect rollout
        for t in range(args.num_steps):
            global_step += args.num_envs

            with torch.no_grad():
                action, log_prob, entropy = agent.get_action(obs)
                value = agent.get_value(obs)

            next_obs, reward, terminated, truncated, infos = envs.step(action)
            assert next_obs.ndim == 2
            next_done = torch.logical_or(terminated, truncated)

            cur_rewards_sum += reward
            cur_episode_length += 1
            episode_rewards.update(cur_rewards_sum[next_done])
            episode_lengths.update(cur_episode_length[next_done])
            if next_done.any():
                writer.add_scalar("charts/episodic_return", cur_rewards_sum[next_done].mean().item(), global_step)
            cur_rewards_sum[next_done] = 0
            cur_episode_length[next_done] = 0

            actions[t] = action
            log_probs[t] = log_prob
            values[t] = value.view(-1)
            rewards[t] = reward.view(-1)
            obss[t + 1] = next_obs
            dones[t + 1] = next_done

            obs = obss[t + 1]

        values[args.num_steps] = agent.get_value(obss[args.num_steps]).view(-1)

        ## bootstrap (XXX: what is this?)
        with torch.no_grad():
            advantages[args.num_steps] = 0
            for t in reversed(range(args.num_steps)):
                # Generalized Advantage Estimation (core-5, see original paper)
                delta = rewards[t] + args.gamma * values[t + 1] * (1 - dones[t + 1]) - values[t]
                advantages[t] = delta + args.gamma * args.gae_lambda * advantages[t + 1] * (1 - dones[t + 1])
            returns = advantages + values

        ## flatten and random shuffle
        random_indices = torch.randperm(args.num_envs * args.num_steps)
        b_obs = obss[:-1].view(-1, *envs.single_observation_space.shape)[random_indices]
        b_log_probs = log_probs[:-1].view(-1)[random_indices]
        b_actions = actions[:-1].view(-1, *envs.single_action_space.shape)[random_indices]
        b_returns = returns[:-1].view(-1)[random_indices]
        b_advantages = advantages[:-1].view(-1)[random_indices]
        b_values = values[:-1].view(-1)[random_indices]

        ## optimize
        for epoch in range(args.epochs):
            # Mini-batch updates (core-6)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size

                #! important to use old actions
                _, new_log_probs, entropy = agent.get_action(b_obs[start:end], b_actions[start:end])
                new_value = agent.get_value(b_obs[start:end]).view(-1)
                logratio = new_log_probs - b_log_probs[start:end]
                ratio = logratio.exp()

                mb_advantages = b_advantages[start:end]

                # Normalize the advantages (core-7)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # actor loss
                a_loss1 = mb_advantages * ratio
                a_loss2 = mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                a_loss = -torch.min(a_loss1, a_loss2).mean()

                # critic loss
                c_loss = 0.5 * (new_value - b_returns[start:end]).pow(2).mean()

                # total loss
                loss = a_loss + 0.5 * c_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        ## log
        writer.add_scalar("charts/episodic_return_mean", episode_rewards.mean, global_step)
        writer.add_scalar("charts/episodic_return_std", episode_rewards.std, global_step)
        writer.add_scalar("charts/episodic_length_mean", episode_lengths.mean, global_step)
        writer.add_scalar("charts/episodic_length_std", episode_lengths.std, global_step)

        print(f"SPS: {global_step / (time.time() - start_time):.2f}")

    writer.close()


if __name__ == "__main__":
    main()
