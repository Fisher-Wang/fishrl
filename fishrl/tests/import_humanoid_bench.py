import gymnasium as gym
import humanoid_bench  # noqa: F401

env_ids = [
    ## Tasks used in Fast-TD3 paper
    "h1hand-reach-v0",
    "h1hand-balance_simple-v0",
    "h1hand-balance_hard-v0",
    "h1hand-pole-v0",
    "h1hand-truck-v0",
    "h1hand-maze-v0",
    "h1hand-push-v0",
    "h1hand-basketball-v0",
    "h1hand-window-v0",
    "h1hand-package-v0",
    "h1hand-truck-v0",
]

for env_id in env_ids:
    env = gym.make(env_id)
    print("Task:", env_id)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("Max episode steps:", env.spec.max_episode_steps)
    print()
    env.close()
