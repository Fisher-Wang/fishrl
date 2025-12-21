import torch

from fishrl.envs.maniskill_env import ManiskillVecEnv

env = ManiskillVecEnv("TurnFaucet-v1", num_envs=4, seed=0, device="cuda", obs_mode="rgb", image_size=(64, 64))
print(env.single_action_space)
print(env.action_space)
print(env.single_observation_space)
print(env.observation_space)
print(env.reset()[0].shape)
obs, reward, terminated, truncated, info = env.step(torch.randn(env.action_space.shape))
print("obs.shape", obs.shape)
print("Reward", reward)
env.close()

print()

env = ManiskillVecEnv("PickCube-v1", num_envs=4, seed=0, device="cuda", obs_mode="prop")
print(env.single_action_space)
print(env.action_space)
print(env.single_observation_space)
print(env.observation_space)
print(env.reset()[0].shape)
obs, reward, terminated, truncated, info = env.step(torch.randn(env.action_space.shape))
print("obs.shape", obs.shape)
print("Reward", reward)
env.close()
