########################
## PPO
########################

## DMControl Tasks
python ppo.py --env-id=dmc/cartpole-balance-v0 --seed 2 --num-envs=4 --lr=1e-3 --num_steps=2048 --total_timesteps=1000000

## Humanoid-bench Tasks
python ppo.py --env-id=humanoid_bench/h1hand-reach-v0 --num-envs=4 --num-steps=1000

## IsaacGymEnv Tasks
python ppo.py --env-id=isaacgymenv/Cartpole --seed 2 --num-envs=512 --lr=3e-4 --num_steps=16 --total_timesteps=1500000

## IsaacLab Tasks
python ppo.py --env-id=isaaclab/Isaac-Reach-Franka-v0 --seed 1 --num-envs=32 --lr=1e-3 --num_steps=360 --total_timesteps=3000000

## RoboVerse Tasks
python ppo.py --env-id=roboverse/reach_origin --num-envs=2048 --num-steps=64 --total-timesteps=10000000


########################
## SAC
########################

## Gymnasium Tasks
python sac.py --env-id=gym/Hopper-v4


########################
## DDPG
########################

## Gymnasium Tasks
python ddpg.py --env-id=gym/Hopper-v4


########################
## DreamerV1
########################

## DMControl Tasks
python dm1.py --env-id dmc/walker-walk-v0


########################
## DreamerV3
########################

## DMControl Tasks
python dm3.py --env-id dmc/walker-walk-v0
