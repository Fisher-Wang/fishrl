# FishRL - AI Agent Guide

## Project Overview

FishRL is a CleanRL-style deep RL library focused on robotics/embodied AI. Each algorithm lives in a single self-contained file.

## Key Files

| File | Description |
|------|-------------|
| `dm3.py` | DreamerV3 implementation |
| `dm1.py` | DreamerV1 implementation |
| `sac.py` / `ddpg.py` / `ppo.py` | Other algorithms |
| `fishrl/envs/` | Unified env interface (`BaseVecEnv`) |
| `fishrl/envs/env_factory.py` | `create_vector_env()` dispatcher |
| `fishrl/utils/` | Logger, timer, metrics, reproducibility |
| `fishrl/wrapper/` | Env wrappers (NumpyToTorch, UnwrapDict, etc.) |

## Running Experiments

See `script.sh` for examples.

Supported env prefixes: `dmc/`, `humanoid_bench/`, `isaaclab/`, `isaacgymenv/`, `gym/`, `maniskill/`, `client/`

## RL Conventions

Two conventions for indexing rewards (and values, termintations, timeouts, ...):
- Aligning with **RL theory**: `reward[t]` = $r(s_t, a_t)$
- Aligning with **gym env**: `reward[t]` = $r(s_{t-1}, a_{t-1})$

In this repo, the algorithm implementation should follow the "RL theory" convention, the replay buffer implementation should follow the "gym env" convention. Should be careful when loading data from the replay buffer for the algorithm.
