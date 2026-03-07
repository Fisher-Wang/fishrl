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
