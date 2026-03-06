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

## TODO

- **[dm3.py] Episode boundary not reset in `dynamic_learning`** (line 854): When a sampled chunk crosses a `done=True` boundary mid-sequence, `posterior`/`deterministic` should reset to zero at that point but currently don't. The RSSM learns spurious transitions between episodes. Fix: check `data["done"]` at each timestep in the loop and reset state for terminated envs.
