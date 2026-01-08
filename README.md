# 🐟 FishRL

FishRL is a collection of popular deep reinforcement learning algorithms implemented in PyTorch, designed for robotics and embodied AI research. It supports both state-based and visual (RGB) observations, and provides clean, reference implementations of methods ranging from PPO and SAC to more advanced approaches such as Dreamer v3.

In addition to algorithm implementations, FishRL focuses on the following practical aspects:

- **Unified environment interface.** FishRL supports multiple robotics and embodied AI environments through a unified interface. This interface can be reused to quickly test new algorithms across different simulation environments with minimal additional engineering for research purposes.

- **Efficient benchmarking.** For large-scale benchmarking, FishRL implements optimized training pipelines and common acceleration techniques. Inspired by [LeanRL](https://github.com/meta-pytorch/LeanRL), FishRL typically achieves 3–10× speedup compared to naïve PyTorch implementations, and is competitive with JAX-based implementations in terms of training throughput.

- **Single-file, readable implementations.** Following the philosophy of [CleanRL](https://github.com/vwxyzjn/cleanrl), FishRL keeps each algorithm in a single, self-contained file whenever possible, while sharing common RL utilities. This design emphasizes readability and ease of modification, making it convenient to understand, debug, and extend the algorithms.


## Installation

### Recommended Setup
```bash
conda create -n fishrl python=3.11 -y && conda activate fishrl
uv pip install -e ".[dmc,benchmark]"
```
The `benchmark` option specifies the packages version for the reproducibility of benchmarking results.

### Advanced Setup

<details><summary>Gymnasium-Robotics environment</summary>

```bash
uv pip install -e ".[gymnasium_robotics]"
```
</details>

<details><summary>ManiSkill environment</summary>

```bash
uv pip install -e ".[maniskill]"
```
</details>

<details><summary>Humanoid-bench environment</summary>

```bash
cd third_party && git clone --depth 1 https://github.com/Fisher-Wang/humanoid-bench && cd ..
uv pip install -e ".[humanoid_bench]" -e third_party/humanoid-bench
```
</details>

<details><summary>IsaacLab 2.3.1 environment</summary>

```bash
cd third_party && git clone --depth 1 --branch v2.3.1 https://github.com/isaac-sim/IsaacLab.git IsaacLab231 && cd ..
sed -i 's/gymnasium==1\.2\.0/gymnasium/g' third_party/IsaacLab231/source/isaaclab/setup.py
uv pip install -e ".[isaaclab]" -e "third_party/IsaacLab231/source/isaaclab" -e "third_party/IsaacLab231/source/isaaclab_tasks"
```
</details>

<details><summary>IsaacGymEnvs environment</summary>

```bash
conda create -n fishrl_gym python=3.8 -y && conda activate fishrl_gym
cd third_party
wget https://developer.nvidia.com/isaac-gym-preview-4 \
    && tar -xf isaac-gym-preview-4 \
    && rm isaac-gym-preview-4
find isaacgym/python -type f -name "*.py" -exec sed -i 's/np\.float/np.float32/g' {} +
uv pip install isaacgym/python
git clone --depth 1 https://github.com/isaac-sim/IsaacGymEnvs && cd IsaacGymEnvs
uv pip install -e .
cd ../..
uv pip install networkx==2.1
```
</details>

## Get Started

To run the Dreamerv3 agent on the DMControl Walker-walk task with rgb observation, you can use the following command:
```bash
python dm3.py --env-id=dmc/walker-walk-v0 --obs-mode=rgb
```

For more examples, please refer to [`script.sh`](./script.sh).

## Algorithms Implemented
| Algorithm  | File     | Recommended<br>Acceleration Option |
|------------|----------| ---------------------------------- |
| [PPO](https://arxiv.org/pdf/1707.06347)        | [`ppo.py`](./ppo.py)     | TODO                    |
| [DDPG](https://arxiv.org/pdf/1509.02971)       | [`ddpg.py`](./ddpg.py)   | `--compile --cudagraph` |
| [SAC](https://arxiv.org/pdf/1812.05905)        | [`sac.py`](./sac.py)     | `--compile --cudagraph` |
| [DrQ v2](https://arxiv.org/pdf/2107.09645)     | [`drqv2.py`](./drqv2.py) | `--compile --cudagraph` |
| [Dreamer v1](https://arxiv.org/pdf/1912.01603) | [`dm1.py`](./dm1.py)     | TODO                    |
| [Dreamer v3](https://arxiv.org/pdf/2301.04104) | [`dm3.py`](./dm3.py)     | `--compile --amp`       |

## Acknowledgments

FishRL is inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) and [SheepRL](https://github.com/Eclectic-Sheep/sheeprl).

Its Dreamer v1 and Dreamer v3 implementation has referred to [NaturalDreamer](https://github.com/InexperiencedMe/NaturalDreamer), [SimpleDreamer](https://github.com/kc-ml2/SimpleDreamer), [SheepRL](https://github.com/Eclectic-Sheep/sheeprl), [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch), and [dreamerv3](https://github.com/danijar/dreamerv3).

Its acceleration techniques have referred to [LeanRL](https://github.com/meta-pytorch/LeanRL).

FishRL's previous name before the public release was `RVRL`.
