# 🐟 FishRL

FishRL is a collection of popular deep RL algorithms implemented in PyTorch. It supports various RL environments, supports both state-based and RGB-based observations, for robotics and embodied AI research.

## Installation

### Recommended Setup
```bash
conda create -n fishrl python=3.11 -y && conda activate fishrl
uv pip install -e ".[dmc,benchmark]"
```
The `benchmark` option specifies the packages version for the reproducibility of benchmarking results.

### Advanced Setup

<details><summary>Humanoid-bench environment</summary>

```bash
conda activate fishrl
cd third_party && git clone --depth 1 https://github.com/Fisher-Wang/humanoid-bench && cd ..
uv pip install -e ".[humanoid_bench]" -e third_party/humanoid-bench
```
</details>

<details><summary>IsaacLab 2.3.1 environment</summary>

```bash
conda activate fishrl
cd third_party && git clone --depth 1 --branch v2.3.1 https://github.com/isaac-sim/IsaacLab.git IsaacLab231 && cd ..
sed -i 's/gymnasium==1\.2\.0/gymnasium/g' third_party/IsaacLab231/source/isaaclab/setup.py
uv pip install -e ".[isaaclab]" -e "third_party/IsaacLab231/source/isaaclab" -e "third_party/IsaacLab231/source/isaaclab_tasks"
```
</details>

<details><summary>IsaacGym environment</summary>

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

For more examples, please refer to the [scripts](./scripts).

## Acknowledgments

FishRL is inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) and [SheepRL](https://github.com/Eclectic-Sheep/sheeprl).

Its Dreamerv1 and Dreamerv3 implementation has referred to [NaturalDreamer](https://github.com/InexperiencedMe/NaturalDreamer), [SimpleDreamer](https://github.com/kc-ml2/SimpleDreamer), [SheepRL](https://github.com/Eclectic-Sheep/sheeprl), [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch), and [dreamerv3](https://github.com/danijar/dreamerv3).

Its acceleration techniques have referred to [LeanRL](https://github.com/meta-pytorch/LeanRL).
