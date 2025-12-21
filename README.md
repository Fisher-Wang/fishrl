# 🐟 FishRL

## Installation

Main environment (Including DM-Control, ManiSkill):
```bash
conda create -n fishrl python=3.11 -y && conda activate fishrl
uv pip install -e ".[dmc,maniskill]"
```

RoboVerse environment:
```bash
conda create -n fishrl_rv python=3.11 -y && conda activate fishrl_rv
cd third_party
git clone https://github.com/RoboVerseOrg/RoboVerse && cd RoboVerse
git checkout 8952c9a  # ensure compatibility
uv pip install -e ".[mjx]" "jax[cuda]==0.6.2"
cd ../..
uv pip install -e .
```

Humanoid-bench environment:
```bash
conda activate fishrl
cd third_party && git clone --depth 1 https://github.com/Fisher-Wang/humanoid-bench && cd ..
uv pip install -e ".[dmc,maniskill,humanoid_bench]" -e third_party/humanoid-bench
```

IsaacLab 2.2.0 environment:
```bash
conda activate fishrl
cd third_party && git clone --depth 1 --branch v2.2.0 https://github.com/isaac-sim/IsaacLab.git IsaacLab220 && cd ..
sed -i 's/gymnasium==1\.2\.0/gymnasium/g' third_party/IsaacLab220/source/isaaclab/setup.py
uv pip install -e ".[dmc,maniskill,humanoid_bench,isaaclab]" -e "third_party/IsaacLab220/source/isaaclab" -e "third_party/IsaacLab220/source/isaaclab_tasks"
```

Gymnasium-robotics environment:
```bash
conda create -n fishrl_gr python=3.11 -y && conda activate fishrl_gr
uv pip install -e ".[gymnasium_robotics]"
```

IsaacGym environment:
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

IsaacLab 2.1.1 environment:
```bash
conda create -n fishrl_lab python=3.10 -y && conda activate fishrl_lab
uv pip install -e ".[isaaclab]"
cd third_party
git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git IsaacLab210 && cd IsaacLab210
sed -i '/^EXTRAS_REQUIRE = {/,/^}$/c\EXTRAS_REQUIRE = {\n    "sb3": [],\n    "skrl": [],\n    "rl-games": [],\n    "rsl-rl": [],\n}' source/isaaclab_rl/setup.py
sed -i 's/if platform\.system() == "Linux":/if False:/' source/isaaclab_mimic/setup.py
./isaaclab.sh -i
cd ../..
```

## Usage
see [script.sh](./script.sh)
