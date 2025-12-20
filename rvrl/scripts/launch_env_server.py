from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import tyro

from rvrl.envs import create_vector_env
from rvrl.envs.client_env import Server


def main(
    port: int = 8888,
    env_id: str = "maniskill/TurnFaucet-v1",
    seed: int = 0,
    image_size: tuple[int, int] = (64, 64),
    num_envs: int = 1,
    device: str = "cuda",
    obs_mode: str = "rgb,prop",
    rv_sim: str = "mujoco",
):
    env = create_vector_env(
        env_id,
        num_envs=num_envs,
        seed=seed,
        device=device,
        obs_mode=obs_mode,
        image_size=image_size,
        scenario_cfg=dict(simulator=rv_sim),
    )
    server = Server(env, port=port)
    server.run()


if __name__ == "__main__":
    tyro.cli(main)
