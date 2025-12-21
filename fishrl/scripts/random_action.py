from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import imageio as iio
import numpy as np
import torch
import tyro
from torch import Tensor
from torchvision.utils import make_grid

from fishrl.envs import create_vector_env


class VideoWriter:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.images: list[np.ndarray] = []

    def add(self, rgb: Tensor):
        """
        Args:
            rgb: (B, C, H, W)
        """
        image = make_grid(rgb, nrow=int(rgb.shape[0] ** 0.5))  # (C, H, W)
        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image = (image.clip(0, 1) * 255).astype(np.uint8)
        self.images.append(image)

    def save(self, fps: int = 30):
        iio.mimsave(self.video_path, self.images, fps=fps)


def main(
    env_id: str = "maniskill/TurnFaucet-v1",
    image_size: tuple[int, int] = (64, 64),
    rv_sim: str = "mujoco",
    num_envs: int = 1,
):
    env = create_vector_env(
        env_id,
        num_envs=num_envs,
        seed=0,
        device="cuda",
        obs_mode="rgb,prop",
        image_size=image_size,
        scenario_cfg=dict(simulator=rv_sim),
    )
    obs, _ = env.reset()
    video_writer = VideoWriter(f"random_action_{env_id.replace('/', '_')}.mp4")
    video_writer.add(obs["rgb"])
    for _ in range(100):
        action = torch.rand(env.action_space.shape, device=env.device) * 2 - 1
        obs, _, _, _, _ = env.step(action)
        video_writer.add(obs["rgb"])
    video_writer.save()
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
