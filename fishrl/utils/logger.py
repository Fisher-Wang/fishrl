from __future__ import annotations

import concurrent.futures
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

TensorArray = torch.Tensor | np.ndarray


class Logger:
    def __init__(self, outputs: list[BaseOutput], init_step: int = 0):
        self._outputs = outputs
        self._metrics = []
        self._step = init_step

    def add(self, mapping: dict[str, Any]):
        assert isinstance(mapping, dict), "logger.add() only takes a dict"
        if not mapping:
            return
        self._metrics += [(self.step, name, value) for name, value in mapping.items()]

    def write(self):
        if not self._metrics:
            return
        for output in self._outputs:
            output.write(self._metrics)
        self._metrics = []

    def close(self):
        self.write()
        for output in self._outputs:
            if isinstance(output, AsyncOutput):
                output.wait()

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int):
        self._step = value


class BaseOutput:
    def write(self, summaries: list[int, str, Any]):
        pass


class WandbOutput(BaseOutput):
    def __init__(self, name: str, pattern: str = r".*", **kargs):
        import wandb

        self._wandb = wandb
        self.pattern = re.compile(pattern)
        self.run = wandb.init(name=name, **kargs)

    def write(self, summaries: list[int, str, Any]):
        wandb = self._wandb
        bystep = defaultdict(dict)
        for step, name, value in summaries:
            if not self.pattern.match(name):
                continue
            if isinstance(value, TensorArray) and len(value.shape) == 4:
                assert value.shape[1] in [1, 3, 4], "Video shape should be (T, C, H, W)"
                bystep[step][name] = wandb.Video(value, fps=15, format="mp4")
            elif isinstance(value, TensorArray) and len(value.shape) == 3:
                if value.shape[2] in [1, 3, 4]:
                    value = value.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                assert value.shape[0] in [1, 3, 4], "Image shape should be (C, H, W)"
                bystep[step][name] = wandb.Image(value)
            else:
                bystep[step][name] = value
        for step, metrics in bystep.items():
            self.run.log(metrics, step=step)


class AsyncOutput(BaseOutput):
    def __init__(self, callback, parallel=True):
        self._callback = callback
        self._parallel = parallel
        if parallel:
            name = type(self).__name__
            self._worker = concurrent.futures.ThreadPoolExecutor(1, f"logger_{name}_async")
            self._future = None

    def write(self, summaries):
        if self._parallel:
            self._future and self._future.result()
            self._future = self._worker.submit(self._callback, copy.deepcopy(summaries))
        else:
            self._callback(summaries)

    def wait(self):
        if self._parallel and self._future:
            concurrent.futures.wait([self._future])


class TensorboardOutput(AsyncOutput):
    def __init__(self, logdir: str, config: dict[str, Any], pattern: str = r".*"):
        super().__init__(self._write)
        from torch.utils.tensorboard import SummaryWriter

        self.logdir = str(logdir)
        self.pattern = re.compile(pattern)
        self.writer = SummaryWriter(logdir)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
        )

    def _write(self, summaries: list[int, str, Any]):
        for step, name, value in summaries:
            if not self.pattern.match(name):
                continue
            if isinstance(value, TensorArray) and len(value.shape) == 4:
                assert value.shape[1] in [1, 3, 4], "Video shape should be (T, C, H, W)"
                value = value.unsqueeze(0)  # (T, C, H, W) -> (1, T, C, H, W)
                self.writer.add_video(name, value, step, fps=15)
            elif isinstance(value, TensorArray) and len(value.shape) == 3:
                if value.shape[2] in [1, 3, 4]:
                    value = value.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                assert value.shape[0] in [1, 3, 4], "Image shape should be (C, H, W)"
                self.writer.add_image(name, value, step, dataformats="CHW")
            else:
                self.writer.add_scalar(name, value, step)


class JsonlOutput(AsyncOutput):
    def __init__(self, logdir: Path, filename: str, pattern: str = r".*"):
        super().__init__(self._write)
        self.filepath = logdir / Path(filename)
        self.pattern = re.compile(pattern)

    def _write(self, summaries: list[int, str, Any]):
        bystep = defaultdict(dict)
        for step, name, value in summaries:
            if not self.pattern.match(name):
                continue
            if isinstance(value, str):
                bystep[step][name] = value
            elif isinstance(value, (int, float)):
                bystep[step][name] = value
            elif isinstance(value, np.ndarray) and len(value.shape) == 0:
                bystep[step][name] = value.item()
            elif isinstance(value, torch.Tensor) and len(value.shape) == 0:
                bystep[step][name] = value.item()

        lines = "".join([json.dumps({"step": step, **scalars}) + "\n" for step, scalars in bystep.items()])
        with open(self.filepath, "a") as f:
            f.write(lines)
