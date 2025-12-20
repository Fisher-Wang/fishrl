from __future__ import annotations

import io
from typing import Any

import numpy as np
from loguru import logger as log
from matplotlib.figure import Figure


class Ratio:
    """Directly taken from Hafner et al. (2023) implementation:
    https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/embodied/core/when.py#L26
    """

    def __init__(self, ratio: float, pretrain_steps: int = 0):
        if pretrain_steps < 0:
            raise ValueError(f"'pretrain_steps' must be non-negative, got {pretrain_steps}")
        if ratio < 0:
            raise ValueError(f"'ratio' must be non-negative, got {ratio}")
        self._pretrain_steps = pretrain_steps
        self._ratio = ratio
        self._prev = None

    def __call__(self, step: int) -> int:
        if self._ratio == 0:
            return 0
        if self._prev is None:
            self._prev = step
            repeats = int(step * self._ratio)
            if self._pretrain_steps > 0:
                if step < self._pretrain_steps:
                    log.warning(
                        "The number of pretrain steps is greater than the number of current steps. This could lead to "
                        f"a higher ratio than the one specified ({self._ratio}). Setting the 'pretrain_steps' equal to "
                        "the number of current steps."
                    )
                    self._pretrain_steps = step
                repeats = int(self._pretrain_steps * self._ratio)
            return repeats
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats

    def state_dict(self) -> dict[str, Any]:
        return {"_ratio": self._ratio, "_prev": self._prev, "_pretrain_steps": self._pretrain_steps}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._ratio = state_dict["_ratio"]
        self._prev = state_dict["_prev"]
        self._pretrain_steps = state_dict["_pretrain_steps"]
        return self


def fig_to_array(fig: Figure) -> np.ndarray:
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    fig_array = data.reshape((int(h), int(w), -1))
    return fig_array
