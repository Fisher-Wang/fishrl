from __future__ import annotations

from typing import Any

import zmq
from loguru import logger as log
from torch import Tensor

from .base import BaseVecEnv


class Server:
    def __init__(self, env: BaseVecEnv, port: int, host: str = "localhost"):
        self.env = env
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        log.info(f"[Server] Bound to {host}:{port}")

    def _close(self):
        self.socket.close()
        self.context.term()
        log.info("[Server] Closed")

    def run(self):
        try:
            while True:
                cmd, data = self.socket.recv_pyobj()
                log.debug(f"[Server] Received command: {cmd}")
                if cmd == "init":
                    self.socket.send_pyobj([
                        self.env.device,
                        self.env.num_envs,
                        self.env.single_action_space,
                        self.env.action_space,
                        self.env.single_observation_space,
                        self.env.observation_space,
                    ])
                elif cmd == "reset":
                    seed, options = data
                    rst = self.env.reset(seed=seed, options=options)
                    self.socket.send_pyobj(rst)
                elif cmd == "step":
                    action = data.to(self.env.device)
                    rst = self.env.step(action)
                    self.socket.send_pyobj(rst)
                elif cmd == "close":
                    break
                else:
                    raise ValueError(f"[Server] Invalid command: {cmd}")
        except KeyboardInterrupt:
            log.info("[Server] Interrupted by user.")
        except Exception as e:
            raise "[Server] Error" from e
        finally:
            self._close()


class ClientEnv(BaseVecEnv):
    def __init__(self, port: int, host: str = "localhost"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        log.info(f"[Client] Connecting to {host}:{port}")
        self.socket.connect(f"tcp://{host}:{port}")
        log.info(f"[Client] Connected to {host}:{port}")
        self.socket.send_pyobj(("init", None))
        (
            self._device,
            self._num_envs,
            self._single_action_space,
            self._action_space,
            self._single_obs_space,
            self._obs_space,
        ) = self.socket.recv_pyobj()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.socket.send_pyobj(("reset", (seed, options)))
        return self.socket.recv_pyobj()

    def step(self, action: Tensor):
        self.socket.send_pyobj(("step", action.cpu()))
        return self.socket.recv_pyobj()

    def render(self):
        raise NotImplementedError

    def close(self):
        self.socket.send_pyobj(("close", None))
        self.socket.close()
        self.context.term()

    @property
    def single_action_space(self):
        return self._single_action_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def single_observation_space(self):
        return self._single_obs_space

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def device(self):
        return self._device
