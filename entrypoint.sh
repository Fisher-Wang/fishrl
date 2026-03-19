#!/bin/bash
uv pip install -e ".[dmc,benchmark,dmlab]" ${HOME}/pkg/deepmind_lab-1.0-py3-none-any.whl
exec "$@"
