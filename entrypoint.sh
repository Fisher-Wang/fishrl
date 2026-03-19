#!/bin/bash
uv pip install --reinstall -e ".[dmc,benchmark,dmlab]" ~/pkg/deepmind_lab-1.0-py3-none-any.whl
exec "$@"
