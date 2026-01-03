#!/bin/bash

source .venv/bin/activate
export MAX_JOBS=4
cd 3rdparty/flash-attention/
uv pip install . --no-build-isolation
