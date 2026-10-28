#!/bin/sh

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.12
git clone git@github.com:kworathur/mamba.git
cd mamba
uv venv
source .venv/bin/activate
uv pip install torch_tpu
uv pip install --no-build-isolation .