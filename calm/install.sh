apt update
apt install unzip vim
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv
source .venv/bin/activate
uv pip install torch ipython datasets hf_transfer transformers zstandard