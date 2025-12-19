apt update
apt install -y vim unzip
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt