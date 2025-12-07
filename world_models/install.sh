uv venv
source .venv/bin/activate
sudo apt update
sudo apt install -y ffmpeg xvfb swig python3-opengl
uv pip install gymnasium pyvirtualdisplay swig gymnasium[box2d] torch numpy ipython cma matplotlib