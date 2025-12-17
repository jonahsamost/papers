apt update
apt install unzip vim
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv
source .venv/bin/activate
uv pip install torch torchvision tqdm matplotlib ipython numpy datasets h5py


savedir="kitti_data"
mkdir -p -- "$savedir"
wget https://www.dropbox.com/s/rpwlnn6j39jjme4/kitti_data.zip?dl=0 -O $savedir/prednet_kitti_data.zip
unzip $savedir/prednet_kitti_data.zip -d $savedir