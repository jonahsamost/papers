import torch
import os
import glob
from model import VAE, MDN_RNN


VAE_CHECKPOINTS = 'checkpoints/vae'
RNN_CHECKPOINTS = 'checkpoints/rnn'


def load_rnn(device="cuda"):
    checkpoint_path = f'{RNN_CHECKPOINTS}/rnn*.pth'
    return load_model(checkpoint_path, device, MDN_RNN)


def load_vae(device="cuda"):
    checkpoint_path = f'{VAE_CHECKPOINTS}/vae*.pth'
    return load_model(checkpoint_path, device, VAE)


def load_model(checkpoint_path, device, model_class):
    list_of_files = glob.glob(checkpoint_path)
    if not list_of_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading checkpoint: {latest_file}")

    checkpoint = torch.load(latest_file, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[10:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model = model_class()
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model