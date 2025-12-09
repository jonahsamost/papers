import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from model import MDN_RNN
from build_dataset import SequenceDataset
from load_model import load_vae

TOTAL_STEPS = 100_000
WARMUP_STEPS = 5_000
SAVE_FREQ = 5_000
LOG_FREQ = 100

BATCH_SIZE = 32
LR = 1e-4
RNN_DIR = 'checkpoints/rnn'
os.makedirs(RNN_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = load_vae(device=device)
vae.eval()
for param in vae.parameters():
    param.requires_grad = False

rnn = MDN_RNN().to(device)
rnn = torch.compile(rnn) 

optimizer = optim.Adam(rnn.parameters(), lr=LR)
scaler = GradScaler()
autocast_ctx = autocast(device_type=device.type, dtype=torch.bfloat16)

scheduler1 = LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS
)
scheduler2 = CosineAnnealingLR(
    optimizer, T_max=TOTAL_STEPS - WARMUP_STEPS, eta_min=1e-6
)
scheduler = SequentialLR(
    optimizer, schedulers=[scheduler1, scheduler2], milestones=[WARMUP_STEPS]
)

dataset = SequenceDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True)
data_iter = iter(dataloader)

print(f"Starting RNN training for {TOTAL_STEPS} steps...")
rnn.train()
current_step = 0

while current_step < TOTAL_STEPS:
    try:
        obs_seq, action_seq = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        obs_seq, action_seq = next(data_iter)

    obs_seq = obs_seq.to(device)
    action_seq = action_seq.to(device)

    optimizer.zero_grad()

    with autocast_ctx:
        # A. VAE Encode (Frozen)
        b, s, c, h, w = obs_seq.shape
        flat_obs = obs_seq.view(-1, c, h, w)

        with torch.no_grad():
            mu, logvar = vae.encode(flat_obs)
            z = vae.reparameterize(mu, logvar)
            z_seq = z.view(b, s, -1)

        z_input = z_seq[:, :-1, :]
        action_input = action_seq[:, :-1, :] 
        z_target = z_seq[:, 1:, :]

        # B. RNN Forward
        pi, sigma, mu_pred, _ = rnn(z_input, action_input)
        
        if current_step < WARMUP_STEPS:
            pi_exp = pi.unsqueeze(-1) 
            avg_mu = torch.sum(pi_exp * mu_pred, dim=2) 
            loss = F.mse_loss(avg_mu, z_target)
            loss_type = "MSE"
        else:
            loss = rnn.mdn_loss_fn(pi, sigma, mu_pred, z_target)
            loss_type = "MDN"

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    
    scheduler.step()
    current_step += 1

    if current_step % LOG_FREQ == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f'Step {current_step}/{TOTAL_STEPS} | [{loss_type}] Loss: {loss.item():.4f} | LR: {lr:.2e}')

    if current_step % SAVE_FREQ == 0:
        print(f"Saving checkpoint at step {current_step}...")
        torch.save(rnn.state_dict(), f'{RNN_DIR}/rnn_step_{current_step}.pth')

print("Training Complete.")
torch.save(rnn.state_dict(), f'{RNN_DIR}/rnn_final.pth')
