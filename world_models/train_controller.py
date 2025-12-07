from poplib import POP3
import gymnasium as gym
import cma
import os
import numpy as np
import torch
from multiprocessing import Pool, cpu_count

from model import VAE, MDN_RNN, Controller
from load_model import load_rnn, load_vae

CHECKPOINT_DIR = 'checkpoints/controller'

NUM_ENVIRON_RUNS = 1000
POPULATION_SIZE = 64 
N_GENERATIONS = 2000
TARGET_REWARD = 900
PARAM_COUNT = 867

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

vae = load_vae(device=device)
vae.eval()

rnn = load_rnn(device=device)
rnn.eval()


def collect_rewards(args):
    candidate_weights, seed = args
    controller = Controller().to(device)
    controller.load_flat_params(candidate_weights)
    controller.eval()
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=True, continuous=True)
    steps = 0
    done = False

    obs, _ = env.reset(seed=seed)
    h = rnn.get_initial_state(batch_size=1)
    cumulative_reward = 0
    with torch.no_grad():
        while not done and steps < NUM_ENVIRON_RUNS:
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0) / 255.0
            obs_tensor = obs_tensor.view(-1, vae.in_channels, vae.in_height_width, vae.in_height_width)

            # vae
            mu, logvar = vae.encode(obs_tensor.detach().clone())
            z = vae.reparameterize(mu, logvar)

            # rnn
            h_state, _ = h
            flat_h = h_state[-1]
            cx = torch.cat([z, flat_h], dim=1)
            action = controller(cx)
            action_np = action.detach().cpu().numpy().flatten()

            # step
            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            cumulative_reward += reward

            # rnn
            z_seq = z.unsqueeze(1)
            action_seq = action.unsqueeze(1)
            _, _, _, h = rnn(z_seq, action_seq, hidden_state=h)

            steps += 1

    return cumulative_reward


def run():
    es = cma.CMAEvolutionStrategy(x0=np.zeros(PARAM_COUNT), sigma0=0.1, inopts={'popsize': POPULATION_SIZE})

    for generation in range(N_GENERATIONS):
        print(f'Running generation: {generation}')
        solutions = es.ask()
        worker_args = [(sol, np.random.randint(10000)) for sol in solutions]
        rewards = []
        for i, wargs in enumerate(worker_args):
            print(f'\trun: {i}')
            rewards.append(collect_rewards(wargs))
        
        rewards = np.array(rewards)
        es.tell(solutions, -rewards)
        
        best_reward = np.max(rewards)
        mean_reward = np.mean(rewards)
        print(f"Gen {generation}: Best: {best_reward:.2f}, Mean: {mean_reward:.2f}")
        
        if best_reward > 400:
            best_idx = np.argmax(rewards)
            best_params = solutions[best_idx]
            np.save(f"{CHECKPOINT_DIR}/controller_gen_{generation}.npy", best_params)
            
        if best_reward > TARGET_REWARD:
            print(f"Solved! Reward {best_reward} > {TARGET_REWARD}")
            break


