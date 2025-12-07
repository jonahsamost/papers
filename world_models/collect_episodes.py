import gymnasium as gym

import traceback
import os
import numpy as np
from multiprocessing import Pool, cpu_count

NUM_ROLLOUTS = 10_000
DATA_DIR = "data/rollouts"
os.makedirs(DATA_DIR, exist_ok=True)

def collect_one_episode(episode_id):
    try:
        env = gym.make(
            "CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95,
            domain_randomize=False, continuous=True
        )
        
        obs, _ = env.reset()
        done = False
        
        episode_obs = []
        episode_actions = []
        episode_dones = []
        
        cnt = 0
        action = env.action_space.sample()
        while not done:
            action[0] += np.random.normal(0, 0.2) # Steer noise
            action[1] += np.random.normal(0, 0.1) # Gas noise
            action[2] = 0 # Disable brake for collection usually helps coverage
            
            action[0] = np.clip(action[0], -1.0, 1.0) # Steer
            action[1] = np.clip(action[1], 0.0, 1.0)  # Gas

            episode_obs.append(obs)
            episode_actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            cnt += 1
            done = terminated or truncated
            episode_dones.append(done)
            if cnt > 500:
                break

        np.savez_compressed(
            f"{DATA_DIR}/rollout_{episode_id}.npz",
            obs=np.array(episode_obs, dtype=np.uint8), 
            actions=np.array(episode_actions, dtype=np.float32),
            dones=np.array(episode_dones, dtype=bool)
        )
        
        env.close()
        return True
    except Exception as e:
        print(f'Exception: {e}')
        traceback.print_exc()
        return False

def run_parallel_collection(total_episodes):
    num_processes = cpu_count() // 2
    print(f"Starting data collection on {num_processes} cores...")
    with Pool(processes=num_processes) as pool:
        cnt = 0
        for _ in pool.imap_unordered(collect_one_episode, range(total_episodes)):
            cnt += 1
            if cnt % 10 == 0:
                print(f"Collected {cnt}/{total_episodes} episodes...", end='\r')


run_parallel_collection(10000)