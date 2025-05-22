import torch
import gymnasium as gym
from src.hex_env import HexEnv
from src.ppo_agent import PPOAgent, RolloutMemory
import numpy as np
import os

# Hyperparameters
HEX_BOARD_SIZE = 7
LEARNING_RATE = 0.0003
GAMMA = 0.99
K_EPOCHS = 4
EPS_CLIP = 0.2
GAE_LAMBDA = 0.95
NUM_EPISODES = 10000 # Number of episodes to train
SAVE_INTERVAL = 100 # Save model every X episodes
MODEL_DIR = "./models"

def train():
    # Determine the device to use (CUDA if available, else MPS if available, else CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for training.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU device for training.")

    # Create environment
    env = HexEnv(size=HEX_BOARD_SIZE)
    
    # Get observation and action space sizes
    obs_shape = env.observation_space.shape
    action_space_size = env.action_space.n

    # Initialize PPO agent and memory, passing the device
    agent = PPOAgent(obs_shape, action_space_size, LEARNING_RATE, GAMMA, K_EPOCHS, EPS_CLIP, GAE_LAMBDA, device)
    memory = RolloutMemory()

    # Create directory for saving models
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Starting PPO training for Hex...")
    for i_episode in range(1, NUM_EPISODES + 1):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not done and not truncated:
            valid_actions = info["valid_actions"]
            action, log_prob, value = agent.select_action(state, valid_actions)
            
            next_state, reward, done, truncated, info = env.step(action)

            # Store in memory
            memory.add(state, action, log_prob, reward, done) # done is equivalent to is_terminal here

            state = next_state
            episode_reward += reward

            # If game is done, update agent
            if done or truncated:
                agent.update(memory)
                memory.clear_memory()
        
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}, Reward: {episode_reward}")

        # Save model
        if i_episode % SAVE_INTERVAL == 0:
            model_path = os.path.join(MODEL_DIR, f"ppo_hex_agent_episode_{i_episode}.pth")
            # Save model state dict to CPU to ensure portability
            torch.save(agent.policy.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    env.close()
    print("Training finished.")

if __name__ == '__main__':
    train()
