import torch
import gymnasium as gym
from src.hex_env import HexEnv
from src.ppo_agent import PPOAgent, RolloutMemory
from src.ppo_model import ActorCritic # For evaluation
from hex_engine import hexPosition # For evaluation
from torch.optim.lr_scheduler import StepLR # Import StepLR
import numpy as np
import os
import random # For random agent in evaluation
import time

# Hyperparameters
HEX_BOARD_SIZE = 7
INITIAL_LEARNING_RATE = 0.001 
GAMMA = 0.99
K_EPOCHS = 10 
EPS_CLIP = 0.2
GAE_LAMBDA = 0.95 # Currently not used in advantage calculation, but kept for completeness

MAX_TOTAL_TIMESTEPS = 200000  # Total timesteps to train for
TIMESTEPS_PER_BATCH = 2048   # Timesteps to collect per batch before updating
UPDATES_PER_EVAL = 50        # Evaluate model every X updates (e.g., 50 updates * 2048 steps/update = ~100k steps)
UPDATES_PER_SAVE = 250       # Save model every X updates (e.g., 250 updates * 2048 steps/update = ~500k steps)
# LR Scheduler: step_size is now in terms of number of updates
LR_SCHEDULER_STEP_SIZE = 50 # Decay LR every X updates (e.g. 50 updates)
LR_SCHEDULER_GAMMA = 0.9    # Multiplicative factor of LR decay

NUM_EVAL_GAMES = 100 # Number of games for periodic evaluation
MODEL_DIR = "./models"

# --- Random Agent for Evaluation ---
def random_agent_eval(board, action_set):
    return random.choice(action_set)

# --- Evaluation Function (integrated) ---
def evaluate_against_random(ppo_policy_net, device, num_games=NUM_EVAL_GAMES): # Renamed ppo_policy to ppo_policy_net for clarity
    print(f"\n--- Evaluating PPO Agent vs Random Agent for {num_games} games ---")
    ppo_wins = 0
    game_engine = hexPosition(size=HEX_BOARD_SIZE)
    ppo_policy_net.eval() # Ensure ppo_policy_net is in eval mode

    for i in range(num_games):
        game_engine.reset()
        # PPO agent always plays, alternates starting position
        # Player1 is White, Player2 is Black
        if i % 2 == 0:
            # PPO is White
            current_player1_policy = ppo_policy_net
            current_player2_policy = random_agent_eval
            ppo_plays_as_player = 1 # PPO is player 1 (White)
        else:
            # PPO is Black
            current_player1_policy = random_agent_eval
            current_player2_policy = ppo_policy_net
            ppo_plays_as_player = -1 # PPO is player -1 (Black)

        while game_engine.winner == 0:
            current_board_for_nn = torch.FloatTensor(game_engine.board).unsqueeze(0).unsqueeze(1).to(device)
            
            if game_engine.player == 1: # White's turn
                if current_player1_policy == ppo_policy_net:
                    with torch.no_grad():
                        # Use the act method of the ActorCritic model instance
                        action_scalar, _, _ = current_player1_policy.act(current_board_for_nn) 
                    action_coords = game_engine.scalar_to_coordinates(action_scalar)
                else: # Random agent
                    action_coords = current_player1_policy(game_engine.board, game_engine.get_action_space())
            else: # Black's turn (player == -1)
                if current_player2_policy == ppo_policy_net:
                    with torch.no_grad():
                        action_scalar, _, _ = current_player2_policy.act(current_board_for_nn)
                    action_coords = game_engine.scalar_to_coordinates(action_scalar)
                else: # Random agent
                    action_coords = current_player2_policy(game_engine.board, game_engine.get_action_space())
            
            game_engine.move(action_coords)
            game_engine.evaluate()

        # Check if PPO won
        if game_engine.winner == ppo_plays_as_player:
            ppo_wins += 1
            
    win_rate = (ppo_wins / num_games) * 100
    print(f"PPO Agent win rate vs Random: {win_rate:.2f}% ({ppo_wins}/{num_games})")
    print("--- Evaluation Finished ---")
    ppo_policy_net.train() # Set policy back to train mode
    return win_rate

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for training.")
    elif torch.backends.mps.is_available(): # Check for MPS
        device = torch.device("mps")
        print("Using MPS device for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU device for training.")
    device = torch.device("cpu")  # User requested to keep forced CPU
    print(f"Training will run on: {device}")

    env = HexEnv(size=HEX_BOARD_SIZE)
    obs_shape = env.observation_space.shape
    action_space_size = env.action_space.n

    agent = PPOAgent(obs_shape, action_space_size, INITIAL_LEARNING_RATE, GAMMA, K_EPOCHS, EPS_CLIP, GAE_LAMBDA, device)
    memory = RolloutMemory()
    lr_scheduler = StepLR(agent.optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Starting PPO training for Hex for {MAX_TOTAL_TIMESTEPS} timesteps...")
    print(f"Batch size: {TIMESTEPS_PER_BATCH}, Updates per batch: {K_EPOCHS}")

    total_timesteps_collected = 0
    num_updates = 0
    
    all_episode_rewards = [] # To store rewards of all completed episodes for averaging

    state, info = env.reset()
    current_episode_reward_accumulator = 0.0 # Accumulates reward for the current episode

    while total_timesteps_collected < MAX_TOTAL_TIMESTEPS:
        # Collect TIMESTEPS_PER_BATCH
        for _ in range(TIMESTEPS_PER_BATCH):
            valid_actions = info["valid_actions"]
            action_scalar, log_prob, _ = agent.select_action(state, valid_actions) 
            
            next_state, step_reward, done, truncated, next_info = env.step(action_scalar)

            memory.add(state, action_scalar, log_prob, step_reward, done or truncated)
            current_episode_reward_accumulator += step_reward
            
            state = next_state
            info = next_info
            total_timesteps_collected += 1

            if done or truncated:
                all_episode_rewards.append(current_episode_reward_accumulator)
                current_episode_reward_accumulator = 0.0 # Reset for next episode
                state, info = env.reset()
            
            if total_timesteps_collected >= MAX_TOTAL_TIMESTEPS:
                break
        
        # Batch is full (or training is ending), update the agent
        if len(memory.states) > 0: # Ensure memory is not empty if MAX_TOTAL_TIMESTEPS is not multiple of TIMESTEPS_PER_BATCH
            p_loss, v_loss, ent = agent.update(memory)
            memory.clear_memory()
            num_updates += 1
            lr_scheduler.step() 

            # Logging
            if num_updates % 10 == 0: # Log losses every 10 updates
                current_lr = agent.optimizer.param_groups[0]['lr']
                avg_ep_reward_str = ""
                # Log average reward of episodes completed in the last ~TIMESTEPS_PER_BATCH steps
                # This is an approximation. For more precise per-batch episode rewards,
                # one would need to track episodes ending within the batch collection.
                # For now, using the last N episodes from all_episode_rewards.
                if len(all_episode_rewards) > 0:
                    # Log average of last, say, 50 episodes if available, or all if fewer
                    lookback_episodes = min(50, len(all_episode_rewards))
                    avg_recent_ep_reward = np.mean(all_episode_rewards[-lookback_episodes:])
                    avg_ep_reward_str = f", Avg Ep Reward (last ~{lookback_episodes}): {avg_recent_ep_reward:.2f}"

                print(f"Update {num_updates}, Timesteps: {total_timesteps_collected}, LR: {current_lr:.7f}{avg_ep_reward_str}")
                print(f"  Losses: Policy: {p_loss:.4f}, Value: {v_loss:.4f}, Entropy: {ent:.4f}")

            if num_updates > 0 and num_updates % UPDATES_PER_EVAL == 0:
                evaluate_against_random(agent.policy, device, NUM_EVAL_GAMES)

            if num_updates > 0 and num_updates % UPDATES_PER_SAVE == 0:
                model_path = os.path.join(MODEL_DIR, f"ppo_hex_agent_update_{num_updates}_steps_{total_timesteps_collected}.pth")
                torch.save(agent.policy.state_dict(), model_path)
                print(f"Model saved to {model_path}")
        
        if total_timesteps_collected >= MAX_TOTAL_TIMESTEPS:
            break

    env.close()
    print("Training finished.")

if __name__ == '__main__':
    train()
