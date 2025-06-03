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

MAX_TOTAL_TIMESTEPS = 2000000  # Total timesteps to train for
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
def evaluate_against_random(ppo_policy_net, device, num_games=NUM_EVAL_GAMES):
    print(f"\n--- Evaluating PPO Agent vs Random Agent for {num_games} games ---")
    ppo_wins = 0
    game_engine = hexPosition(size=HEX_BOARD_SIZE)
    ppo_policy_net.eval() # Ensure ppo_policy_net is in eval mode

    for i in range(num_games):
        game_engine.reset()
        if i % 2 == 0:
            current_player1_is_ppo = True
            ppo_plays_as_player = 1 # PPO is player 1 (White)
        else:
            current_player1_is_ppo = False
            ppo_plays_as_player = -1 # PPO is player -1 (Black)

        while game_engine.winner == 0:
            current_board_for_nn = torch.FloatTensor(game_engine.board).unsqueeze(0).unsqueeze(1).to(device)
            action_coords = None

            is_ppo_turn_now = (game_engine.player == 1 and current_player1_is_ppo) or \
                              (game_engine.player == -1 and not current_player1_is_ppo)

            if is_ppo_turn_now:
                with torch.no_grad():
                    action_logits, _ = ppo_policy_net(current_board_for_nn) # Get logits from model.forward()
                    
                    valid_actions_tuples = game_engine.get_action_space()
                    valid_actions_scalar = [game_engine.coordinate_to_scalar(a) for a in valid_actions_tuples]

                    mask = torch.full(action_logits.shape, -float('inf'), device=device)
                    if valid_actions_scalar: # Ensure there are valid actions
                        mask[0, valid_actions_scalar] = 0
                    
                    masked_action_logits = action_logits + mask
                    probs = torch.nn.functional.softmax(masked_action_logits, dim=-1)
                    
                    # Check for NaN in probs, can happen if all logits are -inf (no valid moves, though env should prevent this)
                    if torch.isnan(probs).any():
                        # Fallback to random action if probs are NaN (should ideally not happen if valid_actions is managed well)
                        print("Warning: NaN in probabilities during evaluation, choosing random action.")
                        chosen_action_tuple = random.choice(valid_actions_tuples)
                        action_coords = chosen_action_tuple
                    else:
                        dist = torch.distributions.Categorical(probs)
                        action_scalar = dist.sample().item()
                        action_coords = game_engine.scalar_to_coordinates(action_scalar)
            else: # Random agent's turn
                action_coords = random_agent_eval(game_engine.board, game_engine.get_action_space())
            
            if action_coords is None: # Should not happen if logic is correct
                print("Error: action_coords is None. Defaulting to random.")
                action_coords = random.choice(game_engine.get_action_space())

            game_engine.move(action_coords)
            game_engine.evaluate()

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
    elif torch.backends.mps.is_available(): 
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
    
    all_episode_rewards = [] 

    state, info = env.reset()
    current_episode_reward_accumulator = 0.0 

    while total_timesteps_collected < MAX_TOTAL_TIMESTEPS:
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
                current_episode_reward_accumulator = 0.0 
                state, info = env.reset()
            
            if total_timesteps_collected >= MAX_TOTAL_TIMESTEPS:
                break
        
        if len(memory.states) > 0: 
            p_loss, v_loss, ent = agent.update(memory)
            memory.clear_memory()
            num_updates += 1
            lr_scheduler.step() 

            if num_updates % 10 == 0: 
                current_lr = agent.optimizer.param_groups[0]['lr']
                avg_ep_reward_str = ""
                if len(all_episode_rewards) > 0:
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
