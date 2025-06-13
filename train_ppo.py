import torch
import gymnasium as gym
from src.hex_env import HexEnv
from src.ppo_agent import PPOAgent, RolloutMemory
from src.ppo_model import ActorCritic # For evaluation
from hex_engine import hexPosition # For evaluation
from torch.optim.lr_scheduler import StepLR # Import StepLR
import numpy as np
import os
import random # For random agent in evaluation and mixed training
import time

# Hyperparameters
HEX_BOARD_SIZE = 7
INITIAL_LEARNING_RATE = 0.01 
GAMMA = 0.99
K_EPOCHS = 10 
EPS_CLIP = 0.2
GAE_LAMBDA = 0.95 

MAX_TOTAL_TIMESTEPS = 1000000  
TIMESTEPS_PER_BATCH = 2048   
UPDATES_PER_EVAL = 50        
UPDATES_PER_SAVE = 250       
LR_SCHEDULER_STEP_SIZE = 50 
LR_SCHEDULER_GAMMA = 0.9    

RANDOM_OPPONENT_RATIO = 0.2 # Play against random opponent for this fraction of episodes

NUM_EVAL_GAMES = 100 
MODEL_DIR = "./models"

# --- Random Agent for Evaluation & Mixed Training ---
def random_opponent_action_logic(game_engine_instance): 
    action_set_tuples = game_engine_instance.get_action_space()
    if not action_set_tuples: # Should not happen in a valid game state before end
        return None # Or handle error appropriately
    chosen_coords = random.choice(action_set_tuples)
    return game_engine_instance.coordinate_to_scalar(chosen_coords)

# --- Evaluation Function (integrated) ---
def evaluate_against_random(ppo_policy_net, device, num_games=NUM_EVAL_GAMES):
    print(f"\n--- Evaluating PPO Agent vs Random Agent for {num_games} games ---")
    ppo_wins = 0
    game_engine = hexPosition(size=HEX_BOARD_SIZE)
    ppo_policy_net.eval() 

    for i in range(num_games):
        game_engine.reset()
        if i % 2 == 0:
            current_player1_is_ppo = True
            ppo_plays_as_player = 1 
        else:
            current_player1_is_ppo = False
            ppo_plays_as_player = -1 

        while game_engine.winner == 0:
            current_board_for_nn = torch.FloatTensor(game_engine.board).unsqueeze(0).unsqueeze(1).to(device)
            action_coords = None
            is_ppo_turn_now = (game_engine.player == 1 and current_player1_is_ppo) or \
                              (game_engine.player == -1 and not current_player1_is_ppo)

            if is_ppo_turn_now:
                with torch.no_grad():
                    action_logits, _ = ppo_policy_net(current_board_for_nn) 
                    valid_actions_tuples = game_engine.get_action_space()
                    valid_actions_scalar = [game_engine.coordinate_to_scalar(a) for a in valid_actions_tuples]
                    mask = torch.full(action_logits.shape, -float('inf'), device=device)
                    if valid_actions_scalar: 
                        mask[0, valid_actions_scalar] = 0
                    masked_action_logits = action_logits + mask
                    probs = torch.nn.functional.softmax(masked_action_logits, dim=-1)
                    if torch.isnan(probs).any():
                        print("Warning: NaN in probabilities during evaluation, choosing random valid action.")
                        chosen_action_tuple = random.choice(valid_actions_tuples) if valid_actions_tuples else game_engine.scalar_to_coordinates(0) # Fallback
                        action_coords = chosen_action_tuple
                    else:
                        dist = torch.distributions.Categorical(probs)
                        action_scalar_ppo = dist.sample().item()
                        action_coords = game_engine.scalar_to_coordinates(action_scalar_ppo)
            else: 
                action_coords = random.choice(game_engine.get_action_space()) # random_agent_eval simplified
            
            if action_coords is None: # Fallback if something went wrong
                 valid_actions = game_engine.get_action_space()
                 action_coords = random.choice(valid_actions) if valid_actions else (0,0)


            game_engine.move(action_coords)
            game_engine.evaluate()

        if game_engine.winner == ppo_plays_as_player:
            ppo_wins += 1
            
    win_rate = (ppo_wins / num_games) * 100
    print(f"PPO Agent win rate vs Random: {win_rate:.2f}% ({ppo_wins}/{num_games})")
    print("--- Evaluation Finished ---")
    ppo_policy_net.train() 
    return win_rate

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")  # User requested to keep forced CPU. Comment out to use detected device.
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
    print(f"Playing against random opponent for {RANDOM_OPPONENT_RATIO*100}% of episodes.")

    total_timesteps_collected = 0
    num_updates = 0
    all_episode_rewards = [] 
    state, info = env.reset()
    current_episode_reward_accumulator = 0.0
    
    # Determine opponent for the first episode
    opponent_type = "self"
    ppo_agent_player_id = 1 # Default, will be set if opponent is random
    if random.random() < RANDOM_OPPONENT_RATIO:
        opponent_type = "random"
        ppo_agent_player_id = random.choice([1, -1])
        print(f"New episode: PPO vs Random. PPO plays as Player {ppo_agent_player_id}")
    else:
        print("New episode: PPO vs Self.")

    while total_timesteps_collected < MAX_TOTAL_TIMESTEPS:
        for _ in range(TIMESTEPS_PER_BATCH):
            valid_actions = info["valid_actions"]
            action_scalar_to_env = -1 # Placeholder
            
            is_ppo_turn_for_memory = False
            current_player_in_game = env.hex_game.player # Player whose turn it is in hex_engine

            if opponent_type == "self":
                is_ppo_turn_for_memory = True
                action_scalar_ppo, log_prob_ppo, _ = agent.select_action(state, valid_actions)
                action_scalar_to_env = action_scalar_ppo
            elif opponent_type == "random":
                if current_player_in_game == ppo_agent_player_id: # PPO's turn
                    is_ppo_turn_for_memory = True
                    action_scalar_ppo, log_prob_ppo, _ = agent.select_action(state, valid_actions)
                    action_scalar_to_env = action_scalar_ppo
                else: # Random opponent's turn
                    is_ppo_turn_for_memory = False
                    action_scalar_to_env = random_opponent_action_logic(env.hex_game)
            
            next_state, step_reward, done, truncated, next_info = env.step(action_scalar_to_env)

            if is_ppo_turn_for_memory: # Only add to memory if PPO made the move
                memory.add(state, action_scalar_to_env, log_prob_ppo, step_reward, done or truncated)
            
            current_episode_reward_accumulator += step_reward # Accumulate for episode outcome logging
            
            state = next_state
            info = next_info
            total_timesteps_collected += 1

            if done or truncated:
                all_episode_rewards.append(current_episode_reward_accumulator)
                current_episode_reward_accumulator = 0.0 
                state, info = env.reset()
                # Determine opponent for the new episode
                if random.random() < RANDOM_OPPONENT_RATIO:
                    opponent_type = "random"
                    ppo_agent_player_id = random.choice([1, -1])
                    # print(f"New episode: PPO vs Random. PPO plays as Player {ppo_agent_player_id}") # Can be verbose
                else:
                    opponent_type = "self"
                    # print("New episode: PPO vs Self.") # Can be verbose
            
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
