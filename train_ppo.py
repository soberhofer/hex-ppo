import torch
import gymnasium as gym
from src.hex_env import HexEnv
from src.ppo_agent import PPOAgent, RolloutMemory
from src.ppo_model import ActorCritic # For evaluation
from hex_engine import hexPosition # For evaluation
import numpy as np
import os
import random # For random agent in evaluation

# Hyperparameters
HEX_BOARD_SIZE = 7
LEARNING_RATE = 0.0003
GAMMA = 0.99
K_EPOCHS = 4
EPS_CLIP = 0.2
GAE_LAMBDA = 0.95
NUM_EPISODES = 10000 # Number of episodes to train
SAVE_INTERVAL = 100 # Save model every X episodes
EVAL_INTERVAL = 500 # Evaluate model every X episodes
NUM_EVAL_GAMES = 20 # Number of games for periodic evaluation
MODEL_DIR = "./models"

# --- Random Agent for Evaluation ---
def random_agent_eval(board, action_set):
    return random.choice(action_set)

# --- Evaluation Function (integrated) ---
def evaluate_against_random(ppo_policy, device, num_games=NUM_EVAL_GAMES):
    print(f"\n--- Evaluating PPO Agent vs Random Agent for {num_games} games ---")
    ppo_wins = 0
    game_engine = hexPosition(size=HEX_BOARD_SIZE)
    
    # Ensure ppo_policy is in eval mode
    ppo_policy.eval()

    for i in range(num_games):
        game_engine.reset()
        
        # PPO agent always plays, alternates starting position
        if i % 2 == 0:
            player1 = ppo_policy # PPO is White
            player2 = random_agent_eval # Random is Black
            ppo_is_white = True
        else:
            player1 = random_agent_eval # Random is White
            player2 = ppo_policy # PPO is Black
            ppo_is_white = False

        while game_engine.winner == 0:
            current_board_state = torch.FloatTensor(game_engine.board).unsqueeze(0).unsqueeze(1).to(device)
            
            if game_engine.player == 1: # White's turn
                if player1 == ppo_policy:
                    with torch.no_grad():
                        action_logits, _ = player1(current_board_state)
                        mask = torch.full(action_logits.shape, -float('inf'), device=device)
                        valid_actions_scalar = [game_engine.coordinate_to_scalar(a) for a in game_engine.get_action_space()]
                        for act_s in valid_actions_scalar: mask[0, act_s] = 0
                        masked_action_logits = action_logits + mask
                        probs = torch.nn.functional.softmax(masked_action_logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        action_scalar = dist.sample().item()
                    action = game_engine.scalar_to_coordinates(action_scalar)
                else: # Random agent
                    action = player1(game_engine.board, game_engine.get_action_space())
            else: # Black's turn
                if player2 == ppo_policy:
                    with torch.no_grad():
                        action_logits, _ = player2(current_board_state)
                        mask = torch.full(action_logits.shape, -float('inf'), device=device)
                        valid_actions_scalar = [game_engine.coordinate_to_scalar(a) for a in game_engine.get_action_space()]
                        for act_s in valid_actions_scalar: mask[0, act_s] = 0
                        masked_action_logits = action_logits + mask
                        probs = torch.nn.functional.softmax(masked_action_logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        action_scalar = dist.sample().item()
                    action = game_engine.scalar_to_coordinates(action_scalar)
                else: # Random agent
                    action = player2(game_engine.board, game_engine.get_action_space())
            
            game_engine.move(action)
            game_engine.evaluate()

        if (game_engine.winner == 1 and ppo_is_white) or \
           (game_engine.winner == -1 and not ppo_is_white):
            ppo_wins += 1
            
    win_rate = (ppo_wins / num_games) * 100
    print(f"PPO Agent win rate vs Random: {win_rate:.2f}% ({ppo_wins}/{num_games})")
    print("--- Evaluation Finished ---")
    # Set policy back to train mode if it was changed
    ppo_policy.train()
    return win_rate

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

    # Lists to store losses and evaluation results
    policy_losses, value_losses, entropies = [], [], []
    eval_win_rates = []

    print("Starting PPO training for Hex...")
    for i_episode in range(1, NUM_EPISODES + 1):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        current_episode_policy_losses, current_episode_value_losses, current_episode_entropies = [], [], []

        while not done and not truncated:
            valid_actions = info["valid_actions"]
            action, log_prob, value = agent.select_action(state, valid_actions)
            
            next_state, reward, done, truncated, info = env.step(action)

            # Store in memory
            memory.add(state, action, log_prob, reward, done) 

            state = next_state
            episode_reward += reward

            # If game is done, update agent
            if done or truncated:
                p_loss, v_loss, ent = agent.update(memory)
                current_episode_policy_losses.append(p_loss)
                current_episode_value_losses.append(v_loss)
                current_episode_entropies.append(ent)
                memory.clear_memory()
        
        if current_episode_policy_losses: # if update was called
            policy_losses.append(np.mean(current_episode_policy_losses))
            value_losses.append(np.mean(current_episode_value_losses))
            entropies.append(np.mean(current_episode_entropies))

        if i_episode % 100 == 0:
            print(f"Episode {i_episode}, Reward: {episode_reward}", end="")
            if policy_losses: # Print losses if available
                print(f", Avg Policy Loss: {policy_losses[-1]:.4f}, Avg Value Loss: {value_losses[-1]:.4f}, Avg Entropy: {entropies[-1]:.4f}")
            else:
                print()


        # Save model
        if i_episode % SAVE_INTERVAL == 0:
            model_path = os.path.join(MODEL_DIR, f"ppo_hex_agent_episode_{i_episode}.pth")
            torch.save(agent.policy.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        # Periodic evaluation
        if i_episode % EVAL_INTERVAL == 0:
            win_rate = evaluate_against_random(agent.policy, device, NUM_EVAL_GAMES)
            eval_win_rates.append(win_rate)
            # Here you could also save eval_win_rates to a file for plotting

    env.close()
    print("Training finished.")
    # You can print or plot policy_losses, value_losses, entropies, eval_win_rates here

if __name__ == '__main__':
    train()
