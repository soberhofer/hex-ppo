import torch
import os
import random
from hex_engine import hexPosition
from submission.ppo_agent_facade import ppo_agent_logic # To use the PPO agent

# --- Configuration ---
NUM_EVAL_GAMES = 100  # Number of games to play for evaluation
HEX_BOARD_SIZE = 7   # Must match the board size used for training
MODEL_DIR = "./models" # Directory where trained models are saved
# Specify a model path or let it pick the latest.
# For specific model: LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_hex_agent_episode_1000.pth")
LATEST_MODEL_PATH = None # Set to None to automatically find the latest

# --- Random Agent ---
def random_agent(board, action_set):
    """A simple agent that picks a random valid move."""
    return random.choice(action_set)

# --- Evaluation Function ---
def evaluate_ppo_agent(num_games=NUM_EVAL_GAMES):
    print(f"Starting evaluation of PPO agent against random agent for {num_games} games...")

    # Load the PPO agent (ppo_agent_logic will handle loading the model)
    # The ppo_agent_logic in facade will load the latest model if LATEST_MODEL_PATH is not set in its own logic
    # For this script, we ensure ppo_agent_logic uses a specific or latest model by pre-loading it if needed.
    # However, ppo_agent_logic already finds the latest model.

    ppo_wins = 0
    random_wins = 0
    draws = 0 # Hex doesn't have draws, but good practice for other games

    game_engine = hexPosition(size=HEX_BOARD_SIZE)

    for i in range(num_games):
        game_engine.reset()
        
        # Alternate who starts
        if i % 2 == 0:
            player1 = ppo_agent_logic
            player1_name = "PPO Agent"
            player2 = random_agent
            player2_name = "Random Agent"
        else:
            player1 = random_agent
            player1_name = "Random Agent"
            player2 = ppo_agent_logic
            player2_name = "PPO Agent"

        print(f"\n--- Game {i+1}/{num_games} ---")
        print(f"{player1_name} (White) vs {player2_name} (Black)")
        
        current_player_is_ppo = (player1 == ppo_agent_logic)

        while game_engine.winner == 0:
            # game_engine.print() # Optional: print board each turn
            
            if game_engine.player == 1: # White's turn
                action = player1(game_engine.board, game_engine.get_action_space())
            else: # Black's turn
                action = player2(game_engine.board, game_engine.get_action_space())
            
            game_engine.move(action)
            game_engine.evaluate() # Check for winner

        # Determine winner
        if game_engine.winner == 1: # White won
            if player1 == ppo_agent_logic:
                ppo_wins += 1
                print(f"Game {i+1}: PPO Agent (White) won.")
            else:
                random_wins += 1
                print(f"Game {i+1}: Random Agent (White) won.")
        elif game_engine.winner == -1: # Black won
            if player2 == ppo_agent_logic:
                ppo_wins += 1
                print(f"Game {i+1}: PPO Agent (Black) won.")
            else:
                random_wins += 1
                print(f"Game {i+1}: Random Agent (Black) won.")
        else: # Should not happen in Hex
            draws +=1
            print(f"Game {i+1}: Draw.")
            
    print("\n--- Evaluation Summary ---")
    print(f"Total games played: {num_games}")
    print(f"PPO Agent wins: {ppo_wins} ({ (ppo_wins/num_games)*100 :.2f}%)")
    print(f"Random Agent wins: {random_wins} ({ (random_wins/num_games)*100 :.2f}%)")
    if draws > 0:
        print(f"Draws: {draws} ({ (draws/num_games)*100 :.2f}%)")

if __name__ == '__main__':
    # Ensure the ppo_agent_logic can find the model
    # It automatically loads the latest from MODEL_DIR
    evaluate_ppo_agent()
