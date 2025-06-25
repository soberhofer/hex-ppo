import torch
import numpy as np
import os
from hex_engine import hexPosition # For scalar_to_coordinates
from src.ppo_model import ActorCritic # Assuming src is in PYTHONPATH or accessible
from torch.distributions import Categorical # Import Categorical

#from train_ppo import HexStrategicAgent

# Define the board size (must match the size used during training)
HEX_BOARD_SIZE = 7 
MODEL_DIR = "Group_G" # Directory where models are saved
SUB_DIR = "agent"
MODEL_PATH = "ppo_voodoo_agent.pth"

# Global variable to hold the loaded model
_ppo_model = None

def load_ppo_model(model_path):
    """Loads the trained PPO model."""
    global _ppo_model
    if _ppo_model is None:
        # Initialize a dummy hexPosition to get obs_shape and action_space_size
        dummy_game = hexPosition(size=HEX_BOARD_SIZE)
        obs_shape = (HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        action_space_size = HEX_BOARD_SIZE * HEX_BOARD_SIZE

        _ppo_model = ActorCritic(obs_shape, action_space_size)

        # could lead to problems, since training on gpu enabled
        _ppo_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        _ppo_model.eval() # Set to evaluation mode
        print(f"PPO model loaded from {model_path}")
    return _ppo_model

def ppo_agent_logic(board, action_set, path= os.path.join(".", MODEL_DIR, SUB_DIR, MODEL_PATH)):
    """
    The logic for the PPO agent to select an action.
    This function will be called by the hex_engine.
    """
    # Ensure the model is loaded
    # You might want to specify the exact model path here, e.g., the latest one
    # For demonstration, let's assume we load the latest saved model
    
    # Find the latest model file
    #list_of_files = os.listdir(MODEL_DIR)
    # full_paths = [os.path.join(MODEL_DIR, f) for f in list_of_files if f.startswith("ppo_hex_agent_episode_") and f.endswith(".pth")]

    """if not full_paths:
        print("No trained PPO model found. Using random action.")
        from random import choice
        return choice(action_set)
    """

    # latest_file = max(full_paths, key=os.path.getctime)
    model = load_ppo_model(path)

    # Convert board to tensor
    obs_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(1) # Add batch and channel dimensions

    # Get action logits from the model
    with torch.no_grad():
        action_logits, _ = model(obs_tensor)

    # Mask invalid actions
    mask = torch.full(action_logits.shape, -float('inf'))
    valid_actions_scalar = [hexPosition(HEX_BOARD_SIZE).coordinate_to_scalar(a) for a in action_set]
    for action_scalar in valid_actions_scalar:
        mask[0, action_scalar] = 0
    
    masked_action_logits = action_logits + mask
    
    probs = torch.nn.functional.softmax(masked_action_logits, dim=-1)
    
    # Sample action from the distribution
    dist = Categorical(probs)
    action_scalar = dist.sample().item()

    # Convert scalar action back to coordinates
    chosen_coordinates = hexPosition(HEX_BOARD_SIZE).scalar_to_coordinates(action_scalar)
    
    return chosen_coordinates

def greedy_agent_logic(board, action_set, path= os.path.join(".", MODEL_DIR, MODEL_PATH)):

    # latest_file = max(full_paths, key=os.path.getctime)
    model = load_ppo_model(path)

    # Convert board to tensor
    obs_tensor = torch.FloatTensor(board).unsqueeze(0).unsqueeze(1) # Add batch and channel dimensions

    # Get action logits from the model
    with torch.no_grad():
        action_logits, _ = model(obs_tensor)

    # Mask invalid actions
    mask = torch.full(action_logits.shape, -float('inf'))
    valid_actions_scalar = [hexPosition(HEX_BOARD_SIZE).coordinate_to_scalar(a) for a in action_set]
    for action_scalar in valid_actions_scalar:
        mask[0, action_scalar] = 0

    masked_action_logits = action_logits + mask


    probs = torch.nn.functional.softmax(masked_action_logits, dim=-1)
    best_action_index = torch.argmax(probs).item()

    # Convert scalar action back to coordinates
    chosen_coordinates = hexPosition(HEX_BOARD_SIZE).scalar_to_coordinates(best_action_index)

    return chosen_coordinates


# TODO: adapt other agents with player id so that it can be used
"""
def select_action(board_state, actions, player_id):
    '''
    Determines the "best" legal move using a simple heuristic evaluation.

    board_state: the board state (2D array or similar structure),
                 0 for empty, 1 for player 1, 2 for player 2 (or adapt as needed).
    player_id:   the agent's ID (1 or 2).

    Returns: (row, col) as the chosen move.
    '''

    agent = HexStrategicAgent(7)
    legal_moves = agent.get_legal_moves(board_state)
    if not legal_moves:
        return None  # No legal moves available

    best_move = None
    best_score = float('-inf')

    for move in legal_moves:
        sim_board = agent.simulate_move(board_state, move, player_id)
        score = agent.evaluate_position(sim_board, player_id)
        if score > best_score:
            best_score = score
            best_move = move

    return best_move
"""