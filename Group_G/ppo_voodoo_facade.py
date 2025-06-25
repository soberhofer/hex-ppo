import os
from hex_engine import hexPosition
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import sys
sys.path.append(os.path.abspath(os.path.join('..')))
HEX_BOARD_SIZE = 7
MODEL_DIR = "agent"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'agent', 'ppo_voodoo_agent.pth')


# Global variable to hold the loaded model
_ppo_model = None


def load_ppo_model(model_path):
    """Loads the trained PPO model."""
    global _ppo_model
    if _ppo_model is None:
        obs_shape = (HEX_BOARD_SIZE, HEX_BOARD_SIZE)
        action_space_size = HEX_BOARD_SIZE * HEX_BOARD_SIZE

        _ppo_model = ActorCritic(obs_shape, action_space_size)

        _ppo_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        _ppo_model.eval() # Set to evaluation mode
        print(f"PPO model loaded from {model_path}")
    return _ppo_model

def ppo_agent_logic(board, action_set, path=os.path.join(MODEL_PATH)):
    """
    The logic for the PPO agent to select an action.
    This function will be called by the hex_engine.
    """
    # Ensure the model is loaded
    # You might want to specify the exact model path here, e.g., the latest one
    print("Check LOADING")
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



class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space_size):
        super(ActorCritic, self).__init__()

        # Assuming obs_shape is (board_size, board_size)
        board_size = obs_shape[0]

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 1x1 Convolution for the residual connection to match channel dimensions
        self.residual_projection = nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0)

        # Calculate the output size of the convolutional layers
        # The size remains the same due to padding=1 and stride=1
        conv_output_size = 64 * board_size * board_size

        # Actor (Policy) network
        self.actor_fc1 = nn.Linear(conv_output_size, 256)
        self.actor_fc2 = nn.Linear(256, action_space_size)

        # Critic (Value) network
        self.critic_fc1 = nn.Linear(conv_output_size, 256)
        self.critic_fc2 = nn.Linear(256, 1)

    def forward(self, obs):
        # obs is expected to be (batch_size, 1, board_size, board_size)
        # Ensure it's float
        obs_float = obs.float()  # Use a different variable name to keep original obs for residual

        # Main path
        x = F.leaky_relu(self.bn1(self.conv1(obs_float)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))

        # Residual connection
        residual = self.residual_projection(obs_float)
        x = x + residual
        x = F.leaky_relu(x)  # Apply ReLU after adding residual

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch

        # Actor
        actor_hidden = F.relu(self.actor_fc1(x))
        action_logits = self.actor_fc2(actor_hidden)

        # Critic
        critic_hidden = F.relu(self.critic_fc1(x))
        value = self.critic_fc2(critic_hidden)

        return action_logits, value

    def act(self, obs):
        action_logits, value = self.forward(obs)

        # Mask invalid actions (assuming invalid actions are represented by -inf or very small negative numbers in logits)
        # This part needs to be handled carefully. For now, assuming all actions in action_space are valid.
        # If invalid actions need to be masked, the environment should provide a mask.

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value

    def evaluate(self, obs, action):
        action_logits, value = self.forward(obs)

        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()  # Ensure entropy is a scalar

        return action_log_probs, value, dist_entropy


def agent_group_G(board, action_set):
    """
    This function serves as the entry point for your trained PPO agent.
    It calls the ppo_agent_logic from ppo_agent_facade to select an action.
    """
    print("Returning ppo_agent_logic")
    return ppo_agent_logic(board, action_set, MODEL_PATH)