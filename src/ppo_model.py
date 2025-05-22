import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space_size):
        super(ActorCritic, self).__init__()
        
        # Assuming obs_shape is (board_size, board_size)
        board_size = obs_shape[0]

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
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
        obs = obs.float()
        
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch

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
        dist_entropy = dist.entropy().mean() # Ensure entropy is a scalar
        
        return action_log_probs, value, dist_entropy
