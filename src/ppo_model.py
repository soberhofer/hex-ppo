from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    """Residual block for ResNet architecture"""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.leaky_relu(out)
        return out

class ActorCritic(nn.Module):
    """ Actor Critic with Resnet 40 Architecture """
    def __init__(self, obs_shape: Tuple[int, int], action_space_size: int, num_channels: int = 256):
        super(ActorCritic, self).__init__()

        board_size = obs_shape[0]
        self.board_size = board_size
        self.action_space_size = action_space_size

        # Initial convolution
        self.conv_input = nn.Conv2d(1, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # 39 residual blocks (40 layers total including input conv)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(39)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_space_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)


    def forward(self, x):
        # Initial convolution
        x = F.leaky_relu(self.bn_input(self.conv_input(x)))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Policy head
        policy = F.leaky_relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = F.leaky_relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.leaky_relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value




class ActorCriticOld(nn.Module):
    def __init__(self, obs_shape, action_space_size):
        super(ActorCriticOld, self).__init__()
        
        # Assuming obs_shape is (board_size, board_size)
        board_size = obs_shape[0]

        # Convolutional layers for feature extraction
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # test bigger network
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        
        # 1x1 Convolution for the residual connection to match channel dimensions
        # self.residual_projection = nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0)

        # convolution residual layer
        self.residual = nn.Conv2d(1, 128, kernel_size=1)  # Projection for skip


        # Calculate the output size of the convolutional layers
        # The size remains the same due to padding=1 and stride=1
        # conv_output_size = 64 * board_size * board_size
        conv_output_size = 128 * board_size ** 2 # without attention pooling
        # conv_output_size = 128 # with attention pooling

        # Actor (Policy) network
        #self.actor_fc1 = nn.Linear(conv_output_size, 256)
        #self.actor_fc2 = nn.Linear(256, action_space_size)

        self.actor = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.SiLU(),
            nn.Linear(512, action_space_size))

        # Critic (Value) network
        #self.critic_fc1 = nn.Linear(conv_output_size, 256)
        #self.critic_fc2 = nn.Linear(256, 1)

        self.critic = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.SiLU(),
            nn.Linear(512, 1))

    def forward(self, x):
        # Residual Conv Block
        residual = self.residual(x)
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = self.conv3(x) + residual  # Skip connection
        x = F.silu(x)  # Apply silu after adding residual

        x = x.view(x.size(0), -1)


        # Heads
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

    """
    def forward(self, obs):
        # obs is expected to be (batch_size, 1, board_size, board_size)
        # Ensure it's float
        obs_float = obs.float() # Use a different variable name to keep original obs for residual

        # Main path
        x = F.silu(self.conv1(obs_float))
        x = F.silu(self.conv2(x))
        
        # Residual connection
        residual = self.residual_projection(obs_float)
        x = x + residual
        x = F.silu(x) # Apply silu after adding residual
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch

        # Actor
        actor_hidden = F.silu(self.actor_fc1(x))
        action_logits = self.actor_fc2(actor_hidden)
        
        # Critic
        critic_hidden = F.silu(self.critic_fc1(x))
        value = self.critic_fc2(critic_hidden)

        return action_logits, value
    """

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
