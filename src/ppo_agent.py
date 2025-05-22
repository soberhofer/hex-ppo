import torch
import torch.optim as optim
from torch.distributions import Categorical
from src.ppo_model import ActorCritic
import numpy as np

class PPOAgent:
    def __init__(self, obs_shape, action_space_size, lr=3e-4, gamma=0.99, k_epochs=4, eps_clip=0.2, gae_lambda=0.95, device=torch.device("cpu")):
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.device = device

        self.policy = ActorCritic(obs_shape, action_space_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_shape, action_space_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = torch.nn.MSELoss()

    def select_action(self, observation, valid_actions):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device) # Add batch dimension and move to device
            action_logits, value = self.policy_old(obs_tensor)
            
            # Mask invalid actions
            mask = torch.full(action_logits.shape, -float('inf'), device=self.device) # Move mask to device
            for action_scalar in valid_actions:
                mask[0, action_scalar] = 0
            
            masked_action_logits = action_logits + mask
            
            probs = torch.nn.functional.softmax(masked_action_logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        
        return action.item(), action_log_prob.item(), value.item()

    def update(self, memory):
        # Convert lists to tensors and move to device
        old_states = torch.stack(memory.states).float().to(self.device)
        old_actions = torch.stack(memory.actions).long().to(self.device)
        old_logprobs = torch.stack(memory.logprobs).float().to(self.device)
        old_rewards = torch.stack(memory.rewards).float().to(self.device)
        old_is_terminals = torch.stack(memory.is_terminals).float().to(self.device)
        
        # Calculate advantages
        advantages = self._calculate_advantages(old_rewards, old_is_terminals, old_states)

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # PPO clip objective
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.MseLoss(state_values.squeeze(), old_rewards) # Assuming old_rewards are already returns

            loss = policy_loss + 0.5 * value_loss - 0.01 * dist_entropy # Add entropy bonus

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _calculate_advantages(self, rewards, is_terminals, states):
        # Calculate discounted rewards (returns)
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device) # Move returns to device

        # Calculate advantages using GAE
        with torch.no_grad():
            values = self.policy_old(states)[1].squeeze() # Get values from the old policy
        
        advantages = returns - values
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        return advantages

class RolloutMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def add(self, state, action, log_prob, reward, is_terminal):
        # Store tensors on CPU, move to device during update
        self.states.append(torch.tensor(state, dtype=torch.float32))
        self.actions.append(torch.tensor(action))
        self.logprobs.append(torch.tensor(log_prob))
        self.rewards.append(torch.tensor(reward, dtype=torch.float32))
        self.is_terminals.append(torch.tensor(is_terminal, dtype=torch.bool))
