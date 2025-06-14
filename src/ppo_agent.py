import torch
import torch.optim as optim
from torch.distributions import Categorical
from src.ppo_model import ActorCritic
import numpy as np

class PPOAgent:
    def __init__(self, obs_shape, action_space_size, lr=3e-4, gamma=0.99, k_epochs=4, eps_clip=0.2, gae_lambda=0.95, device=torch.device("cpu"), entropy_coef=0.05, scaler: bool = False):
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.device = device
        self.entropy_coef = entropy_coef
        self.use_mixed_precision = scaler
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler()

        self.policy = ActorCritic(obs_shape, action_space_size).to(device)
        self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=lr,
                weight_decay=1e-3,  # l2 regularization, reduce overfitting
                betas=(0.9, 0.999)  # control momentum (gradient average), reduce oszillation, control adaptive lrs
            )

        #self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_shape, action_space_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = torch.nn.MSELoss()

    def select_action(self, observation, valid_actions, temperature):
        with torch.no_grad():
            # Add batch and channel dimensions, then move to device
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(1).to(self.device)
            #action_logits, value = self.policy_old(obs_tensor)

            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    action_logits, value = self.policy_old(obs_tensor)
            else:
                action_logits, value = self.policy_old(obs_tensor)




            # Mask invalid actions
            mask = torch.full(action_logits.shape, -float('inf'), device=self.device) # Move mask to device
            for action_scalar in valid_actions:
                mask[0, action_scalar] = 0
            
            masked_action_logits = action_logits + mask
            
            probs = torch.nn.functional.softmax(masked_action_logits / temperature, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        
        return action.item(), action_log_prob.item(), value.item()

    def update(self, memory):

        # entropy coeff annealing --> supports more exploration in the beginning, reduced throughout training
        self.entropy_coef = max(1e-4, self.entropy_coef * 0.95)

        # Convert lists to tensors, add channel dimension, and move to device
        old_states = torch.stack(memory.states).float().unsqueeze(1).to(self.device)
        old_actions = torch.stack(memory.actions).long().to(self.device)
        old_logprobs = torch.stack(memory.logprobs).float().to(self.device)
        old_rewards = torch.stack(memory.rewards).float().to(self.device)
        old_is_terminals = torch.stack(memory.is_terminals).float().to(self.device)
        
        # Calculate advantages
        advantages = self._calculate_advantages(old_rewards, old_is_terminals, old_states)

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):

            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
                    ratios = torch.exp(logprobs - old_logprobs.detach())
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = self.MseLoss(state_values.squeeze(), old_rewards)
                    loss = policy_loss + 0.5 * value_loss - self.entropy_coef * dist_entropy

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Evaluate old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

                # PPO clip objective
                ratios = torch.exp(logprobs - old_logprobs.detach())

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.MseLoss(state_values.squeeze(), old_rewards) # Assuming old_rewards are already returns

                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * dist_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5) # Gradient Clipping
                self.optimizer.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        return policy_loss.item(), value_loss.item(), dist_entropy.item(), loss

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
