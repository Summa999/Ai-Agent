import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple

class TradingEnvironment:
    """Custom trading environment for RL"""
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.position_size = 0
        self.entry_price = 0
        self.trades = []
        self.done = False
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state observation"""
        # Get technical indicators and market data
        current_data = self.data.iloc[self.current_step]
        
        state = np.array([
            current_data['returns'],
            current_data['rsi_14'],
            current_data['macd'],
            current_data['atr_14'],
            current_data['volume_ratio'],
            self.position,
            self.position_size / self.balance,  # Position size ratio
            (current_data['close'] - self.entry_price) / self.entry_price if self.position != 0 else 0
        ])
        
        return state
    
    def step(self, action):
        """Execute action and return new state, reward, done"""
        current_price = self.data.iloc[self.current_step]['close']
        
        # Actions: 0=hold, 1=buy, 2=sell
        reward = 0
        
        if action == 1 and self.position <= 0:  # Buy
            if self.position < 0:  # Close short position
                profit = (self.entry_price - current_price) * self.position_size
                self.balance += profit - (self.transaction_cost * self.position_size * current_price)
                reward = profit / self.initial_balance
                
            # Open long position
            self.position = 1
            self.position_size = self.balance * 0.95  # Use 95% of balance
            self.entry_price = current_price
            self.balance = self.balance * 0.05  # Keep 5% as cash
            
        elif action == 2 and self.position >= 0:  # Sell
            if self.position > 0:  # Close long position
                profit = (current_price - self.entry_price) * (self.position_size / self.entry_price)
                self.balance += self.position_size + profit - (self.transaction_cost * self.position_size)
                reward = profit / self.initial_balance
                
            # Open short position
            self.position = -1
            self.position_size = self.balance * 0.95
            self.entry_price = current_price
            self.balance = self.balance * 0.05
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.data) - 1:
            self.done = True
            # Close any open positions
            if self.position != 0:
                if self.position > 0:
                    profit = (current_price - self.entry_price) * (self.position_size / self.entry_price)
                else:
                    profit = (self.entry_price - current_price) * self.position_size
                self.balance += self.position_size + profit
        
        # Calculate total return as additional reward
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        return self._get_state(), reward, self.done, {'total_return': total_return}

class DQNNetwork(nn.Module):
    """Deep Q-Network for trading"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        
        # Dueling DQN architecture
        self.value_head = nn.Linear(hidden_size // 2, 1)
        self.advantage_head = nn.Linear(hidden_size // 2, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = torch.relu(self.bn1(self.fc1(state)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        
        # Combine value and advantage (Dueling DQN)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        
    def push(self, experience):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6

class DQNTradingAgent:
    """DQN-based trading agent with advanced features"""
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # Neural networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.get('learning_rate', 0.001))
        
        # Replay buffer
        self.memory = PrioritizedReplayBuffer(config.get('buffer_size', 100000))
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.tau = config.get('tau', 0.005)
        
        self.update_counter = 0
        self.losses = []
        
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = (state, action, reward, next_state, done)
        self.memory.push(experience)
        
    def replay(self, batch_size=32):
        """Train on batch of experiences"""
        if len(self.memory.buffer) < batch_size:
            return
            
        experiences, indices, weights = self.memory.sample(batch_size)
        
        states = torch.FloatTensor([e[0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN
        next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
        
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate loss with importance sampling weights
        loss = (weights * nn.functional.mse_loss(current_q_values.squeeze(), 
                                                 target_q_values, reduction='none')).mean()
        
        # Update priorities
        priorities = loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, priorities)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Soft update target network
        self.soft_update()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.update_counter += 1
        
    def soft_update(self):
        """Soft update target network"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)
    
    def train(self, env, episodes=1000):
        """Train the agent"""
        scores = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            while not env.done:
                action = self.act(state)
                next_state, reward, done, info = env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if len(self.memory.buffer) > self.config.get('batch_size', 32):
                    self.replay(self.config.get('batch_size', 32))
            
            scores.append(info['total_return'])
            
            # Log progress
            if episode % 10 == 0:
                avg_score = np.mean(scores[-100:])
                print(f"Episode {episode}, Average Return: {avg_score:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        return scores
    
    def save(self, path):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']