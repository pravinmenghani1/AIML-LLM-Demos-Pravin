#!/usr/bin/env python3
"""
Reinforcement Learning Demo: CartPole Balance
Problem: Balance a pole on a cart using Q-Learning
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class QLearningAgent:
    def __init__(self, n_actions=2, lr=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
    def discretize_state(self, state):
        # Discretize continuous state space
        bins = [10, 10, 10, 10]
        ranges = [(-2.4, 2.4), (-3, 3), (-0.2, 0.2), (-3, 3)]
        
        discrete = []
        for i, (val, (low, high)) in enumerate(zip(state, ranges)):
            val = max(low, min(high, val))
            discrete.append(int((val - low) / (high - low) * (bins[i] - 1)))
        
        return tuple(discrete)
    
    def choose_action(self, state):
        discrete_state = self.discretize_state(state)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[discrete_state])
    
    def learn(self, state, action, reward, next_state, done):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        current_q = self.q_table[discrete_state][action]
        next_max_q = 0 if done else np.max(self.q_table[discrete_next_state])
        target = reward + self.gamma * next_max_q
        
        self.q_table[discrete_state][action] = current_q + self.lr * (target - current_q)
        
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

def test_random_agent(env, episodes=5):
    print("Testing Random Agent:")
    scores = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        scores.append(total_reward)
        print(f"  Episode {episode + 1}: {total_reward} steps")
    
    avg = np.mean(scores)
    print(f"  Random agent average: {avg:.1f} steps\n")
    return scores

def train_agent(env, agent, episodes=1000):
    print("Training Q-Learning Agent:")
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        
        if (episode + 1) % 200 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"  Episode {episode + 1}: Avg = {avg_score:.1f}, Epsilon = {agent.epsilon:.3f}")
    
    print("  Training complete!\n")
    return scores

def test_trained_agent(env, agent, episodes=10):
    print("Testing Trained Agent:")
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration
    
    scores = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            action = agent.choose_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        scores.append(total_reward)
        print(f"  Test {episode + 1}: {total_reward} steps")
    
    agent.epsilon = original_epsilon
    avg = np.mean(scores)
    print(f"  Trained agent average: {avg:.1f} steps\n")
    return scores

def main():
    print("🚀 Reinforcement Learning Demo: CartPole Balance")
    print("=" * 50)
    
    # Create environment
    env = gym.make('CartPole-v1')
    print(f"Environment: CartPole-v1")
    print(f"Actions: {env.action_space.n} (0=Left, 1=Right)")
    print(f"Goal: Keep pole upright as long as possible\n")
    
    # Test random agent
    random_scores = test_random_agent(env)
    
    # Create and train agent
    agent = QLearningAgent()
    training_scores = train_agent(env, agent)
    
    # Test trained agent
    trained_scores = test_trained_agent(env, agent)
    
    # Results
    print("📊 RESULTS:")
    print("=" * 50)
    random_avg = np.mean(random_scores)
    trained_avg = np.mean(trained_scores)
    improvement = trained_avg - random_avg
    improvement_pct = (improvement / random_avg) * 100
    
    print(f"Random Agent Average:  {random_avg:.1f} steps")
    print(f"Trained Agent Average: {trained_avg:.1f} steps")
    print(f"Improvement: {improvement:.1f} steps ({improvement_pct:.1f}% better)")
    
    success_rate = sum(1 for s in trained_scores if s >= 475) / len(trained_scores) * 100
    print(f"Success Rate (≥475 steps): {success_rate:.1f}%")
    
    # Simple plot
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_scores, alpha=0.6)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Steps Survived')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.bar(['Random', 'Trained'], [random_avg, trained_avg], 
                color=['red', 'green'], alpha=0.7)
        plt.title('Performance Comparison')
        plt.ylabel('Average Steps')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('/Users/pravinmenghani/Downloads/demos/reinforcement/results.png', dpi=150)
        print(f"\n📈 Results saved to: results.png")
        plt.show()
    except:
        print("\n📈 Plotting skipped (display not available)")
    
    env.close()
    print("\n🎉 Demo complete! The agent learned to balance the pole!")

if __name__ == "__main__":
    main()
