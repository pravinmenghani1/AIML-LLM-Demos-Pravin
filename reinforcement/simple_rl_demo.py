#!/usr/bin/env python3
"""
Simple Reinforcement Learning Demo: Grid World
A robot learns to navigate to a goal while avoiding obstacles
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        self.reset()
    
    def reset(self):
        self.pos = self.start
        return self.pos
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_pos = (self.pos[0] + moves[action][0], self.pos[1] + moves[action][1])
        
        # Check boundaries
        if (new_pos[0] < 0 or new_pos[0] >= self.size or 
            new_pos[1] < 0 or new_pos[1] >= self.size):
            new_pos = self.pos  # Stay in place
        
        # Check obstacles
        if new_pos in self.obstacles:
            new_pos = self.pos  # Stay in place
        
        self.pos = new_pos
        
        # Rewards
        if self.pos == self.goal:
            reward = 100  # Big reward for reaching goal
            done = True
        elif self.pos in self.obstacles:
            reward = -10  # Penalty for hitting obstacle
            done = False
        else:
            reward = -1   # Small penalty for each step
            done = False
        
        return self.pos, reward, done

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))
    
    def state_to_index(self, state, grid_size):
        return state[0] * grid_size + state[1]
    
    def choose_action(self, state, grid_size):
        state_idx = self.state_to_index(state, grid_size)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[state_idx])
    
    def learn(self, state, action, reward, next_state, done, grid_size):
        state_idx = self.state_to_index(state, grid_size)
        next_state_idx = self.state_to_index(next_state, grid_size)
        
        current_q = self.q_table[state_idx, action]
        next_max_q = 0 if done else np.max(self.q_table[next_state_idx])
        target = reward + self.gamma * next_max_q
        
        self.q_table[state_idx, action] = current_q + self.lr * (target - current_q)
        
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

def test_random_agent(env, episodes=10):
    print("🎲 Testing Random Agent:")
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(50):  # Max 50 steps per episode
            action = random.randint(0, 3)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        status = "🎯 GOAL!" if done and state == env.goal else "❌ Failed"
        print(f"  Episode {episode + 1}: {total_reward:.0f} points, {steps} steps - {status}")
    
    avg = np.mean(scores)
    success_rate = sum(1 for s in scores if s > 0) / len(scores) * 100
    print(f"  Average: {avg:.1f} points, Success rate: {success_rate:.0f}%\n")
    return scores

def train_agent(env, agent, episodes=500):
    print("🧠 Training Q-Learning Agent:")
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(50):
            action = agent.choose_action(state, env.size)
            next_state, reward, done = env.step(action)
            
            agent.learn(state, action, reward, next_state, done, env.size)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            recent_avg = np.mean(scores[-50:])
            success_rate = sum(1 for s in scores[-50:] if s > 0) / 50 * 100
            print(f"  Episode {episode + 1}: Avg = {recent_avg:.1f}, Success = {success_rate:.0f}%, ε = {agent.epsilon:.3f}")
    
    print("  Training complete!\n")
    return scores

def test_trained_agent(env, agent, episodes=10):
    print("🎯 Testing Trained Agent:")
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration
    
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        path = [state]
        
        for step in range(50):
            action = agent.choose_action(state, env.size)
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            path.append(state)
            
            if done:
                break
        
        scores.append(total_reward)
        status = "🎯 GOAL!" if done and state == env.goal else "❌ Failed"
        print(f"  Test {episode + 1}: {total_reward:.0f} points, {steps} steps - {status}")
    
    agent.epsilon = original_epsilon
    avg = np.mean(scores)
    success_rate = sum(1 for s in scores if s > 0) / len(scores) * 100
    print(f"  Average: {avg:.1f} points, Success rate: {success_rate:.0f}%\n")
    return scores

def visualize_policy(env, agent):
    """Show the learned policy as arrows"""
    print("🗺️  Learned Policy (arrows show best action):")
    
    actions = ['↑', '→', '↓', '←']
    grid = np.full((env.size, env.size), ' ', dtype=str)
    
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == env.start:
                grid[i, j] = 'S'
            elif (i, j) == env.goal:
                grid[i, j] = 'G'
            elif (i, j) in env.obstacles:
                grid[i, j] = '█'
            else:
                state_idx = agent.state_to_index((i, j), env.size)
                best_action = np.argmax(agent.q_table[state_idx])
                grid[i, j] = actions[best_action]
    
    print("   " + " ".join([str(i) for i in range(env.size)]))
    for i, row in enumerate(grid):
        print(f"{i}  " + " ".join(row))
    
    print("\nLegend: S=Start, G=Goal, █=Obstacle, ↑→↓←=Best action")

def main():
    print("🤖 Simple Reinforcement Learning Demo: Grid World")
    print("=" * 55)
    print("Goal: Robot learns to navigate from start (0,0) to goal (4,4)")
    print("Obstacles: █ blocks, Rewards: +100 goal, -10 obstacle, -1 step\n")
    
    # Create environment and agent
    env = GridWorld(size=5)
    agent = QLearningAgent(n_states=25, n_actions=4)
    
    # Test random agent
    random_scores = test_random_agent(env)
    
    # Train agent
    training_scores = train_agent(env, agent)
    
    # Test trained agent
    trained_scores = test_trained_agent(env, agent)
    
    # Show learned policy
    visualize_policy(env, agent)
    
    # Results
    print("\n📊 RESULTS:")
    print("=" * 55)
    random_avg = np.mean(random_scores)
    trained_avg = np.mean(trained_scores)
    improvement = trained_avg - random_avg
    
    random_success = sum(1 for s in random_scores if s > 0) / len(random_scores) * 100
    trained_success = sum(1 for s in trained_scores if s > 0) / len(trained_scores) * 100
    
    print(f"Random Agent:  {random_avg:.1f} points, {random_success:.0f}% success")
    print(f"Trained Agent: {trained_avg:.1f} points, {trained_success:.0f}% success")
    print(f"Improvement: {improvement:.1f} points, {trained_success - random_success:.0f}% better success rate")
    
    # Simple plot
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        # Smooth the training curve
        window = 20
        smoothed = []
        for i in range(len(training_scores)):
            start = max(0, i - window)
            smoothed.append(np.mean(training_scores[start:i+1]))
        
        plt.plot(smoothed, color='blue', linewidth=2)
        plt.title('Learning Progress')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.bar(['Random', 'Trained'], [random_avg, trained_avg], 
                color=['red', 'green'], alpha=0.7)
        plt.title('Performance Comparison')
        plt.ylabel('Average Reward')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('/Users/pravinmenghani/Downloads/demos/reinforcement/gridworld_results.png', dpi=150)
        print(f"\n📈 Results saved to: gridworld_results.png")
        plt.show()
    except:
        print("\n📈 Plotting skipped (display not available)")
    
    print("\n🎉 Demo complete! The agent learned to navigate to the goal!")

if __name__ == "__main__":
    main()
