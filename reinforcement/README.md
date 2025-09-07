# 🤖 Reinforcement Learning Demos

This directory contains **working** reinforcement learning demonstrations that showcase step-by-step agent training with clear visualizations.

## 📁 Available Demos

### 1. 🗺️ **Grid World Navigation** (Recommended)
- **Files**: `gridworld_rl_demo.ipynb`, `simple_rl_demo.py`
- **Problem**: Navigate from start to goal while avoiding obstacles
- **Environment**: 5×5 grid with fixed obstacles
- **Algorithm**: Q-Learning with discrete states
- **Success Rate**: 100% after training
- **Best For**: Understanding core RL concepts

**Key Features**:
- ✅ Live training visualization
- ✅ Policy visualization with arrows
- ✅ Step-by-step learning progress
- ✅ Clear before/after comparison
- ✅ Optimal path discovery

### 2. 🎯 **CartPole Balance**
- **Files**: `cartpole_rl_demo.ipynb`, `rl_demo.py`  
- **Problem**: Balance a pole on a moving cart
- **Environment**: OpenAI Gymnasium CartPole-v1
- **Algorithm**: Q-Learning with state discretization
- **Challenge**: Continuous state space discretization
- **Best For**: Understanding state discretization

**Key Features**:
- ✅ Real physics simulation
- ✅ Continuous to discrete state mapping
- ✅ Performance tracking over episodes
- ✅ Success metrics (≥475 steps)

## 🚀 Quick Start

### Option 1: Jupyter Notebooks (Interactive)
```bash
# Start Jupyter
jupyter notebook

# Open either:
# - gridworld_rl_demo.ipynb (Recommended)
# - cartpole_rl_demo.ipynb
```

### Option 2: Python Scripts (Direct Run)
```bash
# Grid World (Shows clear learning)
python simple_rl_demo.py

# CartPole (Classic RL problem)
python rl_demo.py
```

## 📊 Expected Results

### Grid World Demo
- **Random Agent**: ~30% success rate, negative scores
- **Trained Agent**: 100% success rate, +93 points average
- **Learning Time**: ~200 episodes
- **Optimal Path**: 8 steps from start to goal

### CartPole Demo  
- **Random Agent**: ~20-30 steps average
- **Trained Agent**: Variable (depends on discretization)
- **Learning Time**: ~1000 episodes
- **Success Metric**: 475+ steps consistently

## 🎓 Learning Concepts Demonstrated

### Core RL Principles
1. **Trial and Error Learning**: Agent improves through experience
2. **Reward Signals**: Positive/negative feedback guides behavior
3. **Exploration vs Exploitation**: Balance between trying new actions and using known good ones
4. **Policy Learning**: Mapping from states to optimal actions

### Technical Concepts
1. **Q-Learning Algorithm**: Value-based learning method
2. **Epsilon-Greedy Strategy**: Exploration strategy
3. **State Discretization**: Converting continuous to discrete states
4. **Convergence**: How agents reach optimal performance

### Visualizations
1. **Learning Curves**: Performance improvement over time
2. **Policy Maps**: Visual representation of learned strategies
3. **Success Rates**: Quantitative performance metrics
4. **Path Visualization**: Optimal routes discovered

## 🔧 Requirements

```bash
# Core packages (already installed via conda)
gymnasium>=1.0.0
numpy>=1.21.0
matplotlib>=3.5.0

# For Jupyter notebooks
jupyter>=1.0.0
ipython>=7.0.0
```

## 🎯 Recommended Learning Path

1. **Start with Grid World** (`simple_rl_demo.py` or `gridworld_rl_demo.ipynb`)
   - Clearest learning demonstration
   - 100% success rate achievable
   - Visual policy representation

2. **Progress to CartPole** (`rl_demo.py` or `cartpole_rl_demo.ipynb`)
   - More challenging continuous state space
   - Classic RL benchmark problem
   - State discretization techniques

3. **Experiment with Parameters**
   - Modify learning rates, epsilon decay
   - Try different discretization strategies
   - Observe impact on learning speed

## 🎉 Success Indicators

You'll know the demos are working when you see:

- **Grid World**: Agent finds optimal 8-step path consistently
- **CartPole**: Agent balances pole for 200+ steps regularly
- **Learning Curves**: Clear upward trend in performance
- **Policy Visualization**: Logical action choices in each state

## 🚀 Next Steps

After mastering these demos, explore:
- Deep Q-Networks (DQN) for complex state spaces
- Policy Gradient methods (REINFORCE, PPO)
- Multi-agent reinforcement learning
- Real-world applications (robotics, game AI, trading)

---

**Note**: All demos are self-contained and don't require external dependencies beyond the standard scientific Python stack.
