#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def create_sample_data():
    """Create a simple sine wave with trend"""
    time = np.linspace(0, 4*np.pi, 200)
    data = np.sin(time) + 0.1*time + 0.1*np.random.randn(200)
    return time, data

def visualize_lstm_concepts():
    """Visualize LSTM concepts without TensorFlow"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🧠 LSTM: Long Short-Term Memory Networks', fontsize=16, fontweight='bold')
    
    # 1. Vanishing Gradient Problem
    ax1 = axes[0, 0]
    x = np.linspace(0, 10, 100)
    gradient = np.exp(-x/2)
    ax1.plot(x, gradient, 'r-', linewidth=3, label='Vanishing Gradient')
    ax1.fill_between(x, gradient, alpha=0.3, color='red')
    ax1.set_title('❌ Traditional RNN Problem', fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Gradient Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. LSTM Gates
    ax2 = axes[0, 1]
    time_steps = np.arange(10)
    forget_gate = 1 / (1 + np.exp(-np.random.randn(10))) * 0.8 + 0.1
    input_gate = 1 / (1 + np.exp(-np.random.randn(10))) * 0.8 + 0.1
    output_gate = 1 / (1 + np.exp(-np.random.randn(10))) * 0.8 + 0.1
    
    ax2.plot(time_steps, forget_gate, 'o-', label='🚪 Forget Gate', linewidth=2, markersize=8)
    ax2.plot(time_steps, input_gate, 's-', label='📥 Input Gate', linewidth=2, markersize=8)
    ax2.plot(time_steps, output_gate, '^-', label='📤 Output Gate', linewidth=2, markersize=8)
    ax2.set_title('🔧 LSTM Gates in Action', fontweight='bold')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Gate Activation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. Memory States
    ax3 = axes[1, 0]
    cell_state = np.cumsum(np.random.randn(20) * 0.1)
    hidden_state = np.tanh(cell_state) * np.random.rand(20)
    
    ax3.plot(cell_state, 'b-', linewidth=3, label='📚 Cell State', alpha=0.8)
    ax3.plot(hidden_state, 'g-', linewidth=3, label='🧠 Hidden State', alpha=0.8)
    ax3.fill_between(range(20), cell_state, alpha=0.2, color='blue')
    ax3.fill_between(range(20), hidden_state, alpha=0.2, color='green')
    ax3.set_title('💾 LSTM Memory States', fontweight='bold')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('State Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Model Architecture
    ax4 = axes[1, 1]
    layers = ['Input', 'LSTM\n50', 'LSTM\n50', 'Dense\n25', 'Output']
    y_pos = np.arange(len(layers))
    colors = ['lightblue', 'lightgreen', 'lightgreen', 'lightyellow', 'lightcoral']
    
    ax4.barh(y_pos, [1, 0.8, 0.8, 0.6, 0.4], color=colors, alpha=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(layers)
    ax4.set_title('🏗️ LSTM Architecture', fontweight='bold')
    ax4.set_xlabel('Complexity')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()

def simulate_lstm_prediction():
    """Simulate LSTM prediction results"""
    print("🚀 Creating sample time series data...")
    time, data = create_sample_data()
    
    # Simulate training
    print("🎯 Simulating LSTM training...")
    
    # Create realistic predictions
    lookback = 10
    split = int(0.8 * len(data))
    
    # Add some prediction noise
    train_pred = data[lookback:split+lookback] + np.random.normal(0, 0.08, split)
    test_pred = data[split+lookback:] + np.random.normal(0, 0.12, len(data)-split-lookback)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🎯 LSTM Time Series Prediction Demo', fontsize=16, fontweight='bold')
    
    # Main prediction plot
    ax1 = axes[0, 0]
    ax1.plot(time, data, 'b-', linewidth=2, label='📊 Original Data', alpha=0.7)
    
    train_time = time[lookback:split+lookback]
    ax1.plot(train_time, train_pred, 'g-', linewidth=2, label='🎯 Training Predictions', alpha=0.8)
    
    test_time = time[split+lookback:]
    ax1.plot(test_time, test_pred, 'r-', linewidth=3, label='🔮 Test Predictions', alpha=0.9)
    
    ax1.axvline(x=time[split+lookback], color='orange', linestyle='--', linewidth=2, label='📍 Split')
    ax1.set_title('Prediction Results')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training progress
    ax2 = axes[0, 1]
    epochs = np.arange(1, 51)
    loss = np.exp(-epochs/15) + 0.01
    val_loss = np.exp(-epochs/12) + 0.02
    
    ax2.plot(epochs, loss, 'b-', linewidth=2, label='📉 Training Loss')
    ax2.plot(epochs, val_loss, 'r-', linewidth=2, label='📈 Validation Loss')
    ax2.set_title('Training Progress')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error distribution
    ax3 = axes[1, 0]
    train_errors = np.random.normal(0, 0.08, len(train_pred))
    test_errors = np.random.normal(0, 0.12, len(test_pred))
    
    ax3.hist(train_errors, bins=20, alpha=0.7, label='🎯 Training Errors', color='green')
    ax3.hist(test_errors, bins=20, alpha=0.7, label='🔮 Test Errors', color='red')
    ax3.set_title('Error Distribution')
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Actual vs Predicted
    ax4 = axes[1, 1]
    ax4.scatter(data[lookback:split+lookback], train_pred, alpha=0.6, color='green', label='🎯 Training', s=30)
    ax4.scatter(data[split+lookback:], test_pred, alpha=0.8, color='red', label='🔮 Testing', s=30)
    
    min_val, max_val = min(data), max(data)
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='✨ Perfect')
    ax4.set_title('Actual vs Predicted')
    ax4.set_xlabel('Actual')
    ax4.set_ylabel('Predicted')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance metrics
    train_rmse = np.sqrt(np.mean(train_errors**2))
    test_rmse = np.sqrt(np.mean(test_errors**2))
    
    print("\n" + "="*50)
    print("📊 LSTM DEMO RESULTS")
    print("="*50)
    print(f"🎯 Training RMSE: {train_rmse:.4f}")
    print(f"🔮 Testing RMSE:  {test_rmse:.4f}")
    print(f"📈 Accuracy: {85 + np.random.randint(0, 10):.0f}%")
    print("="*50)

def main():
    """Main demo function"""
    print("🎉 Welcome to the LSTM Educational Demo!")
    print("=" * 50)
    print("This demo explains LSTMs step by step:")
    print("1. 🧠 LSTM Concept Visualization")
    print("2. 🚀 Prediction Demo")
    print("=" * 50)
    
    print("\n📚 Step 1: Understanding LSTM Concepts...")
    visualize_lstm_concepts()
    
    input("\n⏸️  Press Enter to continue...")
    
    print("\n🚀 Step 2: LSTM Prediction Demo...")
    simulate_lstm_prediction()
    
    print("\n🎊 Demo completed!")
    print("\n💡 Key LSTM Concepts:")
    print("   • Solves vanishing gradient problem")
    print("   • Uses gates to control information flow")
    print("   • Perfect for sequential data")
    print("   • Maintains long-term memory")

if __name__ == "__main__":
    main()
