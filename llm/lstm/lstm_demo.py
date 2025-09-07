import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Check TensorFlow availability
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available. Running visualization-only mode.")
    TF_AVAILABLE = False

# Set style for better visuals
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LSTMDemo:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def create_sample_data(self):
        """Create a simple sine wave with trend for demonstration"""
        time = np.linspace(0, 4*np.pi, 200)
        # Sine wave with trend and noise
        data = np.sin(time) + 0.1*time + 0.1*np.random.randn(200)
        return time, data
    
    def prepare_lstm_data(self, data, lookback=10):
        """Prepare data for LSTM training"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaled_data
    
    def build_lstm_model(self, input_shape):
        """Build a simple LSTM model"""
        if not TF_AVAILABLE:
            return None
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def visualize_lstm_concept(self):
        """Create visualization explaining LSTM concept"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🧠 LSTM: Long Short-Term Memory Networks Explained', fontsize=16, fontweight='bold')
        
        # 1. Problem with traditional RNNs
        ax1 = axes[0, 0]
        x = np.linspace(0, 10, 100)
        gradient = np.exp(-x/2)  # Vanishing gradient
        ax1.plot(x, gradient, 'r-', linewidth=3, label='Vanishing Gradient')
        ax1.fill_between(x, gradient, alpha=0.3, color='red')
        ax1.set_title('❌ Traditional RNN Problem\nVanishing Gradients', fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Gradient Magnitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. LSTM Cell Structure
        ax2 = axes[0, 1]
        # Simulate LSTM gates
        time_steps = np.arange(10)
        forget_gate = np.random.sigmoid(np.random.randn(10)) * 0.8 + 0.1
        input_gate = np.random.sigmoid(np.random.randn(10)) * 0.8 + 0.1
        output_gate = np.random.sigmoid(np.random.randn(10)) * 0.8 + 0.1
        
        ax2.plot(time_steps, forget_gate, 'o-', label='🚪 Forget Gate', linewidth=2, markersize=8)
        ax2.plot(time_steps, input_gate, 's-', label='📥 Input Gate', linewidth=2, markersize=8)
        ax2.plot(time_steps, output_gate, '^-', label='📤 Output Gate', linewidth=2, markersize=8)
        ax2.set_title('🔧 LSTM Gates in Action', fontweight='bold')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Gate Activation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. Memory Cell State
        ax3 = axes[1, 0]
        cell_state = np.cumsum(np.random.randn(20) * 0.1)
        hidden_state = np.tanh(cell_state) * np.random.rand(20)
        
        ax3.plot(cell_state, 'b-', linewidth=3, label='📚 Cell State (Long-term memory)', alpha=0.8)
        ax3.plot(hidden_state, 'g-', linewidth=3, label='🧠 Hidden State (Short-term)', alpha=0.8)
        ax3.fill_between(range(20), cell_state, alpha=0.2, color='blue')
        ax3.fill_between(range(20), hidden_state, alpha=0.2, color='green')
        ax3.set_title('💾 LSTM Memory States', fontweight='bold')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('State Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Information Flow
        ax4 = axes[1, 1]
        # Create a flow diagram effect
        x = np.linspace(0, 10, 100)
        flow1 = np.sin(x) * np.exp(-x/10) + 2
        flow2 = np.cos(x) * np.exp(-x/8) + 1
        flow3 = np.sin(x + np.pi/4) * np.exp(-x/12)
        
        ax4.plot(x, flow1, linewidth=4, alpha=0.8, label='🌊 Information Flow')
        ax4.plot(x, flow2, linewidth=4, alpha=0.8, label='🔄 Selective Memory')
        ax4.plot(x, flow3, linewidth=4, alpha=0.8, label='⚡ Gradient Flow')
        ax4.fill_between(x, flow1, alpha=0.2)
        ax4.fill_between(x, flow2, alpha=0.2)
        ax4.fill_between(x, flow3, alpha=0.2)
        ax4.set_title('🌊 Information & Gradient Flow', fontweight='bold')
        ax4.set_xlabel('Network Depth')
        ax4.set_ylabel('Signal Strength')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_prediction_demo(self):
        """Run the actual LSTM prediction demo"""
        if not TF_AVAILABLE:
            print("⚠️  TensorFlow not available. Showing simulated results...")
            self.run_simulated_demo()
            return
            
        print("🚀 Creating sample time series data...")
        time, data = self.create_sample_data()
        
        # Prepare data
        lookback = 10
        X, y, scaled_data = self.prepare_lstm_data(data, lookback)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        print("🏗️  Building LSTM model...")
        model = self.build_lstm_model((X_train.shape[1], 1))
        
        print("🎯 Training LSTM model...")
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                          validation_split=0.2, verbose=0)
        
        # Make predictions
        train_pred = model.predict(X_train, verbose=0)
        test_pred = model.predict(X_test, verbose=0)
        
        # Inverse transform
        train_pred = self.scaler.inverse_transform(train_pred)
        test_pred = self.scaler.inverse_transform(test_pred)
        y_train_actual = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Visualize results
        self.plot_results(time, data, train_pred, test_pred, y_train_actual, 
                         y_test_actual, history, lookback, split)
    
    def run_simulated_demo(self):
        """Run simulated demo when TensorFlow is not available"""
        print("🚀 Creating sample time series data...")
        time, data = self.create_sample_data()
        
        # Create simulated predictions
        lookback = 10
        split = int(0.8 * len(data))
        
        # Simulate LSTM predictions with some noise
        train_pred = data[lookback:split+lookback] + np.random.normal(0, 0.1, split)
        test_pred = data[split+lookback:] + np.random.normal(0, 0.15, len(data)-split-lookback)
        
        # Create fake history
        class FakeHistory:
            def __init__(self):
                epochs = 50
                self.history = {
                    'loss': np.exp(-np.linspace(0, 3, epochs)) + 0.01,
                    'val_loss': np.exp(-np.linspace(0, 2.8, epochs)) + 0.02
                }
        
        history = FakeHistory()
        
        # Plot simulated results
        self.plot_simulated_results(time, data, train_pred, test_pred, history, lookback, split)
    
    def plot_results(self, time, original_data, train_pred, test_pred, 
                    y_train_actual, y_test_actual, history, lookback, split):
        """Create comprehensive visualization of results"""
        fig = plt.figure(figsize=(18, 12))
        
        # Main prediction plot
        ax1 = plt.subplot(2, 3, (1, 2))
        
        # Plot original data
        plt.plot(time, original_data, 'b-', linewidth=2, label='📊 Original Data', alpha=0.7)
        
        # Plot training predictions
        train_time = time[lookback:split+lookback]
        plt.plot(train_time, train_pred.flatten(), 'g-', linewidth=2, 
                label='🎯 Training Predictions', alpha=0.8)
        
        # Plot test predictions
        test_time = time[split+lookback:]
        plt.plot(test_time, test_pred.flatten(), 'r-', linewidth=3, 
                label='🔮 Test Predictions', alpha=0.9)
        
        # Add vertical line to separate train/test
        plt.axvline(x=time[split+lookback], color='orange', linestyle='--', 
                   linewidth=2, label='📍 Train/Test Split')
        
        plt.title('🎯 LSTM Time Series Prediction Results', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training history
        ax2 = plt.subplot(2, 3, 3)
        plt.plot(history.history['loss'], 'b-', linewidth=2, label='📉 Training Loss')
        plt.plot(history.history['val_loss'], 'r-', linewidth=2, label='📈 Validation Loss')
        plt.title('📊 Model Training Progress', fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Error analysis
        ax3 = plt.subplot(2, 3, 4)
        train_errors = y_train_actual.flatten() - train_pred.flatten()
        test_errors = y_test_actual.flatten() - test_pred.flatten()
        
        plt.hist(train_errors, bins=20, alpha=0.7, label='🎯 Training Errors', color='green')
        plt.hist(test_errors, bins=20, alpha=0.7, label='🔮 Test Errors', color='red')
        plt.title('📊 Prediction Error Distribution', fontweight='bold')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot: Actual vs Predicted
        ax4 = plt.subplot(2, 3, 5)
        plt.scatter(y_train_actual, train_pred, alpha=0.6, color='green', 
                   label='🎯 Training', s=30)
        plt.scatter(y_test_actual, test_pred, alpha=0.8, color='red', 
                   label='🔮 Testing', s=30)
        
        # Perfect prediction line
        min_val = min(np.min(y_train_actual), np.min(y_test_actual))
        max_val = max(np.max(y_train_actual), np.max(y_test_actual))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', 
                linewidth=2, label='✨ Perfect Prediction')
        
        plt.title('🎯 Actual vs Predicted Values', fontweight='bold')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Model architecture visualization
        ax5 = plt.subplot(2, 3, 6)
        layers = ['Input\n(Sequence)', 'LSTM\n(50 units)', 'LSTM\n(50 units)', 
                 'Dense\n(25 units)', 'Output\n(1 unit)']
        y_pos = np.arange(len(layers))
        colors = ['lightblue', 'lightgreen', 'lightgreen', 'lightyellow', 'lightcoral']
        
        bars = plt.barh(y_pos, [1, 0.8, 0.8, 0.6, 0.4], color=colors, alpha=0.8)
        plt.yticks(y_pos, layers)
        plt.title('🏗️ LSTM Model Architecture', fontweight='bold')
        plt.xlabel('Relative Complexity')
        
        # Add arrows between layers
        for i in range(len(layers)-1):
            plt.annotate('', xy=(0.1, i+1), xytext=(0.1, i),
                        arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
        
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        train_rmse = np.sqrt(np.mean((y_train_actual - train_pred)**2))
        test_rmse = np.sqrt(np.mean((y_test_actual - test_pred)**2))
        
        print("\n" + "="*50)
        print("📊 LSTM MODEL PERFORMANCE SUMMARY")
        print("="*50)
        print(f"🎯 Training RMSE: {train_rmse:.4f}")
        print(f"🔮 Testing RMSE:  {test_rmse:.4f}")
        print(f"📈 Model Accuracy: {max(0, 100*(1-test_rmse/np.std(y_test_actual))):.2f}%")
        print("="*50)
    
    def plot_simulated_results(self, time, original_data, train_pred, test_pred, history, lookback, split):
        """Plot simulated results when TensorFlow is not available"""
        fig = plt.figure(figsize=(15, 10))
        
        # Main prediction plot
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(time, original_data, 'b-', linewidth=2, label='📊 Original Data', alpha=0.7)
        
        train_time = time[lookback:split+lookback]
        plt.plot(train_time, train_pred, 'g-', linewidth=2, label='🎯 Simulated Training', alpha=0.8)
        
        test_time = time[split+lookback:]
        plt.plot(test_time, test_pred, 'r-', linewidth=3, label='🔮 Simulated Test', alpha=0.9)
        
        plt.axvline(x=time[split+lookback], color='orange', linestyle='--', linewidth=2, label='📍 Train/Test Split')
        plt.title('🎯 Simulated LSTM Predictions', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training history
        ax2 = plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'], 'b-', linewidth=2, label='📉 Training Loss')
        plt.plot(history.history['val_loss'], 'r-', linewidth=2, label='📈 Validation Loss')
        plt.title('📊 Simulated Training Progress', fontweight='bold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Error analysis
        ax3 = plt.subplot(2, 2, 3)
        train_errors = np.random.normal(0, 0.1, len(train_pred))
        test_errors = np.random.normal(0, 0.15, len(test_pred))
        
        plt.hist(train_errors, bins=20, alpha=0.7, label='🎯 Training Errors', color='green')
        plt.hist(test_errors, bins=20, alpha=0.7, label='🔮 Test Errors', color='red')
        plt.title('📊 Simulated Error Distribution', fontweight='bold')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Model architecture
        ax4 = plt.subplot(2, 2, 4)
        layers = ['Input\n(Sequence)', 'LSTM\n(50 units)', 'LSTM\n(50 units)', 'Dense\n(25 units)', 'Output\n(1 unit)']
        y_pos = np.arange(len(layers))
        colors = ['lightblue', 'lightgreen', 'lightgreen', 'lightyellow', 'lightcoral']
        
        plt.barh(y_pos, [1, 0.8, 0.8, 0.6, 0.4], color=colors, alpha=0.8)
        plt.yticks(y_pos, layers)
        plt.title('🏗️ LSTM Model Architecture', fontweight='bold')
        plt.xlabel('Relative Complexity')
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*50)
        print("📊 SIMULATED LSTM PERFORMANCE")
        print("="*50)
        print("🎯 Training RMSE: 0.0856")
        print("🔮 Testing RMSE:  0.1124")
        print("📈 Model Accuracy: 89.23%")
        print("⚠️  Note: These are simulated results for demonstration")
        print("="*50)

def main():
    """Main function to run the LSTM demo"""
    print("🎉 Welcome to the Interactive LSTM Demo!")
    print("=" * 50)
    print("This demo will teach you about LSTMs step by step:")
    print("1. 🧠 LSTM Concept Visualization")
    print("2. 🚀 Live Prediction Demo")
    print("=" * 50)
    
    demo = LSTMDemo()
    
    # Show concept visualization
    print("\n📚 Step 1: Understanding LSTM Concepts...")
    demo.visualize_lstm_concept()
    
    input("\n⏸️  Press Enter to continue to the prediction demo...")
    
    # Run prediction demo
    print("\n🚀 Step 2: Running LSTM Prediction Demo...")
    demo.run_prediction_demo()
    
    print("\n🎊 Demo completed! You now understand how LSTMs work!")
    print("\n💡 Key Takeaways:")
    print("   • LSTMs solve vanishing gradient problem")
    print("   • They use gates to control information flow")
    print("   • Perfect for sequential data like time series")
    print("   • Memory cells maintain long-term dependencies")

if __name__ == "__main__":
    main()
