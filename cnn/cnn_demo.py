#!/usr/bin/env python3
"""
🧠 CNN Magic Demo for Kids! 🎨
A simple, visual demonstration of how CNNs work
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

def create_simple_image():
    """Create a simple 8x8 image with a clear letter 'X'"""
    image = np.zeros((8, 8))
    
    # Create a clear letter 'X'
    # Top-left to bottom-right diagonal
    for i in range(8):
        image[i, i] = 1
    
    # Top-right to bottom-left diagonal  
    for i in range(8):
        image[i, 7-i] = 1
        
    return image

def create_edge_filter():
    """Create a simple edge detection filter"""
    return np.array([[-1, -1, -1],
                     [-1,  8, -1],
                     [-1, -1, -1]])

def apply_convolution(image, kernel):
    """Apply convolution operation (simplified)"""
    result = np.zeros((6, 6))  # Output will be smaller
    
    for i in range(6):
        for j in range(6):
            # Extract 3x3 patch
            patch = image[i:i+3, j:j+3]
            # Apply filter
            result[i, j] = np.sum(patch * kernel)
    
    return result

def visualize_cnn_process():
    """Create a visual demonstration of CNN process"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('How CNNs See Pictures - Like Magic!', fontsize=16, fontweight='bold')
    
    # Step 1: Original Image
    original_image = create_simple_image()
    axes[0, 0].imshow(original_image, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('1. Original Picture\n(Letter X)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add numbers to show pixel values
    for i in range(8):
        for j in range(8):
            axes[0, 0].text(j, i, f'{original_image[i, j]:.1f}', 
                           ha='center', va='center', fontsize=8, color='white')
    
    # Step 2: Filter/Kernel
    edge_filter = create_edge_filter()
    axes[0, 1].imshow(edge_filter, cmap='RdBu', interpolation='nearest')
    axes[0, 1].set_title('2. Magic Filter\n(Edge Detector)', fontsize=12, fontweight='bold')
    
    # Add numbers to filter
    for i in range(3):
        for j in range(3):
            color = 'white' if edge_filter[i, j] < 0 else 'black'
            axes[0, 1].text(j, i, f'{edge_filter[i, j]}', 
                           ha='center', va='center', fontsize=10, color=color, fontweight='bold')
    
    # Step 3: Convolution Result
    conv_result = apply_convolution(original_image, edge_filter)
    axes[0, 2].imshow(conv_result, cmap='plasma', interpolation='nearest')
    axes[0, 2].set_title('3. After Magic Filter\n(Edges Found!)', fontsize=12, fontweight='bold')
    
    # Step 4: Pooling (Max Pooling)
    pooled = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            patch = conv_result[i*2:(i+1)*2, j*2:(j+1)*2]
            pooled[i, j] = np.max(patch)
    
    axes[1, 0].imshow(pooled, cmap='hot', interpolation='nearest')
    axes[1, 0].set_title('4. After Pooling\n(Keep Important Stuff)', fontsize=12, fontweight='bold')
    
    # Add numbers to pooled result
    for i in range(3):
        for j in range(3):
            axes[1, 0].text(j, i, f'{pooled[i, j]:.1f}', 
                           ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # Step 5: Show the process
    axes[1, 1].text(0.5, 0.7, 'CNN Brain Process:', ha='center', va='center', 
                    fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    
    process_text = """
    1. Look at picture (Letter X)
    2. Use magic filters
    3. Find patterns (diagonal lines)
    4. Keep important parts
    5. Make a guess!
    
    "I think this is the letter X!"
    """
    
    axes[1, 1].text(0.5, 0.3, process_text, ha='center', va='center', 
                    fontsize=11, transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    axes[1, 1].axis('off')
    
    # Step 6: Final prediction
    axes[1, 2].text(0.5, 0.5, 'Final Answer:\n\nLETTER X!\n\nConfidence: 95%', 
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    axes[1, 2].axis('off')
    
    # Remove axis ticks for cleaner look
    for ax in axes.flat:
        if ax != axes[1, 1] and ax != axes[1, 2]:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def create_interactive_demo():
    """Create an interactive demonstration"""
    print("🎨 Welcome to the CNN Magic Show! 🎨")
    print("=" * 50)
    print()
    
    print("🧠 What is a CNN?")
    print("A CNN is like giving a computer magical glasses")
    print("that help it understand what's in pictures!")
    print()
    
    input("Press Enter to see how it works... 🎬")
    
    # Show the visualization
    visualize_cnn_process()
    
    print("\n🎉 Amazing! You just saw how a CNN works!")
    print("\nHere's what happened:")
    print("1. 👁️  We showed the computer a letter 'X'")
    print("2. 🔍  The computer used special filters to find edges")
    print("3. ✨  It found the important patterns")
    print("4. 🏊‍♀️  It kept only the most important information")
    print("5. 🎯  It made a smart guess: 'This is the letter X!'")
    print()
    print("🌟 CNNs can learn to recognize:")
    print("   • 🐱 Cats and dogs")
    print("   • 🚗 Cars and trucks") 
    print("   • 👨‍👩‍👧‍👦 People's faces")
    print("   • 🏠 Houses and buildings")
    print("   • 📝 Letters and numbers")
    print("   • And much more!")
    print()
    print("🎓 You're now a CNN expert! Great job! 🎉")

if __name__ == "__main__":
    create_interactive_demo()
