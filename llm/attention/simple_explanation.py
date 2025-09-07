import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def draw_attention_story():
    """Explain attention like a story with pictures"""
    
    # Story 1: What is Attention?
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("🧠 ATTENTION: How Computers Read Like Humans", fontsize=16, fontweight='bold')
    
    # Panel 1: Human reading
    ax1 = axes[0, 0]
    ax1.set_title("👶 How YOU read a sentence", fontsize=12, fontweight='bold')
    
    sentence = ["The", "cat", "sat", "on", "mat"]
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
    
    for i, (word, color) in enumerate(zip(sentence, colors)):
        rect = patches.Rectangle((i, 0), 0.8, 0.5, facecolor=color, edgecolor='black')
        ax1.add_patch(rect)
        ax1.text(i+0.4, 0.25, word, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw attention arrows
    ax1.annotate('', xy=(1.4, 0.6), xytext=(0.4, 0.6), 
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax1.text(0.9, 0.8, 'You look at "cat"\nand think about "The"', ha='center', fontsize=9, color='red')
    
    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-0.2, 1.2)
    ax1.axis('off')
    
    # Panel 2: Computer attention
    ax2 = axes[0, 1]
    ax2.set_title("🤖 How COMPUTER reads (Attention)", fontsize=12, fontweight='bold')
    
    # Create attention matrix
    attention_matrix = np.array([
        [0.1, 0.8, 0.05, 0.03, 0.02],  # "The" pays attention to...
        [0.3, 0.4, 0.2, 0.05, 0.05],   # "cat" pays attention to...
        [0.1, 0.3, 0.4, 0.15, 0.05],   # "sat" pays attention to...
        [0.05, 0.1, 0.3, 0.5, 0.05],   # "on" pays attention to...
        [0.02, 0.08, 0.1, 0.2, 0.6]    # "mat" pays attention to...
    ])
    
    im = ax2.imshow(attention_matrix, cmap='Reds', interpolation='nearest')
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(sentence)
    ax2.set_yticklabels(sentence)
    ax2.set_xlabel('Looks at →')
    ax2.set_ylabel('Word ↓')
    
    # Add numbers
    for i in range(5):
        for j in range(5):
            ax2.text(j, i, f'{attention_matrix[i,j]:.1f}', ha='center', va='center', 
                    color='white' if attention_matrix[i,j] > 0.3 else 'black', fontweight='bold')
    
    # Panel 3: Step by step process
    ax3 = axes[1, 0]
    ax3.set_title("🔍 Step-by-Step: How Attention Works", fontsize=12, fontweight='bold')
    
    steps = [
        "1. 📝 Read all words",
        "2. 🤔 For each word, ask:",
        "   'Which other words help",
        "    me understand this word?'",
        "3. 📊 Give scores (0-1)",
        "4. 🎯 Focus on high scores"
    ]
    
    for i, step in enumerate(steps):
        ax3.text(0.1, 0.9 - i*0.15, step, fontsize=10, transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax3.axis('off')
    
    # Panel 4: Why it's useful
    ax4 = axes[1, 1]
    ax4.set_title("✨ Why Attention is MAGIC!", fontsize=12, fontweight='bold')
    
    benefits = [
        "🎯 Understands relationships",
        "📍 Knows word positions matter", 
        "🧩 Connects distant words",
        "🚀 Works for any sentence length",
        "🎨 Can focus on multiple things"
    ]
    
    for i, benefit in enumerate(benefits):
        ax4.text(0.1, 0.9 - i*0.18, benefit, fontsize=10, transform=ax4.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/pravinmenghani/Downloads/demos/llm/attention/attention_story.png', dpi=150, bbox_inches='tight')
    plt.show()

def draw_transformer_building_blocks():
    """Show Transformer like building blocks"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_title("🏗️ TRANSFORMER: Like Building with LEGO Blocks!", fontsize=16, fontweight='bold')
    
    # Draw the transformer stack
    blocks = [
        ("📤 OUTPUT", "lightgreen", "Final answer!"),
        ("🧠 TRANSFORMER BLOCK 6", "lightblue", "Think more..."),
        ("🧠 TRANSFORMER BLOCK 5", "lightblue", "Think more..."),
        ("🧠 TRANSFORMER BLOCK 4", "lightblue", "Think more..."),
        ("🧠 TRANSFORMER BLOCK 3", "lightblue", "Think more..."),
        ("🧠 TRANSFORMER BLOCK 2", "lightblue", "Think more..."),
        ("🧠 TRANSFORMER BLOCK 1", "lightblue", "First thoughts"),
        ("📍 ADD POSITION INFO", "lightyellow", "Where is each word?"),
        ("📝 WORD EMBEDDINGS", "lightcoral", "Turn words into numbers"),
        ("📥 INPUT WORDS", "lightgray", "The cat sat on mat")
    ]
    
    y_pos = 0
    for i, (label, color, description) in enumerate(blocks):
        # Main block
        rect = patches.Rectangle((1, y_pos), 6, 0.8, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(4, y_pos + 0.4, label, ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Description
        ax.text(8, y_pos + 0.4, description, ha='left', va='center', fontsize=10, style='italic')
        
        # Arrow (except for last block)
        if i < len(blocks) - 1:
            ax.annotate('', xy=(4, y_pos + 0.9), xytext=(4, y_pos + 1.1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
        
        y_pos += 1.2
    
    # Add detailed view of one transformer block
    ax.text(0.5, len(blocks) * 1.2 + 1, "🔍 INSIDE ONE TRANSFORMER BLOCK:", 
            fontsize=14, fontweight='bold', color='darkblue')
    
    mini_blocks = [
        ("🎯 ATTENTION", "What words to focus on?"),
        ("➕ ADD & NORMALIZE", "Combine and clean up"),
        ("🧮 FEED FORWARD", "Do math calculations"),
        ("➕ ADD & NORMALIZE", "Combine and clean up again")
    ]
    
    x_start = 0.5
    for i, (name, desc) in enumerate(mini_blocks):
        rect = patches.Rectangle((x_start + i*2.2, len(blocks) * 1.2 + 2), 2, 0.6, 
                               facecolor='lightsteelblue', edgecolor='navy', linewidth=1)
        ax.add_patch(rect)
        ax.text(x_start + i*2.2 + 1, len(blocks) * 1.2 + 2.3, name, 
               ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x_start + i*2.2 + 1, len(blocks) * 1.2 + 1.5, desc, 
               ha='center', va='center', fontsize=8, style='italic')
        
        if i < len(mini_blocks) - 1:
            ax.annotate('', xy=(x_start + (i+1)*2.2 - 0.1, len(blocks) * 1.2 + 2.3), 
                       xytext=(x_start + i*2.2 + 2.1, len(blocks) * 1.2 + 2.3),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='navy'))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, len(blocks) * 1.2 + 4)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/pravinmenghani/Downloads/demos/llm/attention/transformer_blocks.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_simple_demo():
    """Create a super simple working example"""
    
    print("🎉 WELCOME TO TRANSFORMER KINDERGARTEN! 🎉")
    print("=" * 50)
    print()
    
    # Simple example
    sentence = "The cat sat"
    words = sentence.split()
    
    print(f"📝 Our sentence: '{sentence}'")
    print(f"📚 Words: {words}")
    print()
    
    # Show how each word looks at others
    print("🔍 HOW EACH WORD PAYS ATTENTION:")
    print("-" * 30)
    
    # Simple attention scores (made up for demo)
    attention_scores = {
        "The": {"The": 0.1, "cat": 0.8, "sat": 0.1},
        "cat": {"The": 0.3, "cat": 0.5, "sat": 0.2}, 
        "sat": {"The": 0.2, "cat": 0.4, "sat": 0.4}
    }
    
    for word in words:
        print(f"\n'{word}' looks at:")
        for other_word, score in attention_scores[word].items():
            bars = "█" * int(score * 10)
            print(f"  '{other_word}': {score:.1f} {bars}")
    
    print("\n" + "=" * 50)
    print("🧠 WHAT THIS MEANS:")
    print("• 'The' pays most attention to 'cat' (0.8)")
    print("• 'cat' pays most attention to itself (0.5)")  
    print("• 'sat' pays equal attention to 'cat' and itself")
    print("\n✨ This helps the computer understand the sentence better!")

if __name__ == "__main__":
    print("Creating visual explanations...")
    create_simple_demo()
    print("\nGenerating pictures...")
    draw_attention_story()
    draw_transformer_building_blocks()
    print("\n🎨 Pictures saved! Check the files:")
    print("• attention_story.png")
    print("• transformer_blocks.png")
