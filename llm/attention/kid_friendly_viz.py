"""
Kid-Friendly Transformer Visualization
Shows how words "talk" to each other with colorful connections!
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from translation_demo import TranslationTransformer

def create_attention_heatmap(attention_weights, source_words, target_words, title):
    """Create a colorful heatmap showing which words pay attention to which"""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    # Set labels
    ax.set_xticks(range(len(source_words)))
    ax.set_yticks(range(len(target_words)))
    ax.set_xticklabels(source_words)
    ax.set_yticklabels(target_words)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Attention Strength')
    
    # Add text annotations
    for i in range(len(target_words)):
        for j in range(len(source_words)):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                         ha="center", va="center", color="red", fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Source Words (What we\'re reading)', fontweight='bold')
    ax.set_ylabel('Target Words (What we\'re writing)', fontweight='bold')
    
    plt.tight_layout()
    return fig

def explain_like_five():
    """Explain transformer attention like talking to a 5-year-old"""
    
    print("🎨 TRANSFORMER ATTENTION - EXPLAINED FOR KIDS! 🎨")
    print("=" * 60)
    print()
    
    print("🤔 Imagine you're learning to translate...")
    print("   You have a French sentence: 'Je suis étudiant'")
    print("   You want to write in English: 'I am a student'")
    print()
    
    print("🧠 The computer has two helpers:")
    print("   1. 📖 READER (Encoder): Reads and understands French")
    print("   2. ✏️  WRITER (Decoder): Writes the English translation")
    print()
    
    # Create model and get attention
    french_vocab = {"<pad>": 0, "Je": 1, "suis": 2, "étudiant": 3}
    english_vocab = {"<pad>": 0, "<start>": 1, "I": 2, "am": 3, "a": 4, "student": 5}
    
    model = TranslationTransformer(len(french_vocab), len(english_vocab))
    
    french_input = torch.tensor([[1, 2, 3]])  # [Je, suis, étudiant]
    english_target = torch.tensor([[1, 2, 3, 4, 5]])  # [<start>, I, am, a, student]
    
    with torch.no_grad():
        output, enc_attn, dec_attn = model(french_input, english_target)
    
    # Prepare words for visualization
    french_words = ["Je", "suis", "étudiant"]
    english_words = ["<start>", "I", "am", "a", "student"]
    
    print("👀 STEP 1: READER looks at French words")
    print("   Each French word 'talks' to other French words")
    print("   This helps understand: 'Je' goes with 'suis', 'suis' goes with 'étudiant'")
    print()
    
    # Show encoder self-attention
    enc_attention = enc_attn[0].numpy()  # First batch, all heads averaged or first head
    if len(enc_attention.shape) == 3:  # If multiple heads, take first head
        enc_attention = enc_attention[0]
    
    print("📊 READER'S ATTENTION MAP:")
    print("   Numbers show how much each word pays attention to others")
    for i, word in enumerate(french_words):
        attention_str = " ".join([f"{french_words[j]}({enc_attention[i,j]:.2f})" 
                                for j in range(len(french_words))])
        print(f"   {word} looks at: {attention_str}")
    print()
    
    print("✏️  STEP 2: WRITER creates English translation")
    print("   While writing each English word, it looks back at French words")
    print("   This helps decide: 'Je' → 'I', 'suis' → 'am', 'étudiant' → 'student'")
    print()
    
    # Show decoder cross-attention
    dec_attention = dec_attn[0].numpy()  # First batch
    if len(dec_attention.shape) == 3:  # If multiple heads, take first head
        dec_attention = dec_attention[0]
    
    print("📊 WRITER'S ATTENTION MAP:")
    print("   Shows which French words help write each English word")
    for i, eng_word in enumerate(english_words):
        attention_str = " ".join([f"{french_words[j]}({dec_attention[i,j]:.2f})" 
                                for j in range(len(french_words))])
        print(f"   Writing '{eng_word}' by looking at: {attention_str}")
    print()
    
    print("🎯 WHAT THIS MEANS:")
    print("   ✨ Higher numbers = 'I'm paying more attention to this word!'")
    print("   ✨ The computer learns which words are important for translation")
    print("   ✨ Just like you might look back at French while writing English!")
    print()
    
    # Create visualizations
    fig1 = create_attention_heatmap(
        enc_attention, 
        french_words, 
        french_words,
        "🇫🇷 READER: How French words talk to each other"
    )
    
    fig2 = create_attention_heatmap(
        dec_attention, 
        french_words, 
        english_words,
        "🇺🇸 WRITER: How English words look at French words"
    )
    
    # Save plots
    fig1.savefig('/Users/pravinmenghani/Downloads/demos/llm/attention/encoder_attention.png', 
                 dpi=150, bbox_inches='tight')
    fig2.savefig('/Users/pravinmenghani/Downloads/demos/llm/attention/decoder_attention.png', 
                 dpi=150, bbox_inches='tight')
    
    print("📸 SAVED COLORFUL PICTURES:")
    print("   📁 encoder_attention.png - Shows how French words connect")
    print("   📁 decoder_attention.png - Shows how English looks at French")
    print()
    print("🎉 Now you understand how transformers translate languages!")
    print("   It's like having a super-smart reading and writing assistant!")

if __name__ == "__main__":
    explain_like_five()
