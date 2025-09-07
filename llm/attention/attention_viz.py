import torch
import matplotlib.pyplot as plt
import numpy as np
from transformer_demo import MultiHeadAttention, PositionalEncoding

def visualize_attention():
    # Create sample data
    d_model = 64
    n_heads = 8
    seq_len = 12
    
    # Sample sentence tokens (imagine: "The cat sat on the mat")
    tokens = ["The", "cat", "sat", "on", "the", "mat", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>"]
    
    # Create attention layer
    attention = MultiHeadAttention(d_model, n_heads)
    pos_encoding = PositionalEncoding(d_model)
    
    # Random embeddings for demo
    embeddings = torch.randn(1, seq_len, d_model)
    embeddings = pos_encoding(embeddings)
    
    # Get attention weights
    with torch.no_grad():
        _, attn_weights = attention(embeddings, embeddings, embeddings)
    
    # Visualize first head
    attn_matrix = attn_weights[0, 0].numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.title('Self-Attention Visualization (Head 1)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Add token labels
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    
    # Add values as text
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            plt.text(j, i, f'{attn_matrix[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/Users/pravinmenghani/Downloads/demos/llm/attention/attention_heatmap.png', dpi=150)
    plt.show()
    
    print("Attention heatmap saved as 'attention_heatmap.png'")

def compare_positional_encodings():
    d_model = 64
    max_len = 100
    
    pos_encoding = PositionalEncoding(d_model, max_len)
    
    # Get positional encodings
    positions = torch.arange(max_len).unsqueeze(0)
    dummy_input = torch.zeros(1, max_len, d_model)
    encoded = pos_encoding(dummy_input)
    
    plt.figure(figsize=(12, 6))
    
    # Plot first few dimensions
    for i in range(0, min(8, d_model), 2):
        plt.plot(encoded[0, :50, i].numpy(), label=f'dim {i}')
        plt.plot(encoded[0, :50, i+1].numpy(), label=f'dim {i+1}', linestyle='--')
    
    plt.title('Positional Encoding Patterns')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/Users/pravinmenghani/Downloads/demos/llm/attention/positional_encoding.png', dpi=150)
    plt.show()
    
    print("Positional encoding plot saved as 'positional_encoding.png'")

if __name__ == "__main__":
    print("Generating attention visualizations...")
    visualize_attention()
    compare_positional_encodings()
