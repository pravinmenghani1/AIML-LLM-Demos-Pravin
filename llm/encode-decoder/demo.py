import torch
import torch.nn as nn
from transformer_demo import Transformer, create_causal_mask

def simple_translation_demo():
    """Demo showing encoder-decoder for simple sequence translation"""
    
    # Simple vocabulary mapping
    src_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "hello": 3, "world": 4, "how": 5, "are": 6, "you": 7}
    tgt_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "hola": 3, "mundo": 4, "como": 5, "estas": 6, "tu": 7}
    
    # Create model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256
    )
    
    # Sample sequences: "hello world" -> "hola mundo"
    src_seq = torch.tensor([[1, 3, 4, 2]])  # <sos> hello world <eos>
    tgt_seq = torch.tensor([[1, 3, 4]])     # <sos> hola mundo (without <eos> for training)
    
    # Create causal mask for decoder
    tgt_mask = create_causal_mask(tgt_seq.size(1)).unsqueeze(0).unsqueeze(0)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(src_seq, tgt_seq, tgt_mask=tgt_mask)
        predictions = torch.argmax(output, dim=-1)
    
    print("=== Encoder-Decoder Translation Demo ===")
    print(f"Source sequence: {src_seq}")
    print(f"Target sequence: {tgt_seq}")
    print(f"Model output shape: {output.shape}")
    print(f"Predicted tokens: {predictions}")
    
    return model, output

def inference_demo():
    """Demo showing step-by-step inference"""
    
    # Create model
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=128,
        n_heads=4,
        n_layers=2
    )
    
    # Source sequence
    src = torch.randint(1, 50, (1, 8))  # Random source sequence
    
    print("\n=== Step-by-step Inference Demo ===")
    print(f"Source sequence: {src}")
    
    # Start with <sos> token
    tgt = torch.tensor([[1]])  # Start with <sos>
    max_length = 10
    
    model.eval()
    with torch.no_grad():
        for step in range(max_length):
            # Create causal mask
            tgt_mask = create_causal_mask(tgt.size(1)).unsqueeze(0).unsqueeze(0)
            
            # Forward pass
            output = model(src, tgt, tgt_mask=tgt_mask)
            
            # Get next token
            next_token = torch.argmax(output[:, -1, :], dim=-1, keepdim=True)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            print(f"Step {step + 1}: Generated token {next_token.item()}, Current sequence: {tgt}")
            
            # Stop if we generate <eos> token (assuming token 2 is <eos>)
            if next_token.item() == 2:
                break
    
    print(f"Final generated sequence: {tgt}")

if __name__ == "__main__":
    # Run demos
    simple_translation_demo()
    inference_demo()
