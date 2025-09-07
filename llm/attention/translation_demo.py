"""
Transformer Translation Demo: "Je suis étudiant" → "I am a student"
Simplified for understanding - like explaining to a 5-year-old!
"""

import torch
import torch.nn as nn
import math

class SimpleEncoder(nn.Module):
    """The Encoder - like a smart reader that understands French"""
    
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(10, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        print(f"🇫🇷 ENCODER: Reading French words: {x.tolist()}")
        
        # Step 1: Convert words to vectors (embeddings)
        embedded = self.embedding(x)
        print(f"   📝 Converted to vectors: shape {embedded.shape}")
        
        # Step 2: Add position information
        seq_len = x.size(1)
        positioned = embedded + self.pos_encoding[:, :seq_len, :]
        print(f"   📍 Added position info (where each word sits)")
        
        # Step 3: Self-attention (words look at each other)
        attn_out, attn_weights = self.attention(positioned, positioned, positioned)
        x = self.norm1(positioned + attn_out)
        print(f"   👀 Words looked at each other (self-attention)")
        
        # Step 4: Feed-forward processing
        ff_out = self.feedforward(x)
        output = self.norm2(x + ff_out)
        print(f"   🧠 Processed through neural network")
        print(f"   ✅ Encoder output shape: {output.shape}")
        
        return output, attn_weights

class SimpleDecoder(nn.Module):
    """The Decoder - like a smart translator that writes English"""
    
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(10, d_model)
        
        # Self-attention (masked)
        self.self_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Encoder-decoder attention
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def create_causal_mask(self, seq_len):
        """Creates mask so decoder can only look at previous words"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def forward(self, target, encoder_output):
        print(f"\n🇺🇸 DECODER: Starting to write English...")
        
        # Step 1: Embed target tokens
        embedded = self.embedding(target)
        seq_len = target.size(1)
        positioned = embedded + self.pos_encoding[:, :seq_len, :]
        print(f"   📝 Prepared target embeddings: {target.tolist()}")
        
        # Step 2: Masked self-attention (can only look at previous words)
        causal_mask = self.create_causal_mask(seq_len)
        self_attn_out, _ = self.self_attention(positioned, positioned, positioned, 
                                             attn_mask=causal_mask)
        x = self.norm1(positioned + self_attn_out)
        print(f"   👀 Self-attention (only looking at previous English words)")
        
        # Step 3: Cross-attention (look at French input)
        cross_attn_out, cross_weights = self.cross_attention(x, encoder_output, encoder_output)
        x = self.norm2(x + cross_attn_out)
        print(f"   🔍 Cross-attention (looking at French words for context)")
        
        # Step 4: Feed-forward
        ff_out = self.feedforward(x)
        x = self.norm3(x + ff_out)
        
        # Step 5: Generate word probabilities
        output = self.output_projection(x)
        print(f"   🎯 Generated word probabilities: shape {output.shape}")
        
        return output, cross_weights

class TranslationTransformer(nn.Module):
    """Complete Transformer for Translation"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64):
        super().__init__()
        self.encoder = SimpleEncoder(src_vocab_size, d_model)
        self.decoder = SimpleDecoder(tgt_vocab_size, d_model)
    
    def forward(self, src, tgt):
        encoder_output, enc_attn = self.encoder(src)
        decoder_output, dec_attn = self.decoder(tgt, encoder_output)
        return decoder_output, enc_attn, dec_attn

def demo_translation():
    """Demo: 'Je suis étudiant' → 'I am a student'"""
    
    print("🌟 TRANSFORMER TRANSLATION DEMO 🌟")
    print("=" * 50)
    print("Teaching a computer to translate like a 5-year-old would understand!")
    print()
    
    # Simple vocabulary
    french_vocab = {"<pad>": 0, "Je": 1, "suis": 2, "étudiant": 3}
    english_vocab = {"<pad>": 0, "<start>": 1, "I": 2, "am": 3, "a": 4, "student": 5, "<end>": 6}
    
    # Create model
    model = TranslationTransformer(len(french_vocab), len(english_vocab))
    
    # Input: "Je suis étudiant" 
    french_input = torch.tensor([[1, 2, 3]])  # [Je, suis, étudiant]
    
    # Target: "<start> I am a student"
    english_target = torch.tensor([[1, 2, 3, 4, 5]])  # [<start>, I, am, a, student]
    
    print("📚 VOCABULARY:")
    print(f"French: {french_vocab}")
    print(f"English: {english_vocab}")
    print()
    
    print("🔄 TRANSLATION PROCESS:")
    print("Input (French): Je suis étudiant")
    print("Target (English): I am a student")
    print()
    
    # Forward pass
    with torch.no_grad():
        output, enc_attn, dec_attn = model(french_input, english_target)
    
    print(f"\n📊 RESULTS:")
    print(f"Encoder attention shape: {enc_attn.shape}")
    print(f"Decoder cross-attention shape: {dec_attn.shape}")
    print(f"Final output shape: {output.shape}")
    
    # Show attention patterns
    print(f"\n🎯 ATTENTION PATTERNS:")
    print("Encoder self-attention (how French words relate):")
    print(torch.round(enc_attn[0, 0], decimals=2).numpy())  # First head
    
    print("\nDecoder cross-attention (English words looking at French):")
    print(torch.round(dec_attn[0, 0], decimals=2).numpy())  # First head
    
    print(f"\n🎉 SUMMARY:")
    print("✅ Encoder read and understood French")
    print("✅ Decoder generated English while looking at French")
    print("✅ Attention shows which words the model focuses on!")

if __name__ == "__main__":
    demo_translation()
