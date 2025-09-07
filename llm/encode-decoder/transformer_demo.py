import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        return self.norm2(x + ff_output)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # Self-attention
        attn_output = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + attn_output)
        
        # Cross-attention
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + cross_attn_output)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        return self.norm3(x + ff_output)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        # Encoder and Decoder layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb += self.pos_encoding[:, :src.size(1), :].to(src.device)
        
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
            
        # Decoder
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb += self.pos_encoding[:, :tgt.size(1), :].to(tgt.device)
        
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_mask)
            
        return self.output_projection(decoder_output)

def create_causal_mask(size):
    """Create causal mask for decoder self-attention"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

# Demo usage
if __name__ == "__main__":
    # Model parameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    n_heads = 8
    n_layers = 6
    
    # Create model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers)
    
    # Sample input
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # Create causal mask for decoder
    tgt_mask = create_causal_mask(tgt_seq_len).unsqueeze(0).unsqueeze(0)
    
    # Forward pass
    output = model(src, tgt, tgt_mask=tgt_mask)
    
    print(f"Input source shape: {src.shape}")
    print(f"Input target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
