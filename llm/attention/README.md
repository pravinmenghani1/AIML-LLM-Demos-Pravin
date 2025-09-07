# Transformer Demo: "Attention is All You Need"

This demo implements the core concepts from the seminal paper "Attention is All You Need" by Vaswani et al.

## Key Components Implemented

- **Multi-Head Self-Attention**: The core mechanism that allows the model to focus on different parts of the input
- **Positional Encoding**: Sinusoidal encodings to give the model information about token positions
- **Transformer Blocks**: Complete encoder blocks with attention, normalization, and feed-forward layers
- **Scaled Dot-Product Attention**: The fundamental attention mechanism with Q, K, V matrices
- **Encoder-Decoder Architecture**: Full translation model with cross-attention

## Files

- `transformer_demo.py` - Core Transformer implementation (encoder-only)
- `translation_demo.py` - **NEW!** Complete encoder-decoder for translation
- `kid_friendly_viz.py` - **NEW!** Kid-friendly explanation with visualizations
- `attention_viz.py` - Visualization tools for attention patterns
- `requirements.txt` - Dependencies

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic encoder demo
python transformer_demo.py

# Run translation demo (encoder-decoder)
python translation_demo.py

# Kid-friendly explanation with pictures
python kid_friendly_viz.py

# Generate attention visualizations
python attention_viz.py
```

## Translation Demo Features

The new translation demo shows:

1. **Encoder Processing**: How "Je suis étudiant" gets processed
   - Word embeddings + positional encoding
   - Self-attention (French words talking to each other)
   - Feed-forward processing

2. **Decoder Processing**: How "I am a student" gets generated
   - Masked self-attention (can only see previous English words)
   - Cross-attention (English words looking at French words)
   - Output generation with probability distributions

3. **Kid-Friendly Explanation**: 
   - Simple language explaining each step
   - Colorful attention heatmaps
   - Shows exactly which words pay attention to which

## Key Concepts Demonstrated

1. **Self-Attention Formula**: Attention(Q,K,V) = softmax(QK^T/√d_k)V
2. **Multi-Head Attention**: Parallel attention mechanisms with different learned projections
3. **Positional Encoding**: PE(pos,2i) = sin(pos/10000^(2i/d_model))
4. **Residual Connections**: Skip connections around each sub-layer
5. **Layer Normalization**: Applied after each residual connection
6. **Cross-Attention**: Decoder attending to encoder outputs
7. **Causal Masking**: Preventing decoder from seeing future tokens

## Architecture Overview

### Encoder-Decoder (Translation)
```
French Input: "Je suis étudiant"
    ↓
Input Embeddings + Positional Encoding
    ↓
Multi-Head Self-Attention (French ↔ French)
    ↓
Add & Norm
    ↓
Feed Forward Network
    ↓
Add & Norm
    ↓
ENCODER OUTPUT
    ↓
English Target: "<start> I am a student"
    ↓
Target Embeddings + Positional Encoding
    ↓
Masked Multi-Head Self-Attention (English ↔ English)
    ↓
Add & Norm
    ↓
Multi-Head Cross-Attention (English ↔ French)
    ↓
Add & Norm
    ↓
Feed Forward Network
    ↓
Add & Norm
    ↓
Output Projection → "I am a student"
```
