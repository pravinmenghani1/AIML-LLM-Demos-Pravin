# Transformer Demo: "Attention is All You Need"

This demo implements the core concepts from the seminal paper "Attention is All You Need" by Vaswani et al.

## Key Components Implemented

- **Multi-Head Self-Attention**: The core mechanism that allows the model to focus on different parts of the input
- **Positional Encoding**: Sinusoidal encodings to give the model information about token positions
- **Transformer Blocks**: Complete encoder blocks with attention, normalization, and feed-forward layers
- **Scaled Dot-Product Attention**: The fundamental attention mechanism with Q, K, V matrices

## Files

- `transformer_demo.py` - Core Transformer implementation
- `attention_viz.py` - Visualization tools for attention patterns
- `requirements.txt` - Dependencies

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic demo
python transformer_demo.py

# Generate attention visualizations
python attention_viz.py
```

## Key Concepts Demonstrated

1. **Self-Attention Formula**: Attention(Q,K,V) = softmax(QK^T/√d_k)V
2. **Multi-Head Attention**: Parallel attention mechanisms with different learned projections
3. **Positional Encoding**: PE(pos,2i) = sin(pos/10000^(2i/d_model))
4. **Residual Connections**: Skip connections around each sub-layer
5. **Layer Normalization**: Applied after each residual connection

## Architecture Overview

```
Input Embeddings + Positional Encoding
    ↓
Multi-Head Self-Attention
    ↓
Add & Norm
    ↓
Feed Forward Network
    ↓
Add & Norm
    ↓
(Repeat N times)
    ↓
Output Projection
```
