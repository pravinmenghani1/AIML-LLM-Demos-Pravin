# 🔓 Interactive Decoder Demo

Multiple decoder demonstrations including general encoding schemes and LLM-specific decoders.

## 🌐 General Decoders (`decoder_demo.html`)

Interactive browser-based decoder with visual effects.

**Features:**
- Base64, Morse code, Binary, Caesar cipher, URL decoders
- Visual effects and color-coded output
- Real-time decoding feedback

## 🤖 LLM Decoder Demo (`llm_decoder_demo.html`)

Interactive visualization of how Large Language Models process information.

**Features:**
- **Tokenization Decoder:** Shows text → token conversion
- **Logits to Text:** Visualizes probability distributions
- **Attention Patterns:** Interactive attention matrix visualization
- **Embedding Decoder:** Vector representation visualization
- **Generation Process:** Step-by-step text generation simulation

## 🖥️ Terminal Demos

### General Decoder (`terminal_decoder.py`)
Animated command-line decoder with step-by-step visualization.

### LLM Terminal Decoder (`llm_terminal_decoder.py`)
Deep dive into transformer architecture with animations.

**Features:**
- Tokenization process breakdown
- Softmax probability calculations
- Attention weight visualization
- Embedding vector display
- Layer-by-layer transformer processing
- Generation step simulation

## 🎯 Sample Inputs

**General Decoders:**
- Base64: `SGVsbG8gV29ybGQ=` → "Hello World"
- Morse: `.... . .-.. .-.. --- / .-- --- .-. .-.. -..` → "HELLO WORLD"
- Binary: `01001000 01100101 01101100 01101100 01101111` → "Hello"

**LLM Decoders:**
- Tokenization: "Hello, world!" → [1009, 1002, 1010, 1002]
- Logits: [2.1, 1.8, 0.5, 3.2] → probability distribution
- Generation: "The future of AI is" → "exciting and transformative"

## 🚀 Quick Start

```bash
# General decoders
open decoder_demo.html
python3 terminal_decoder.py

# LLM-focused decoders
open llm_decoder_demo.html
python3 llm_terminal_decoder.py
```

Perfect for understanding both traditional encoding schemes and modern LLM internals!
