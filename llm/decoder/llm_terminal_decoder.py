#!/usr/bin/env python3
import time
import math
import random
import numpy as np
from typing import List, Dict, Tuple

class LLMDecoder:
    def __init__(self):
        self.vocab = {
            1000: '<BOS>', 1001: '<EOS>', 1002: '<UNK>', 1003: '<PAD>',
            1004: 'the', 1005: 'cat', 1006: 'dog', 1007: 'is', 1008: 'running',
            1009: 'hello', 1010: 'world', 1011: 'AI', 1012: 'model', 1013: 'neural'
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def animate_text(self, text: str, delay: float = 0.03):
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()
    
    def tokenize_demo(self, text: str):
        print("\n🔤 TOKENIZATION DECODER")
        print("=" * 60)
        print(f"Input text: '{text}'")
        
        # Simulate tokenization
        words = text.lower().replace(',', ' ,').replace('.', ' .').split()
        tokens = []
        
        print("\nTokenization process:")
        for word in words:
            if word in self.reverse_vocab:
                token_id = self.reverse_vocab[word]
                tokens.append((token_id, word))
                print(f"  '{word}' → Token ID: {token_id}")
            else:
                token_id = 1002  # <UNK>
                tokens.append((token_id, '<UNK>'))
                print(f"  '{word}' → Token ID: {token_id} (<UNK>)")
            time.sleep(0.3)
        
        print(f"\n✅ Final tokens: {[t[0] for t in tokens]}")
        return tokens
    
    def softmax(self, logits: List[float]) -> List[float]:
        max_logit = max(logits)
        exp_logits = [math.exp(x - max_logit) for x in logits]
        sum_exp = sum(exp_logits)
        return [x / sum_exp for x in exp_logits]
    
    def logits_to_tokens_demo(self):
        print("\n📊 LOGITS TO TOKEN DECODER")
        print("=" * 60)
        
        # Simulate model output logits
        logits = [2.1, 1.8, 0.5, 3.2, 1.1, 0.8]
        vocab_subset = ['the', 'cat', 'is', 'running', 'fast', 'slowly']
        
        print("Model output logits:")
        for i, (word, logit) in enumerate(zip(vocab_subset, logits)):
            print(f"  {word}: {logit:.2f}")
        
        print("\nApplying softmax...")
        time.sleep(1)
        
        probs = self.softmax(logits)
        
        print("\nProbability distribution:")
        for word, prob in zip(vocab_subset, probs):
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {word:8} {prob:.3f} |{bar}|")
            time.sleep(0.4)
        
        # Select token
        selected_idx = probs.index(max(probs))
        selected_token = vocab_subset[selected_idx]
        print(f"\n✅ Selected token: '{selected_token}' (prob: {max(probs):.3f})")
    
    def attention_visualization(self, tokens: List[str]):
        print("\n🎯 ATTENTION PATTERN DECODER")
        print("=" * 60)
        print(f"Tokens: {tokens}")
        
        n = len(tokens)
        print(f"\nGenerating {n}x{n} attention matrix...")
        
        # Simulate attention weights
        attention_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                # Simulate attention pattern (higher for nearby tokens)
                distance = abs(i - j)
                base_attention = 1.0 / (1 + distance * 0.5)
                noise = random.uniform(0.8, 1.2)
                attention = base_attention * noise
                row.append(attention)
            
            # Normalize row to sum to 1
            row_sum = sum(row)
            row = [x / row_sum for x in row]
            attention_matrix.append(row)
        
        print("\nAttention weights (query → key):")
        print("     ", end="")
        for token in tokens:
            print(f"{token:>8}", end="")
        print()
        
        for i, (query_token, row) in enumerate(zip(tokens, attention_matrix)):
            print(f"{query_token:>4} ", end="")
            for attention in row:
                intensity = "█" if attention > 0.3 else "▓" if attention > 0.15 else "░"
                print(f"{intensity:>7}", end=" ")
            print(f" (max: {max(row):.2f})")
            time.sleep(0.5)
    
    def embedding_visualization(self, word: str, dim: int = 16):
        print(f"\n🧠 EMBEDDING DECODER: '{word}'")
        print("=" * 60)
        
        # Simulate embedding vector
        random.seed(hash(word) % 1000)  # Consistent for same word
        embedding = [random.uniform(-1, 1) for _ in range(dim)]
        
        print(f"Embedding vector ({dim} dimensions):")
        for i, value in enumerate(embedding):
            bar_length = int(abs(value) * 20)
            direction = "+" if value >= 0 else "-"
            bar = "█" * bar_length
            print(f"  dim_{i:2d}: {value:6.3f} {direction}{bar}")
            time.sleep(0.2)
        
        # Show similarity to other words
        print(f"\nSimilarity to other words:")
        similar_words = ['neural', 'network', 'model', 'AI']
        for other_word in similar_words:
            if other_word != word:
                # Simulate cosine similarity
                similarity = random.uniform(0.1, 0.9)
                print(f"  {word} ↔ {other_word}: {similarity:.3f}")
                time.sleep(0.3)
    
    def generation_process_demo(self, prompt: str):
        print(f"\n🔄 GENERATION PROCESS DECODER")
        print("=" * 60)
        print(f"Prompt: '{prompt}'")
        
        generated_tokens = ['exciting', 'and', 'transformative', 'technology']
        current_sequence = prompt.split()
        
        for step, next_token in enumerate(generated_tokens, 1):
            print(f"\n--- Generation Step {step} ---")
            print(f"Current sequence: {' '.join(current_sequence)}")
            
            # Simulate forward pass
            print("Forward pass through transformer...")
            for layer in range(1, 4):
                print(f"  Layer {layer}: Processing attention & feedforward")
                time.sleep(0.3)
            
            # Simulate logits computation
            print("Computing output logits...")
            time.sleep(0.5)
            
            # Show top candidates
            candidates = [next_token, 'amazing', 'powerful', 'innovative']
            logits = [3.2, 2.1, 1.8, 1.5]
            probs = self.softmax(logits)
            
            print("Top candidates:")
            for candidate, prob in zip(candidates, probs):
                print(f"  {candidate}: {prob:.3f}")
            
            print(f"✅ Selected: '{next_token}'")
            current_sequence.append(next_token)
            time.sleep(1)
        
        print(f"\n🎉 Final output: '{' '.join(current_sequence)}'")
    
    def transformer_layer_demo(self):
        print("\n⚙️ TRANSFORMER LAYER DECODER")
        print("=" * 60)
        
        sequence = ['The', 'cat', 'is', 'sleeping']
        print(f"Input sequence: {sequence}")
        
        for layer_num in range(1, 4):
            print(f"\n--- Layer {layer_num} ---")
            
            # Multi-head attention
            print("1. Multi-Head Attention:")
            for head in range(1, 5):
                print(f"   Head {head}: Computing attention weights...")
                time.sleep(0.2)
            print("   Concatenating heads and projecting...")
            time.sleep(0.3)
            
            # Add & Norm
            print("2. Add & Norm (residual connection)")
            time.sleep(0.3)
            
            # Feed Forward
            print("3. Feed Forward Network:")
            print("   Linear → ReLU → Linear")
            time.sleep(0.4)
            
            # Add & Norm
            print("4. Add & Norm (residual connection)")
            time.sleep(0.3)
            
            print(f"   ✅ Layer {layer_num} complete")
        
        print("\n🎯 Final layer norm and output projection")
        time.sleep(0.5)
    
    def run_demo(self):
        print("🤖 LLM DECODER DEMONSTRATION")
        print("=" * 70)
        
        demos = [
            ("Tokenization", lambda: self.tokenize_demo("Hello, world! The cat is running.")),
            ("Logits to Tokens", self.logits_to_tokens_demo),
            ("Attention Patterns", lambda: self.attention_visualization(['The', 'cat', 'is', 'sleeping'])),
            ("Embeddings", lambda: self.embedding_visualization('neural')),
            ("Generation Process", lambda: self.generation_process_demo("The future of AI is")),
            ("Transformer Layers", self.transformer_layer_demo)
        ]
        
        for name, demo_func in demos:
            input(f"\nPress Enter to see {name} demo...")
            demo_func()
            time.sleep(1)
        
        print("\n🎉 LLM Decoder demo complete!")
        print("This demonstrates how language models process and decode information internally.")

if __name__ == "__main__":
    decoder = LLMDecoder()
    decoder.run_demo()
