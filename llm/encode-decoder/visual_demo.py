import torch
import torch.nn.functional as F
from transformer_demo import Transformer, create_causal_mask
import time

class VisualTransformerDemo:
    def __init__(self):
        # Simple vocabularies for demo
        self.en_vocab = {
            "<pad>": 0, "<sos>": 1, "<eos>": 2,
            "hello": 3, "world": 4, "how": 5, "are": 6, "you": 7,
            "good": 8, "morning": 9, "nice": 10, "day": 11
        }
        
        self.es_vocab = {
            "<pad>": 0, "<sos>": 1, "<eos>": 2,
            "hola": 3, "mundo": 4, "como": 5, "estas": 6, "tu": 7,
            "buenos": 8, "dias": 9, "buen": 10, "dia": 11
        }
        
        # Reverse mappings
        self.en_idx_to_word = {v: k for k, v in self.en_vocab.items()}
        self.es_idx_to_word = {v: k for k, v in self.es_vocab.items()}
        
        # Create model
        self.model = Transformer(
            src_vocab_size=len(self.en_vocab),
            tgt_vocab_size=len(self.es_vocab),
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=256
        )
        
    def print_header(self, title):
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)
        
    def print_step(self, step, description):
        print(f"\n🔸 Step {step}: {description}")
        print("-" * 40)
        
    def tokens_to_words(self, tokens, vocab_map):
        return [vocab_map.get(t.item(), f"UNK({t.item()})") for t in tokens[0]]
        
    def demonstrate_encoding(self, english_sentence):
        self.print_header("ENCODER PHASE")
        
        # Convert sentence to tokens
        words = english_sentence.lower().split()
        tokens = [self.en_vocab["<sos>"]] + [self.en_vocab.get(w, 0) for w in words] + [self.en_vocab["<eos>"]]
        src_tensor = torch.tensor([tokens])
        
        print(f"📝 Input: '{english_sentence}'")
        print(f"🔤 Tokens: {self.tokens_to_words(src_tensor, self.en_idx_to_word)}")
        print(f"🔢 Token IDs: {tokens}")
        
        self.print_step(1, "Encoder processes entire English sentence")
        
        # Get encoder embeddings
        src_emb = self.model.src_embedding(src_tensor) * (self.model.d_model ** 0.5)
        src_emb += self.model.pos_encoding[:, :src_tensor.size(1), :].to(src_tensor.device)
        
        print(f"📊 Embedding shape: {src_emb.shape}")
        print("✅ Each word now has rich 128-dimensional representation")
        
        # Pass through encoder layers
        encoder_output = src_emb
        for i, layer in enumerate(self.model.encoder_layers):
            encoder_output = layer(encoder_output)
            print(f"🔄 After encoder layer {i+1}: Enhanced representations")
            
        print(f"🎯 Final encoder output shape: {encoder_output.shape}")
        print("✅ Encoder created context-aware representations of English sentence")
        
        return src_tensor, encoder_output
        
    def demonstrate_decoding(self, src_tensor, encoder_output, target_sentence):
        self.print_header("DECODER PHASE")
        
        print(f"🎯 Target: '{target_sentence}'")
        print("🔄 Generating Spanish translation word by word...")
        
        # Start with <sos> token
        generated = torch.tensor([[self.es_vocab["<sos>"]]])
        max_length = 10
        
        self.model.eval()
        with torch.no_grad():
            for step in range(max_length):
                self.print_step(step + 1, f"Generating word {step + 1}")
                
                # Show current state
                current_words = self.tokens_to_words(generated, self.es_idx_to_word)
                print(f"📝 Current Spanish: {' '.join(current_words)}")
                
                # Create causal mask
                tgt_mask = create_causal_mask(generated.size(1)).unsqueeze(0).unsqueeze(0)
                
                # Get decoder embeddings
                tgt_emb = self.model.tgt_embedding(generated) * (self.model.d_model ** 0.5)
                tgt_emb += self.model.pos_encoding[:, :generated.size(1), :].to(generated.device)
                
                # Pass through decoder layers
                decoder_output = tgt_emb
                for i, layer in enumerate(self.model.decoder_layers):
                    decoder_output = layer(decoder_output, encoder_output, tgt_mask)
                
                # Get predictions
                logits = self.model.output_projection(decoder_output)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # Show top predictions
                top_probs, top_indices = torch.topk(probs, 3)
                print("🤔 Top 3 predictions:")
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    word = self.es_idx_to_word.get(idx.item(), f"UNK({idx.item()})")
                    print(f"   {word}: {prob.item():.3f}")
                
                # Add to sequence
                generated = torch.cat([generated, next_token], dim=1)
                next_word = self.es_idx_to_word.get(next_token.item(), f"UNK({next_token.item()})")
                print(f"✅ Selected: '{next_word}'")
                
                # Stop if <eos>
                if next_token.item() == self.es_vocab["<eos>"]:
                    break
                    
                time.sleep(0.5)  # Pause for effect
                
        final_words = self.tokens_to_words(generated, self.es_idx_to_word)
        print(f"\n🎉 Final translation: {' '.join(final_words[1:-1])}")  # Remove <sos> and <eos>
        
    def run_demo(self, english_sentence, spanish_sentence):
        self.print_header("TRANSFORMER ENCODER-DECODER DEMO")
        print(f"🌍 Task: Translate '{english_sentence}' to Spanish")
        print("📚 Note: This is a toy model with random weights for demonstration")
        
        # Encoding phase
        src_tensor, encoder_output = self.demonstrate_encoding(english_sentence)
        
        # Decoding phase
        self.demonstrate_decoding(src_tensor, encoder_output, spanish_sentence)
        
        self.print_header("SUMMARY")
        print("🔍 What happened:")
        print("1. 🏗️  ENCODER: Read entire English sentence, created rich representations")
        print("2. 🎯 DECODER: Generated Spanish words one by one")
        print("3. 🔗 CROSS-ATTENTION: Decoder looked at English context for each Spanish word")
        print("4. 🚫 CAUSAL MASK: Decoder couldn't peek at future Spanish words")
        print("\n💡 In a real trained model, this would produce accurate translations!")

def main():
    print("🤖 TRANSFORMER ENCODER-DECODER ARCHITECTURE DEMO")
    print("\n📖 PROBLEM: Machine Translation (English → Spanish)")
    print("\n🎯 GOAL: Show how encoder-decoder attention works step by step")
    print("\nThe encoder reads the entire input sentence and creates context.")
    print("The decoder generates output word by word, attending to:")
    print("  • Previous output words (self-attention)")
    print("  • Input sentence context (cross-attention)")
    
    response = input("\n🚀 Would you like to run the visual demo? (y/n): ").lower().strip()
    
    if response == 'y' or response == 'yes':
        demo = VisualTransformerDemo()
        demo.run_demo("hello world", "hola mundo")
    else:
        print("👋 Demo cancelled. Run again when ready!")

if __name__ == "__main__":
    main()
