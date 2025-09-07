#!/usr/bin/env python3
import time
import base64
import urllib.parse
from typing import Dict, List

class TerminalDecoder:
    def __init__(self):
        self.morse_code = {
            '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
            '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
            '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
            '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
            '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
            '--..': 'Z', ' ': ' '
        }
        
    def animate_text(self, text: str, delay: float = 0.05):
        """Animate text character by character"""
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()
    
    def decode_base64_animated(self, encoded: str):
        print("\n🔤 BASE64 DECODER")
        print("=" * 50)
        print(f"Input: {encoded}")
        
        # Animate decoding process
        print("\nDecoding", end="")
        for _ in range(3):
            print(".", end="", flush=True)
            time.sleep(0.5)
        
        try:
            decoded = base64.b64decode(encoded).decode('utf-8')
            print(f"\n✅ Decoded: ", end="")
            self.animate_text(decoded, 0.1)
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def decode_morse_animated(self, morse: str):
        print("\n📡 MORSE CODE DECODER")
        print("=" * 50)
        print(f"Input: {morse}")
        
        words = morse.split(' / ')
        decoded_words = []
        
        for word in words:
            letters = word.split(' ')
            decoded_word = ""
            
            print(f"\nDecoding word: {word}")
            for letter in letters:
                if letter in self.morse_code:
                    char = self.morse_code[letter]
                    decoded_word += char
                    print(f"  {letter} → {char}")
                    time.sleep(0.3)
            
            decoded_words.append(decoded_word)
        
        result = ' '.join(decoded_words)
        print(f"\n✅ Final result: ", end="")
        self.animate_text(result, 0.1)
    
    def decode_binary_animated(self, binary: str):
        print("\n🔢 BINARY DECODER")
        print("=" * 50)
        print(f"Input: {binary}")
        
        bytes_list = binary.split()
        decoded = ""
        
        print("\nDecoding each byte:")
        for byte_str in bytes_list:
            if len(byte_str) == 8 and all(c in '01' for c in byte_str):
                decimal = int(byte_str, 2)
                char = chr(decimal)
                decoded += char
                
                print(f"  {byte_str} → {decimal:3d} → '{char}'")
                time.sleep(0.4)
        
        print(f"\n✅ Decoded text: ", end="")
        self.animate_text(decoded, 0.1)
    
    def decode_caesar_animated(self, text: str, shift: int = 3):
        print(f"\n🔐 CAESAR CIPHER DECODER (Shift: -{shift})")
        print("=" * 50)
        print(f"Input: {text}")
        
        decoded = ""
        print("\nDecoding character by character:")
        
        for char in text:
            if char.isalpha():
                start = ord('A') if char.isupper() else ord('a')
                shifted = chr((ord(char) - start - shift) % 26 + start)
                decoded += shifted
                print(f"  '{char}' → '{shifted}'")
                time.sleep(0.2)
            else:
                decoded += char
                print(f"  '{char}' → '{char}' (unchanged)")
                time.sleep(0.1)
        
        print(f"\n✅ Decoded: ", end="")
        self.animate_text(decoded, 0.1)
    
    def visual_binary_breakdown(self, text: str):
        print(f"\n🎨 VISUAL BINARY BREAKDOWN: '{text}'")
        print("=" * 50)
        
        for char in text:
            binary = format(ord(char), '08b')
            print(f"'{char}' → {ord(char):3d} → {binary}")
            
            # Visual representation
            visual = ""
            for bit in binary:
                visual += "█" if bit == '1' else "░"
            print(f"     Visual: {visual}")
            time.sleep(0.5)
    
    def run_demo(self):
        print("🔓 INTERACTIVE DECODER DEMO")
        print("=" * 60)
        
        demos = [
            ("Base64", "SGVsbG8gV29ybGQ=", self.decode_base64_animated),
            ("Morse", ".... . .-.. .-.. --- / .-- --- .-. .-.. -..", self.decode_morse_animated),
            ("Binary", "01001000 01100101 01101100 01101100 01101111", self.decode_binary_animated),
            ("Caesar", "Khoor Zruog", lambda x: self.decode_caesar_animated(x, 3))
        ]
        
        for name, sample, decoder in demos:
            input(f"\nPress Enter to see {name} decoding demo...")
            decoder(sample)
            time.sleep(1)
        
        # Bonus: Visual binary breakdown
        input("\nPress Enter for visual binary breakdown...")
        self.visual_binary_breakdown("Hello")
        
        print("\n🎉 Demo complete! Try running with your own inputs.")

if __name__ == "__main__":
    decoder = TerminalDecoder()
    decoder.run_demo()
