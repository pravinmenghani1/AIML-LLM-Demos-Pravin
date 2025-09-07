def print_architecture():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    TRANSFORMER ARCHITECTURE                  ║
╚══════════════════════════════════════════════════════════════╝

INPUT: "Hello world"                    OUTPUT: "Hola mundo"
       ↓                                        ↑
   ┌─────────┐                              ┌─────────┐
   │ Tokens  │                              │ Tokens  │
   │ [1,3,4] │                              │ [1,3,4] │
   └─────────┘                              └─────────┘
       ↓                                        ↑
   ┌─────────┐                              ┌─────────┐
   │Embedding│                              │Linear   │
   │+ Pos    │                              │Layer    │
   └─────────┘                              └─────────┘
       ↓                                        ↑
┌───────────────┐                        ┌───────────────┐
│   ENCODER     │                        │   DECODER     │
│               │                        │               │
│ ┌───────────┐ │                        │ ┌───────────┐ │
│ │Self-Attn  │ │                        │ │Self-Attn  │ │
│ │(all words │ │                        │ │(causal)   │ │
│ │see each   │ │                        │ │           │ │
│ │other)     │ │                        │ └───────────┘ │
│ └───────────┘ │                        │       ↓       │
│       ↓       │                        │ ┌───────────┐ │
│ ┌───────────┐ │                        │ │Cross-Attn │ │
│ │Feed       │ │                        │ │(looks at  │ │
│ │Forward    │ │                        │ │encoder)   │ │
│ └───────────┘ │                        │ └───────────┘ │
│               │                        │       ↓       │
│ (Repeat N     │                        │ ┌───────────┐ │
│  layers)      │                        │ │Feed       │ │
│               │                        │ │Forward    │ │
└───────────────┘                        │ └───────────┘ │
       ↓                                 │               │
   ┌─────────┐                          │ (Repeat N     │
   │Context  │ ─────────────────────────→│  layers)      │
   │Vector   │                          │               │
   └─────────┘                          └───────────────┘

KEY CONCEPTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 SELF-ATTENTION (Encoder):
   • Each word can see ALL other words in input
   • "Hello" can attend to "world" and vice versa
   • Creates rich contextual understanding

🔍 SELF-ATTENTION (Decoder):  
   • Each word can only see PREVIOUS words
   • When generating "mundo", can only see "hola"
   • Prevents cheating by looking ahead

🔍 CROSS-ATTENTION (Decoder):
   • Spanish words attend to English context
   • "hola" looks at "Hello world" to decide what to generate
   • This is where translation magic happens!

🎯 GENERATION PROCESS:
   1. Encoder processes "Hello world" → context
   2. Decoder starts with <start> token
   3. Uses context + <start> → generates "hola"  
   4. Uses context + <start> + "hola" → generates "mundo"
   5. Uses context + <start> + "hola" + "mundo" → generates <end>
""")

if __name__ == "__main__":
    print_architecture()
