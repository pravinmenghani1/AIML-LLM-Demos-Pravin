# Rule-Based AI System (1950s-1980s Style)

A simple implementation of classic expert systems that dominated AI from 1950-1980s.

## Core Components

- **Rule Engine**: Forward-chaining inference engine
- **Knowledge Base**: IF-THEN rules
- **Working Memory**: Current facts
- **Inference Engine**: Pattern matching and rule firing

## Files

- `rule_engine.py` - Core rule-based AI implementation
- `medical_demo.py` - Medical diagnosis example
- `visualizer.html` - Interactive web visualization

## Usage

```bash
# Run medical diagnosis demo
python medical_demo.py

# Open visualizer in browser
open visualizer.html
```

## How It Works

1. **Facts** are added to working memory
2. **Rules** are checked against current facts
3. **Inference engine** fires matching rules
4. New facts are derived and added
5. Process repeats until no new rules fire

This demonstrates the fundamental approach used in early expert systems like MYCIN and DENDRAL.
