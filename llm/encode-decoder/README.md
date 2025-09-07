# 🤖 Transformer Encoder-Decoder Interactive Demo

A beginner-friendly interactive web demo explaining how Transformer architecture works for machine translation.

## 🎯 What This Demo Shows

- **Problem**: Translate "Hello world" (English) → "Hola mundo" (Spanish)
- **Solution**: Encoder-decoder Transformer architecture
- **Learning**: Visual step-by-step process with animations

## 🚀 Quick Start (3 Steps)

### Option 1: Simple HTML File (Recommended)
```bash
# 1. Open the HTML file directly in your browser
open transformer_demo.html
# OR double-click transformer_demo.html in file explorer
```

### Option 2: Python Web Server
```bash
# 1. Install Python (if not already installed)
# 2. Run the server
python3 serve_demo.py

# 3. Browser will open automatically at http://localhost:8000
```

### Option 3: Manual Browser Opening
```bash
# 1. Start server manually
python3 -m http.server 8000

# 2. Open browser and go to:
# http://localhost:8000/transformer_demo.html
```

## 📖 How to Use the Demo

1. **Read the Problem**: Understand what we're trying to solve
2. **Study Architecture**: See how encoder and decoder work together  
3. **Click "Run Translation Demo"**: Watch the step-by-step animation
4. **Observe**: 7 animated steps showing the translation process

## 🎨 What You'll See

- 🔤 **Tokenization**: Words → Numbers
- 📖 **Encoder**: Reads entire English sentence
- 🧠 **Self-Attention**: Words looking at each other
- ✍️ **Decoder**: Generates Spanish word by word
- 🔍 **Cross-Attention**: Using English context for Spanish generation
- ✅ **Result**: "Hola mundo" translation

## 🛠️ Files Included

- `transformer_demo.html` - Interactive web demo (main file)
- `serve_demo.py` - Python web server (optional)
- `transformer_demo.py` - PyTorch implementation
- `visual_demo.py` - Command-line visual demo
- `demo.py` - Basic usage examples

## 🔧 Troubleshooting

**"Site can't be reached" error?**
- Try Option 1: Just double-click `transformer_demo.html`
- The HTML file works standalone without a server!

**Python server not working?**
- Check if Python 3 is installed: `python3 --version`
- Try different port: `python3 -m http.server 8080`

**Browser not opening automatically?**
- Manually open: http://localhost:8000/transformer_demo.html

## 🎓 Learning Path

1. **Start Here**: Open `transformer_demo.html` in browser
2. **Understand Code**: Look at `transformer_demo.py` 
3. **Run Examples**: Try `python3 demo.py`
4. **Deep Dive**: Explore `visual_demo.py`

## 💡 Perfect For

- AI/ML beginners
- Students learning Transformers
- Anyone curious about how Google Translate works
- Teachers explaining attention mechanisms

---

**Just want to see it?** → Double-click `transformer_demo.html` 🚀
