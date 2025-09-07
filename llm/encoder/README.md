# 🧠 Encoder Architecture Demo

Interactive visualization of transformer encoder components with multiple demo options.

## 🚀 Quick Start

### Option 1: HTML Demo (Recommended - No Setup Required)
```bash
python run_demo.py
# Select option 1
```
Or directly open `encoder_demo.html` in your browser.

### Option 2: Interactive Streamlit Demo
```bash
python run_demo.py
# Select option 4 to install dependencies
# Then select option 2 to run
```

### Option 3: Original Jupyter Notebook
```bash
jupyter notebook encoder_demo.ipynb
```

## 📋 Requirements

### HTML Demo
- No additional requirements (runs in any modern browser)

### Streamlit Demo
```bash
pip install -r requirements_visual.txt
```

### Jupyter Demo
```bash
pip install -r requirements.txt
```

## 🎯 Features

- **Interactive Controls**: Adjust model parameters in real-time
- **Attention Visualization**: Heatmaps showing token relationships
- **Embedding Analysis**: Visual representation of token embeddings
- **Architecture Overview**: Interactive model structure diagram

## 📁 Files

- `encoder_demo.html` - Standalone HTML demo
- `streamlit_encoder_demo.py` - Interactive web app
- `encoder_demo.ipynb` - Original Jupyter notebook
- `run_demo.py` - Easy launcher script
- `requirements_visual.txt` - Dependencies for visual demos

## 🎮 Usage

1. Run the launcher: `python run_demo.py`
2. Choose your preferred demo type
3. Enter text and adjust parameters
4. Click "Process with Encoder" to see visualizations

## 🔧 Troubleshooting

**Missing dependencies?**
```bash
python run_demo.py
# Select option 4 to auto-install
```

**HTML demo not opening?**
- Ensure you have a modern web browser
- Try opening `encoder_demo.html` directly

**Streamlit issues?**
```bash
pip install streamlit plotly pandas
streamlit run streamlit_encoder_demo.py
```
