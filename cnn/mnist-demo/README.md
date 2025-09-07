# CNN vs DNN MNIST Comparison Demo

## 🎯 Objective
Compare Convolutional Neural Network (CNN) vs Deep Neural Network (DNN) performance on MNIST digit classification.

## 🚀 Quick Start

```bash
cd /Users/pravinmenghani/Downloads/demos/cnn/mnist-demo
pip install -r requirements.txt
jupyter notebook cnn_mnist_demo.ipynb
```

## 📊 Expected Results

| Model | Accuracy | Parameters | Key Advantage |
|-------|----------|------------|---------------|
| **CNN** | ~96-99% | ~50K | Spatial awareness |
| **DNN** | ~95-98% | ~100K | General purpose |

## 🧠 What You'll Learn

### CNN Architecture:
- **Convolution**: Feature detection with filters
- **Pooling**: Dimensionality reduction
- **Feature Maps**: Visual representation of learned features
- **Parameter Sharing**: Efficiency through weight reuse

### Key Concepts:
1. **Spatial Preservation**: 2D structure maintained
2. **Translation Invariance**: Features detected anywhere
3. **Hierarchical Learning**: Edges → Shapes → Objects
4. **Local Connectivity**: Neurons connect to local regions

## 🔍 Demo Features

- **From-scratch CNN implementation**
- **Visual filter analysis**
- **Feature map visualization**
- **Performance comparison with DNN**
- **Confusion matrix analysis**

## 💡 Key Takeaways

**Why CNN > DNN for Images:**
- Preserves spatial relationships
- Fewer parameters needed
- Better feature extraction
- More robust to translations

**When to Use:**
- **CNN**: Images, spatial data, computer vision
- **DNN**: Tabular data, general classification

---
**Result**: CNN achieves higher accuracy with fewer parameters for image tasks!
