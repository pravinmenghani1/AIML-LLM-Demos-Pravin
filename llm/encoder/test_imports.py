#!/usr/bin/env python3

print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    from transformers import AutoTokenizer
    print("✓ Transformers")
except ImportError as e:
    print(f"✗ Transformers: {e}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib")
except ImportError as e:
    print(f"✗ Matplotlib: {e}")

try:
    import seaborn as sns
    print("✓ Seaborn")
except ImportError as e:
    print(f"✗ Seaborn: {e}")

try:
    from sklearn.decomposition import PCA
    print("✓ Scikit-learn")
except ImportError as e:
    print(f"✗ Scikit-learn: {e}")

print("\nEnvironment test complete!")
