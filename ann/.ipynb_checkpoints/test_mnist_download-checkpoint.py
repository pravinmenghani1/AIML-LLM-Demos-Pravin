#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import fetch_openml
import time

def download_mnist_with_timeout():
    """Download MNIST with timeout and progress indication"""
    print("Starting MNIST download...")
    print("This may take a few minutes for the first download...")
    
    try:
        start_time = time.time()
        
        # Download with timeout handling
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        
        end_time = time.time()
        print(f"✅ Download completed in {end_time - start_time:.2f} seconds")
        print(f"Dataset shape: {mnist.data.shape}")
        print(f"Labels shape: {mnist.target.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_mnist_with_timeout()
    if success:
        print("MNIST dataset is ready to use!")
    else:
        print("Please check your internet connection and try again.")
