#!/usr/bin/env python3
"""
Encoder Demo Launcher
Choose between different visualization options for the encoder demo.
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import plotly
        return True
    except ImportError:
        return False

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_visual.txt"])
    print("✅ Installation complete!")

def run_streamlit_demo():
    """Launch the Streamlit demo."""
    print("🚀 Launching Streamlit demo...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_encoder_demo.py"])

def open_html_demo():
    """Open the HTML demo in browser."""
    html_path = Path("encoder_demo.html").absolute()
    print(f"🌐 Opening HTML demo: {html_path}")
    webbrowser.open(f"file://{html_path}")

def main():
    print("🧠 Encoder Architecture Demo Launcher")
    print("=" * 40)
    
    while True:
        print("\nChoose your demo option:")
        print("1. 🌐 HTML Demo (No dependencies, opens in browser)")
        print("2. 🚀 Streamlit Demo (Interactive, requires installation)")
        print("3. 📊 Jupyter Notebook (Original demo)")
        print("4. 🛠️ Install requirements for Streamlit demo")
        print("5. ❌ Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            open_html_demo()
            break
        elif choice == "2":
            if check_requirements():
                run_streamlit_demo()
            else:
                print("❌ Required packages not found.")
                install_choice = input("Install now? (y/n): ").lower()
                if install_choice == 'y':
                    install_requirements()
                    run_streamlit_demo()
                else:
                    print("Please install requirements first (option 4)")
            break
        elif choice == "3":
            print("🔗 Opening Jupyter notebook...")
            subprocess.run([sys.executable, "-m", "jupyter", "notebook", "encoder_demo.ipynb"])
            break
        elif choice == "4":
            install_requirements()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
