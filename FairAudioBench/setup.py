#!/usr/bin/env python3

"""
FairAudioBench Setup Script
Sets up the environment and validates installation
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        print("Installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
            print("✓ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e}")
            return False
    else:
        print("Warning: requirements.txt not found")
        return False

def validate_imports():
    """Validate that all modules can be imported"""
    try:
        from src.frontends import create_frontend
        from src.models import create_model
        from src.datasets import CrossCulturalDataset
        from src.bias_evaluation import BiasMetrics
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"Error importing modules: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ["data", "experiments", "results", "logs"]
    for dir_name in dirs:
        dir_path = Path(__file__).parent / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def main():
    """Main setup function"""
    print("Setting up FairAudioBench...")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("Warning: Could not install all requirements")
    
    # Create directories
    create_directories()
    
    # Validate imports
    if not validate_imports():
        print("Error: Module validation failed")
        sys.exit(1)
    
    print("="*50)
    print("✓ FairAudioBench setup complete!")
    print("\nNext steps:")
    print("1. Download datasets: python scripts/download_datasets.py --all")
    print("2. Preprocess data: python scripts/preprocess_datasets.py --all")
    print("3. Run experiments: python run_experiments.py --domains all --frontends all")
    print("4. View results: python demonstrate_experiments.py")

if __name__ == "__main__":
    main()
