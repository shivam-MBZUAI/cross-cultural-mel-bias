#!/usr/bin/env python3
"""Test minimal imports"""
import sys

print("Testing basic imports...")

try:
    import numpy as np
    print("✓ NumPy OK")
    
    import torch
    print("✓ PyTorch OK")
    
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    print("✓ Matplotlib OK")
    
    import seaborn as sns
    print("✓ Seaborn OK")
    
    from dataclasses import dataclass, field
    from typing import Dict, List
    print("✓ Typing OK")
    
    @dataclass
    class TestConfig:
        sample_rate: int = 22050
        tasks: List[str] = field(default_factory=lambda: ["test"])
        
    config = TestConfig()
    print(f"✓ Dataclass OK: {config.sample_rate}")
    
    print("All basic imports working!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
