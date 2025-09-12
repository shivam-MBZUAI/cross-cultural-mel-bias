#!/usr/bin/env python3
"""Simple test"""
print("Testing imports...")

try:
    from frontends import ExperimentConfig
    print("✓ Config import successful")
    
    config = ExperimentConfig()
    print(f"✓ Config created: {config.sample_rate} Hz")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
