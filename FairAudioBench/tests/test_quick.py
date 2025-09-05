#!/usr/bin/env python3

"""
Quick test to verify the experiment implementation is working
"""

import sys
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.frontends import create_frontend
from src.datasets import CrossCulturalDataset, speech_collate_fn, classification_collate_fn
from src.models import create_model

def test_speech_dataset(sample_mode=False):
    """Test speech dataset loading."""
    print("Testing Speech Dataset...")
    
    # Create frontend
    frontend = create_frontend('mel', sample_rate=22050)
    
    # Choose data path based on mode
    if sample_mode:
        data_path = '../processed_data/samples/speech'
    else:
        data_path = 'processed_data/speech/en'
    
    # Test dataset
    try:
        dataset = CrossCulturalDataset(
            data_path,
            frontend=frontend,
            task_type='speech',
            split='test',  # Only evaluation, no training
            max_samples=5
        )
        
        print(f"✓ English dataset loaded: {len(dataset)} samples")
        print(f"✓ Vocab size: {dataset.get_vocab_size()}")
        
        # Test loading a sample
        features, targets, input_len, target_len = dataset[0]
        print(f"✓ Sample shape: {features.shape}, Target length: {target_len.item()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Speech dataset test failed: {e}")
        return False

def test_music_dataset(sample_mode=False):
    """Test music dataset loading."""
    print("\nTesting Music Dataset...")
    
    # Create frontend
    frontend = create_frontend('mel', sample_rate=22050)
    
    # Choose data path based on mode
    if sample_mode:
        data_path = '../processed_data/samples/music'
    else:
        data_path = 'processed_data/music/gtzan'
    
    # Test dataset
    try:
        dataset = CrossCulturalDataset(
            data_path,
            frontend=frontend,
            task_type='music',
            split='test',  # Only evaluation, no training
            max_samples=5
        )
        
        print(f"✓ Music dataset loaded: {len(dataset)} samples")
        print(f"✓ Number of classes: {dataset.get_num_classes()}")
        
        # Test loading a sample
        features, label = dataset[0]
        print(f"✓ Sample shape: {features.shape}, Label: {label}")
        
        return True
        
    except Exception as e:
        print(f"✗ Music dataset test failed: {e}")
        return False

def test_scene_dataset(sample_mode=False):
    """Test scene dataset loading."""
    print("\nTesting Scene Dataset...")
    
    # Create frontend
    frontend = create_frontend('mel', sample_rate=22050)
    
    # Choose data path based on mode
    if sample_mode:
        data_path = '../processed_data/samples/scenes'
    else:
        data_path = 'processed_data/scenes/tau_urban'
    
    # Test dataset
    try:
        dataset = CrossCulturalDataset(
            data_path,
            frontend=frontend,
            task_type='scenes',
            split='test',  # Only evaluation, no training
            max_samples=5
        )
        
        print(f"✓ Scene dataset loaded: {len(dataset)} samples")
        print(f"✓ Number of classes: {dataset.get_num_classes()}")
        
        # Test loading a sample
        sample = dataset[0]
        if len(sample) == 3:  # features, label, city
            features, label, city = sample
            print(f"✓ Sample shape: {features.shape}, Label: {label}, City: {city}")
        else:
            features, label = sample
            print(f"✓ Sample shape: {features.shape}, Label: {label}")
        
        return True
        
    except Exception as e:
        print(f"✗ Scene dataset test failed: {e}")
        return False

def test_frontends():
    """Test different frontends."""
    print("\nTesting Frontends...")
    
    # Create dummy audio
    audio = torch.randn(1, 22050)  # 1 second of audio
    
    frontends = ['mel', 'erb', 'bark', 'cqt']
    
    for frontend_name in frontends:
        try:
            frontend = create_frontend(frontend_name, sample_rate=22050)
            features = frontend(audio)
            print(f"✓ {frontend_name.upper()}: {features.shape}")
        except Exception as e:
            print(f"✗ {frontend_name.upper()} failed: {e}")

def test_models():
    """Test model creation."""
    print("\nTesting Models...")
    
    try:
        # Test speech model
        speech_model = create_model('speech', input_dim=128, num_classes=50)
        print(f"✓ Speech model created: {type(speech_model).__name__}")
        
        # Test music model
        music_model = create_model('music', input_dim=128, num_classes=10)
        print(f"✓ Music model created: {type(music_model).__name__}")
        
        # Test scene model
        scene_model = create_model('scenes', input_dim=128, num_classes=10)
        print(f"✓ Scene model created: {type(scene_model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Quick FairAudioBench test')
    parser.add_argument('--sample_mode', action='store_true', 
                       help='Test with sample data instead of full datasets')
    args = parser.parse_args()
    
    print("Cross-Cultural Bias Experiment Implementation Test")
    print("=" * 50)
    
    if args.sample_mode:
        print("🔬 Running in SAMPLE MODE - testing with included sample files")
        print()
    
    tests = [
        test_frontends,
        test_models,
        lambda: test_speech_dataset(args.sample_mode),
        lambda: test_music_dataset(args.sample_mode),
        lambda: test_scene_dataset(args.sample_mode)
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        if args.sample_mode:
            print("✓ All sample tests passed! Sample files are working correctly.")
            print("📥 To test with full datasets, download them first:")
            print("   python scripts/download_datasets.py --all")
            print("   python scripts/preprocess_datasets.py --all")
        else:
            print("✓ All tests passed! Ready to run experiments.")
    else:
        print("✗ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
