#!/usr/bin/env python3

"""
Test Script for Cross-Cultural Bias Research Implementation
ICASSP 2026 Paper

This script tests the core components of our experimental framework
to ensure everything is working correctly before running full experiments.

Author: Shivam Chauhan, Ajay Pundhir
Organization: Presight AI, Abu Dhabi, UAE
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our modules
try:
    from frontends import create_frontend
    from models import create_model, ModelTrainer
    from bias_evaluation import BiasMetrics
    from datasets import analyze_dataset_distribution
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_frontends():
    """Test all audio front-ends."""
    print("\n" + "="*50)
    print("TESTING AUDIO FRONT-ENDS")
    print("="*50)
    
    sample_rate = 22050
    audio_length = 3 * sample_rate  # 3 seconds
    batch_size = 2
    
    # Create dummy audio
    audio = torch.randn(batch_size, audio_length)
    print(f"Input audio shape: {audio.shape}")
    
    frontends = ['mel', 'erb', 'bark', 'cqt', 'leaf', 'sincnet']
    results = {}
    
    for frontend_name in frontends:
        try:
            frontend = create_frontend(frontend_name, sample_rate=sample_rate)
            features = frontend(audio)
            results[frontend_name] = {
                'success': True,
                'output_shape': features.shape,
                'feature_dim': frontend.get_feature_dim()
            }
            print(f"✓ {frontend_name.upper():>8}: {features.shape} - Feature dim: {frontend.get_feature_dim()}")
        except Exception as e:
            results[frontend_name] = {
                'success': False,
                'error': str(e)
            }
            print(f"✗ {frontend_name.upper():>8}: Error - {e}")
    
    return results

def test_models():
    """Test all model architectures."""
    print("\n" + "="*50)
    print("TESTING MODEL ARCHITECTURES")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    input_dim = 128
    batch_size = 4
    freq_bins = input_dim
    time_frames = 100
    
    dummy_input = torch.randn(batch_size, freq_bins, time_frames).to(device)
    
    models_config = [
        ('speech', 50, {'hidden_dim': 256, 'num_layers': 2}),
        ('music', 10, {'dropout': 0.1}),
        ('scenes', 10, {'dropout': 0.1})
    ]
    
    results = {}
    
    for task, num_classes, kwargs in models_config:
        try:
            model = create_model(task, input_dim, num_classes, **kwargs).to(device)
            
            if task == 'speech':
                # Test speech model with dummy lengths
                input_lengths = torch.LongTensor([time_frames] * batch_size).to(device)
                output = model(dummy_input, input_lengths)
            else:
                output = model(dummy_input)
            
            num_params = sum(p.numel() for p in model.parameters())
            
            results[task] = {
                'success': True,
                'output_shape': output.shape,
                'num_parameters': num_params
            }
            
            print(f"✓ {task.upper():>8}: Output {output.shape} - {num_params:,} parameters")
            
        except Exception as e:
            results[task] = {
                'success': False,
                'error': str(e)
            }
            print(f"✗ {task.upper():>8}: Error - {e}")
            traceback.print_exc()
    
    return results

def test_bias_evaluation():
    """Test bias evaluation framework."""
    print("\n" + "="*50)
    print("TESTING BIAS EVALUATION FRAMEWORK")
    print("="*50)
    
    bias_metrics = BiasMetrics()
    
    # Dummy results matching paper expectations
    dummy_speech_results = {
        'vi': {'mel': 0.312, 'erb': 0.219, 'leaf': 0.238},
        'th': {'mel': 0.287, 'erb': 0.201, 'leaf': 0.219},
        'en': {'mel': 0.187, 'erb': 0.175, 'leaf': 0.172},
        'es': {'mel': 0.169, 'erb': 0.161, 'leaf': 0.158}
    }
    
    dummy_music_results = {
        'gtzan': {'mel': 0.85, 'erb': 0.87, 'leaf': 0.89},
        'carnatic': {'mel': 0.72, 'erb': 0.78, 'leaf': 0.82},
        'fma': {'mel': 0.83, 'erb': 0.85, 'leaf': 0.87},
        'hindustani': {'mel': 0.69, 'erb': 0.75, 'leaf': 0.79}
    }
    
    dummy_scene_results = {
        'mel': {'barcelona': 0.78, 'helsinki': 0.76, 'london': 0.80},
        'erb': {'barcelona': 0.79, 'helsinki': 0.77, 'london': 0.81},
        'leaf': {'barcelona': 0.81, 'helsinki': 0.80, 'london': 0.83}
    }
    
    try:
        # Test speech bias analysis
        speech_bias = bias_metrics.evaluate_speech_bias(dummy_speech_results)
        print("✓ Speech bias analysis completed")
        
        # Print key results
        for frontend, analysis in speech_bias.items():
            gap = analysis['bias_metrics']['relative_gap_percent']
            print(f"  {frontend.upper()}: {gap:+.1f}% bias gap")
        
        # Test music bias analysis
        music_bias = bias_metrics.evaluate_music_bias(dummy_music_results)
        print("✓ Music bias analysis completed")
        
        for frontend, analysis in music_bias.items():
            gap = analysis['bias_metrics']['relative_gap_percent']
            print(f"  {frontend.upper()}: {gap:+.1f}% bias gap")
        
        # Test scene bias analysis
        scene_bias = bias_metrics.evaluate_scene_bias(dummy_scene_results)
        print("✓ Scene bias analysis completed")
        
        for frontend, analysis in scene_bias.items():
            cv = analysis['overall_performance']['coefficient_of_variation']
            print(f"  {frontend.upper()}: {cv:.3f} geographic variance")
        
        # Test bias mitigation analysis
        mitigation = bias_metrics.compute_bias_mitigation_effectiveness(
            dummy_speech_results, dummy_speech_results, 'speech'
        )
        print("✓ Bias mitigation analysis completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Bias evaluation error: {e}")
        traceback.print_exc()
        return False

def test_data_availability():
    """Test data availability and structure."""
    print("\n" + "="*50)
    print("TESTING DATA AVAILABILITY")
    print("="*50)
    
    data_dir = Path("processed_data")
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False
    
    # Check each domain
    domains = ['speech', 'music', 'scenes']
    available_data = {}
    
    for domain in domains:
        try:
            analysis = analyze_dataset_distribution(data_dir, domain)
            available_data[domain] = analysis
            
            if analysis:
                print(f"✓ {domain.upper()} data available:")
                for dataset, info in analysis.items():
                    if isinstance(info, dict) and 'total_samples' in info:
                        print(f"  - {dataset}: {info['total_samples']} samples")
                    else:
                        print(f"  - {dataset}: {info}")
            else:
                print(f"✗ No {domain} data found")
                
        except Exception as e:
            print(f"✗ Error analyzing {domain} data: {e}")
            available_data[domain] = {}
    
    return available_data

def test_integration():
    """Test integration between components."""
    print("\n" + "="*50)
    print("TESTING COMPONENT INTEGRATION")
    print("="*50)
    
    try:
        # Test frontend + model integration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create frontend
        frontend = create_frontend('mel', sample_rate=22050)
        
        # Create dummy audio
        audio = torch.randn(2, 22050 * 3)  # 2 samples, 3 seconds each
        
        # Extract features
        features = frontend(audio)
        print(f"✓ Features extracted: {features.shape}")
        
        # Test with music model
        music_model = create_model('music', frontend.get_feature_dim(), 10).to(device)
        features_gpu = features.to(device)
        
        with torch.no_grad():
            output = music_model(features_gpu)
        
        print(f"✓ Model inference successful: {output.shape}")
        print(f"✓ Integration test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print("CROSS-CULTURAL BIAS RESEARCH - IMPLEMENTATION TEST")
    print("="*60)
    print("ICASSP 2026 Paper Implementation")
    print("Authors: Shivam Chauhan, Ajay Pundhir")
    print("Organization: Presight AI, Abu Dhabi, UAE")
    print("="*60)
    
    test_results = {}
    
    # Run tests
    test_results['frontends'] = test_frontends()
    test_results['models'] = test_models()
    test_results['bias_evaluation'] = test_bias_evaluation()
    test_results['data_availability'] = test_data_availability()
    test_results['integration'] = test_integration()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    # Count successes
    frontend_successes = sum(1 for result in test_results['frontends'].values() 
                           if result.get('success', False))
    model_successes = sum(1 for result in test_results['models'].values() 
                        if result.get('success', False))
    
    print(f"Audio Front-ends: {frontend_successes}/6 working")
    print(f"Model Architectures: {model_successes}/3 working") 
    print(f"Bias Evaluation: {'✓' if test_results['bias_evaluation'] else '✗'}")
    print(f"Data Availability: {'✓' if test_results['data_availability'] else '✗'}")
    print(f"Integration: {'✓' if test_results['integration'] else '✗'}")
    
    # Overall status
    total_components = frontend_successes + model_successes + \
                      (1 if test_results['bias_evaluation'] else 0) + \
                      (1 if test_results['data_availability'] else 0) + \
                      (1 if test_results['integration'] else 0)
    
    print(f"\nOverall: {total_components}/13 components working")
    
    if total_components >= 10:
        print("🎉 Implementation ready for experiments!")
        return True
    elif total_components >= 7:
        print("⚠️  Implementation mostly ready, some issues to fix")
        return False
    else:
        print("❌ Implementation needs significant work")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
