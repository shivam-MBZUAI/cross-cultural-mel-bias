#!/usr/bin/env python3

"""
FairAudioBench Example Usage
Demonstrates how to use the benchmark for custom experiments
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from src.frontends import create_frontend
from src.models import create_model, ModelTrainer
from src.datasets import CrossCulturalDataset, speech_collate_fn
from src.bias_evaluation import BiasMetrics
from config.fairaudiobench_config import get_config, get_domain_config

def example_frontend_comparison():
    """Example: Compare different audio front-ends"""
    print("=== Frontend Comparison Example ===")
    
    # Create different front-ends
    frontends = {
        'mel': create_frontend('mel', sample_rate=16000),
        'erb': create_frontend('erb', sample_rate=16000),
        'bark': create_frontend('bark', sample_rate=16000)
    }
    
    # Generate dummy audio data
    dummy_audio = torch.randn(16000 * 3)  # 3 seconds at 16kHz
    
    print("Processing audio with different front-ends:")
    for name, frontend in frontends.items():
        features = frontend(dummy_audio)
        print(f"  {name.upper()}: {features.shape}")
    
    return frontends

def example_bias_evaluation():
    """Example: Evaluate bias in model predictions"""
    print("\n=== Bias Evaluation Example ===")
    
    # Create bias evaluator
    evaluator = BiasMetrics()
    
    # Simulate model predictions and group labels
    num_samples = 1000
    predictions = torch.randint(0, 10, (num_samples,))
    targets = torch.randint(0, 10, (num_samples,))
    
    # Simulate cultural groups (e.g., language families)
    group_labels = torch.randint(0, 3, (num_samples,))
    group_names = ['Germanic', 'Romance', 'Sino-Tibetan']
    
    # Calculate bias metrics using available method
    # Simulate language results
    results_by_language = {
        'english': {'accuracy': 0.95, 'f1_score': 0.93},
        'spanish': {'accuracy': 0.92, 'f1_score': 0.90},
        'chinese': {'accuracy': 0.88, 'f1_score': 0.86}
    }
    
    bias_metrics = evaluator.evaluate_speech_bias(results_by_language)
    
    print("Bias Metrics:")
    for metric, value in bias_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    return bias_metrics

def example_custom_experiment():
    """Example: Run a custom experiment"""
    print("\n=== Custom Experiment Example ===")
    
    # Get configuration
    config = get_domain_config('speech')
    
    # Create frontend and model
    frontend = create_frontend('mel', sample_rate=16000)
    model = create_model('speech', frontend.output_dim, num_classes=10)
    
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Simulate training data
    batch_size = 32
    sequence_length = 100
    feature_dim = frontend.output_dim
    
    # Dummy batch
    features = torch.randn(batch_size, sequence_length, feature_dim)
    labels = torch.randint(0, 10, (batch_size,))
    lengths = torch.randint(50, sequence_length, (batch_size,))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(features, lengths)
        predictions = torch.argmax(outputs, dim=1)
    
    # Calculate accuracy
    accuracy = (predictions == labels).float().mean()
    print(f"Dummy batch accuracy: {accuracy:.4f}")
    
    return model, accuracy

def example_dataset_usage():
    """Example: Load and use datasets"""
    print("\n=== Dataset Usage Example ===")
    
    # Note: This example requires actual processed data
    data_dir = Path("../processed_data/speech/english")
    
    if data_dir.exists():
        print(f"Loading dataset from: {data_dir}")
        
        # Create frontend
        frontend = create_frontend('mel', sample_rate=16000)
        
        # Create dataset
        dataset = CrossCulturalDataset(
            data_dir=data_dir,
            frontend=frontend,
            task_type='speech',
            max_length=16000 * 3  # 3 seconds
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Get a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Audio shape: {sample['audio'].shape}")
            print(f"Label: {sample['label']}")
    else:
        print(f"Dataset directory not found: {data_dir}")
        print("Please run data download and preprocessing first")

def example_model_training():
    """Example: Train a model"""
    print("\n=== Model Training Example ===")
    
    # Create components
    frontend = create_frontend('mel', sample_rate=16000)
    model = create_model('speech', frontend.output_dim, num_classes=10)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        learning_rate=0.001,
        device='cpu'  # Use CPU for example
    )
    
    # Simulate training data
    num_batches = 5
    batch_size = 16
    sequence_length = 100
    
    print("Simulating training...")
    for epoch in range(2):
        epoch_loss = 0
        
        for batch_idx in range(num_batches):
            # Generate dummy batch
            features = torch.randn(batch_size, sequence_length, frontend.output_dim)
            labels = torch.randint(0, 10, (batch_size,))
            lengths = torch.randint(50, sequence_length, (batch_size,))
            
            # Training step
            loss = trainer.train_step(features, labels, lengths)
            epoch_loss += loss
        
        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    print("Training simulation complete")
    return trainer

def example_configuration_usage():
    """Example: Use configuration system"""
    print("\n=== Configuration Usage Example ===")
    
    # Get full configuration
    full_config = get_config()
    
    print("Available configuration sections:")
    for section in full_config.keys():
        print(f"  - {section}")
    
    # Get domain-specific configuration
    speech_config = get_domain_config('speech')
    
    print(f"\nSpeech domain configuration:")
    print(f"  Languages: {speech_config['domain_specific']['languages']}")
    print(f"  Task: {speech_config['domain_specific']['task']}")
    print(f"  Model type: {speech_config['model']['type']}")
    
    return speech_config

def main():
    """Run all examples"""
    print("FairAudioBench Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        frontends = example_frontend_comparison()
        bias_metrics = example_bias_evaluation()
        model, accuracy = example_custom_experiment()
        example_dataset_usage()
        trainer = example_model_training()
        config = example_configuration_usage()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nNext steps:")
        print("1. Download and preprocess datasets")
        print("2. Run full experiments with: python run_experiments.py")
        print("3. Analyze results with: python analyze_results.py")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all dependencies are installed and modules are available")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
