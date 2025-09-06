"""
Training Script for FairAudioBench Models
Trains all six front-ends with matched hyperparameters for fair comparison

⚠️  IMPORTANT NOTE:
This is a REFERENCE implementation for model training. The primary purpose of 
FairAudioBench is EVALUATION of bias in audio front-ends, not training new models.

This script is provided for researchers who want to:
1. Train their own models for comparison with the paper's results
2. Understand the training pipeline used in the paper
3. Adapt the benchmark for their own front-ends or tasks

For the main benchmark evaluation (measuring bias across front-ends):
Use: python run_benchmark.py --evaluation-only

The core contribution is the bias measurement framework, not the training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import argparse
import time
import os

from models.frontends import create_frontend, get_model_summary

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FairAudioDataset(Dataset):
    """Dataset class for FairAudioBench."""
    
    def __init__(self, 
                 metadata_file: str, 
                 data_dir: str,
                 domain: str = "speech",
                 split: str = "train",
                 max_samples: Optional[int] = None):
        
        self.data_dir = Path(data_dir)
        self.domain = domain
        self.split = split
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        
        if max_samples:
            self.metadata = self.metadata.head(max_samples)
        
        # Create label mapping
        self.create_label_mapping()
        
        logger.info(f"Loaded {len(self.metadata)} samples for {domain} {split}")
    
    def create_label_mapping(self):
        """Create label mapping based on domain."""
        if self.domain == "speech":
            # For speech: classify language or tonal/non-tonal
            unique_labels = self.metadata['language'].unique()
        elif self.domain == "music":
            # For music: classify musical tradition
            unique_labels = self.metadata['tradition'].unique()
        elif self.domain == "urban_sounds":
            # For urban sounds: classify city or country
            unique_labels = self.metadata['city'].unique()
        else:
            raise ValueError(f"Unknown domain: {self.domain}")
        
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)
        
        logger.info(f"Created {self.num_classes} classes for {self.domain}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """Get a sample from the dataset."""
        row = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = self.data_dir / row['file_path']
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Ensure mono and correct sample rate
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)  # Remove channel dimension
        
        # Get label
        if self.domain == "speech":
            label = self.label_to_idx[row['language']]
        elif self.domain == "music":
            label = self.label_to_idx[row['tradition']]
        elif self.domain == "urban_sounds":
            label = self.label_to_idx[row['city']]
        
        # Additional metadata
        metadata = {
            'file_path': row['file_path'],
            'domain': self.domain
        }
        
        if self.domain == "speech":
            metadata.update({
                'language': row['language'],
                'is_tonal': row['is_tonal']
            })
        elif self.domain == "music":
            metadata.update({
                'tradition': row['tradition'],
                'cultural_origin': row.get('cultural_origin', 'Unknown')
            })
        elif self.domain == "urban_sounds":
            metadata.update({
                'city': row['city'],
                'country': row['country'],
                'population': row.get('population', 0)
            })
        
        return waveform, label, metadata

class ClassificationHead(nn.Module):
    """Classification head for front-end models."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling over time dimension
        x = torch.mean(x, dim=1)  # [batch_size, feature_dim]
        return self.classifier(x)

class FairAudioBenchTrainer:
    """Trainer for FairAudioBench models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model components
        self.frontend = None
        self.classifier = None
        self.model = None
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Results storage
        self.results = {
            'train_history': [],
            'val_history': [],
            'test_results': {},
            'model_info': {}
        }
    
    def setup_model(self, frontend_name: str, num_classes: int):
        """Setup front-end and classification model."""
        logger.info(f"Setting up {frontend_name} frontend...")
        
        # Create front-end
        self.frontend = create_frontend(
            frontend_name,
            sample_rate=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            n_mels=self.config['n_mels'],
            f_min=self.config['f_min'],
            f_max=self.config['f_max'],
            target_params=self.config['target_params']
        )
        
        # Create classification head
        self.classifier = ClassificationHead(
            input_dim=512,  # Frontend output dimension
            num_classes=num_classes,
            dropout=self.config['dropout']
        )
        
        # Combine into full model
        self.model = nn.Sequential(self.frontend, self.classifier)
        self.model.to(self.device)
        
        # Log model info
        frontend_summary = get_model_summary(self.frontend)
        self.results['model_info'] = {
            'frontend_name': frontend_name,
            'frontend_summary': frontend_summary,
            'num_classes': num_classes,
            'total_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        logger.info(f"Model setup complete. Total parameters: {self.results['model_info']['total_parameters']:,}")
    
    def setup_data(self, data_dir: str, domain: str):
        """Setup data loaders."""
        logger.info(f"Setting up data loaders for {domain}...")
        
        # Dataset paths
        splits_dir = Path(data_dir) / "splits" / domain
        
        # Create datasets
        train_dataset = FairAudioDataset(
            metadata_file=splits_dir / "train.csv",
            data_dir=data_dir,
            domain=domain,
            split="train",
            max_samples=self.config.get('max_train_samples', None)
        )
        
        val_dataset = FairAudioDataset(
            metadata_file=splits_dir / "val.csv",
            data_dir=data_dir,
            domain=domain,
            split="val",
            max_samples=self.config.get('max_val_samples', None)
        )
        
        test_dataset = FairAudioDataset(
            metadata_file=splits_dir / "test.csv",
            data_dir=data_dir,
            domain=domain,
            split="test",
            max_samples=self.config.get('max_test_samples', None)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        return train_dataset.num_classes
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (waveforms, labels, metadata) in enumerate(progress_bar):
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(waveforms)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        epoch_metrics = {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100.0 * correct / total
        }
        
        return epoch_metrics
    
    def validate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Validate model on given data loader."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        all_metadata = []
        
        with torch.no_grad():
            for waveforms, labels, metadata in tqdm(data_loader, desc="Validating"):
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(waveforms)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store predictions for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_metadata.extend(metadata)
        
        metrics = {
            'loss': total_loss / len(data_loader),
            'accuracy': 100.0 * correct / total,
            'predictions': all_predictions,
            'labels': all_labels,
            'metadata': all_metadata
        }
        
        return metrics
    
    def train(self, data_dir: str, domain: str, frontend_name: str, output_dir: str) -> bool:
        """Main training loop."""
        logger.info(f"Starting training: {frontend_name} on {domain}")
        
        # Setup
        num_classes = self.setup_data(data_dir, domain)
        self.setup_model(frontend_name, num_classes)
        self.setup_optimizer()
        
        # Create output directory
        output_path = Path(output_dir) / f"{frontend_name}_{domain}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        best_val_acc = 0.0
        best_model_path = output_path / "best_model.pth"
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch results
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_metrics['accuracy'],
                    'config': self.config
                }, best_model_path)
                logger.info(f"New best model saved with val acc: {best_val_acc:.2f}%")
            
            # Store history
            self.results['train_history'].append(train_metrics)
            self.results['val_history'].append(val_metrics)
        
        # Load best model for testing
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        test_metrics = self.validate(self.test_loader)
        self.results['test_results'] = test_metrics
        
        logger.info(f"Training completed. Test accuracy: {test_metrics['accuracy']:.2f}%")
        
        # Save results
        self.save_results(output_path, frontend_name, domain)
        
        return True
    
    def save_results(self, output_path: Path, frontend_name: str, domain: str):
        """Save training results and model info."""
        # Remove non-serializable items for JSON
        json_results = {
            'model_info': self.results['model_info'],
            'train_history': [{k: v for k, v in epoch.items() if k not in ['predictions', 'labels', 'metadata']} 
                             for epoch in self.results['train_history']],
            'val_history': [{k: v for k, v in epoch.items() if k not in ['predictions', 'labels', 'metadata']} 
                           for epoch in self.results['val_history']],
            'test_results': {k: v for k, v in self.results['test_results'].items() 
                           if k not in ['predictions', 'labels', 'metadata']},
            'config': self.config
        }
        
        # Save JSON results
        with open(output_path / "results.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save detailed results with predictions
        detailed_results = {
            'predictions': self.results['test_results']['predictions'],
            'labels': self.results['test_results']['labels'],
            'metadata': self.results['test_results']['metadata']
        }
        
        torch.save(detailed_results, output_path / "detailed_results.pth")
        
        logger.info(f"Results saved to {output_path}")

def load_config(config_file: str) -> Dict[str, Any]:
    """Load training configuration."""
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            # Model parameters
            'sample_rate': 16000,
            'n_fft': 512,
            'hop_length': 160,
            'n_mels': 80,
            'f_min': 80.0,
            'f_max': 8000.0,
            'target_params': 5000000,
            'dropout': 0.1,
            
            # Training parameters
            'batch_size': 32,
            'num_epochs': 50,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'num_workers': 4,
            
            # Data parameters
            'max_train_samples': None,
            'max_val_samples': None,
            'max_test_samples': None
        }
    
    return config

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train FairAudioBench models")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing processed datasets"
    )
    parser.add_argument(
        "--output-dir",
        default="./trained_models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--config",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--frontend",
        choices=['standard_mel', 'erb_scale', 'gammatone', 'cochlear', 'bark_scale', 'learnable_mel'],
        help="Specific frontend to train (if not specified, trains all)"
    )
    parser.add_argument(
        "--domain",
        choices=['speech', 'music', 'urban_sounds'],
        help="Specific domain to train on (if not specified, trains all)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with limited data"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Modify config for quick test
    if args.quick_test:
        config.update({
            'num_epochs': 2,
            'max_train_samples': 100,
            'max_val_samples': 50,
            'max_test_samples': 50
        })
        logger.info("Running in quick test mode")
    
    # Define frontends and domains to train
    frontends = [args.frontend] if args.frontend else [
        'standard_mel', 'erb_scale', 'gammatone', 
        'cochlear', 'bark_scale', 'learnable_mel'
    ]
    
    domains = [args.domain] if args.domain else [
        'speech', 'music', 'urban_sounds'
    ]
    
    # Train all combinations
    for frontend in frontends:
        for domain in domains:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {frontend} on {domain}")
            logger.info(f"{'='*50}")
            
            try:
                trainer = FairAudioBenchTrainer(config)
                success = trainer.train(
                    data_dir=args.data_dir,
                    domain=domain,
                    frontend_name=frontend,
                    output_dir=args.output_dir
                )
                
                if success:
                    logger.info(f"✓ Successfully trained {frontend} on {domain}")
                else:
                    logger.error(f"✗ Failed to train {frontend} on {domain}")
                    
            except Exception as e:
                logger.error(f"✗ Error training {frontend} on {domain}: {e}")
                continue
    
    logger.info("\nTraining completed for all specified combinations!")

if __name__ == "__main__":
    main()
