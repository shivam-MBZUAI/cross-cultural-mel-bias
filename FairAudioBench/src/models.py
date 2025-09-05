#!/usr/bin/env python3

"""
Model Implementations for Cross-Cultural Bias Research
ICASSP 2026 Paper

This module implements the neural network models for speech recognition,
music classification, and acoustic scene classification tasks.

Author: Shivam Chauhan, Ajay Pundhir
Organization: Presight AI, Abu Dhabi, UAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

class BaseModel(nn.Module):
    """Base model class for all tasks."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class SpeechRecognitionModel(BaseModel):
    """
    Speech recognition model for character/phoneme level prediction.
    Uses CNN + RNN architecture suitable for sequence prediction.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 vocab_size: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__(input_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CNN feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
        )
        
        # Calculate CNN output dimension
        self.cnn_output_dim = self._get_cnn_output_dim()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # CTC loss for sequence alignment
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    def _get_cnn_output_dim(self) -> int:
        """Calculate the output dimension after CNN layers."""
        # Dummy input to calculate dimensions
        dummy_input = torch.randn(1, 1, self.input_dim, 100)  # (batch, channel, freq, time)
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
        return dummy_output.shape[2]  # frequency dimension after conv
    
    def forward(self, x: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for speech recognition.
        
        Args:
            x: Input features (batch_size, freq_bins, time_frames)
            input_lengths: Actual lengths of sequences (for CTC)
        
        Returns:
            Log probabilities for each time step and character
        """
        batch_size, freq_bins, time_frames = x.shape
        
        # Add channel dimension for CNN
        x = x.unsqueeze(1)  # (batch_size, 1, freq_bins, time_frames)
        
        # CNN feature extraction
        conv_features = self.conv_layers(x)  # (batch_size, channels, freq', time')
        
        # Reshape for RNN: (batch_size, time', freq' * channels)
        conv_features = conv_features.permute(0, 3, 1, 2)  # (batch_size, time', channels, freq')
        conv_features = conv_features.flatten(2)  # (batch_size, time', channels * freq')
        
        # LSTM processing
        lstm_out, _ = self.lstm(conv_features)  # (batch_size, time', hidden_dim * 2)
        
        # Classification
        logits = self.classifier(lstm_out)  # (batch_size, time', vocab_size)
        
        # Log softmax for CTC
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
    
    def compute_loss(self, 
                    log_probs: torch.Tensor, 
                    targets: torch.Tensor,
                    input_lengths: torch.Tensor,
                    target_lengths: torch.Tensor) -> torch.Tensor:
        """Compute CTC loss."""
        # log_probs: (batch_size, time, vocab_size)
        # targets: (batch_size, target_length)
        # input_lengths: (batch_size,)
        # target_lengths: (batch_size,)
        
        # CTC expects (time, batch_size, vocab_size)
        log_probs = log_probs.permute(1, 0, 2)
        
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

class MusicClassificationModel(BaseModel):
    """
    Music classification model for genre/raga/makam classification.
    Uses CNN architecture with attention mechanism.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int,
                 dropout: float = 0.1):
        super().__init__(input_dim, num_classes)
        
        # CNN feature extraction
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(dropout)
        )
        
        # Attention mechanism for temporal modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for music classification.
        
        Args:
            x: Input features (batch_size, freq_bins, time_frames)
        
        Returns:
            Class logits (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch_size, 1, freq_bins, time_frames)
        
        # CNN feature extraction
        features = self.conv_blocks(x)  # (batch_size, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch_size, 512)
        
        # For attention, we need sequence dimension
        # Here we'll use the feature as a single-step sequence
        features = features.unsqueeze(1)  # (batch_size, 1, 512)
        
        # Self-attention (for single timestep, this acts as feature refinement)
        attended_features, _ = self.attention(features, features, features)
        attended_features = attended_features.squeeze(1)  # (batch_size, 512)
        
        # Classification
        logits = self.classifier(attended_features)
        
        return logits

class SceneClassificationModel(BaseModel):
    """
    Acoustic scene classification model.
    Uses CNN with spatial attention for scene recognition.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int,
                 dropout: float = 0.1):
        super().__init__(input_dim, num_classes)
        
        # CNN backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Fixed spatial size
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for scene classification.
        
        Args:
            x: Input features (batch_size, freq_bins, time_frames)
        
        Returns:
            Class logits (batch_size, num_classes)
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch_size, 1, freq_bins, time_frames)
        
        # CNN feature extraction
        features = self.backbone(x)  # (batch_size, 256, 4, 4)
        
        # Spatial attention
        attention_weights = self.spatial_attention(features)  # (batch_size, 1, 4, 4)
        attended_features = features * attention_weights  # (batch_size, 256, 4, 4)
        
        # Global pooling
        global_features = self.global_pool(attended_features)  # (batch_size, 256, 1, 1)
        global_features = global_features.squeeze(-1).squeeze(-1)  # (batch_size, 256)
        
        # Classification
        logits = self.classifier(global_features)
        
        return logits

# Factory function for creating models
def create_model(task: str, input_dim: int, num_classes: int, **kwargs) -> BaseModel:
    """
    Factory function to create models for different tasks.
    
    Args:
        task: One of 'speech', 'music', 'scenes'
        input_dim: Input feature dimension
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        Initialized model
    """
    models = {
        'speech': SpeechRecognitionModel,
        'music': MusicClassificationModel,
        'scenes': SceneClassificationModel
    }
    
    if task not in models:
        raise ValueError(f"Unknown task: {task}. Available: {list(models.keys())}")
    
    return models[task](input_dim, num_classes, **kwargs)

# Model utilities
class ModelTrainer:
    """Utility class for training models."""
    
    def __init__(self, model: BaseModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
    
    def train_epoch(self, 
                   dataloader, 
                   optimizer, 
                   criterion, 
                   task: str = 'classification') -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            if task == 'speech':
                # Speech recognition with CTC loss
                features, targets, input_lengths, target_lengths = batch
                features = features.to(self.device)
                targets = targets.to(self.device)
                input_lengths = input_lengths.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                optimizer.zero_grad()
                log_probs = self.model(features, input_lengths)
                loss = self.model.compute_loss(log_probs, targets, input_lengths, target_lengths)
                
            else:
                # Classification tasks
                features, targets = batch
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, dataloader, task: str = 'classification') -> Tuple[float, float]:
        """Evaluate model performance."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if task == 'speech':
                    # Speech evaluation would need specialized decoding
                    # For now, return dummy values
                    return 0.0, 0.0
                else:
                    features, targets = batch
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.model(features)
                    loss = F.cross_entropy(outputs, targets)
                    
                    predictions = torch.argmax(outputs, dim=1)
                    correct += (predictions == targets).sum().item()
                    total += targets.size(0)
                    total_loss += loss.item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        
        return accuracy, avg_loss

if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test speech model
    speech_model = create_model('speech', input_dim=128, num_classes=50)  # 50 characters/phonemes
    print(f"Speech model: {sum(p.numel() for p in speech_model.parameters())} parameters")
    
    # Test music model
    music_model = create_model('music', input_dim=128, num_classes=10)  # 10 genres/ragas
    print(f"Music model: {sum(p.numel() for p in music_model.parameters())} parameters")
    
    # Test scene model
    scene_model = create_model('scenes', input_dim=128, num_classes=10)  # 10 scene types
    print(f"Scene model: {sum(p.numel() for p in scene_model.parameters())} parameters")
    
    # Test forward passes
    batch_size = 4
    freq_bins = 128
    time_frames = 100
    
    dummy_input = torch.randn(batch_size, freq_bins, time_frames)
    
    print("\nTesting forward passes:")
    
    # Music classification
    music_output = music_model(dummy_input)
    print(f"Music output shape: {music_output.shape}")
    
    # Scene classification
    scene_output = scene_model(dummy_input)
    print(f"Scene output shape: {scene_output.shape}")
    
    # Speech recognition
    speech_output = speech_model(dummy_input)
    print(f"Speech output shape: {speech_output.shape}")
