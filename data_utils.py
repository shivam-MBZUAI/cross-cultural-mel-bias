#!/usr/bin/env python3

"""
Dataset Classes for Cross-Cultural Bias Research
ICASSP 2026 Paper

This module implements PyTorch Dataset classes for loading and preprocessing
speech, music, and acoustic scene data with different audio frontends.

Author: Shivam Chauhan, Ajay Pundhir
Organization: Presight AI, Abu Dhabi, UAE
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class CrossCulturalDataset(Dataset):
    """
    Base dataset class for cross-cultural audio analysis.
    """
    
    def __init__(self,
                 data_dir: Union[Path, str],
                 frontend,
                 task_type: str = 'classification',
                 split: str = 'train',
                 max_samples: Optional[int] = None,
                 transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to dataset directory (containing metadata.csv)
            frontend: Audio frontend for feature extraction
            task_type: 'speech', 'music', or 'scenes'
            split: 'train', 'val', or 'test'
            max_samples: Maximum number of samples to load (for efficiency)
            transform: Optional data transformations
        """
        self.data_dir = Path(data_dir)
        self.frontend = frontend
        self.task_type = task_type
        self.split = split
        self.max_samples = max_samples
        self.transform = transform
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.metadata = pd.read_csv(metadata_path)
        
        # Prepare labels based on task type
        self._prepare_labels()
        
        # Create train/val/test splits
        self._create_splits()
        
        # Limit samples if specified
        if max_samples and len(self.split_indices) > max_samples:
            self.split_indices = np.random.choice(
                self.split_indices, size=max_samples, replace=False
            ).tolist()
        
        logger.info(f"Loaded {len(self.split_indices)} {split} samples from {metadata_path}")
    
    def _prepare_labels(self):
        """Prepare labels based on task type."""
        if self.task_type == 'speech':
            self._prepare_speech_labels()
        elif self.task_type in ['music', 'scenes']:
            self._prepare_classification_labels()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _prepare_speech_labels(self):
        """Prepare character-level labels for speech recognition."""
        # Build character vocabulary
        if 'text' in self.metadata.columns:
            all_text = ' '.join(self.metadata['text'].fillna('').astype(str))
            unique_chars = sorted(set(all_text.lower()))
        else:
            # Default character set for multiple languages
            unique_chars = list(' abcdefghijklmnopqrstuvwxyz0123456789.,!?-')
        
        # Add special tokens
        special_tokens = ['<blank>', '<unk>']  # CTC blank token and unknown
        self.vocab = special_tokens + unique_chars
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
        logger.info(f"Built vocabulary with {self.vocab_size} characters")
    
    def _prepare_classification_labels(self):
        """Prepare labels for classification tasks."""
        # Try different label column names
        label_columns = ['label', 'class', 'genre', 'raga', 'makam', 'mizan', 'scene']
        label_column = None
        
        for col in label_columns:
            if col in self.metadata.columns:
                label_column = col
                break
        
        if label_column is not None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.metadata[label_column])
            self.num_classes = len(self.label_encoder.classes_)
            logger.info(f"Found {self.num_classes} classes in column '{label_column}': {list(self.label_encoder.classes_)}")
        else:
            # Create dummy labels if no label column found
            logger.warning("No label column found, creating dummy labels")
            self.labels = np.zeros(len(self.metadata))
            self.num_classes = 1
    
    def _create_splits(self):
        """Create train/val/test splits."""
        n_samples = len(self.metadata)
        indices = np.arange(n_samples)
        
        # Use stratified split for classification tasks
        if self.task_type in ['music', 'scenes'] and hasattr(self, 'labels'):
            try:
                from sklearn.model_selection import train_test_split
                
                # Train: 70%, Val: 15%, Test: 15%
                train_idx, temp_idx = train_test_split(
                    indices, test_size=0.3, stratify=self.labels, random_state=42
                )
                val_idx, test_idx = train_test_split(
                    temp_idx, test_size=0.5, 
                    stratify=[self.labels[i] for i in temp_idx], 
                    random_state=42
                )
            except Exception as e:
                logger.warning(f"Stratified split failed: {e}, using random split")
                train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
                val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        else:
            # Random split for speech tasks or when stratification fails
            from sklearn.model_selection import train_test_split
            train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        
        # Select indices based on split
        if self.split == 'train':
            self.split_indices = train_idx.tolist()
        elif self.split == 'val':
            self.split_indices = val_idx.tolist()
        elif self.split == 'test':
            self.split_indices = test_idx.tolist()
        else:
            self.split_indices = indices.tolist()
    
    def __len__(self):
        return len(self.split_indices)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        actual_idx = self.split_indices[idx]
        row = self.metadata.iloc[actual_idx]
        
        # Load audio
        audio_path = row['audio_path']
        if not Path(audio_path).is_absolute():
            audio_path = self.data_dir / audio_path
        
        try:
            audio, sr = sf.read(str(audio_path))
            
            # Handle stereo audio by taking mean across channels
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio)
            
            # Extract features using frontend
            with torch.no_grad():
                features = self.frontend(audio_tensor.unsqueeze(0)).squeeze(0)
            
            # Apply transforms
            if self.transform:
                features = self.transform(features)
            
            # Return based on task type
            if self.task_type == 'speech':
                # For speech recognition, return features and encoded text
                if 'text' in row and pd.notna(row['text']):
                    text = str(row['text']).lower()
                    # Convert text to character indices
                    char_indices = []
                    for char in text:
                        if char in self.char_to_idx:
                            char_indices.append(self.char_to_idx[char])
                        else:
                            char_indices.append(self.char_to_idx['<unk>'])
                    
                    # Return features, targets, input_length, target_length
                    target_tensor = torch.LongTensor(char_indices)
                    input_length = torch.LongTensor([features.size(1)])  # Time dimension
                    target_length = torch.LongTensor([len(char_indices)])
                    
                    return features, target_tensor, input_length, target_length
                else:
                    # Return dummy targets if no text available
                    target_tensor = torch.LongTensor([])
                    input_length = torch.LongTensor([features.size(1)])
                    target_length = torch.LongTensor([0])
                    return features, target_tensor, input_length, target_length
            
            else:
                # For classification tasks
                label = self.labels[actual_idx]
                
                # For scene classification, also return city information if available
                if self.task_type == 'scenes' and 'city' in row:
                    return features, label, row['city']
                else:
                    return features, label
        
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            # Return dummy data in case of error
            dummy_features = torch.zeros(128, 100)  # 128 features, 100 time steps
            
            if self.task_type == 'speech':
                return dummy_features, torch.LongTensor([]), torch.LongTensor([100]), torch.LongTensor([0])
            else:
                return dummy_features, 0
    
    def get_vocab_size(self):
        """Get vocabulary size for speech tasks."""
        if hasattr(self, 'vocab_size'):
            return self.vocab_size
        return 0
    
    def get_num_classes(self):
        """Get number of classes for classification tasks."""
        if hasattr(self, 'num_classes'):
            return self.num_classes
        return 0

class SpeechDataset(CrossCulturalDataset):
    """Dataset for speech recognition tasks."""
    
    def __init__(self, data_dir: Path, metadata_file: str, frontend, split: str = 'train'):
        super().__init__(data_dir, metadata_file, frontend, 'sequence', split)
        
        # Build character vocabulary
        self._build_vocabulary()
    
    def _prepare_labels(self):
        """Prepare character-level labels for speech recognition."""
        # For now, we'll use dummy labels
        # In practice, you would extract characters from transcripts
        self.labels = list(range(len(self.metadata)))
    
    def _build_vocabulary(self):
        """Build character vocabulary from transcripts."""
        # Simplified vocabulary - in practice, build from actual transcripts
        if 'text' in self.metadata.columns:
            all_text = ' '.join(self.metadata['text'].fillna('').astype(str))
            unique_chars = sorted(set(all_text))
        else:
            # Default character set for multiple languages
            unique_chars = list(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-')
        
        # Add special tokens
        special_tokens = ['<blank>', '<sos>', '<eos>', '<unk>']
        self.vocab = special_tokens + unique_chars
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
        logger.info(f"Built vocabulary with {self.vocab_size} characters")
    
    def __getitem__(self, idx):
        """Get a speech sample with character-level targets."""
        actual_idx = self.indices[idx]
        row = self.metadata.iloc[actual_idx]
        
        # Load audio and extract features
        audio_path = row['audio_path']
        if not Path(audio_path).is_absolute():
            audio_path = self.data_dir / audio_path
        
        audio, sr = sf.read(str(audio_path))
        audio_tensor = torch.FloatTensor(audio)
        
        with torch.no_grad():
            features = self.frontend(audio_tensor.unsqueeze(0)).squeeze(0)
        
        # Get transcript
        if 'text' in row and pd.notna(row['text']):
            transcript = str(row['text'])
        else:
            transcript = "dummy text"  # Fallback
        
        # Convert transcript to indices
        char_indices = [self.char_to_idx.get(char, self.char_to_idx['<unk>']) 
                       for char in transcript]
        target = torch.LongTensor(char_indices)
        
        # Return features, target, and lengths for CTC
        feature_length = features.shape[-1]  # Time dimension
        target_length = len(char_indices)
        
        return features, target, feature_length, target_length

class MusicDataset(CrossCulturalDataset):
    """Dataset for music classification tasks."""
    
    def __init__(self, data_dir: Path, metadata_file: str, frontend, split: str = 'train'):
        super().__init__(data_dir, metadata_file, frontend, 'classification', split)
    
    def _prepare_labels(self):
        """Prepare labels for music classification."""
        # Determine label column based on tradition
        label_columns = ['genre', 'raga', 'makam', 'mizan', 'class', 'label']
        
        label_column = None
        for col in label_columns:
            if col in self.metadata.columns:
                label_column = col
                break
        
        if label_column is None:
            raise ValueError(f"No suitable label column found. Available columns: {self.metadata.columns.tolist()}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.metadata[label_column])
        
        logger.info(f"Found {len(self.label_encoder.classes_)} classes: {self.label_encoder.classes_}")

class SceneDataset(CrossCulturalDataset):
    """Dataset for acoustic scene classification."""
    
    def __init__(self, data_dir: Path, metadata_file: str, frontend, split: str = 'train'):
        super().__init__(data_dir, metadata_file, frontend, 'classification', split)
    
    def _prepare_labels(self):
        """Prepare labels for scene classification."""
        # Use scene labels
        if 'scene' in self.metadata.columns:
            label_column = 'scene'
        elif 'scene_label' in self.metadata.columns:
            label_column = 'scene_label'
        else:
            raise ValueError("No scene label column found")
        
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.metadata[label_column])
        
        logger.info(f"Found {len(self.label_encoder.classes_)} scene types: {self.label_encoder.classes_}")

# Collate functions for DataLoader
def speech_collate_fn(batch):
    """
    Collate function for speech data with variable-length sequences.
    
    Args:
        batch: List of tuples (features, targets, input_length, target_length)
    
    Returns:
        Batched tensors with proper padding for CTC loss
    """
    features_list = []
    targets_list = []
    input_lengths = []
    target_lengths = []
    
    for features, targets, inp_len, tgt_len in batch:
        features_list.append(features)
        targets_list.append(targets)
        input_lengths.append(inp_len)
        target_lengths.append(tgt_len)
    
    # Pad features to same length
    max_feature_len = max(f.size(1) for f in features_list)
    feature_dim = features_list[0].size(0)
    
    padded_features = torch.zeros(len(batch), feature_dim, max_feature_len)
    for i, features in enumerate(features_list):
        length = features.size(1)
        padded_features[i, :, :length] = features
    
    # Concatenate all targets for CTC loss
    concatenated_targets = torch.cat(targets_list, dim=0)
    
    # Stack lengths
    input_lengths = torch.cat(input_lengths, dim=0)
    target_lengths = torch.cat(target_lengths, dim=0)
    
    return padded_features, concatenated_targets, input_lengths, target_lengths

def classification_collate_fn(batch):
    """
    Collate function for classification tasks.
    """
    if len(batch[0]) == 3:  # With extra info (e.g., city for scenes)
        features, labels, extra = zip(*batch)
        
        # Pad features to same length
        max_feature_len = max(f.size(1) for f in features)
        feature_dim = features[0].size(0)
        
        padded_features = torch.zeros(len(batch), feature_dim, max_feature_len)
        for i, f in enumerate(features):
            length = f.size(1)
            padded_features[i, :, :length] = f
        
        labels = torch.LongTensor(labels)
        
        return padded_features, labels, list(extra)
    
    else:  # Standard classification
        features, labels = zip(*batch)
        
        # Pad features to same length
        max_feature_len = max(f.size(1) for f in features)
        feature_dim = features[0].size(0)
        
        padded_features = torch.zeros(len(batch), feature_dim, max_feature_len)
        for i, f in enumerate(features):
            length = f.size(1)
            padded_features[i, :, :length] = f
        
        labels = torch.LongTensor(labels)
        
        return padded_features, labels

# Factory functions for creating datasets and dataloaders
def create_speech_dataset(data_dir: Path, language: str, frontend, split: str = 'train'):
    """Create speech dataset for a specific language."""
    language_dir = data_dir / "speech" / language
    return SpeechDataset(language_dir, "metadata.csv", frontend, split)

def create_music_dataset(data_dir: Path, tradition: str, frontend, split: str = 'train'):
    """Create music dataset for a specific tradition."""
    tradition_dir = data_dir / "music" / tradition
    return MusicDataset(tradition_dir, "metadata.csv", frontend, split)

def create_scene_dataset(data_dir: Path, frontend, split: str = 'train'):
    """Create scene dataset."""
    scene_dir = data_dir / "scenes" / "tau_urban"
    return SceneDataset(scene_dir, "metadata.csv", frontend, split)

def create_dataloaders(dataset_train, dataset_val, dataset_test, 
                      batch_size: int = 32, task_type: str = 'classification'):
    """
    Create DataLoaders for train/val/test datasets.
    
    Args:
        dataset_train: Training dataset
        dataset_val: Validation dataset  
        dataset_test: Test dataset
        batch_size: Batch size
        task_type: 'classification' or 'speech'
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes/vocab_size)
    """
    # Choose collate function
    if task_type == 'speech':
        collate_fn = speech_collate_fn
        num_outputs = dataset_train.vocab_size
    else:
        collate_fn = classification_collate_fn
        num_outputs = dataset_train.get_num_classes()
    
    # Create dataloaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_outputs

# Utility functions for data analysis
def analyze_dataset_distribution(data_dir: Path, domain: str):
    """Analyze the distribution of samples across different categories."""
    if domain == 'speech':
        speech_dir = data_dir / "speech"
        analysis = {}
        
        for lang_dir in speech_dir.iterdir():
            if lang_dir.is_dir():
                metadata_file = lang_dir / "metadata.csv"
                if metadata_file.exists():
                    df = pd.read_csv(metadata_file)
                    analysis[lang_dir.name] = {
                        'total_samples': len(df),
                        'avg_duration': df['duration'].mean() if 'duration' in df.columns else None,
                        'total_duration': df['duration'].sum() if 'duration' in df.columns else None
                    }
        
        return analysis
    
    elif domain == 'music':
        music_dir = data_dir / "music"
        analysis = {}
        
        for tradition_dir in music_dir.iterdir():
            if tradition_dir.is_dir():
                metadata_file = tradition_dir / "metadata.csv"
                if metadata_file.exists():
                    df = pd.read_csv(metadata_file)
                    
                    # Find label column
                    label_columns = ['genre', 'raga', 'makam', 'mizan', 'class', 'label']
                    label_column = None
                    for col in label_columns:
                        if col in df.columns:
                            label_column = col
                            break
                    
                    analysis[tradition_dir.name] = {
                        'total_samples': len(df),
                        'num_classes': df[label_column].nunique() if label_column else None,
                        'class_distribution': df[label_column].value_counts().to_dict() if label_column else None
                    }
        
        return analysis
    
    elif domain == 'scenes':
        scene_dir = data_dir / "scenes" / "tau_urban"
        metadata_file = scene_dir / "metadata.csv"
        
        if metadata_file.exists():
            df = pd.read_csv(metadata_file)
            
            analysis = {
                'total_samples': len(df),
                'num_scenes': df['scene'].nunique() if 'scene' in df.columns else None,
                'num_cities': df['city'].nunique() if 'city' in df.columns else None,
                'scene_distribution': df['scene'].value_counts().to_dict() if 'scene' in df.columns else None,
                'city_distribution': df['city'].value_counts().to_dict() if 'city' in df.columns else None
            }
            
            return analysis
    
    return {}

if __name__ == "__main__":
    # Test dataset loading
    from src.frontends import create_frontend
    
    data_dir = Path("processed_data")
    
    # Test speech dataset
    if (data_dir / "speech" / "en").exists():
        frontend = create_frontend('mel')
        speech_dataset = create_speech_dataset(data_dir, 'en', frontend, 'train')
        print(f"Speech dataset: {len(speech_dataset)} samples")
        print(f"Vocabulary size: {speech_dataset.vocab_size}")
        
        # Test sample
        if len(speech_dataset) > 0:
            sample = speech_dataset[0]
            print(f"Sample features shape: {sample[0].shape}")
    
    # Test music dataset
    if (data_dir / "music" / "gtzan").exists():
        frontend = create_frontend('mel')
        music_dataset = create_music_dataset(data_dir, 'gtzan', frontend, 'train')
        print(f"Music dataset: {len(music_dataset)} samples")
        print(f"Number of classes: {music_dataset.get_num_classes()}")
        
        # Test sample
        if len(music_dataset) > 0:
            sample = music_dataset[0]
            print(f"Sample features shape: {sample[0].shape}")
    
    # Analyze dataset distributions
    print("\nDataset Analysis:")
    for domain in ['speech', 'music', 'scenes']:
        analysis = analyze_dataset_distribution(data_dir, domain)
        print(f"\n{domain.upper()} Distribution:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
