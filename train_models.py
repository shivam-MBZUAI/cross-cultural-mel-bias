#!/usr/bin/env python3
"""
Train CRNN Models for Audio Tasks
This script trains the CRNN models needed for the evaluation pipeline
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============== DATASET CLASS ==============

class AudioDataset(Dataset):
    """Dataset for loading audio files and labels"""
    
    def __init__(self, file_paths, labels, frontend, max_length_seconds=10):
        self.file_paths = file_paths
        self.labels = labels
        self.frontend = frontend
        self.sample_rate = 16000
        self.max_length = max_length_seconds * self.sample_rate
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Trim or pad
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Extract features using frontend
        with torch.no_grad():
            features = self.frontend(waveform.squeeze(0))
        
        return features, self.labels[idx]


# ============== MODEL ARCHITECTURES ==============

class CRNN(nn.Module):
    """CRNN model for all audio tasks"""
    
    def __init__(self, input_dim=80, num_classes=10, dropout_rate=0.3):
        super().__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate RNN input size (after 3 pooling layers)
        rnn_input_size = (input_dim // 8) * 128
        
        # RNN layers
        self.lstm = nn.LSTM(rnn_input_size, 256, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # Output layers
        self.fc = nn.Linear(512, num_classes)
        self.activation = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        # Reshape for RNN: (batch, channels, freq, time) -> (batch, time, features)
        batch, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, -1)
        
        # RNN processing
        x, _ = self.lstm(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.fc(x)
        return x


# ============== AUDIO FRONT-ENDS ==============

class MelFilterbank(nn.Module):
    """Standard mel-scale filterbank"""
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=80, hop_length=160):
        super().__init__()
        self.mel_scale = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length
        )
    
    def forward(self, waveform):
        mel_spec = self.mel_scale(waveform)
        log_mel = torch.log(mel_spec + 1e-9)
        return log_mel


# ============== TRAINING FUNCTIONS ==============

def load_music_data(data_dir='data/music', max_per_genre=None):
    """Load music classification data"""
    print("Loading music data...")
    
    genres = ['arab_andalusian', 'carnatic', 'fma_small', 'gtzan', 
              'hindustani', 'turkish_makam']
    
    file_paths = []
    labels = []
    
    for genre_idx, genre in enumerate(genres):
        genre_dir = Path(data_dir) / genre
        if not genre_dir.exists():
            print(f"  Warning: {genre} not found")
            continue
        
        audio_files = list(genre_dir.glob('*.wav')) + list(genre_dir.glob('*.mp3'))
        if max_per_genre:
            audio_files = audio_files[:max_per_genre]
        
        print(f"  Found {len(audio_files)} files in {genre}")
        
        for audio_file in audio_files:
            file_paths.append(str(audio_file))
            labels.append(genre_idx)
    
    print(f"Total music files: {len(file_paths)}")
    return file_paths, labels, len(genres)


def load_scene_data(data_dir='data/scenes', max_per_scene=None):
    """Load scene classification data"""
    print("Loading scene data...")
    
    scenes = ['european-1', 'european-2']
    
    file_paths = []
    labels = []
    
    for scene_idx, scene in enumerate(scenes):
        scene_dir = Path(data_dir) / scene
        if not scene_dir.exists():
            print(f"  Warning: {scene} not found")
            continue
        
        audio_files = list(scene_dir.glob('*.wav'))
        if max_per_scene:
            audio_files = audio_files[:max_per_scene]
        
        print(f"  Found {len(audio_files)} files in {scene}")
        
        for audio_file in audio_files:
            file_paths.append(str(audio_file))
            labels.append(scene_idx)
    
    print(f"Total scene files: {len(file_paths)}")
    return file_paths, labels, len(scenes)


def load_speech_data(data_dir='data/speech', max_per_language=None):
    """Load speech data for language classification"""
    print("Loading speech data...")
    
    languages = ['de', 'en', 'es', 'fr', 'it', 'nl', 'pa-IN', 'th', 'vi', 
                 'yue', 'zh-CN']
    
    file_paths = []
    labels = []
    
    for lang_idx, lang in enumerate(languages):
        lang_dir = Path(data_dir) / lang
        if not lang_dir.exists():
            print(f"  Warning: {lang} not found")
            continue
        
        audio_files = list(lang_dir.glob('*.wav'))
        if max_per_language:
            audio_files = audio_files[:max_per_language]
        
        print(f"  Found {len(audio_files)} files in {lang}")
        
        for audio_file in audio_files:
            file_paths.append(str(audio_file))
            labels.append(lang_idx)
    
    print(f"Total speech files: {len(file_paths)}")
    return file_paths, labels, len(languages)


def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001, device='cuda'):
    """Train a CRNN model"""
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 
                             'acc': train_correct/train_total})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history, best_val_acc


def main():
    print("="*60)
    print("TRAINING CRNN MODELS FOR AUDIO TASKS")
    print("="*60)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize frontend (using Mel for training all models)
    frontend = MelFilterbank()
    
    # Training parameters
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    
    # ============== TRAIN MUSIC MODEL ==============
    print("\n" + "="*60)
    print("TRAINING MUSIC CLASSIFICATION MODEL")
    print("="*60)
    
    # Load data
    file_paths, labels, num_classes = load_music_data(max_per_genre=100)
    
    if len(file_paths) > 0:
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            file_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Number of classes: {num_classes}")
        
        # Create datasets
        train_dataset = AudioDataset(X_train, y_train, frontend, max_length_seconds=10)
        val_dataset = AudioDataset(X_val, y_val, frontend, max_length_seconds=10)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=2)
        
        # Create and train model
        model = CRNN(input_dim=80, num_classes=num_classes)
        model, history, best_acc = train_model(
            model, train_loader, val_loader, 
            num_epochs=num_epochs, learning_rate=learning_rate, device=device
        )
        
        # Save model
        torch.save(model.state_dict(), 'models/crnn_music.pth')
        print(f"\n✓ Music model saved (Best Val Acc: {best_acc:.4f})")
        
        # Save training history
        with open('models/music_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    else:
        print("⚠️  No music data found, skipping music model training")
    
    # ============== TRAIN SCENE MODEL ==============
    print("\n" + "="*60)
    print("TRAINING SCENE CLASSIFICATION MODEL")
    print("="*60)
    
    # Load data
    file_paths, labels, num_classes = load_scene_data(max_per_scene=100)
    
    if len(file_paths) > 0:
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            file_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Number of classes: {num_classes}")
        
        # Create datasets
        train_dataset = AudioDataset(X_train, y_train, frontend, max_length_seconds=10)
        val_dataset = AudioDataset(X_val, y_val, frontend, max_length_seconds=10)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2)
        
        # Create and train model
        model = CRNN(input_dim=80, num_classes=num_classes)
        model, history, best_acc = train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs, learning_rate=learning_rate, device=device
        )
        
        # Save model
        torch.save(model.state_dict(), 'models/crnn_scene.pth')
        print(f"\n✓ Scene model saved (Best Val Acc: {best_acc:.4f})")
        
        # Save training history
        with open('models/scene_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    else:
        print("⚠️  No scene data found, skipping scene model training")
    
    # ============== TRAIN SPEECH MODEL ==============
    print("\n" + "="*60)
    print("TRAINING SPEECH LANGUAGE CLASSIFICATION MODEL")
    print("="*60)
    
    # Load data
    file_paths, labels, num_classes = load_speech_data(max_per_language=100)
    
    if len(file_paths) > 0:
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            file_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Number of classes: {num_classes}")
        
        # Create datasets
        train_dataset = AudioDataset(X_train, y_train, frontend, max_length_seconds=5)
        val_dataset = AudioDataset(X_val, y_val, frontend, max_length_seconds=5)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2)
        
        # Create and train model
        model = CRNN(input_dim=80, num_classes=num_classes)
        model, history, best_acc = train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs, learning_rate=learning_rate, device=device
        )
        
        # Save model
        torch.save(model.state_dict(), 'models/crnn_speech.pth')
        print(f"\n✓ Speech model saved (Best Val Acc: {best_acc:.4f})")
        
        # Save training history
        with open('models/speech_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    else:
        print("⚠️  No speech data found, skipping speech model training")
    
    # ============== SUMMARY ==============
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    print("\nModels saved in 'models/' directory:")
    if os.path.exists('models/crnn_music.pth'):
        print("  ✓ crnn_music.pth")
    if os.path.exists('models/crnn_scene.pth'):
        print("  ✓ crnn_scene.pth")
    if os.path.exists('models/crnn_speech.pth'):
        print("  ✓ crnn_speech.pth")
    
    print("\nYou can now run the evaluation pipeline:")
    print("  python run_experiments.py")


if __name__ == "__main__":
    main()