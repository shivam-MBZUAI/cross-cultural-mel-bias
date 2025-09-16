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
import librosa
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
        # Load audio with better error handling
        audio_path = self.file_paths[idx]
        
        try:
            # Try torchaudio first (usually most reliable)
            waveform, sr = torchaudio.load(audio_path)
        except:
            try:
                # Fallback to librosa with audioread
                import librosa
                import audioread
                waveform, sr = librosa.load(audio_path, sr=None, mono=False)
                waveform = torch.from_numpy(waveform).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
            except:
                # If all fails, return zeros (will be skipped in training)
                print(f"Warning: Could not load {audio_path}, using silence")
                waveform = torch.zeros(1, self.max_length)
                sr = self.sample_rate
        
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
    """CRNN with better spectro-temporal modeling"""
    def __init__(self, input_dim=80, num_classes=10, task_type='classification'):
        super().__init__()
        self.task_type = task_type
        self.input_dim = input_dim  # Store for later use
        
        # Frequency-aware CNN blocks
        self.conv1a = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2a = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3a = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4a = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling strategies
        self.freq_pool = nn.MaxPool2d((2, 1))  # Pool frequency only
        self.time_pool = nn.MaxPool2d((1, 2))  # Pool time only
        self.both_pool = nn.MaxPool2d((2, 2))  # Pool both
        
        # Calculate actual frequency dimension after pooling
        # After: both_pool -> freq_pool -> time_pool -> both_pool
        # Frequency dimension: input_dim / 2 / 2 / 1 / 2 = input_dim / 8
        freq_dim_after_pool = input_dim // 8
        
        # Attention mechanism for frequency dimension - FIX THE DIMENSIONS
        self.freq_attention = nn.Sequential(
            nn.Linear(freq_dim_after_pool * 256, 128),  # Fixed input size
            nn.ReLU(),
            nn.Linear(128, freq_dim_after_pool),  # Output per frequency bin
            nn.Sigmoid()
        )
        
        # Dropout with different rates
        self.dropout_conv = nn.Dropout2d(0.2)
        self.dropout_rnn = nn.Dropout(0.3)
        
        # Calculate RNN input size
        rnn_input_size = freq_dim_after_pool * 256
        
        # Multi-scale temporal modeling
        self.lstm1 = nn.LSTM(rnn_input_size, 256, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, num_layers=1,
                            batch_first=True, bidirectional=True)
        
        # Layer normalization for LSTM outputs
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(512)
        
        # Task-specific heads
        if task_type == 'classification':
            self.output_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        else:
            self.output_head = nn.Linear(512, num_classes)
        
        # Learnable positional encoding for time dimension
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, 1))
        
    def forward(self, x):
        # Input shape: (batch, freq_bins, time_frames) or (batch, 1, freq, time)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        
        # Block 1: Extract low-level features
        conv1 = self.conv1a(x)
        conv1 = torch.relu(conv1)
        conv1 = self.conv1b(conv1)
        conv1 = self.bn1(conv1)
        conv1 = torch.relu(conv1)
        x1 = self.both_pool(conv1)  # Frequency: input_dim/2
        x1 = self.dropout_conv(x1)
        
        # Block 2: Mid-level features
        conv2 = self.conv2a(x1)
        conv2 = torch.relu(conv2)
        conv2 = self.conv2b(conv2)
        conv2 = self.bn2(conv2)
        conv2 = torch.relu(conv2)
        x2 = self.freq_pool(conv2)  # Frequency: input_dim/4
        x2 = self.dropout_conv(x2)
        
        # Block 3: High-level features
        conv3 = self.conv3a(x2)
        conv3 = torch.relu(conv3)
        conv3 = self.conv3b(conv3)
        conv3 = self.bn3(conv3)
        conv3 = torch.relu(conv3)
        x3 = self.time_pool(conv3)  # Frequency: input_dim/4 (time pooling only)
        x3 = self.dropout_conv(x3)
        
        # Block 4: Final CNN features
        conv4 = self.conv4a(x3)
        conv4 = torch.relu(conv4)
        conv4 = self.conv4b(conv4)
        conv4 = self.bn4(conv4)
        conv4 = torch.relu(conv4)
        x4 = self.both_pool(conv4)  # Frequency: input_dim/8
        x4 = self.dropout_conv(x4)
        
        # Get dimensions
        batch, channels, freq, time = x4.size()
        
        # Apply frequency attention
        freq_features = x4.mean(dim=3)  # Average over time: (batch, channels, freq)
        freq_features_flat = freq_features.view(batch, -1)  # (batch, channels*freq)
        freq_weights = self.freq_attention(freq_features_flat)  # (batch, freq)
        freq_weights = freq_weights.view(batch, 1, freq, 1)  # Reshape for broadcasting
        x4 = x4 * freq_weights  # Apply attention weights
        
        # Prepare for RNN
        x4 = x4.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x4 = x4.reshape(batch, time, -1)  # (batch, time, features)
        
        # Add positional encoding (fix dimension)
        if time <= self.positional_encoding.size(1):
            pos_enc = self.positional_encoding[:, :time, :].expand(batch, time, 1)
            x4 = x4 + pos_enc
        
        # Two-layer bidirectional LSTM
        lstm_out1, _ = self.lstm1(x4)
        lstm_out1 = self.ln1(lstm_out1)
        lstm_out1 = self.dropout_rnn(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.ln2(lstm_out2)
        
        # Combine different temporal aggregations
        avg_pool = torch.mean(lstm_out2, dim=1)
        max_pool = torch.max(lstm_out2, dim=1)[0]
        last_hidden = lstm_out2[:, -1, :]
        
        # Attention-weighted average
        attention_weights = torch.softmax(lstm_out2.mean(dim=2, keepdim=True), dim=1)
        attention_pool = (lstm_out2 * attention_weights).sum(dim=1)
        
        # Combine all pooling strategies
        combined = (avg_pool + max_pool + last_hidden + attention_pool) / 4
        
        # Task-specific output
        output = self.output_head(combined)
        
        return output


# ============== AUDIO FRONT-ENDS ==============

class MelFilterbank(nn.Module):
    """Mel-scale filterbank front-end"""
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=40, hop_length=160):
        super().__init__()
        self.n_mels = n_mels
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

class ERBFilterbank(nn.Module):
    """ERB-scale filterbank"""
    def __init__(self, sample_rate=16000, n_filters=32, n_fft=512, hop_length=160):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_filters = n_filters
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Create ERB filterbank
        min_freq = 50
        max_freq = sample_rate / 2
        erb_freqs = self._erb_space(min_freq, max_freq, n_filters)
        self.filterbank = self._make_erb_filters(erb_freqs, sample_rate, n_fft)
    
    def _erb_space(self, low_freq, high_freq, n_filters):
        """Generate ERB-spaced frequencies"""
        ear_q = 9.26449
        min_bw = 24.7
        
        erb_low = (low_freq / ear_q) + min_bw
        erb_high = (high_freq / ear_q) + min_bw
        
        erb_freqs = np.linspace(erb_low, erb_high, n_filters)
        freqs = (erb_freqs - min_bw) * ear_q
        return freqs
    
    def _make_erb_filters(self, center_freqs, sample_rate, n_fft):
        """Create ERB filterbank matrix"""
        n_freqs = n_fft // 2 + 1
        freqs = np.linspace(0, sample_rate / 2, n_freqs)
        
        filterbank = np.zeros((len(center_freqs), n_freqs))
        
        for i, cf in enumerate(center_freqs):
            erb_width = 24.7 * (0.00437 * cf + 1)
            response = np.exp(-0.5 * ((freqs - cf) / (0.5 * erb_width)) ** 2)
            filterbank[i] = response / response.sum()
        
        return torch.FloatTensor(filterbank)
    
    def forward(self, waveform):
        # Compute STFT
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, 
                         return_complex=True)
        magnitude = torch.abs(stft)
        
        # Apply ERB filterbank
        erb_spec = torch.matmul(self.filterbank, magnitude)
        log_erb = torch.log(erb_spec + 1e-9)
        return log_erb

class BarkFilterbank(nn.Module):
    """Bark-scale filterbank"""
    def __init__(self, sample_rate=16000, n_filters=24, n_fft=512, hop_length=160):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_filters = n_filters
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Create Bark filterbank
        max_bark = self._hz_to_bark(sample_rate / 2)
        bark_points = np.linspace(0, max_bark, n_filters + 2)
        hz_points = [self._bark_to_hz(bark) for bark in bark_points]
        self.filterbank = self._make_bark_filters(hz_points, sample_rate, n_fft)
    
    def _hz_to_bark(self, freq):
        """Convert Hz to Bark scale"""
        return 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500) ** 2)
    
    def _bark_to_hz(self, bark):
        """Convert Bark to Hz"""
        return 600 * np.sinh(bark / 4)
    
    def _make_bark_filters(self, hz_points, sample_rate, n_fft):
        """Create Bark filterbank matrix"""
        n_freqs = n_fft // 2 + 1
        freqs = np.linspace(0, sample_rate / 2, n_freqs)
        
        filterbank = np.zeros((len(hz_points) - 2, n_freqs))
        
        for i in range(1, len(hz_points) - 1):
            lower = hz_points[i - 1]
            center = hz_points[i]
            upper = hz_points[i + 1]
            
            # Triangular filter
            for j, freq in enumerate(freqs):
                if lower <= freq <= center:
                    filterbank[i-1, j] = (freq - lower) / (center - lower)
                elif center < freq <= upper:
                    filterbank[i-1, j] = (upper - freq) / (upper - center)
        
        return torch.FloatTensor(filterbank)
    
    def forward(self, waveform):
        # Compute STFT
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                         return_complex=True)
        magnitude = torch.abs(stft)
        
        # Apply Bark filterbank
        bark_spec = torch.matmul(self.filterbank, magnitude)
        log_bark = torch.log(bark_spec + 1e-9)
        return log_bark

class CQTFrontend(nn.Module):
    """Constant-Q Transform frontend"""
    def __init__(self, sample_rate=16000, hop_length=160, n_bins=84):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.fmin = 50
    
    def forward(self, waveform):
        # Convert to numpy for librosa
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        # Compute CQT
        cqt = librosa.cqt(waveform, sr=self.sample_rate, hop_length=self.hop_length,
                         n_bins=self.n_bins, fmin=self.fmin)
        cqt_mag = np.abs(cqt)
        log_cqt = np.log(cqt_mag + 1e-9)
        
        return torch.FloatTensor(log_cqt)

class PCEN(nn.Module):
    """Per-Channel Energy Normalization"""
    def __init__(self, alpha=0.98, delta=2.0, r=0.5, s=0.025, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.s = s
        self.eps = eps
    
    def forward(self, x):
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # Add batch dimension if missing
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
        
        smooth = torch.zeros_like(x)
        smooth[:, :, 0] = x[:, :, 0]
        
        for t in range(1, x.shape[-1]):
            smooth[:, :, t] = (1 - self.s) * smooth[:, :, t-1] + self.s * x[:, :, t]
        
        pcen = (x / (smooth + self.eps) ** self.alpha + self.delta) ** self.r - self.delta ** self.r    
        if squeeze_batch:
            pcen = pcen.squeeze(0)
        
        return pcen

class MelPCEN(nn.Module):
    """Mel + PCEN frontend"""
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=40, hop_length=160):
        super().__init__()
        self.n_mels = n_mels
        self.mel = MelFilterbank(sample_rate, n_fft, n_mels, hop_length)
        self.pcen = PCEN()
    
    def forward(self, waveform):
        mel_spec = self.mel(waveform)
        mel_linear = torch.exp(mel_spec)
        pcen_spec = self.pcen(mel_linear)
        log_pcen = torch.log(pcen_spec + 1e-9)
        return log_pcen


# ============== TRAINING FUNCTIONS ==============

def load_music_data(data_dir='../ICASSP/data/music', max_per_genre=None):
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


def load_scene_data(data_dir='../ICASSP/data/scenes', max_per_scene=None):
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


def load_speech_data(data_dir='../ICASSP/data/speech', max_per_language=None):
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

def get_frontend_configs():
    """Get all frontend configurations matching the paper specifications"""
    configs = {
        'Mel': {
            'class': MelFilterbank,
            'params': {'n_mels': 40},
            'input_dim': 40
        },
        'ERB': {
            'class': ERBFilterbank,
            'params': {'n_filters': 32},
            'input_dim': 32
        },
        'Bark': {
            'class': BarkFilterbank,
            'params': {'n_filters': 24},
            'input_dim': 24
        },
        'CQT': {
            'class': CQTFrontend,
            'params': {'n_bins': 84},
            'input_dim': 84
        },
        'Mel_PCEN': {
            'class': MelPCEN,
            'params': {'n_mels': 40},
            'input_dim': 40
        }
    }
    return configs

def train_frontend_models(frontend_name, frontend_config, tasks_data, 
                         batch_size=32, num_epochs=30, learning_rate=0.001, device='cuda'):
    """Train models for a specific frontend across all tasks"""
    
    print(f"\n{'='*60}")
    print(f"TRAINING MODELS FOR {frontend_name} FRONTEND")
    print(f"{'='*60}")
    
    # Initialize frontend
    frontend_class = frontend_config['class']
    frontend_params = frontend_config['params']
    input_dim = frontend_config['input_dim']
    frontend = frontend_class(**frontend_params)
    
    print(f"Frontend: {frontend_name}")
    print(f"Input dimension: {input_dim}")
    
    results = {}
    
    # Train model for each task
    for task_name, (file_paths, labels, num_classes) in tasks_data.items():
        if len(file_paths) == 0:
            print(f"⚠️  No data for {task_name}, skipping...")
            continue
            
        print(f"\n--- Training {task_name} model with {frontend_name} ---")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            file_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Classes: {num_classes}")
        
        # Determine max length based on task
        max_length = 5 if task_name == 'speech' else 10
        
        # Create datasets
        train_dataset = AudioDataset(X_train, y_train, frontend, max_length_seconds=max_length)
        val_dataset = AudioDataset(X_val, y_val, frontend, max_length_seconds=max_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=2)
        
        # Create model with correct input dimension
        model = CRNN(input_dim=input_dim, num_classes=num_classes)
        
        # Train model
        model, history, best_acc = train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs, learning_rate=learning_rate, device=device
        )
        
        # Save model with frontend-specific name
        model_path = f'models/crnn_{task_name}_{frontend_name}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model saved: {model_path} (Best Acc: {best_acc:.4f})")
        
        # Save history
        history_path = f'models/{task_name}_{frontend_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        results[task_name] = {
            'best_acc': best_acc,
            'model_path': model_path
        }
    
    return results

def main():
    print("="*60)
    print("TRAINING FRONTEND-SPECIFIC CRNN MODELS")
    print("Matching Paper Specifications")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load all task data first
    print("\n1. Loading all datasets...")
    print("-"*40)
    
    tasks_data = {}
    
    # Load music data
    file_paths, labels, num_classes = load_music_data(max_per_genre=500)  # Reduced for faster training
    if len(file_paths) > 0:
        tasks_data['music'] = (file_paths, labels, num_classes)
    
    # Load scene data  
    file_paths, labels, num_classes = load_scene_data(max_per_scene=500)
    if len(file_paths) > 0:
        tasks_data['scene'] = (file_paths, labels, num_classes)
    
    # Load speech data
    file_paths, labels, num_classes = load_speech_data(max_per_language=500)
    if len(file_paths) > 0:
        tasks_data['speech'] = (file_paths, labels, num_classes)
    
    # Get frontend configurations
    frontend_configs = get_frontend_configs()
    
    # Training parameters
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    
    # Train models for each frontend
    print("\n2. Training models for each frontend...")
    print("-"*40)
    
    all_results = {}
    
    for frontend_name, frontend_config in frontend_configs.items():
        results = train_frontend_models(
            frontend_name, 
            frontend_config, 
            tasks_data,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device
        )
        all_results[frontend_name] = results
    
    # Save summary
    with open('models/training_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    for frontend_name, results in all_results.items():
        print(f"\n{frontend_name}:")
        for task_name, task_results in results.items():
            print(f"  {task_name}: Acc={task_results['best_acc']:.4f}")
    
    print("\nAll models saved in 'models/' directory")
    print("Model naming convention: crnn_[task]_[frontend].pth")
    print("\nReady for evaluation with run_experiments.py")

if __name__ == "__main__":
    main()