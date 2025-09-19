#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'

import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import torchaudio
import librosa
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============== DISTRIBUTED TRAINING SETUP ==============

def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    destroy_process_group()

# ============== OPTIMIZED DATASET CLASS ==============

class AudioDataset(Dataset):
    """Optimized dataset with caching and better error handling"""
    
    def __init__(self, file_paths, labels, frontend, max_length_seconds=10, 
                 cache_features=True, rank=0):
        self.file_paths = file_paths
        self.labels = labels
        self.frontend = frontend
        self.sample_rate = 16000
        self.max_length = max_length_seconds * self.sample_rate
        self.cache_features = cache_features
        self.feature_cache = {}
        self.rank = rank
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Check cache first
        if self.cache_features and idx in self.feature_cache:
            return self.feature_cache[idx], self.labels[idx]
        
        audio_path = self.file_paths[idx]
        
        try:
            # Load audio efficiently
            waveform, sr = torchaudio.load(audio_path, normalize=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Trim or pad
            if waveform.shape[1] > self.max_length:
                # Random crop for training variety
                if self.labels is not None:  # Training mode
                    start = torch.randint(0, waveform.shape[1] - self.max_length + 1, (1,))
                    waveform = waveform[:, start:start + self.max_length]
                else:  # Validation/test mode
                    waveform = waveform[:, :self.max_length]
            else:
                padding = self.max_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            
            # Extract features using frontend
            with torch.no_grad():
                features = self.frontend(waveform.squeeze(0))
            
            # Cache if enabled
            if self.cache_features and len(self.feature_cache) < 10000:  # Limit cache size
                self.feature_cache[idx] = features
            
            return features, self.labels[idx]
            
        except Exception as e:
            if self.rank == 0:
                print(f"Error loading {audio_path}: {e}")
            # Return zero features as fallback
            features = torch.zeros((self.frontend.n_filters if hasattr(self.frontend, 'n_filters') else 40, 
                                   self.max_length // 160))
            return features, self.labels[idx]

# ============== OPTIMIZED CRNN MODEL ==============

class CRNN(nn.Module):
    """CRNN with paper-compliant architecture and optimizations"""
    def __init__(self, input_dim=80, num_classes=10, task_type='classification', dropout_rate=0.3):
        super().__init__()
        self.task_type = task_type
        self.input_dim = input_dim
        
        # Frequency-aware CNN blocks with grouped convolutions for efficiency
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1))  # Only pool frequency
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2))  # Only pool time
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2))
        )
        
        # Calculate RNN input size
        freq_dim_after_pool = max(1, input_dim // 8)
        
        # Frequency attention (as per paper)
        self.freq_attention = nn.Sequential(
            nn.Linear(freq_dim_after_pool * 256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(128, freq_dim_after_pool),
            nn.Sigmoid()
        )
        
        # Dropout layers
        self.dropout_conv = nn.Dropout2d(dropout_rate * 0.6)
        self.dropout_rnn = nn.Dropout(dropout_rate)
        
        # RNN layers
        rnn_input_size = freq_dim_after_pool * 256
        
        # Using LSTM instead of GRU for better gradient flow in deep networks
        self.lstm1 = nn.LSTM(rnn_input_size, 256, num_layers=1,
                            batch_first=True, bidirectional=True, dropout=0)
        self.lstm2 = nn.LSTM(512, 256, num_layers=1,
                            batch_first=True, bidirectional=True, dropout=0)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(512)
        
        # Output head
        if task_type == 'classification':
            self.output_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 1.5),
                nn.Linear(256, num_classes)
            )
        else:
            self.output_head = nn.Linear(512, num_classes)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, 1) * 0.02)
        
    def forward(self, x):
        # Input shape: (batch, freq_bins, time_frames)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        
        # CNN blocks
        x = self.conv1(x)
        x = self.dropout_conv(x)
        
        x = self.conv2(x)
        x = self.dropout_conv(x)
        
        x = self.conv3(x)
        x = self.dropout_conv(x)
        
        x = self.conv4(x)
        x = self.dropout_conv(x)
        
        # Apply frequency attention
        batch, channels, freq, time = x.size()
        
        freq_features = x.mean(dim=3)
        freq_features_flat = freq_features.view(batch, -1)
        freq_weights = self.freq_attention(freq_features_flat)
        freq_weights = freq_weights.view(batch, 1, freq, 1)
        x = x * freq_weights
        
        # Prepare for RNN
        x = x.permute(0, 3, 1, 2).reshape(batch, time, -1)
        
        # Add positional encoding
        if time <= self.positional_encoding.size(1):
            pos_enc = self.positional_encoding[:, :time, :]
        else:
            repeats = (time // self.positional_encoding.size(1)) + 1
            pos_enc = self.positional_encoding.repeat(1, repeats, 1)[:, :time, :]
        x = x + pos_enc
        
        # LSTM layers
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.ln1(lstm_out1)
        lstm_out1 = self.dropout_rnn(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.ln2(lstm_out2)
        
        # Pooling strategies
        avg_pool = torch.mean(lstm_out2, dim=1)
        max_pool, _ = torch.max(lstm_out2, dim=1)
        last_hidden = lstm_out2[:, -1, :]
        
        # Attention pooling
        attention_scores = torch.bmm(lstm_out2, lstm_out2.transpose(1, 2))
        attention_weights = torch.softmax(attention_scores.sum(dim=2, keepdim=True), dim=1)
        attention_pool = (lstm_out2 * attention_weights).sum(dim=1)
        
        # Combine all
        combined = (avg_pool + max_pool + last_hidden + attention_pool) / 4
        
        return self.output_head(combined)

# ============== DATA LOADING FUNCTIONS ==============

def load_music_data(data_dir='../ICASSP/data/music', max_per_genre=None):
    """Load music classification data - keeping original structure"""
    print("Loading music data...")
    
    # Same genre names as original
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
    """Load scene classification data - keeping original structure"""
    print("Loading scene data...")
    
    # Keep it simple like original - just two scene directories
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
    """Load speech data for language classification - keeping original structure"""
    print("Loading speech data...")
    
    # Same language codes as original
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

# ============== TRAINING FUNCTIONS ==============

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, rank, accumulation_steps=1):
    """Single epoch training with mixed precision and gradient accumulation"""
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f'Training', disable=(rank != 0))
    
    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.to(device)
        labels = labels.to(device)
        
        # Mixed precision training
        with autocast():
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Scale loss for gradient accumulation
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Statistics
        train_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        if rank == 0:
            pbar.set_postfix({'loss': loss.item() * accumulation_steps, 
                             'acc': train_correct/train_total})
        
        # Clear cache periodically to prevent memory buildup
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Handle remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    
    return train_loss / len(train_loader), train_correct / train_total

def validate(model, val_loader, criterion, device, rank):
    """Validation with mixed precision"""
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc='Validation', disable=(rank != 0)):
            features = features.to(device)
            labels = labels.to(device)
            
            with autocast():
                outputs = model(features)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    return val_loss / len(val_loader), val_correct / val_total

def train_model_ddp(rank, world_size, frontend_name, frontend_config, 
                    task_name, task_data, num_epochs=30, batch_size=32):
    """Distributed training function with memory optimization"""
    
    # Setup DDP
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    set_seed(42 + rank)
    
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    
    # Unpack task data
    file_paths, labels, num_classes = task_data
    
    if rank == 0:
        print(f"\n--- Training {task_name} with {frontend_name} ---")
        print(f"Total samples: {len(file_paths)}")
    
    # Split data (80/10/10 as per paper)
    X_temp, X_test, y_temp, y_test = train_test_split(
        file_paths, labels, test_size=0.1, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp  # 0.111 * 0.9 ≈ 0.1
    )
    
    if rank == 0:
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize frontend
    frontend_class = frontend_config['class']
    frontend_params = frontend_config['params']
    input_dim = frontend_config['input_dim']
    frontend = frontend_class(**frontend_params)
    
    # Determine max length based on task
    max_length = 5 if task_name == 'speech' else 10
    
    # Create datasets with caching disabled for memory issues
    train_dataset = AudioDataset(X_train, y_train, frontend, 
                                max_length_seconds=max_length, 
                                cache_features=False, rank=rank)  # Disable caching for memory
    val_dataset = AudioDataset(X_val, y_val, frontend, 
                              max_length_seconds=max_length,
                              cache_features=False, rank=rank)
    
    # Distributed samplers
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, 
                                      rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, 
                                    rank=rank, shuffle=False)
    
    # Create dataloaders with fewer workers for memory
    num_workers = 2 if world_size > 4 else 4  # Reduce workers for many GPUs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            sampler=train_sampler, num_workers=num_workers,
                            pin_memory=True, prefetch_factor=1)  # Reduced prefetch
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          sampler=val_sampler, num_workers=num_workers,
                          pin_memory=True, prefetch_factor=1)
    
    # Create model
    model = CRNN(input_dim=input_dim, num_classes=num_classes).to(device)
    model = DDP(model, device_ids=[rank])
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Gradient accumulation steps (increase if memory issues)
    accumulation_steps = 2 if world_size > 4 else 1
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        # Train with gradient accumulation
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, scaler, device, rank,
                                           accumulation_steps=accumulation_steps)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, rank)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model (only on rank 0)
        if rank == 0:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.module.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered")
                break
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()
    
    # Save final model (only on rank 0)
    if rank == 0 and best_model_state:
        os.makedirs('models', exist_ok=True)
        model_path = f'models/crnn_{task_name}_{frontend_name}.pth'
        torch.save(best_model_state, model_path)
        print(f"Model saved: {model_path} (Best Acc: {best_val_acc:.4f})")
        
        # Save training history
        history = {
            'best_val_acc': best_val_acc,
            'final_epoch': epoch + 1
        }
        with open(f'models/{task_name}_{frontend_name}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    cleanup()
    return best_val_acc

# ============== AUDIO FRONT-ENDS ==============

class LEAFFrontend(nn.Module):
    """LEAF: Learnable Audio Frontend"""
    def __init__(self, sample_rate=16000, n_filters=64, window_len=401, 
                 window_stride=160):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_filters = n_filters
        self.window_len = window_len
        self.window_stride = window_stride
        
        # Initialize center frequencies uniformly in mel-scale
        min_freq = 80.0
        max_freq = sample_rate / 2
        mel_min = 2595 * np.log10(1 + min_freq / 700)
        mel_max = 2595 * np.log10(1 + max_freq / 700)
        mel_scale = np.linspace(mel_min, mel_max, n_filters)
        center_freqs = 700 * (10 ** (mel_scale / 2595) - 1)
        
        # Learnable parameters for Gabor filters
        self.center_freqs = nn.Parameter(torch.tensor(center_freqs, dtype=torch.float32))
        self.bandwidths = nn.Parameter(torch.ones(n_filters) * 0.5)
    
    def forward(self, waveform):
        # Store original shape info
        original_dim = waveform.dim()
        device = waveform.device
        
        # Ensure waveform is 1D for filter generation
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        
        # Create time axis on the same device
        t = torch.arange(0, self.window_len, dtype=torch.float32, device=device)
        t = t / self.sample_rate
        t = t - (self.window_len - 1) / (2 * self.sample_rate)
        
        # Constrain frequencies to valid range
        center_freqs = torch.clamp(self.center_freqs, 50, self.sample_rate/2)
        bandwidths = torch.clamp(self.bandwidths, 0.1, 2.0)
        
        # Generate Gabor filters
        filters = []
        for i in range(self.n_filters):
            f_c = center_freqs[i]
            bw = bandwidths[i]
            
            # Gabor filter: Gaussian envelope * sinusoid
            gaussian = torch.exp(-(t ** 2) / (2 * (bw / self.sample_rate) ** 2))
            sinusoid = torch.cos(2 * math.pi * f_c * t)
            gabor = gaussian * sinusoid
            
            # Normalize
            gabor = gabor / (torch.sum(gabor ** 2) ** 0.5 + 1e-8)
            filters.append(gabor)
        
        # Stack filters: shape (n_filters, window_len)
        filters = torch.stack(filters, dim=0)
        
        # Reshape for conv1d: need (out_channels, in_channels, kernel_size)
        filters = filters.unsqueeze(1)  # Now shape: (n_filters, 1, window_len)
        
        # Prepare waveform for conv1d: need (batch, in_channels, length)
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, time)
        
        # Apply convolution
        filtered = F.conv1d(waveform, filters, stride=self.window_stride, padding=self.window_len//2)
        
        # Square for energy
        filtered = filtered ** 2
        
        # Log compression
        output = torch.log(filtered + 1e-9)
        
        # Remove batch dimension to get (n_filters, time_frames)
        output = output.squeeze(0)
        
        return output


class SincNetFrontend(nn.Module):
    """SincNet: Learnable sinc-based filters"""
    def __init__(self, sample_rate=16000, n_filters=64, filter_length=251):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_filters = n_filters
        self.filter_length = filter_length
        
        # Initialize filter parameters using mel-scale
        min_freq = 50
        max_freq = sample_rate / 2
        
        min_mel = 2595 * np.log10(1 + min_freq / 700)
        max_mel = 2595 * np.log10(1 + max_freq / 700)
        
        mel_points = np.linspace(min_mel, max_mel, n_filters + 1)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        # Learnable parameters for band-pass filters
        self.freq_low = nn.Parameter(torch.tensor(hz_points[:-1], dtype=torch.float32))
        self.freq_band = nn.Parameter(torch.tensor(np.diff(hz_points), dtype=torch.float32))
        
        # Hamming window
        n = torch.arange(0, filter_length, dtype=torch.float32)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / filter_length)
        self.register_buffer('window', window)
    
    def forward(self, waveform):
        # Store original shape info
        original_dim = waveform.dim()
        device = waveform.device
        
        # Ensure waveform is 1D for processing
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        
        # Constrain frequencies to valid range
        freq_low = torch.clamp(self.freq_low, 50, self.sample_rate/2 - 100)
        freq_band = torch.clamp(self.freq_band, 50, 1000)
        
        # Normalize frequencies
        freq_low_hz = freq_low / self.sample_rate
        freq_high_hz = (freq_low + freq_band) / self.sample_rate
        
        # Time axis on same device
        n = torch.arange(0, self.filter_length, dtype=torch.float32, device=device)
        n_center = (self.filter_length - 1) / 2
        
        # Generate sinc filters
        filters = []
        for i in range(self.n_filters):
            low = freq_low_hz[i]
            high = freq_high_hz[i]
            
            # Time samples
            t = (n - n_center) / self.sample_rate
            
            # Band-pass = high-pass - low-pass
            high_pass = 2 * high * torch.sinc(2 * high * t)
            low_pass = 2 * low * torch.sinc(2 * low * t)
            band_pass = high_pass - low_pass
            
            # Apply window
            band_pass = band_pass * self.window
            
            # Normalize
            band_pass = band_pass / (torch.sum(band_pass ** 2) ** 0.5 + 1e-8)
            filters.append(band_pass)
        
        # Stack filters: shape (n_filters, filter_length)
        filters = torch.stack(filters, dim=0)
        
        # Reshape for conv1d: (out_channels, in_channels, kernel_size)
        filters = filters.unsqueeze(1)  # Shape: (n_filters, 1, filter_length)
        
        # Prepare waveform for conv1d: (batch, in_channels, length)
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, time)
        
        # Apply convolution
        filtered = F.conv1d(waveform, filters, stride=160, padding=self.filter_length//2)
        
        # Square for energy (consistent with other frontends)
        filtered = filtered ** 2
        
        # Log compression
        output = torch.log(filtered + 1e-9)
        
        # Remove batch dimension to get (n_filters, time_frames)
        output = output.squeeze(0)
        
        return output


class MelFilterbank(nn.Module):
    """Mel-scale filterbank front-end"""
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=40, hop_length=160):
        super().__init__()
        self.n_mels = n_mels
        self.n_filters = n_mels  # Add this for compatibility
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
        self.n_filters = n_bins  # Add for compatibility
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
        self.n_filters = n_mels  # Add for compatibility
        self.mel = MelFilterbank(sample_rate, n_fft, n_mels, hop_length)
        self.pcen = PCEN()
    
    def forward(self, waveform):
        mel_spec = self.mel(waveform)
        mel_linear = torch.exp(mel_spec)
        pcen_spec = self.pcen(mel_linear)
        log_pcen = torch.log(pcen_spec + 1e-9)
        return log_pcen

# ============== FRONTEND CONFIGURATIONS ==============

def get_frontend_configs():
    """Frontend configurations matching paper specifications"""
    configs = {
        'SincNet': {
            'class': SincNetFrontend,
            'params': {'n_filters': 64},
            'input_dim': 64
        },
        'LEAF': {
            'class': LEAFFrontend,
            'params': {'n_filters': 64},
            'input_dim': 64
        },
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

# ============== MAIN TRAINING ORCHESTRATOR ==============

def main():
    print("="*60)
    print("MULTI-GPU OPTIMIZED TRAINING")
    print("ICASSP Paper Implementation")
    print("="*60)
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    print(f"\nDetected {world_size} GPUs")
    
    if world_size == 0:
        print("No GPUs found! Please run on a GPU machine.")
        return
    
    # Check GPU memory status
    print("\nGPU Memory Status:")
    for i in range(world_size):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {allocated:.2f}/{reserved:.2f}/{total:.2f} GB (allocated/reserved/total)")
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # Training configuration
    num_epochs = 30
    
    # Adjust batch size based on GPU count - don't scale linearly for large GPU counts
    if world_size <= 2:
        batch_size_per_gpu = 32
    elif world_size <= 4:
        batch_size_per_gpu = 16
    else:
        batch_size_per_gpu = 8  # For 5+ GPUs, use smaller batch size
    
    batch_size = batch_size_per_gpu  # This is per GPU, DataLoader handles distribution
    max_samples = None  # Use all available data
    
    print(f"Batch size per GPU: {batch_size}")
    print(f"Total effective batch size: {batch_size * world_size}")
    
    # If memory issues, suggest using fewer GPUs
    if world_size > 4:
        print(f"\n  WARNING: Using {world_size} GPUs may cause memory issues.")
        print("   Consider using fewer GPUs by setting CUDA_VISIBLE_DEVICES")
        print(f"   Example: CUDA_VISIBLE_DEVICES=0,1,2,3 python {__file__}")
    
    # Load all datasets
    print("\n1. Loading all datasets...")
    print("-"*40)
    
    tasks_data = {}
    
    # Music task (6 genre collections as per paper)
    file_paths, labels, num_classes = load_music_data(
        max_per_genre=max_samples
    )
    if len(file_paths) > 0:
        tasks_data['music'] = (file_paths, labels, num_classes)
    
    # Scene task (2 groups: european-1 and european-2)
    file_paths, labels, num_classes = load_scene_data(
        max_per_scene=max_samples
    )
    if len(file_paths) > 0:
        tasks_data['scene'] = (file_paths, labels, num_classes)
    
    # Speech task (11 languages as per paper)
    file_paths, labels, num_classes = load_speech_data(
        max_per_language=max_samples
    )
    if len(file_paths) > 0:
        tasks_data['speech'] = (file_paths, labels, num_classes)
    
    # Get frontend configurations
    frontend_configs = get_frontend_configs()
    
    # Train models for each frontend and task
    print("\n2. Training models (Multi-GPU)...")
    print("-"*40)
    
    all_results = {}
    
    for frontend_name, frontend_config in frontend_configs.items():
        print(f"\n{'='*60}")
        print(f"TRAINING {frontend_name} FRONTEND")
        print(f"{'='*60}")
        
        frontend_results = {}
        
        for task_name, task_data in tasks_data.items():
            if len(task_data[0]) == 0:
                print(f"Skipping {task_name} - no data")
                continue
            
            # Launch distributed training
            print(f"\nLaunching {world_size}-GPU training for {task_name}")
            
            # For single-node multi-GPU, we use spawn
            if world_size > 1:
                mp.spawn(train_model_ddp,
                        args=(world_size, frontend_name, frontend_config,
                              task_name, task_data, num_epochs, batch_size),
                        nprocs=world_size,
                        join=True)
            else:
                # Single GPU fallback
                best_acc = train_model_ddp(0, 1, frontend_name, frontend_config,
                                          task_name, task_data, num_epochs, batch_size)
            
            # Load result from saved file
            history_path = f'models/{task_name}_{frontend_name}_history.json'
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                    frontend_results[task_name] = {
                        'best_acc': history['best_val_acc'],
                        'model_path': f'models/crnn_{task_name}_{frontend_name}.pth'
                    }
        
        all_results[frontend_name] = frontend_results
    
    # Save comprehensive summary
    with open('models/training_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    
    for frontend_name, results in all_results.items():
        print(f"\n{frontend_name}:")
        for task_name, task_results in results.items():
            print(f"  {task_name}: Acc={task_results['best_acc']:.4f}")
    
    print("\n All models saved in 'models/' directory")

if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    main()