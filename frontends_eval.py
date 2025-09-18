#!/usr/bin/env python3
"""
Complete Audio Front-End Experimental Pipeline with Multi-Run Significance Testing
Processes real audio files through pre-trained models and validates results across multiple runs
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============== AUDIO FRONT-ENDS ==============

class MelFilterbank(nn.Module):
    """Mel-scale filterbank front-end"""
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=80, hop_length=160):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
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
    """ERB-scale filterbank - best efficiency/fairness tradeoff"""
    def __init__(self, sample_rate=16000, n_filters=80, n_fft=512, hop_length=160):
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
        order = 1
        
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
    def __init__(self, sample_rate=16000, n_filters=80, n_fft=512, hop_length=160):
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
        # Approximation
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
    def __init__(self, sample_rate=16000, hop_length=160, n_bins=80):
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
        
        # Apply PCEN
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


# ============== CRNN MODEL ==============

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


# ============== DATA LOADING ==============

class AudioDataLoader:
    """Load and process audio files from processed_data directory"""
    
    def __init__(self, data_dir='../ICASSP/data', sample_rate=16000):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.audio_cache = {}
    
    def load_music_data(self, max_samples=50):
        """Load music classification data"""
        music_dir = self.data_dir / 'music'
        data = {'files': [], 'labels': [], 'audio': []}
        
        genres = ['arab_andalusian', 'carnatic', 'fma_small', 'gtzan', 
                 'hindustani', 'turkish_makam']
        
        for genre_idx, genre in enumerate(genres):
            genre_dir = music_dir / genre
            if not genre_dir.exists():
                print(f"Warning: {genre_dir} not found")
                continue
            
            audio_files = list(genre_dir.glob('*.wav')) + list(genre_dir.glob('*.mp3'))
            audio_files = audio_files[:max_samples]
            
            print(f"Loading {len(audio_files)} files from {genre}")
            
            for audio_file in tqdm(audio_files, desc=f"Loading {genre}"):
                try:
                    # Convert Path to string for torchaudio
                    waveform, sr = torchaudio.load(str(audio_file))
                    
                    # Resample if needed
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    
                    # Convert to mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Trim or pad to 10 seconds
                    target_length = self.sample_rate * 10
                    if waveform.shape[1] > target_length:
                        waveform = waveform[:, :target_length]
                    else:
                        padding = target_length - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, padding))
                    
                    data['files'].append(str(audio_file))
                    data['labels'].append(genre_idx)
                    data['audio'].append(waveform)
                    
                except Exception as e:
                    print(f"Error loading {audio_file}: {e}")
        
        return data
    
    def load_scene_data(self, max_samples=50):
        """Load scene classification data"""
        scenes_dir = self.data_dir / 'scenes'
        data = {'files': [], 'labels': [], 'audio': []}
        
        scenes = ['european-1', 'european-2']  # From TAU Urban dataset
        
        for scene_idx, scene in enumerate(scenes):
            scene_dir = scenes_dir / scene
            if not scene_dir.exists():
                print(f"Warning: {scene_dir} not found")
                continue
            
            audio_files = list(scene_dir.glob('*.wav'))[:max_samples]
            print(f"Loading {len(audio_files)} files from {scene}")
            
            for audio_file in tqdm(audio_files, desc=f"Loading {scene}"):
                try:
                    # Convert Path to string for torchaudio
                    waveform, sr = torchaudio.load(str(audio_file))
                    
                    # Resample if needed
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    
                    # Convert to mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Trim or pad to 10 seconds
                    target_length = self.sample_rate * 10
                    if waveform.shape[1] > target_length:
                        waveform = waveform[:, :target_length]
                    else:
                        padding = target_length - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, padding))
                    
                    data['files'].append(str(audio_file))
                    data['labels'].append(scene_idx)
                    data['audio'].append(waveform)
                    
                except Exception as e:
                    print(f"Error loading {audio_file}: {e}")
        
        return data
    
    def load_speech_data(self, max_samples=50):
        """Load speech recognition data"""
        speech_dir = self.data_dir / 'speech'
        data = {'files': [], 'languages': [], 'audio': []}
        
        languages = ['de', 'en', 'es', 'fr', 'it', 'nl', 'pa-IN', 'th', 'vi', 
                    'yue', 'zh-CN']
        
        for lang in languages:
            lang_dir = speech_dir / lang
            if not lang_dir.exists():
                print(f"Warning: {lang_dir} not found")
                continue
            
            # Get all audio files
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.flac']:
                audio_files.extend(list(lang_dir.glob(ext)))
            
            audio_files = sorted(audio_files)[:max_samples]
            
            if not audio_files:
                print(f"Warning: No audio files found in {lang_dir}")
                continue
                
            print(f"Loading {len(audio_files)} files from {lang}")
            
            for audio_file in tqdm(audio_files, desc=f"Loading {lang}"):
                try:
                    waveform, sr = torchaudio.load(str(audio_file))
                    
                    # Resample if needed
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    
                    # Convert to mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Trim or pad to 5 seconds for speech
                    target_length = self.sample_rate * 5
                    if waveform.shape[1] > target_length:
                        waveform = waveform[:, :target_length]
                    else:
                        padding = target_length - waveform.shape[1]
                        waveform = torch.nn.functional.pad(waveform, (0, padding))
                    
                    data['files'].append(str(audio_file))
                    data['languages'].append(lang)
                    data['audio'].append(waveform)
                    
                except Exception as e:
                    print(f"Error loading {audio_file}: {e}")
        
        return data


# ============== EVALUATION ==============

class FairnessEvaluator:
    """Calculate fairness metrics from actual predictions with bootstrap significance testing"""
    
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(list))
        # Define language groups for speech task
        self.tonal_languages = ['pa-IN', 'th', 'vi', 'yue', 'zh-CN']
        self.non_tonal_languages = ['de', 'en', 'es', 'fr', 'it', 'nl']
    
    def bootstrap_significance_test(self, group1_scores, group2_scores, n_bootstrap=1000, alpha=0.01):
        """
        Perform bootstrap test to determine if performance difference is significant
        Uses the percentile method for hypothesis testing
        """
        # Convert to arrays
        group1_scores = np.array(group1_scores)
        group2_scores = np.array(group2_scores)
        
        # Calculate observed difference
        observed_diff = np.mean(group1_scores) - np.mean(group2_scores)
        
        # Method 1: Permutation-based bootstrap
        # Under null hypothesis, the two groups are exchangeable
        combined_scores = np.concatenate([group1_scores, group2_scores])
        n1 = len(group1_scores)
        n2 = len(group2_scores)
        
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Shuffle the combined data
            np.random.shuffle(combined_scores)
            
            # Split into two groups of original sizes
            perm_group1 = combined_scores[:n1]
            perm_group2 = combined_scores[n1:n1+n2]
            
            # Calculate difference
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            bootstrap_diffs.append(perm_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value (two-tailed test)
        # Count how many permuted differences are as extreme or more extreme than observed
        p_value = np.sum(np.abs(bootstrap_diffs) >= np.abs(observed_diff)) / n_bootstrap
        
        # Also calculate confidence interval using the standard bootstrap
        # (resampling from each group separately)
        standard_bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Sample with replacement from each group
            boot_group1 = np.random.choice(group1_scores, size=len(group1_scores), replace=True)
            boot_group2 = np.random.choice(group2_scores, size=len(group2_scores), replace=True)
            
            boot_diff = np.mean(boot_group1) - np.mean(boot_group2)
            standard_bootstrap_diffs.append(boot_diff)
        
        standard_bootstrap_diffs = np.array(standard_bootstrap_diffs)
        
        return {
            'p_value': float(p_value),
            'significant': bool(p_value < alpha),
            'observed_diff': float(observed_diff),
            'confidence_interval': [float(x) for x in np.percentile(standard_bootstrap_diffs, [0.5, 99.5])],  # 99% CI for alpha=0.01
            'effect_size': float(observed_diff / np.std(combined_scores)) if np.std(combined_scores) > 0 else 0  # Cohen's d
        }
    
    def evaluate_frontend(self, frontend, model, data, frontend_name, task_name):
        """Evaluate a single frontend on a task with significance testing"""
        model.eval()
        frontend.eval()
        
        all_predictions = []
        all_labels = []
        group_predictions = defaultdict(list)
        group_labels = defaultdict(list)
        group_raw_scores = defaultdict(list)  # Store raw scores for bootstrap
        
        with torch.no_grad():
            for i, audio in enumerate(tqdm(data['audio'], desc=f"Evaluating {frontend_name}")):
                # Process through frontend
                features = frontend(audio)
                
                # Add batch dimension if needed
                if len(features.shape) == 2:
                    features = features.unsqueeze(0)
                
                # Get predictions
                output = model(features)
                pred = torch.argmax(output, dim=-1).item()
                
                all_predictions.append(pred)
                
                if task_name == 'music':
                    label = data['labels'][i]
                    group = data['files'][i].split('/')[-2]  # Genre from path
                    all_labels.append(label)
                    group_predictions[group].append(pred)
                    group_labels[group].append(label)
                    # Store whether this prediction is correct (for bootstrap)
                    group_raw_scores[group].append(1 if pred == label else 0)
                    
                elif task_name == 'scene':
                    label = data['labels'][i]
                    group = data['files'][i].split('/')[-2]  # Scene type from path
                    all_labels.append(label)
                    group_predictions[group].append(pred)
                    group_labels[group].append(label)
                    group_raw_scores[group].append(1 if pred == label else 0)
                    
                elif task_name == 'speech':
                    # Language classification task
                    lang = data['languages'][i]
                    # Create label from language index
                    lang_to_idx = {'de': 0, 'en': 1, 'es': 2, 'fr': 3, 'it': 4, 'nl': 5,
                                  'pa-IN': 6, 'th': 7, 'vi': 8, 'yue': 9, 'zh-CN': 10}
                    label = lang_to_idx[lang]
                    all_labels.append(label)
                    group_predictions[lang].append(pred)
                    group_labels[lang].append(label)
                    group_raw_scores[lang].append(1 if pred == label else 0)
        
        # Calculate metrics
        results = {}
        
        overall_acc = accuracy_score(all_labels, all_predictions)
        overall_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        # Calculate per-group metrics
        group_metrics = {}
        for group in group_predictions:
            if len(group_labels[group]) > 0:
                acc = accuracy_score(group_labels[group], group_predictions[group])
                f1 = f1_score(group_labels[group], group_predictions[group], 
                             average='macro', zero_division=0)
                group_metrics[group] = {
                    'accuracy': acc, 
                    'f1': f1,
                    'n_samples': len(group_labels[group])
                }
        
        # Calculate fairness metrics
        accuracies = [m['accuracy'] for m in group_metrics.values()]
        
        # Worst-Group Score (WGS) - minimum accuracy across groups
        wgs = min(accuracies) if accuracies else 0
        
        # Performance Gap (Δ) - maximum disparity between groups
        gap = max(accuracies) - min(accuracies) if len(accuracies) > 1 else 0
        
        # Disparate Impact (ρ) - ratio of worst to best performance
        di = min(accuracies) / max(accuracies) if len(accuracies) > 1 and max(accuracies) > 0 else 1
        
        results = {
            'overall_accuracy': overall_acc,
            'overall_f1': overall_f1,
            'group_metrics': group_metrics,
            'wgs': wgs,
            'gap': gap,
            'di': di
        }
        
        # Add significance testing based on task
        if task_name == 'music':
            western = ['gtzan', 'fma_small']
            non_western = ['arab_andalusian', 'carnatic', 'hindustani', 'turkish_makam']
            
            western_scores = []
            non_western_scores = []
            
            for g in western:
                if g in group_raw_scores:
                    western_scores.extend(group_raw_scores[g])
            
            for g in non_western:
                if g in group_raw_scores:
                    non_western_scores.extend(group_raw_scores[g])
            
            if western_scores and non_western_scores:
                sig_test = self.bootstrap_significance_test(
                    western_scores, non_western_scores, 
                    n_bootstrap=1000, alpha=0.01
                )
                results['significance_test'] = sig_test
                results['comparison'] = 'Western vs Non-Western'
                
                print(f"    Significance test (Western vs Non-Western):")
                print(f"      p-value: {sig_test['p_value']:.4f}")
                print(f"      Significant at p<0.01: {sig_test['significant']}")
                print(f"      Performance difference: {sig_test['observed_diff']:.4f}")
        
        elif task_name == 'scene':
            europe1_scores = group_raw_scores.get('european-1', [])
            europe2_scores = group_raw_scores.get('european-2', [])
            
            if europe1_scores and europe2_scores:
                sig_test = self.bootstrap_significance_test(
                    europe1_scores, europe2_scores,
                    n_bootstrap=1000, alpha=0.01
                )
                results['significance_test'] = sig_test
                results['comparison'] = 'Europe-1 vs Europe-2'
                
                print(f"    Significance test (Europe-1 vs Europe-2):")
                print(f"      p-value: {sig_test['p_value']:.4f}")
                print(f"      Significant at p<0.01: {sig_test['significant']}")
                print(f"      Performance difference: {sig_test['observed_diff']:.4f}")
        
        elif task_name == 'speech':
            # Test tonal vs non-tonal languages
            tonal_scores = []
            non_tonal_scores = []
            
            for lang in self.tonal_languages:
                if lang in group_raw_scores:
                    tonal_scores.extend(group_raw_scores[lang])
            
            for lang in self.non_tonal_languages:
                if lang in group_raw_scores:
                    non_tonal_scores.extend(group_raw_scores[lang])
            
            if tonal_scores and non_tonal_scores:
                sig_test = self.bootstrap_significance_test(
                    tonal_scores, non_tonal_scores,
                    n_bootstrap=1000, alpha=0.01
                )
                results['significance_test'] = sig_test
                results['comparison'] = 'Tonal vs Non-tonal'
                
                # Calculate average accuracy for each group (for paper reporting)
                tonal_accs = [group_metrics[lang]['accuracy'] for lang in self.tonal_languages 
                             if lang in group_metrics]
                non_tonal_accs = [group_metrics[lang]['accuracy'] for lang in self.non_tonal_languages 
                                  if lang in group_metrics]
                
                if tonal_accs and non_tonal_accs:
                    results['tonal_avg_acc'] = np.mean(tonal_accs)
                    results['non_tonal_avg_acc'] = np.mean(non_tonal_accs)
                    # This is your paper's WER gap (but using accuracy)
                    results['tonal_gap'] = results['non_tonal_avg_acc'] - results['tonal_avg_acc']
                
                print(f"    Significance test (Tonal vs Non-tonal):")
                print(f"      Tonal avg accuracy: {results.get('tonal_avg_acc', 0):.4f}")
                print(f"      Non-tonal avg accuracy: {results.get('non_tonal_avg_acc', 0):.4f}")
                print(f"      Gap: {results.get('tonal_gap', 0):.4f}")
                print(f"      p-value: {sig_test['p_value']:.4f}")
                print(f"      Significant at p<0.01: {sig_test['significant']}")
        
        return results
    
    def run_full_evaluation(self, frontends, tasks_data, run_seed=None):
        """Run complete evaluation across all frontends and tasks with significance testing"""
        
        if run_seed is not None:
            # Set seeds for reproducibility
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)
        
        results = {}
        all_frontend_scores = defaultdict(lambda: defaultdict(dict))
        
        for frontend_name, frontend in frontends.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {frontend_name}")
            print(f"{'='*50}")
            
            # Determine input dimension for this frontend
            if frontend_name in ['Mel', 'Mel+PCEN', 'Mel_PCEN']:
                input_dim = 40
            elif frontend_name == 'ERB':
                input_dim = 32
            elif frontend_name == 'Bark':
                input_dim = 24
            elif frontend_name == 'CQT':
                input_dim = 84
            else:
                input_dim = 80  # fallback
            
            results[frontend_name] = {}
            
            for task_name, data in tasks_data.items():
                if data and len(data['audio']) > 0:
                    print(f"\nTask: {task_name}")
                    
                    # Determine number of classes based on task
                    if task_name == 'music':
                        num_classes = 6
                    elif task_name == 'scene':
                        num_classes = 2
                    elif task_name == 'speech':
                        num_classes = 11
                    else:
                        num_classes = 10  # fallback
                    
                    # Create model with correct input dimension
                    model = CRNN(input_dim=input_dim, num_classes=num_classes, 
                               task_type='classification')
                    
                    # Load frontend-specific weights
                    # Handle both Mel+PCEN and Mel_PCEN naming
                    frontend_file_name = frontend_name.replace('+', '_')
                    model_file = f'models/crnn_{task_name}_{frontend_file_name}.pth'
                    
                    if os.path.exists(model_file):
                        print(f"  Loading weights from {model_file}")
                        model.load_state_dict(torch.load(model_file, map_location='cpu'))
                    else:
                        # Try alternative naming if first attempt fails
                        alt_model_file = f'models/crnn_{task_name}.pth'
                        if os.path.exists(alt_model_file) and frontend_name == 'Mel':
                            print(f"  Loading weights from {alt_model_file} (fallback)")
                            model.load_state_dict(torch.load(alt_model_file, map_location='cpu'))
                        else:
                            print(f"  Warning: No pre-trained weights found at {model_file}")
                            print(f"  Using random initialization - results will be poor!")
                    
                    # Evaluate
                    task_results = self.evaluate_frontend(
                        frontend, model, data, frontend_name, task_name
                    )
                    results[frontend_name][task_name] = task_results
                    
                    # Print results
                    print(f"  Overall Accuracy: {task_results['overall_accuracy']:.4f}")
                    print(f"  Overall F1: {task_results['overall_f1']:.4f}")
                    print(f"  WGS: {task_results['wgs']:.4f}")
                    print(f"  Gap: {task_results['gap']:.4f}")
                    print(f"  DI: {task_results['di']:.4f}")
                    
                    # Store for cross-frontend comparison
                    all_frontend_scores[task_name][frontend_name] = {
                        'gap': task_results['gap'],
                        'wgs': task_results['wgs'],
                        'di': task_results['di']
                    }
                    if task_name == 'speech' and 'tonal_gap' in task_results:
                        all_frontend_scores[task_name][frontend_name]['tonal_gap'] = task_results['tonal_gap']
        
        # Compare frontends (Mel baseline vs others)
        print("\n" + "="*60)
        print("CROSS-FRONTEND SIGNIFICANCE TESTING")
        print("="*60)
        
        if 'Mel' in frontends:
            for task_name in all_frontend_scores:
                print(f"\n{task_name.upper()} - Comparing frontends to Mel baseline:")
                mel_gap = all_frontend_scores[task_name].get('Mel', {}).get('gap', 0)
                
                for frontend_name in frontends:
                    if frontend_name != 'Mel' and frontend_name in all_frontend_scores[task_name]:
                        other_gap = all_frontend_scores[task_name][frontend_name]['gap']
                        reduction = ((mel_gap - other_gap) / mel_gap * 100) if mel_gap > 0 else 0
                        print(f"  {frontend_name}: Gap={other_gap:.4f}, Reduction={reduction:.1f}%")
        
        return results


# ============== MULTI-RUN EVALUATION ==============

def run_multi_run_evaluation(n_runs=5):
    """
    Run the entire experiment multiple times with different random seeds
    to test if bias findings are statistically significant and reproducible
    """
    print("="*70)
    print("MULTI-RUN AUDIO FRONT-END BIAS EVALUATION")
    print(f"Running {n_runs} independent experiments for statistical validation")
    print("="*70)
    
    # Initialize data loader
    data_loader = AudioDataLoader()
    
    # Load datasets once (same data for all runs)
    print("\n1. Loading Audio Data...")
    print("-"*40)
    
    tasks_data = {}
    
    # Load music data (reduced samples for faster multi-run testing)
    print("\nLoading Music Classification Data...")
    music_data = data_loader.load_music_data(max_samples=100)  # Reduced for multi-run
    if music_data['audio']:
        tasks_data['music'] = music_data
        print(f"Loaded {len(music_data['audio'])} music samples")
    
    # Load scene data
    print("\nLoading Scene Classification Data...")
    scene_data = data_loader.load_scene_data(max_samples=50)
    if scene_data['audio']:
        tasks_data['scene'] = scene_data
        print(f"Loaded {len(scene_data['audio'])} scene samples")
    
    # Load speech data
    print("\nLoading Speech Recognition Data...")
    speech_data = data_loader.load_speech_data(max_samples=100)
    if speech_data['audio']:
        tasks_data['speech'] = speech_data
        print(f"Loaded {len(speech_data['audio'])} speech samples")
    
    if not tasks_data:
        print("\nError: No data loaded. Please check your data directory structure.")
        return
    
    print("\n2. Initializing Audio Front-ends...")
    print("-"*40)
    
    frontends = {
        'Mel': MelFilterbank(n_mels=40),
        'ERB': ERBFilterbank(n_filters=32),
        'Bark': BarkFilterbank(n_filters=24),
        'CQT': CQTFrontend(n_bins=84),
        'Mel_PCEN': MelPCEN(n_mels=40)
    }
    
    print(f"Initialized {len(frontends)} front-ends: {list(frontends.keys())}")
    
    # Storage for results across runs
    all_runs_results = []
    
    # Run multiple experiments
    print("\n3. Running Multi-Run Evaluation...")
    print("-"*40)
    
    for run_idx in range(n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{n_runs} (seed={run_idx})")
        print(f"{'='*60}")
        
        # Run evaluation with specific seed
        evaluator = FairnessEvaluator()
        run_results = evaluator.run_full_evaluation(frontends, tasks_data, run_seed=run_idx)
        all_runs_results.append(run_results)
        
        # Save individual run results
        os.makedirs('results/multi_run', exist_ok=True)
        with open(f'results/multi_run/run_{run_idx}_results.json', 'w') as f:
            json.dump(run_results, f, indent=2, default=str)
    
    # Aggregate results across runs
    print("\n" + "="*70)
    print("AGGREGATING MULTI-RUN RESULTS")
    print("="*70)
    
    aggregated_results = aggregate_multirun_results(all_runs_results)
    
    # Perform cross-run significance testing
    print("\n" + "="*70)
    print("CROSS-RUN STATISTICAL SIGNIFICANCE TESTING")
    print("="*70)
    
    significance_results = test_multirun_significance(aggregated_results)
    
    # Generate final report
    generate_multirun_report(aggregated_results, significance_results, n_runs)
    
    # Create visualizations
    plot_multirun_results(aggregated_results)
    
    print("\n" + "="*70)
    print("MULTI-RUN EVALUATION COMPLETE")
    print("="*70)
    print("\nAll results saved in results/multi_run/ directory")
    print("Check multirun_report.txt for detailed statistical analysis")
    
    return aggregated_results, significance_results


def aggregate_multirun_results(all_runs_results):
    """Aggregate results across multiple runs"""
    aggregated = {}
    
    for frontend in all_runs_results[0].keys():
        aggregated[frontend] = {}
        
        for task in all_runs_results[0][frontend].keys():
            metrics_across_runs = {
                'accuracy': [],
                'f1': [],
                'wgs': [],
                'gap': [],
                'di': []
            }
            
            # Collect metrics from all runs
            for run_results in all_runs_results:
                if frontend in run_results and task in run_results[frontend]:
                    task_results = run_results[frontend][task]
                    metrics_across_runs['accuracy'].append(task_results.get('overall_accuracy', 0))
                    metrics_across_runs['f1'].append(task_results.get('overall_f1', 0))
                    metrics_across_runs['wgs'].append(task_results.get('wgs', 0))
                    metrics_across_runs['gap'].append(task_results.get('gap', 0))
                    metrics_across_runs['di'].append(task_results.get('di', 0))
            
            # Calculate statistics
            aggregated[frontend][task] = {}
            for metric, values in metrics_across_runs.items():
                if values:
                    aggregated[frontend][task][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'values': values
                    }
    
    return aggregated


def test_multirun_significance(aggregated_results):
    """Test if differences between frontends are significant across runs"""
    significance_results = {}
    
    baseline = 'Mel'
    
    for task in ['music', 'scene', 'speech']:
        significance_results[task] = {}
        
        if task not in aggregated_results[baseline]:
            continue
        
        for metric in ['gap', 'wgs']:
            significance_results[task][metric] = {}
            
            if metric not in aggregated_results[baseline][task]:
                continue
            
            baseline_values = aggregated_results[baseline][task][metric]['values']
            
            for frontend in aggregated_results.keys():
                if frontend == baseline:
                    continue
                
                if task in aggregated_results[frontend] and metric in aggregated_results[frontend][task]:
                    compare_values = aggregated_results[frontend][task][metric]['values']
                    
                    # Ensure equal length
                    min_len = min(len(baseline_values), len(compare_values))
                    baseline_vals = baseline_values[:min_len]
                    compare_vals = compare_values[:min_len]
                    
                    if len(baseline_vals) > 1:
                        # Paired t-test
                        t_stat, t_pval = stats.ttest_rel(baseline_vals, compare_vals)
                        
                        # Effect size (Cohen's d)
                        diff = np.array(baseline_vals) - np.array(compare_vals)
                        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                        
                        significance_results[task][metric][frontend] = {
                            'baseline_mean': float(np.mean(baseline_vals)),
                            'baseline_std': float(np.std(baseline_vals)),
                            'frontend_mean': float(np.mean(compare_vals)),
                            'frontend_std': float(np.std(compare_vals)),
                            'difference': float(np.mean(baseline_vals) - np.mean(compare_vals)),
                            'p_value': float(t_pval),
                            'cohens_d': float(cohens_d),
                            'significant_p001': bool(t_pval < 0.01),
                            'significant_p005': bool(t_pval < 0.05)
                        }
    
    return significance_results


def generate_multirun_report(aggregated_results, significance_results, n_runs):
    """Generate comprehensive report of multi-run results"""
    report = []
    report.append("="*70)
    report.append("MULTI-RUN SIGNIFICANCE TESTING REPORT")
    report.append(f"Based on {n_runs} independent experimental runs")
    report.append("="*70)
    
    # Metric stability
    report.append("\n1. METRIC STABILITY ACROSS RUNS (mean ± std)")
    report.append("-"*50)
    
    for frontend in aggregated_results.keys():
        report.append(f"\n{frontend}:")
        for task in aggregated_results[frontend].keys():
            report.append(f"  {task}:")
            for metric in ['gap', 'wgs', 'accuracy']:
                if metric in aggregated_results[frontend][task]:
                    stats = aggregated_results[frontend][task][metric]
                    report.append(f"    {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Statistical significance
    report.append("\n2. STATISTICAL SIGNIFICANCE (vs. Mel baseline)")
    report.append("-"*50)
    
    for task in significance_results:
        report.append(f"\n{task.upper()}:")
        for metric in ['gap', 'wgs']:
            if metric in significance_results[task]:
                report.append(f"  {metric.upper()} metric:")
                for frontend in significance_results[task][metric]:
                    results = significance_results[task][metric][frontend]
                    report.append(f"    {frontend}:")
                    report.append(f"      Difference: {results['difference']:.4f}")
                    report.append(f"      p-value: {results['p_value']:.4f}")
                    report.append(f"      Cohen's d: {results['cohens_d']:.3f}")
                    report.append(f"      Significant (p<0.01): {results['significant_p001']}")
    
    # Interpretation
    report.append("\n3. INTERPRETATION")
    report.append("-"*50)
    report.append("- Results are ROBUST if std < 0.05 across runs")
    report.append("- Results are SIGNIFICANT if p < 0.01")
    report.append("- Effect sizes: |d| < 0.2 (small), 0.2-0.8 (medium), > 0.8 (large)")
    
    # Key findings
    report.append("\n4. KEY FINDINGS")
    report.append("-"*50)
    
    robust_findings = []
    for task in significance_results:
        for metric in significance_results[task]:
            for frontend in significance_results[task][metric]:
                if significance_results[task][metric][frontend]['significant_p001']:
                    robust_findings.append(
                        f"- {frontend} shows significant {metric} improvement for {task} "
                        f"(p={significance_results[task][metric][frontend]['p_value']:.4f})"
                    )
    
    if robust_findings:
        report.extend(robust_findings)
    else:
        report.append("- No statistically significant improvements found at p<0.01")
    
    # Save report
    report_text = "\n".join(report)
    
    os.makedirs('results/multi_run', exist_ok=True)
    with open('results/multi_run/multirun_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text


def plot_multirun_results(aggregated_results):
    """Create visualizations of multi-run results"""
    metrics = ['gap', 'wgs']
    tasks = ['music', 'scene', 'speech']
    
    fig, axes = plt.subplots(len(metrics), len(tasks), figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        for j, task in enumerate(tasks):
            ax = axes[i, j] if len(metrics) > 1 else axes[j]
            
            data_to_plot = []
            labels = []
            
            for frontend in aggregated_results.keys():
                if task in aggregated_results[frontend] and metric in aggregated_results[frontend][task]:
                    data_to_plot.append(aggregated_results[frontend][task][metric]['values'])
                    labels.append(frontend)
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # Color Mel differently as baseline
                for patch, label in zip(bp['boxes'], labels):
                    if label == 'Mel':
                        patch.set_facecolor('lightcoral')
                    else:
                        patch.set_facecolor('lightblue')
                
                ax.set_title(f'{task.capitalize()} - {metric.upper()}')
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
                ax.set_xticklabels(labels, rotation=45)
    
    plt.suptitle('Multi-Run Results Distribution')
    plt.tight_layout()
    
    os.makedirs('results/multi_run', exist_ok=True)
    plt.savefig('results/multi_run/multirun_distributions.png', dpi=150)
    plt.show()


# ============== MAIN EXECUTION ==============

def main(run_type='single'):
    """
    Main execution function
    Args:
        run_type: 'single' for single run, 'multi' for multi-run with significance testing
    """
    if run_type == 'multi':
        # Run multi-run evaluation for robust significance testing
        aggregated_results, significance_results = run_multi_run_evaluation(n_runs=5)
    else:
        # Original single-run evaluation
        print("="*60)
        print("AUDIO FRONT-END BIAS EVALUATION PIPELINE")
        print("Processing Real Audio Files Through Pre-trained Models")
        print("="*60)
        
        # Initialize data loader
        data_loader = AudioDataLoader()
        
        # Load datasets
        print("\n1. Loading Audio Data...")
        print("-"*40)
        
        tasks_data = {}
        
        # Load music data
        print("\nLoading Music Classification Data...")
        music_data = data_loader.load_music_data(max_samples=300)
        if music_data['audio']:
            tasks_data['music'] = music_data
            print(f"Loaded {len(music_data['audio'])} music samples")
        
        # Load scene data
        print("\nLoading Scene Classification Data...")
        scene_data = data_loader.load_scene_data(max_samples=100)
        if scene_data['audio']:
            tasks_data['scene'] = scene_data
            print(f"Loaded {len(scene_data['audio'])} scene samples")
        
        # Load speech data
        print("\nLoading Speech Recognition Data...")
        speech_data = data_loader.load_speech_data(max_samples=2000)
        if speech_data['audio']:
            tasks_data['speech'] = speech_data
            print(f"Loaded {len(speech_data['audio'])} speech samples")
        
        if not tasks_data:
            print("\nError: No data loaded. Please check your data directory structure.")
            return
        
        print("\n2. Initializing Audio Front-ends...")
        print("-"*40)
        
        frontends = {
            'Mel': MelFilterbank(n_mels=40),
            'ERB': ERBFilterbank(n_filters=32),
            'Bark': BarkFilterbank(n_filters=24),
            'CQT': CQTFrontend(n_bins=84),
            'Mel_PCEN': MelPCEN(n_mels=40)
        }
        
        print(f"Initialized {len(frontends)} front-ends: {list(frontends.keys())}")
        
        # Run evaluation
        print("\n3. Running Evaluation on Real Audio...")
        print("-"*40)
        
        evaluator = FairnessEvaluator()
        results = evaluator.run_full_evaluation(frontends, tasks_data)
        
        # Save results
        print("\n4. Saving Results...")
        print("-"*40)
        
        os.makedirs('results', exist_ok=True)
        
        # Save detailed results as JSON
        with open('results/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Results saved to results/evaluation_results.json")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        
        print("\nSummary of Results:")
        for frontend in frontends.keys():
            print(f"\n{frontend}:")
            for task in tasks_data.keys():
                if frontend in results and task in results[frontend]:
                    if 'overall_accuracy' in results[frontend][task]:
                        print(f"  {task}: Acc={results[frontend][task]['overall_accuracy']:.4f}, "
                              f"WGS={results[frontend][task]['wgs']:.4f}")
        
        print("\nAll outputs saved in results/ directory")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'multi':
        # Run multi-run evaluation
        main(run_type='multi')
    else:
        # Run single evaluation
        main(run_type='single')