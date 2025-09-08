#!/usr/bin/env python3

"""
Audio Front-End Implementations for Cross-Cultural Bias Research
ICASSP 2026 Paper

This module implements various audio front-ends for comparative analysis:
- Traditional: Mel-scale, ERB, Bark scale
- Perceptual: Constant-Q Transform (CQT)
- Learnable: LEAF, SincNet

Author: Shivam Chauhan, Ajay Pundhir
Organization: Presight AI, Abu Dhabi, UAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
import logging

logger = logging.getLogger(__name__)

class AudioFrontEnd(nn.Module):
    """Base class for all audio front-ends."""
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = 128  # Standard for comparison
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features from audio tensor."""
        raise NotImplementedError
        
    def get_feature_dim(self) -> int:
        """Return the feature dimension."""
        return self.n_mels

class MelScaleFrontEnd(AudioFrontEnd):
    """
    Traditional Mel-scale front-end (baseline for cultural bias analysis).
    Based on 1940s Western psychoacoustic studies.
    """
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048, 
                 hop_length: int = 512, n_mels: int = 128, f_min: float = 0.0, 
                 f_max: Optional[float] = None):
        super().__init__(sample_rate, n_fft, hop_length)
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        
        # Create mel filter bank
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
            power=2.0,
            normalized=True
        )
        
        # Log compression
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel-scale features."""
        # audio shape: (batch_size, time) or (time,)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        mel_spec = self.mel_transform(audio)
        log_mel = self.amplitude_to_db(mel_spec)
        
        # Normalize to [-1, 1] range for stability
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        return log_mel  # Shape: (batch_size, n_mels, time_frames)

class ERBScaleFrontEnd(AudioFrontEnd):
    """
    Equivalent Rectangular Bandwidth (ERB) scale front-end.
    More perceptually accurate than mel-scale across cultures.
    """
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048,
                 hop_length: int = 512, n_filters: int = 128, 
                 f_min: float = 50.0, f_max: Optional[float] = None):
        super().__init__(sample_rate, n_fft, hop_length)
        self.n_mels = n_filters  # Keep same interface
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        
        # Create ERB filter bank
        self.erb_filters = self._create_erb_filterbank()
        self.register_buffer('filters', self.erb_filters)
        
        # STFT for computing spectrograms
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
            normalized=True
        )
        
    def _hz_to_erb(self, freq_hz: float) -> float:
        """Convert frequency in Hz to ERB scale."""
        return 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))
    
    def _erb_to_hz(self, erb: float) -> float:
        """Convert ERB scale to frequency in Hz."""
        return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)
    
    def _create_erb_filterbank(self) -> torch.Tensor:
        """Create ERB-spaced triangular filter bank."""
        # ERB scale frequency points
        erb_min = self._hz_to_erb(self.f_min)
        erb_max = self._hz_to_erb(self.f_max)
        erb_points = np.linspace(erb_min, erb_max, self.n_mels + 2)
        hz_points = np.array([self._erb_to_hz(erb) for erb in erb_points])
        
        # Convert to FFT bin indices
        fft_freqs = np.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
        bin_indices = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create triangular filters
        filters = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        
        for i in range(self.n_mels):
            left = bin_indices[i]
            center = bin_indices[i + 1]
            right = bin_indices[i + 2]
            
            # Left slope
            for j in range(left, center):
                if center != left:
                    filters[i, j] = (j - left) / (center - left)
            
            # Right slope
            for j in range(center, right):
                if right != center:
                    filters[i, j] = (right - j) / (right - center)
        
        return torch.FloatTensor(filters)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract ERB-scale features."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Compute power spectrogram
        power_spec = self.stft(audio)  # (batch_size, freq_bins, time_frames)
        
        # Apply ERB filters
        erb_spec = torch.matmul(self.filters, power_spec)
        
        # Log compression
        log_erb = torch.log(erb_spec + 1e-8)
        
        # Normalize
        log_erb = (log_erb - log_erb.mean()) / (log_erb.std() + 1e-8)
        
        return log_erb

class BarkScaleFrontEnd(AudioFrontEnd):
    """
    Bark scale front-end based on critical bands.
    Alternative perceptual scale for cross-cultural comparison.
    """
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048,
                 hop_length: int = 512, n_filters: int = 128):
        super().__init__(sample_rate, n_fft, hop_length)
        self.n_mels = n_filters
        
        # Create Bark filter bank
        self.bark_filters = self._create_bark_filterbank()
        self.register_buffer('filters', self.bark_filters)
        
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
            normalized=True
        )
    
    def _hz_to_bark(self, freq_hz: float) -> float:
        """Convert frequency in Hz to Bark scale."""
        return 6 * np.arcsinh(freq_hz / 600)
    
    def _bark_to_hz(self, bark: float) -> float:
        """Convert Bark scale to frequency in Hz."""
        return 600 * np.sinh(bark / 6)
    
    def _create_bark_filterbank(self) -> torch.Tensor:
        """Create Bark-spaced triangular filter bank."""
        f_max = self.sample_rate // 2
        bark_min = self._hz_to_bark(50)  # Start from 50 Hz
        bark_max = self._hz_to_bark(f_max)
        
        bark_points = np.linspace(bark_min, bark_max, self.n_mels + 2)
        hz_points = np.array([self._bark_to_hz(bark) for bark in bark_points])
        
        # Convert to FFT bin indices
        bin_indices = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create triangular filters
        filters = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        
        for i in range(self.n_mels):
            left = bin_indices[i]
            center = bin_indices[i + 1]
            right = bin_indices[i + 2]
            
            for j in range(left, center):
                if center != left:
                    filters[i, j] = (j - left) / (center - left)
            
            for j in range(center, right):
                if right != center:
                    filters[i, j] = (right - j) / (right - center)
        
        return torch.FloatTensor(filters)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract Bark-scale features."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        power_spec = self.stft(audio)
        bark_spec = torch.matmul(self.filters, power_spec)
        log_bark = torch.log(bark_spec + 1e-8)
        log_bark = (log_bark - log_bark.mean()) / (log_bark.std() + 1e-8)
        
        return log_bark

class CQTFrontEnd(AudioFrontEnd):
    """
    Constant-Q Transform (CQT) front-end.
    Logarithmic frequency spacing may be more culturally neutral.
    """
    
    def __init__(self, sample_rate: int = 22050, n_bins: int = 128,
                 hop_length: int = 512, f_min: float = 32.7):  # C1 note
        super().__init__(sample_rate, hop_length=hop_length)
        self.n_mels = n_bins
        self.f_min = f_min
        
        # Use librosa for CQT computation
        self.cqt_transform = torchaudio.transforms.ConstantQ(
            sample_rate=sample_rate,
            n_bins=n_bins,
            hop_length=hop_length,
            fmin=f_min
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract CQT features."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Compute CQT
        cqt = self.cqt_transform(audio)
        
        # Convert to magnitude and log scale
        magnitude_cqt = torch.abs(cqt)
        log_cqt = torch.log(magnitude_cqt + 1e-8)
        
        # Normalize
        log_cqt = (log_cqt - log_cqt.mean()) / (log_cqt.std() + 1e-8)
        
        return log_cqt

class LEAFFrontEnd(AudioFrontEnd):
    """
    LEAF (Learnable Frontend) implementation.
    Data-driven approach to learn optimal audio representations.
    """
    
    def __init__(self, sample_rate: int = 22050, n_filters: int = 128,
                 window_len: int = 401, window_stride: int = 160):
        super().__init__(sample_rate)
        self.n_mels = n_filters
        self.window_len = window_len
        self.window_stride = window_stride
        
        # Learnable Gabor filters
        self.gabor_params_real = nn.Parameter(torch.randn(n_filters, window_len))
        self.gabor_params_imag = nn.Parameter(torch.randn(n_filters, window_len))
        
        # Learnable pooling
        self.pooling = nn.Parameter(torch.ones(n_filters))
        
        # Initialize filters
        self._init_gabor_filters()
    
    def _init_gabor_filters(self):
        """Initialize Gabor filters with reasonable values."""
        with torch.no_grad():
            # Initialize with mel-scale-like frequencies
            mel_frequencies = torch.linspace(0, self.sample_rate // 2, self.n_mels)
            
            for i, freq in enumerate(mel_frequencies):
                # Create Gabor filter
                t = torch.arange(self.window_len) - self.window_len // 2
                t = t.float()
                
                # Gaussian envelope
                sigma = self.window_len / 8
                envelope = torch.exp(-0.5 * (t / sigma) ** 2)
                
                # Sinusoidal carrier
                omega = 2 * math.pi * freq / self.sample_rate
                real_part = envelope * torch.cos(omega * t)
                imag_part = envelope * torch.sin(omega * t)
                
                self.gabor_params_real[i] = real_part
                self.gabor_params_imag[i] = imag_part
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract learnable features."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension
        
        batch_size = audio.shape[0]
        
        # Unfold audio for convolution
        unfolded = F.unfold(
            audio.unsqueeze(1).unsqueeze(1),  # (batch, 1, 1, time)
            kernel_size=(1, self.window_len),
            stride=(1, self.window_stride)
        )  # (batch, window_len, n_frames)
        
        # Apply Gabor filters
        real_responses = torch.matmul(self.gabor_params_real, unfolded)
        imag_responses = torch.matmul(self.gabor_params_imag, unfolded)
        
        # Compute magnitude
        magnitude = torch.sqrt(real_responses ** 2 + imag_responses ** 2)
        
        # Learnable pooling (compression)
        pooled = magnitude ** self.pooling.unsqueeze(-1)
        
        # Log compression
        log_features = torch.log(pooled + 1e-8)
        
        # Normalize
        log_features = (log_features - log_features.mean()) / (log_features.std() + 1e-8)
        
        return log_features

class SincNetFrontEnd(AudioFrontEnd):
    """
    SincNet front-end implementation.
    Interpretable 1D convolutional filters based on sinc functions.
    """
    
    def __init__(self, sample_rate: int = 22050, n_filters: int = 128,
                 kernel_size: int = 251, stride: int = 160):
        super().__init__(sample_rate)
        self.n_mels = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Learnable parameters for sinc filters
        self.f1 = nn.Parameter(torch.rand(n_filters) * sample_rate / 4)  # Low cutoff
        self.f2 = nn.Parameter(torch.rand(n_filters) * sample_rate / 4)  # High cutoff
        
        # Hamming window
        self.register_buffer('window', torch.hamming_window(kernel_size))
        
        # Initialize cutoff frequencies
        self._init_cutoff_freqs()
    
    def _init_cutoff_freqs(self):
        """Initialize cutoff frequencies in mel-scale order."""
        with torch.no_grad():
            mel_scale = torch.linspace(0, 2595 * np.log10(1 + (self.sample_rate / 2) / 700), self.n_mels)
            hz_scale = 700 * (10 ** (mel_scale / 2595) - 1)
            
            # Set f1 and f2 to create bandpass filters
            for i in range(self.n_mels):
                if i == 0:
                    self.f1[i] = 0
                    self.f2[i] = hz_scale[i]
                else:
                    self.f1[i] = hz_scale[i-1] 
                    self.f2[i] = hz_scale[i]
    
    def _create_sinc_filters(self) -> torch.Tensor:
        """Create sinc-based bandpass filters."""
        # Ensure f2 > f1
        f1 = torch.abs(self.f1)
        f2 = torch.abs(self.f2)
        f1, f2 = torch.min(f1, f2), torch.max(f1, f2)
        
        # Time axis
        t = torch.arange(-(self.kernel_size // 2), (self.kernel_size // 2) + 1, dtype=torch.float32)
        t = t.to(self.f1.device)
        
        # Create sinc filters
        filters = []
        for i in range(self.n_mels):
            # Normalized cutoff frequencies
            f1_norm = f1[i] / self.sample_rate
            f2_norm = f2[i] / self.sample_rate
            
            # Sinc function
            sinc_filter = torch.zeros_like(t)
            
            # Handle t=0 case separately
            mask = (t != 0)
            sinc_filter[mask] = (torch.sin(2 * math.pi * f2_norm * t[mask]) - 
                                torch.sin(2 * math.pi * f1_norm * t[mask])) / (math.pi * t[mask])
            
            # t=0 case
            sinc_filter[t == 0] = 2 * (f2_norm - f1_norm)
            
            # Apply window
            windowed_filter = sinc_filter * self.window
            
            # Normalize
            windowed_filter = windowed_filter / torch.sum(torch.abs(windowed_filter))
            
            filters.append(windowed_filter)
        
        return torch.stack(filters)  # (n_filters, kernel_size)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract SincNet features."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension
        
        # Create sinc filters
        filters = self._create_sinc_filters()
        
        # Apply 1D convolution
        features = F.conv1d(
            audio.unsqueeze(1),  # (batch, 1, time)
            filters.unsqueeze(1),  # (n_filters, 1, kernel_size)
            stride=self.stride,
            padding=self.kernel_size // 2
        )  # (batch, n_filters, time_frames)
        
        # Apply ReLU activation
        features = F.relu(features)
        
        # Log compression
        log_features = torch.log(features + 1e-8)
        
        # Normalize
        log_features = (log_features - log_features.mean()) / (log_features.std() + 1e-8)
        
        return log_features

# Factory function for creating front-ends
def create_frontend(frontend_type: str, **kwargs) -> AudioFrontEnd:
    """
    Factory function to create audio front-ends.
    
    Args:
        frontend_type: One of 'mel', 'erb', 'bark', 'cqt', 'leaf', 'sincnet'
        **kwargs: Additional arguments passed to the front-end constructor
    
    Returns:
        AudioFrontEnd: Initialized front-end
    """
    frontends = {
        'mel': MelScaleFrontEnd,
        'erb': ERBScaleFrontEnd,
        'bark': BarkScaleFrontEnd,
        'cqt': CQTFrontEnd,
        'leaf': LEAFFrontEnd,
        'sincnet': SincNetFrontEnd
    }
    
    if frontend_type not in frontends:
        raise ValueError(f"Unknown frontend type: {frontend_type}. "
                        f"Available: {list(frontends.keys())}")
    
    return frontends[frontend_type](**kwargs)

if __name__ == "__main__":
    # Test all front-ends
    sample_rate = 22050
    audio_length = 3 * sample_rate  # 3 seconds
    batch_size = 2
    
    # Create dummy audio
    audio = torch.randn(batch_size, audio_length)
    
    frontends = ['mel', 'erb', 'bark', 'cqt', 'leaf', 'sincnet']
    
    print("Testing audio front-ends:")
    print(f"Input audio shape: {audio.shape}")
    print("-" * 50)
    
    for frontend_name in frontends:
        try:
            frontend = create_frontend(frontend_name, sample_rate=sample_rate)
            features = frontend(audio)
            print(f"{frontend_name.upper():>8}: {features.shape} - ✓")
        except Exception as e:
            print(f"{frontend_name.upper():>8}: Error - {e}")
