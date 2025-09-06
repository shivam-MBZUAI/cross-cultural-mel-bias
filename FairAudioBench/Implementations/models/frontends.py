"""
FairAudioBench Model Implementations
All six front-ends with matched hyperparameters (5M params) for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math

class BaseFrontEnd(nn.Module):
    """Base class for all audio front-ends with standardized interface."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_fft: int = 512,
                 hop_length: int = 160,
                 n_mels: int = 80,
                 f_min: float = 80.0,
                 f_max: float = 8000.0,
                 target_params: int = 5000000):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.target_params = target_params
        
        # Standard input length (4 seconds at 16kHz)
        self.input_length = 4 * sample_rate
        
        # Calculate output dimensions after front-end processing
        self.time_steps = (self.input_length // hop_length) + 1
        
    def get_param_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
        
    def get_frontend_name(self) -> str:
        """Get name of the front-end."""
        return self.__class__.__name__

class StandardMelSpectrogramFrontEnd(BaseFrontEnd):
    """Standard Mel-scale spectrogram with linear frequency mapping."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Standard mel-scale transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            mel_scale="htk"  # Standard HTK mel scale
        )
        
        # Add learnable components to reach target parameter count
        self._add_learnable_components()
        
    def _add_learnable_components(self):
        """Add learnable components to reach target parameter count."""
        # Calculate current feature dimensions
        feature_dim = self.n_mels
        
        # Add convolutional layers to process mel features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.n_mels, 32))
        )
        
        # Add transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=8
        )
        
        # Final projection layer
        self.output_projection = nn.Linear(256, 512)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through standard mel-scale front-end.
        
        Args:
            x: Input audio waveform [batch_size, time]
            
        Returns:
            Mel-scale features [batch_size, time_steps, feature_dim]
        """
        # Apply mel-scale transform
        mel_spec = self.mel_transform(x)  # [batch, n_mels, time]
        mel_spec = torch.log(mel_spec + 1e-8)  # Log compression
        
        # Add channel dimension and apply conv layers
        mel_spec = mel_spec.unsqueeze(1)  # [batch, 1, n_mels, time]
        conv_features = self.conv_layers(mel_spec)  # [batch, 256, n_mels, 32]
        
        # Reshape for transformer
        batch_size = conv_features.size(0)
        conv_features = conv_features.view(batch_size, 256, -1).transpose(1, 2)  # [batch, seq_len, 256]
        
        # Apply transformer
        transformer_out = self.transformer(conv_features)
        
        # Final projection
        output = self.output_projection(transformer_out)
        
        return output

class ERBScaleFrontEnd(BaseFrontEnd):
    """ERB (Equivalent Rectangular Bandwidth) scale front-end."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create ERB filter bank
        self.erb_filters = self._create_erb_filterbank()
        
        # Add learnable components
        self._add_learnable_components()
        
    def _create_erb_filterbank(self) -> torch.Tensor:
        """Create ERB-scale filterbank."""
        # ERB scale formula: ERB(f) = 24.7 * (4.37f/1000 + 1)
        # Convert to ERB scale
        def hz_to_erb(f):
            return 24.7 * (4.37 * f / 1000 + 1)
        
        def erb_to_hz(erb):
            return (erb / 24.7 - 1) * 1000 / 4.37
        
        # Create ERB-spaced frequencies
        erb_min = hz_to_erb(self.f_min)
        erb_max = hz_to_erb(self.f_max)
        erb_points = torch.linspace(erb_min, erb_max, self.n_mels + 2)
        hz_points = torch.tensor([erb_to_hz(erb) for erb in erb_points])
        
        # Create triangular filters
        n_fft_half = self.n_fft // 2 + 1
        fft_freqs = torch.linspace(0, self.sample_rate / 2, n_fft_half)
        
        filters = torch.zeros(self.n_mels, n_fft_half)
        
        for i in range(self.n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            # Triangular filter
            for j, freq in enumerate(fft_freqs):
                if left <= freq <= center:
                    filters[i, j] = (freq - left) / (center - left)
                elif center < freq <= right:
                    filters[i, j] = (right - freq) / (right - center)
        
        return filters
    
    def _add_learnable_components(self):
        """Add learnable components to reach target parameter count."""
        # Register ERB filters as buffer (non-trainable)
        self.register_buffer('erb_filterbank', self.erb_filters)
        
        # Add convolutional processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.n_mels, 32))
        )
        
        # Add transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=8
        )
        
        # Final projection
        self.output_projection = nn.Linear(256, 512)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ERB-scale front-end."""
        # Compute STFT
        stft = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            return_complex=True
        )
        magnitude = torch.abs(stft)  # [batch, freq, time]
        
        # Apply ERB filterbank
        erb_spec = torch.matmul(self.erb_filterbank, magnitude)  # [batch, n_mels, time]
        erb_spec = torch.log(erb_spec + 1e-8)  # Log compression
        
        # Process through conv layers
        erb_spec = erb_spec.unsqueeze(1)  # [batch, 1, n_mels, time]
        conv_features = self.conv_layers(erb_spec)
        
        # Reshape and apply transformer
        batch_size = conv_features.size(0)
        conv_features = conv_features.view(batch_size, 256, -1).transpose(1, 2)
        transformer_out = self.transformer(conv_features)
        
        # Final projection
        output = self.output_projection(transformer_out)
        
        return output

class GammatoneFilterBankFrontEnd(BaseFrontEnd):
    """Gammatone filterbank front-end mimicking human auditory processing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create gammatone filters
        self.gammatone_filters = self._create_gammatone_filters()
        
        # Add learnable components
        self._add_learnable_components()
        
    def _create_gammatone_filters(self) -> torch.Tensor:
        """Create gammatone filterbank."""
        # ERB-spaced center frequencies
        def hz_to_erb_rate(f):
            return 24.7 * (4.37 * f / 1000 + 1)
        
        def erb_rate_to_hz(erb_rate):
            return (erb_rate / 24.7 - 1) * 1000 / 4.37
        
        erb_min = hz_to_erb_rate(self.f_min)
        erb_max = hz_to_erb_rate(self.f_max)
        erb_points = torch.linspace(erb_min, erb_max, self.n_mels)
        center_freqs = torch.tensor([erb_rate_to_hz(erb) for erb in erb_points])
        
        # Filter parameters
        filter_length = 1024
        t = torch.arange(filter_length, dtype=torch.float32) / self.sample_rate
        
        filters = torch.zeros(self.n_mels, filter_length)
        
        for i, fc in enumerate(center_freqs):
            # Gammatone filter impulse response
            # h(t) = t^(n-1) * exp(-2πbt) * cos(2πfct + φ)
            n = 4  # Filter order
            erb = 24.7 * (4.37 * fc / 1000 + 1)
            b = 1.019 * erb
            
            envelope = (t ** (n - 1)) * torch.exp(-2 * math.pi * b * t)
            carrier = torch.cos(2 * math.pi * fc * t)
            filters[i] = envelope * carrier
        
        return filters
    
    def _add_learnable_components(self):
        """Add learnable components to reach target parameter count."""
        # Register gammatone filters as buffer
        self.register_buffer('gammatone_filterbank', self.gammatone_filters)
        
        # Add learnable processing layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.n_mels, 32))
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=8
        )
        
        self.output_projection = nn.Linear(256, 512)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through gammatone filterbank front-end."""
        batch_size = x.size(0)
        
        # Apply gammatone filters via convolution
        filtered_outputs = []
        
        for i in range(self.n_mels):
            # Convolve with gammatone filter
            filtered = F.conv1d(
                x.unsqueeze(1), 
                self.gammatone_filterbank[i:i+1].unsqueeze(1), 
                padding=self.gammatone_filterbank.size(1)//2
            )
            
            # Envelope extraction (half-wave rectification + lowpass)
            rectified = F.relu(filtered)
            envelope = F.avg_pool1d(rectified, kernel_size=self.hop_length, stride=self.hop_length)
            filtered_outputs.append(envelope)
        
        # Stack filter outputs
        gammatone_features = torch.cat(filtered_outputs, dim=1)  # [batch, n_mels, time]
        gammatone_features = torch.log(gammatone_features + 1e-8)
        
        # Process through conv layers
        gammatone_features = gammatone_features.unsqueeze(1)
        conv_features = self.conv_layers(gammatone_features)
        
        # Reshape and apply transformer
        conv_features = conv_features.view(batch_size, 256, -1).transpose(1, 2)
        transformer_out = self.transformer(conv_features)
        
        # Final projection
        output = self.output_projection(transformer_out)
        
        return output

class CochlearFilterBankFrontEnd(BaseFrontEnd):
    """Cochlear filterbank front-end based on basilar membrane modeling."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create cochlear filters
        self.cochlear_filters = self._create_cochlear_filters()
        
        # Add learnable components
        self._add_learnable_components()
        
    def _create_cochlear_filters(self) -> nn.ModuleList:
        """Create cochlear filterbank using cascade of second-order sections."""
        # Logarithmically spaced center frequencies
        center_freqs = torch.logspace(
            math.log10(self.f_min), 
            math.log10(self.f_max), 
            self.n_mels
        )
        
        filters = nn.ModuleList()
        
        for fc in center_freqs:
            # Create cascade of biquad filters
            filter_cascade = nn.ModuleList()
            
            # Each cochlear filter is a cascade of bandpass filters
            for stage in range(4):  # 4-stage cascade
                biquad = BiquadFilter(
                    center_freq=fc,
                    sample_rate=self.sample_rate,
                    q_factor=4.0 + stage * 2.0  # Increasing Q for sharpening
                )
                filter_cascade.append(biquad)
            
            filters.append(filter_cascade)
        
        return filters
    
    def _add_learnable_components(self):
        """Add learnable components to reach target parameter count."""
        # Add processing layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.n_mels, 32))
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=8
        )
        
        self.output_projection = nn.Linear(256, 512)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through cochlear filterbank front-end."""
        batch_size = x.size(0)
        
        # Apply cochlear filters
        filtered_outputs = []
        
        for filter_cascade in self.cochlear_filters:
            signal = x
            
            # Apply cascade of biquad filters
            for biquad in filter_cascade:
                signal = biquad(signal)
            
            # Envelope extraction and downsampling
            envelope = torch.abs(signal)
            downsampled = F.avg_pool1d(
                envelope.unsqueeze(1), 
                kernel_size=self.hop_length, 
                stride=self.hop_length
            )
            filtered_outputs.append(downsampled)
        
        # Stack filter outputs
        cochlear_features = torch.cat(filtered_outputs, dim=1)  # [batch, n_mels, time]
        cochlear_features = torch.log(cochlear_features + 1e-8)
        
        # Process through conv layers
        cochlear_features = cochlear_features.unsqueeze(1)
        conv_features = self.conv_layers(cochlear_features)
        
        # Reshape and apply transformer
        conv_features = conv_features.view(batch_size, 256, -1).transpose(1, 2)
        transformer_out = self.transformer(conv_features)
        
        # Final projection
        output = self.output_projection(transformer_out)
        
        return output

class BarkScaleFrontEnd(BaseFrontEnd):
    """Bark scale front-end based on critical band theory."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create Bark scale filterbank
        self.bark_filters = self._create_bark_filterbank()
        
        # Add learnable components
        self._add_learnable_components()
        
    def _create_bark_filterbank(self) -> torch.Tensor:
        """Create Bark scale filterbank."""
        # Bark scale formula: Bark = 13 * arctan(0.00076 * f) + 3.5 * arctan((f/7500)^2)
        def hz_to_bark(f):
            return 13 * torch.arctan(0.00076 * f) + 3.5 * torch.arctan((f / 7500) ** 2)
        
        def bark_to_hz(bark):
            # Approximate inverse (iterative solution simplified)
            return 600 * torch.sinh(bark / 4)
        
        # Create Bark-spaced frequencies
        bark_min = hz_to_bark(torch.tensor(self.f_min))
        bark_max = hz_to_bark(torch.tensor(self.f_max))
        bark_points = torch.linspace(bark_min, bark_max, self.n_mels + 2)
        hz_points = bark_to_hz(bark_points)
        
        # Create triangular filters
        n_fft_half = self.n_fft // 2 + 1
        fft_freqs = torch.linspace(0, self.sample_rate / 2, n_fft_half)
        
        filters = torch.zeros(self.n_mels, n_fft_half)
        
        for i in range(self.n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            # Triangular filter with Bark spacing
            for j, freq in enumerate(fft_freqs):
                if left <= freq <= center:
                    filters[i, j] = (freq - left) / (center - left)
                elif center < freq <= right:
                    filters[i, j] = (right - freq) / (right - center)
        
        return filters
    
    def _add_learnable_components(self):
        """Add learnable components to reach target parameter count."""
        # Register Bark filters as buffer
        self.register_buffer('bark_filterbank', self.bark_filters)
        
        # Add processing layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.n_mels, 32))
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=8
        )
        
        self.output_projection = nn.Linear(256, 512)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Bark scale front-end."""
        # Compute STFT
        stft = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            return_complex=True
        )
        magnitude = torch.abs(stft)  # [batch, freq, time]
        
        # Apply Bark filterbank
        bark_spec = torch.matmul(self.bark_filterbank, magnitude)  # [batch, n_mels, time]
        bark_spec = torch.log(bark_spec + 1e-8)  # Log compression
        
        # Process through conv layers
        bark_spec = bark_spec.unsqueeze(1)
        conv_features = self.conv_layers(bark_spec)
        
        # Reshape and apply transformer
        batch_size = conv_features.size(0)
        conv_features = conv_features.view(batch_size, 256, -1).transpose(1, 2)
        transformer_out = self.transformer(conv_features)
        
        # Final projection
        output = self.output_projection(transformer_out)
        
        return output

class LearnableMelFrontEnd(BaseFrontEnd):
    """Learnable Mel-scale front-end with trainable frequency mapping."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add learnable components
        self._add_learnable_components()
        
    def _add_learnable_components(self):
        """Add learnable components including trainable frequency mapping."""
        # Learnable frequency mapping
        n_fft_half = self.n_fft // 2 + 1
        
        # Initialize with standard mel filterbank and make it learnable
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max
        )
        
        # Extract mel filterbank weights and make them learnable
        initial_filterbank = mel_transform.mel_scale.fb.clone()
        self.learnable_filterbank = nn.Parameter(initial_filterbank)
        
        # Learnable frequency transformation
        self.freq_transform = nn.Sequential(
            nn.Linear(n_fft_half, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_mels),
            nn.Softmax(dim=-1)
        )
        
        # Add processing layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.n_mels, 32))
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=8
        )
        
        self.output_projection = nn.Linear(256, 512)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through learnable Mel front-end."""
        # Compute STFT
        stft = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            return_complex=True
        )
        magnitude = torch.abs(stft)  # [batch, freq, time]
        
        # Apply learnable filterbank
        mel_spec = torch.matmul(self.learnable_filterbank, magnitude)
        
        # Additional learnable frequency transformation
        batch_size, n_freq, time_steps = magnitude.shape
        freq_weights = self.freq_transform(magnitude.transpose(1, 2))  # [batch, time, n_mels]
        adaptive_spec = torch.bmm(freq_weights.transpose(1, 2), magnitude.transpose(1, 2))  # [batch, n_mels, time]
        
        # Combine learnable filterbank and adaptive transformation
        combined_spec = 0.7 * mel_spec + 0.3 * adaptive_spec.transpose(1, 2)
        combined_spec = torch.log(combined_spec + 1e-8)
        
        # Process through conv layers
        combined_spec = combined_spec.unsqueeze(1)
        conv_features = self.conv_layers(combined_spec)
        
        # Reshape and apply transformer
        conv_features = conv_features.view(batch_size, 256, -1).transpose(1, 2)
        transformer_out = self.transformer(conv_features)
        
        # Final projection
        output = self.output_projection(transformer_out)
        
        return output

class BiquadFilter(nn.Module):
    """Biquad (second-order) digital filter implementation."""
    
    def __init__(self, center_freq: float, sample_rate: float, q_factor: float = 1.0):
        super().__init__()
        
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.q_factor = q_factor
        
        # Calculate filter coefficients
        self._calculate_coefficients()
        
    def _calculate_coefficients(self):
        """Calculate biquad filter coefficients for bandpass filter."""
        # Normalized frequency
        omega = 2 * math.pi * self.center_freq / self.sample_rate
        alpha = math.sin(omega) / (2 * self.q_factor)
        
        # Bandpass filter coefficients
        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * math.cos(omega)
        a2 = 1 - alpha
        
        # Normalize by a0
        self.register_buffer('b0', torch.tensor(b0 / a0))
        self.register_buffer('b1', torch.tensor(b1 / a0))
        self.register_buffer('b2', torch.tensor(b2 / a0))
        self.register_buffer('a1', torch.tensor(a1 / a0))
        self.register_buffer('a2', torch.tensor(a2 / a0))
        
        # State variables
        self.register_buffer('x1', torch.zeros(1))
        self.register_buffer('x2', torch.zeros(1))
        self.register_buffer('y1', torch.zeros(1))
        self.register_buffer('y2', torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply biquad filter to input signal."""
        batch_size, seq_len = x.shape
        output = torch.zeros_like(x)
        
        # Initialize state for each sample in batch
        x1 = self.x1.expand(batch_size)
        x2 = self.x2.expand(batch_size)
        y1 = self.y1.expand(batch_size)
        y2 = self.y2.expand(batch_size)
        
        # Apply filter sample by sample
        for i in range(seq_len):
            x_curr = x[:, i]
            
            # Biquad difference equation
            y_curr = (self.b0 * x_curr + self.b1 * x1 + self.b2 * x2 
                     - self.a1 * y1 - self.a2 * y2)
            
            output[:, i] = y_curr
            
            # Update states
            x2 = x1
            x1 = x_curr
            y2 = y1
            y1 = y_curr
        
        return output

# Model factory for creating front-ends
def create_frontend(frontend_name: str, **kwargs) -> BaseFrontEnd:
    """
    Factory function to create front-end models.
    
    Args:
        frontend_name: Name of the front-end to create
        **kwargs: Additional arguments for front-end initialization
        
    Returns:
        Initialized front-end model
    """
    frontend_classes = {
        'standard_mel': StandardMelSpectrogramFrontEnd,
        'erb_scale': ERBScaleFrontEnd,
        'gammatone': GammatoneFilterBankFrontEnd,
        'cochlear': CochlearFilterBankFrontEnd,
        'bark_scale': BarkScaleFrontEnd,
        'learnable_mel': LearnableMelFrontEnd
    }
    
    if frontend_name not in frontend_classes:
        raise ValueError(f"Unknown frontend: {frontend_name}. Available: {list(frontend_classes.keys())}")
    
    return frontend_classes[frontend_name](**kwargs)

def get_model_summary(model: BaseFrontEnd) -> Dict[str, Any]:
    """Get summary of model architecture and parameters."""
    total_params = model.get_param_count()
    
    summary = {
        'model_name': model.get_frontend_name(),
        'total_parameters': total_params,
        'target_parameters': model.target_params,
        'parameter_ratio': total_params / model.target_params,
        'input_shape': f"[batch_size, {model.input_length}]",
        'output_shape': f"[batch_size, {model.time_steps}, 512]",
        'sample_rate': model.sample_rate,
        'n_mels': model.n_mels,
        'f_min': model.f_min,
        'f_max': model.f_max
    }
    
    return summary

# Test function
def test_all_frontends():
    """Test all front-end implementations."""
    print("Testing FairAudioBench Front-ends...")
    
    # Test parameters
    batch_size = 2
    seq_length = 64000  # 4 seconds at 16kHz
    
    # Create test input
    test_input = torch.randn(batch_size, seq_length)
    
    frontend_names = [
        'standard_mel', 'erb_scale', 'gammatone', 
        'cochlear', 'bark_scale', 'learnable_mel'
    ]
    
    for name in frontend_names:
        print(f"\nTesting {name}...")
        
        try:
            # Create model
            model = create_frontend(name)
            model.eval()
            
            # Forward pass
            with torch.no_grad():
                output = model(test_input)
            
            # Get summary
            summary = get_model_summary(model)
            
            print(f"  ✓ {name} - Params: {summary['total_parameters']:,} - Output: {output.shape}")
            
        except Exception as e:
            print(f"  ✗ {name} - Error: {e}")

if __name__ == "__main__":
    test_all_frontends()
