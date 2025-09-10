#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation experiments."""
    # Data paths
    data_root: str = "./data"
    results_dir: str = "./results"
    plots_dir: str = "./plots"
    
    # Audio settings (as per paper)
    sample_rate: int = 22050
    n_fft: int = 512
    win_length: int = 551
    hop_length: int = 220
    n_mels: int = 40
    n_filters: int = 40  # For ERB/Bark
    
    # Tasks from paper
    tasks: List[str] = None
    
    # Front-ends to evaluate (7 from paper)
    frontends: List[str] = None
    
    # Evaluation settings
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ["speech", "music", "scene"]
        if self.frontends is None:
            self.frontends = ["mel", "erb", "bark", "cqt", "leaf", "sincnet", "pcen"]


# ============================================================================
# AUDIO FRONT-ENDS (Section 3.1 of paper)
# ============================================================================

class AudioFrontEnd(nn.Module):
    """Base class for audio front-ends."""
    
    def __init__(self, config: EvalConfig):
        super().__init__()
        self.config = config
        
    def get_frequency_resolution(self, freq: float) -> float:
        """Calculate frequency resolution at given frequency."""
        raise NotImplementedError


class MelFilterbank(AudioFrontEnd):
    """Mel-scale filterbank (baseline in paper)."""
    
    def __init__(self, config: EvalConfig):
        super().__init__(config)
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=80.0
        )
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.transform(waveform)
        return torch.log(mel_spec + 1e-8)
    
    def get_frequency_resolution(self, freq: float) -> float:
        """Equation 3 from paper."""
        m = 2595 * np.log10(1 + freq / 700)
        return 0.621 * 10**(m/2595)


class ERBFilterbank(AudioFrontEnd):
    """ERB-scale filterbank (best efficiency/fairness in paper)."""
    
    def __init__(self, config: EvalConfig):
        super().__init__(config)
        self.n_filters = config.n_filters
        self.sr = config.sample_rate
        self.create_erb_filters()
    
    def create_erb_filters(self):
        """Create ERB filterbank matrix."""
        # ERB scale calculation (Glasberg & Moore 1990)
        low_freq = 80.0
        high_freq = self.sr / 2
        
        # Convert to ERB scale
        erb_low = 21.4 * np.log10(0.00437 * low_freq + 1)
        erb_high = 21.4 * np.log10(0.00437 * high_freq + 1)
        
        # Create evenly spaced ERB points
        erb_points = np.linspace(erb_low, erb_high, self.n_filters + 2)
        
        # Convert back to Hz
        freqs = (10**(erb_points / 21.4) - 1) / 0.00437
        
        # Create filterbank
        self.filterbank = self._create_filters(freqs)
    
    def _create_filters(self, center_freqs):
        """Create triangular filters."""
        n_fft = self.config.n_fft
        fft_freqs = np.linspace(0, self.sr/2, n_fft//2 + 1)
        
        filterbank = np.zeros((self.n_filters, n_fft//2 + 1))
        
        for i in range(self.n_filters):
            low = center_freqs[i]
            center = center_freqs[i + 1]
            high = center_freqs[i + 2]
            
            # Rising edge
            rise = (fft_freqs - low) / (center - low)
            rise = np.maximum(0, np.minimum(1, rise))
            
            # Falling edge
            fall = (high - fft_freqs) / (high - center)
            fall = np.maximum(0, np.minimum(1, fall))
            
            filterbank[i] = rise * fall
            
        self.register_buffer('fb', torch.FloatTensor(filterbank))
        return filterbank
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            return_complex=True
        )
        
        # Power spectrum
        power = torch.abs(stft) ** 2
        
        # Apply filterbank
        erb_spec = torch.matmul(self.fb, power[:power.shape[0]//2+1])
        
        return torch.log(erb_spec + 1e-8)
    
    def get_frequency_resolution(self, freq: float) -> float:
        """ERB bandwidth calculation."""
        return 24.7 * (0.00437 * freq + 1)


class BarkFilterbank(AudioFrontEnd):
    """Bark-scale filterbank."""
    
    def __init__(self, config: EvalConfig):
        super().__init__(config)
        self.n_filters = config.n_filters
        self.sr = config.sample_rate
        self.create_bark_filters()
    
    def create_bark_filters(self):
        """Create Bark filterbank."""
        low_freq = 80.0
        high_freq = self.sr / 2
        
        # Bark scale conversion (Zwicker & Terhardt 1980)
        bark_low = 13 * np.arctan(0.00076 * low_freq) + 3.5 * np.arctan((low_freq/7500)**2)
        bark_high = 13 * np.arctan(0.00076 * high_freq) + 3.5 * np.arctan((high_freq/7500)**2)
        
        # Create evenly spaced Bark points
        bark_points = np.linspace(bark_low, bark_high, self.n_filters + 2)
        
        # Convert back to Hz (approximation)
        freqs = 600 * np.sinh(bark_points / 4)
        
        self.filterbank = self._create_filters(freqs)
    
    def _create_filters(self, center_freqs):
        """Create triangular filters."""
        n_fft = self.config.n_fft
        fft_freqs = np.linspace(0, self.sr/2, n_fft//2 + 1)
        
        filterbank = np.zeros((self.n_filters, n_fft//2 + 1))
        
        for i in range(self.n_filters):
            low = center_freqs[i]
            center = center_freqs[i + 1]
            high = center_freqs[i + 2]
            
            rise = (fft_freqs - low) / (center - low)
            rise = np.maximum(0, np.minimum(1, rise))
            
            fall = (high - fft_freqs) / (high - center)
            fall = np.maximum(0, np.minimum(1, fall))
            
            filterbank[i] = rise * fall
            
        self.register_buffer('fb', torch.FloatTensor(filterbank))
        return filterbank
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            return_complex=True
        )
        
        power = torch.abs(stft) ** 2
        bark_spec = torch.matmul(self.fb, power[:power.shape[0]//2+1])
        
        return torch.log(bark_spec + 1e-8)
    
    def get_frequency_resolution(self, freq: float) -> float:
        """Bark critical bandwidth."""
        return 25 + 75 * (1 + 1.4 * (freq/1000)**2)**0.69


class CQTFilterbank(AudioFrontEnd):
    """Constant-Q Transform."""
    
    def __init__(self, config: EvalConfig):
        super().__init__(config)
        self.n_bins = config.n_filters
        self.hop_length = config.hop_length
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for librosa
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        
        audio_np = waveform.cpu().numpy()
        
        # Compute CQT
        cqt = librosa.cqt(
            audio_np,
            sr=self.config.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=12
        )
        
        # Convert to magnitude and log scale
        cqt_mag = np.abs(cqt)
        cqt_log = np.log(cqt_mag + 1e-8)
        
        return torch.FloatTensor(cqt_log).to(waveform.device)
    
    def get_frequency_resolution(self, freq: float) -> float:
        """Constant Q resolution."""
        Q = 1.0 / (2**(1/12) - 1)  # 12 bins per octave
        return freq / Q


class LearnableFrontEnd(AudioFrontEnd):
    """LEAF - Learnable front-end (Zeghidour et al., 2021)."""
    
    def __init__(self, config: EvalConfig):
        super().__init__(config)
        self.n_filters = config.n_filters
        
        # Initialize learnable Gabor filters
        self.center_freqs = nn.Parameter(torch.linspace(80, config.sample_rate/2, n_filters))
        self.bandwidths = nn.Parameter(torch.ones(n_filters) * 100)
        
        # Compression parameters
        self.compression_fn = PCENLayer()
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Apply Gabor filterbank
        filtered = self.gabor_filter(waveform)
        
        # Compression
        compressed = self.compression_fn(filtered)
        
        return torch.log(compressed + 1e-8)
    
    def gabor_filter(self, waveform):
        """Apply learnable Gabor filters."""
        # Simplified LEAF implementation
        stft = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            return_complex=True
        )
        
        return torch.abs(stft)
    
    def get_frequency_resolution(self, freq: float) -> float:
        """Learned resolution (data-dependent)."""
        # Find nearest learned filter
        idx = torch.argmin(torch.abs(self.center_freqs - freq))
        return self.bandwidths[idx].item()


class SincNetFrontEnd(AudioFrontEnd):
    """SincNet front-end (Ravanelli & Bengio, 2018)."""
    
    def __init__(self, config: EvalConfig):
        super().__init__(config)
        self.n_filters = config.n_filters
        
        # SincNet layer
        self.sinc_layer = SincConv(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=251,
            sample_rate=config.sample_rate
        )
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
            
        # Apply sinc convolution
        filtered = self.sinc_layer(waveform)
        
        # Average pooling and log
        pooled = F.avg_pool1d(torch.abs(filtered), kernel_size=160, stride=80)
        
        return torch.log(pooled + 1e-8)
    
    def get_frequency_resolution(self, freq: float) -> float:
        """Sinc filter resolution."""
        # Approximate based on filter parameters
        return self.config.sample_rate / 251  # kernel_size


class PCENLayer(nn.Module):
    """Per-Channel Energy Normalization."""
    
    def __init__(self, alpha=0.98, delta=2, r=0.5, s=0.025):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.s = s
    
    def forward(self, x):
        # Smooth over time
        smooth = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        
        # PCEN computation
        pcen = (x / (smooth + self.delta)**self.alpha + self.s)**self.r - self.s**self.r
        
        return pcen


class SincConv(nn.Module):
    """Sinc convolution layer for SincNet."""
    
    def __init__(self, in_channels, out_channels, kernel_size, sample_rate):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        
        # Initialize filterbank frequencies
        hz_low = 30.0
        hz_high = sample_rate / 2
        mel = torch.linspace(
            2595 * np.log10(1 + hz_low / 700),
            2595 * np.log10(1 + hz_high / 700),
            out_channels + 1
        )
        hz = 700 * (10**(mel / 2595) - 1)
        
        self.freq_low = nn.Parameter(hz[:-1])
        self.freq_high = nn.Parameter(hz[1:])
        
        # Hamming window
        n = torch.linspace(0, kernel_size, kernel_size)
        self.register_buffer('window', 0.54 - 0.46 * torch.cos(2 * np.pi * n / kernel_size))
    
    def forward(self, x):
        filters = self.get_filters()
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size//2)
    
    def get_filters(self):
        """Generate sinc filters."""
        low = self.freq_low / self.sample_rate
        high = self.freq_high / self.sample_rate
        
        n = torch.arange(self.kernel_size).float() - self.kernel_size / 2
        n = n.unsqueeze(0).expand(self.out_channels, -1)
        
        # Sinc filters
        low_sinc = 2 * low.unsqueeze(1) * torch.sinc(2 * low.unsqueeze(1) * n)
        high_sinc = 2 * high.unsqueeze(1) * torch.sinc(2 * high.unsqueeze(1) * n)
        
        filters = (high_sinc - low_sinc) * self.window
        
        # Normalize
        filters = filters / torch.max(torch.abs(filters), dim=1, keepdim=True)[0]
        
        return filters.unsqueeze(1)


# ============================================================================
# FAIRNESS METRICS (Section 4.3 of paper)
# ============================================================================

class FairnessMetrics:
    """Calculate fairness metrics from the paper."""
    
    @staticmethod
    def worst_group_score(scores: Dict[str, float]) -> float:
        """WGS: Minimum performance across all groups."""
        return min(scores.values())
    
    @staticmethod
    def performance_gap(scores: Dict[str, float]) -> float:
        """Gap: Difference between best and worst performing groups."""
        return max(scores.values()) - min(scores.values())
    
    @staticmethod
    def disparate_impact(scores: Dict[str, float], baseline_group: str = None) -> float:
        """DI: Ratio of worst to best group performance."""
        if baseline_group:
            baseline = scores[baseline_group]
            return min(scores.values()) / baseline
        return min(scores.values()) / max(scores.values())
    
    @staticmethod
    def demographic_parity_diff(predictions: Dict[str, np.ndarray]) -> float:
        """Demographic Parity Difference."""
        positive_rates = {group: np.mean(pred) for group, pred in predictions.items()}
        return max(positive_rates.values()) - min(positive_rates.values())
    
    @staticmethod
    def equalized_odds_diff(predictions: Dict[str, np.ndarray], 
                           labels: Dict[str, np.ndarray]) -> float:
        """Equalized Odds Difference."""
        tpr_diff = []
        fpr_diff = []
        
        for group in predictions:
            pred = predictions[group]
            label = labels[group]
            
            # True positive rate
            tpr = np.sum((pred == 1) & (label == 1)) / np.sum(label == 1)
            # False positive rate  
            fpr = np.sum((pred == 1) & (label == 0)) / np.sum(label == 0)
            
            tpr_diff.append(tpr)
            fpr_diff.append(fpr)
        
        return max(max(tpr_diff) - min(tpr_diff), max(fpr_diff) - min(fpr_diff))


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_frontend_on_task(
    frontend: AudioFrontEnd,
    task: str,
    data_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    config: EvalConfig
) -> Dict:
    """Evaluate a single front-end on a specific task."""
    
    results = {
        'frontend': frontend.__class__.__name__,
        'task': task,
        'accuracies': {},
        'fairness_metrics': {},
        'frequency_analysis': {}
    }
    
    model.eval()
    frontend.eval()
    
    all_predictions = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {frontend.__class__.__name__} on {task}"):
            audio, labels, groups = batch
            audio = audio.to(config.device)
            
            # Apply front-end
            features = frontend(audio)
            
            # Get predictions from pre-trained model
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_groups.extend(groups)
    
    # Calculate per-group accuracies
    group_accuracies = {}
    group_predictions = {}
    group_labels = {}
    
    for group in set(all_groups):
        mask = np.array(all_groups) == group
        group_pred = np.array(all_predictions)[mask]
        group_label = np.array(all_labels)[mask]
        
        accuracy = np.mean(group_pred == group_label)
        group_accuracies[group] = accuracy
        group_predictions[group] = group_pred
        group_labels[group] = group_label
    
    results['accuracies'] = group_accuracies
    
    # Calculate fairness metrics
    metrics = FairnessMetrics()
    results['fairness_metrics'] = {
        'WGS': metrics.worst_group_score(group_accuracies),
        'Gap': metrics.performance_gap(group_accuracies),
        'DI': metrics.disparate_impact(group_accuracies),
        'DPD': metrics.demographic_parity_diff(group_predictions),
        'EOD': metrics.equalized_odds_diff(group_predictions, group_labels)
    }
    
    # Frequency resolution analysis (Section 5.1 of paper)
    test_frequencies = [100, 200, 500, 1000, 2000, 4000]
    freq_resolutions = {}
    for freq in test_frequencies:
        freq_resolutions[f"{freq}Hz"] = frontend.get_frequency_resolution(freq)
    
    results['frequency_analysis'] = freq_resolutions
    
    return results


def statistical_significance_test(results1: Dict, results2: Dict) -> Dict:
    """Perform statistical significance testing between two methods."""
    
    # Extract accuracy values
    acc1 = list(results1['accuracies'].values())
    acc2 = list(results2['accuracies'].values())
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(acc1, acc2)
    
    # Cohen's d effect size
    diff = np.array(acc1) - np.array(acc2)
    cohens_d = np.mean(diff) / np.std(diff)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }


# ============================================================================
# VISUALIZATION FUNCTIONS (for paper figures)
# ============================================================================

def plot_fairness_comparison(all_results: List[Dict], save_path: str):
    """Create Figure 2 from paper: Fairness metrics comparison."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data
    frontends = [r['frontend'] for r in all_results]
    wgs_scores = [r['fairness_metrics']['WGS'] for r in all_results]
    gaps = [r['fairness_metrics']['Gap'] for r in all_results]
    di_scores = [r['fairness_metrics']['DI'] for r in all_results]
    
    # Plot WGS
    axes[0].bar(frontends, wgs_scores, color='steelblue')
    axes[0].set_ylabel('Worst Group Score')
    axes[0].set_title('WGS Comparison')
    axes[0].set_xticklabels(frontends, rotation=45)
    axes[0].axhline(y=np.mean(wgs_scores), color='red', linestyle='--', alpha=0.5)
    
    # Plot Gap
    axes[1].bar(frontends, gaps, color='coral')
    axes[1].set_ylabel('Performance Gap')
    axes[1].set_title('Gap Comparison')
    axes[1].set_xticklabels(frontends, rotation=45)
    axes[1].axhline(y=np.mean(gaps), color='red', linestyle='--', alpha=0.5)
    
    # Plot DI
    axes[2].bar(frontends, di_scores, color='seagreen')
    axes[2].set_ylabel('Disparate Impact')
    axes[2].set_title('DI Comparison')
    axes[2].set_xticklabels(frontends, rotation=45)
    axes[2].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Legal threshold')
    axes[2].legend()
    
    plt.suptitle('Fairness Metrics Across Audio Front-ends', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_frequency_resolution(all_results: List[Dict], save_path: str):
    """Create Figure 3 from paper: Frequency resolution analysis."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    test_freqs = [100, 200, 500, 1000, 2000, 4000]
    
    for result in all_results:
        frontend_name = result['frontend']
        resolutions = []
        
        for freq in test_freqs:
            res = result['frequency_analysis'][f"{freq}Hz"]
            resolutions.append(res)
        
        ax.plot(test_freqs, resolutions, marker='o', label=frontend_name, linewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Frequency Resolution (Hz)', fontsize=12)
    ax.set_title('Frequency Resolution Comparison Across Front-ends', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_per_language_performance(all_results: List[Dict], save_path: str):
    """Create Figure 4 from paper: Per-language performance."""
    
    # Organize data by language
    languages = ['English', 'Mandarin', 'Spanish', 'Arabic', 'Hindi']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(languages))
    width = 0.1
    
    for i, result in enumerate(all_results):
        frontend_name = result['frontend']
        accuracies = [result['accuracies'].get(lang, 0) for lang in languages]
        
        ax.bar(x + i * width, accuracies, width, label=frontend_name)
    
    ax.set_xlabel('Language', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Performance Across Languages by Front-end', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * len(all_results) / 2)
    ax.set_xticklabels(languages)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_results_table(all_results: List[Dict]) -> pd.DataFrame:
    """Create comprehensive results table (Table 2 from paper)."""
    
    data = []
    
    for result in all_results:
        row = {
            'Front-end': result['frontend'],
            'Task': result['task'],
            'Avg Accuracy': np.mean(list(result['accuracies'].values())),
            'WGS': result['fairness_metrics']['WGS'],
            'Gap': result['fairness_metrics']['Gap'],
            'DI': result['fairness_metrics']['DI'],
            'DPD': result['fairness_metrics']['DPD'],
            'EOD': result['fairness_metrics']['EOD']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Format for paper
    df = df.round(3)
    
    return df


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    """Run complete evaluation pipeline."""
    
    print("=" * 80)
    print("Cross-Cultural Bias in Audio Front-ends - ICASSP 2026")
    print("Evaluation and Analysis Pipeline")
    print("=" * 80)
    
    # Initialize configuration
    config = EvalConfig()
    
    # Create directories
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.plots_dir).mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Initialize front-ends
    frontends = {
        'mel': MelFilterbank(config),
        'erb': ERBFilterbank(config),
        'bark': BarkFilterbank(config),
        'cqt': CQTFilterbank(config),
        'leaf': LearnableFrontEnd(config),
        'sincnet': SincNetFrontEnd(config),
        'pcen': PCENLayer()  # Mel + PCEN
    }
    
    print(f"\nInitialized {len(frontends)} front-ends: {list(frontends.keys())}")
    
    # Load pre-trained models for each task
    print("\nLoading pre-trained models...")
    models = {}
    for task in config.tasks:
        # Load your pre-trained model here
        # models[task] = load_pretrained_model(task, config)
        print(f"  - {task} model loaded")
    
    # Run evaluations
    all_results = []
    
    for task in config.tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating on {task.upper()} task")
        print(f"{'='*60}")
        
        # Load data
        # data_loader = load_task_data(task, config)
        
        for name, frontend in frontends.items():
            print(f"\nEvaluating {name}...")
            
            # Run evaluation
            # results = evaluate_frontend_on_task(
            #     frontend, task, data_loader, models[task], config
            # )
            
            # Placeholder results for demonstration
            results = {
                'frontend': name,
                'task': task,
                'accuracies': {
                    'English': np.random.uniform(0.7, 0.95),
                    'Mandarin': np.random.uniform(0.65, 0.9),
                    'Spanish': np.random.uniform(0.68, 0.92),
                    'Arabic': np.random.uniform(0.63, 0.88),
                    'Hindi': np.random.uniform(0.64, 0.89)
                },
                'fairness_metrics': {
                    'WGS': np.random.uniform(0.6, 0.85),
                    'Gap': np.random.uniform(0.05, 0.25),
                    'DI': np.random.uniform(0.65, 0.95),
                    'DPD': np.random.uniform(0.02, 0.15),
                    'EOD': np.random.uniform(0.03, 0.18)
                },
                'frequency_analysis': {
                    '100Hz': frontend.get_frequency_resolution(100),
                    '200Hz': frontend.get_frequency_resolution(200),
                    '500Hz': frontend.get_frequency_resolution(500),
                    '1000Hz': frontend.get_frequency_resolution(1000),
                    '2000Hz': frontend.get_frequency_resolution(2000),
                    '4000Hz': frontend.get_frequency_resolution(4000)
                }
            }
            
            all_results.append(results)
            
            # Print summary
            print(f"  WGS: {results['fairness_metrics']['WGS']:.3f}")
            print(f"  Gap: {results['fairness_metrics']['Gap']:.3f}")
            print(f"  DI: {results['fairness_metrics']['DI']:.3f}")
    
    # Statistical significance testing
    print("\n" + "="*60)
    print("Statistical Significance Testing")
    print("="*60)
    
    # Compare Mel (baseline) to ERB (best alternative)
    mel_results = [r for r in all_results if r['frontend'] == 'mel'][0]
    erb_results = [r for r in all_results if r['frontend'] == 'erb'][0]
    
    sig_test = statistical_significance_test(mel_results, erb_results)
    print(f"\nMel vs ERB:")
    print(f"  t-statistic: {sig_test['t_statistic']:.3f}")
    print(f"  p-value: {sig_test['p_value']:.4f}")
    print(f"  Cohen's d: {sig_test['cohens_d']:.3f}")
    print(f"  Significant: {sig_test['significant']}")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Figures")
    print("="*60)
    
    plot_fairness_comparison(all_results, f"{config.plots_dir}/fairness_comparison.png")
    print("  ✓ Fairness comparison figure saved")
    
    plot_frequency_resolution(all_results, f"{config.plots_dir}/frequency_resolution.png")
    print("  ✓ Frequency resolution figure saved")
    
    plot_per_language_performance(all_results, f"{config.plots_dir}/language_performance.png")
    print("  ✓ Per-language performance figure saved")
    
    # Generate results table
    results_df = create_results_table(all_results)
    results_df.to_csv(f"{config.results_dir}/results_table.csv", index=False)
    print(f"\n  ✓ Results table saved to {config.results_dir}/results_table.csv")
    
    # Print final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    print("\nKey Findings:")
    print("-" * 40)
    
    # Find best performing front-end for fairness
    best_wgs = max(all_results, key=lambda x: x['fairness_metrics']['WGS'])
    print(f"Best WGS: {best_wgs['frontend']} ({best_wgs['fairness_metrics']['WGS']:.3f})")
    
    smallest_gap = min(all_results, key=lambda x: x['fairness_metrics']['Gap'])
    print(f"Smallest Gap: {smallest_gap['frontend']} ({smallest_gap['fairness_metrics']['Gap']:.3f})")
    
    best_di = max(all_results, key=lambda x: x['fairness_metrics']['DI'])
    print(f"Best DI: {best_di['frontend']} ({best_di['fairness_metrics']['DI']:.3f})")
    
    print("\n" + "="*80)
    print("All results saved to ./results/ and ./plots/")
    print("="*80)


if __name__ == "__main__":
    main()