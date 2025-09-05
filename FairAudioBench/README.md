# FairAudioBench: Cross-Cultural Bias Evaluation Framework

**Complete implementation of the ICASSP 2026 paper: "Cross-Cultural Bias in Mel-Scale Audio Front-Ends: Evidence from Speech and Music"**

A production-ready benchmark suite for **evaluating cross-cultural bias** in audio front-ends across speech, music, and environmental sound classification.

**Important: This is an evaluation-only framework** - we analyze bias in existing audio representations, not train new models.

## 🚀 Quick Start

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment and validate
python setup.py
```

### Data Pipeline
```bash
# Option A: Quick test with included samples (no download required)
python run_experiments.py --data_dir ../processed_data/samples --epochs 2

# Option B: Full dataset download (requires HF/Kaggle authentication)
python scripts/download_datasets.py --all --hf_token $HF_TOKEN
python scripts/preprocess_datasets.py --all
python scripts/validate_datasets.py
```

### Run Experiments
```bash
# Quick validation with samples (evaluation only)
python run_experiments.py --data_dir ../processed_data/samples --evaluation_only

# Full paper evaluation (requires full datasets)
python run_experiments.py --domains speech music --frontends mel erb leaf --evaluation_only

# Custom bias analysis
python run_experiments.py --domains speech --frontends mel erb bark --evaluation_only
```

### Analysis and Results
```bash
# Generate comprehensive analysis
python analyze_results.py

# Demonstrate paper results
python demonstrate_experiments.py

# Usage examples
python examples.py
```

## 🏗️ Implementation Overview

### Core Components

**Audio Front-Ends** (`src/frontends.py`)
- Mel-spectrogram, ERB-scale, Bark-scale (traditional)
- Constant-Q Transform (CQT) with logarithmic spacing
- LEAF (Learnable Audio Front-End) and SincNet (learnable filters)

**Neural Architectures** (`src/models.py`)
- Speech: CNN + BiLSTM for language identification  
- Music: ResNet-based architecture for genre classification
- Scenes: EfficientNet for acoustic scene classification

**Dataset Handling** (`src/datasets.py`)
- PyTorch Dataset classes for all domains
- Automated data loading with cultural group annotations
- Balanced sampling and train/val/test splitting

**Bias Evaluation** (`src/bias_evaluation.py`)
- Group gap analysis with statistical testing
- Fairness metrics (equalized odds, demographic parity)
- Comprehensive reporting with confidence intervals

### Experiment Pipeline

1. **Data Processing**: Automated download and preprocessing of all datasets
2. **Training**: Multi-domain experiments with all front-end combinations
3. **Evaluation**: Comprehensive bias analysis across cultural groups
4. **Analysis**: Statistical testing, visualization, and LaTeX report generation

### Key Features

- **Production-ready**: Robust error handling, logging, memory management
- **Modular design**: Easy extension with new front-ends, models, or domains
- **GPU acceleration**: Efficient training on modern hardware
- **Reproducible**: Fixed seeds, deterministic operations, comprehensive configuration
- **Statistical rigor**: Proper hypothesis testing with multiple comparison corrections

## 📊 Datasets and Domains

### 🗣️ Speech Recognition
**Dataset**: Mozilla CommonVoice (10 languages)
- **Languages**: English, German, Spanish, French, Italian, Dutch (non-tonal) + Vietnamese, Thai, Punjabi, Cantonese (tonal)
- **Task**: Language identification (10-class classification)
- **Cultural Bias**: Tonal vs. non-tonal language families
- **Files**: 17,751 processed audio files, 22kHz sampling rate

### 🎵 Music Classification  
**Datasets**: 6 musical traditions
- **Western**: GTZAN (10 genres), FMA (8 genres)
- **Traditional**: Carnatic (Indian classical), Turkish Makam, Hindustani (North Indian), Arab-Andalusian
- **Task**: Genre/style classification
- **Cultural Bias**: Western vs. traditional musical systems
- **Files**: 1,513 processed audio files, 30-second segments

### 🌆 Environmental Sound Classification
**Dataset**: TAU Urban Acoustic Scenes (10 cities)
- **Cities**: Amsterdam, Barcelona, Helsinki, Lisbon, London, Lyon, Milan, Paris, Prague, Vienna
- **Task**: Acoustic scene classification (10 scenes)
- **Cultural Bias**: Geographic and developmental differences
- **Files**: 600 processed audio files, 10-second segments

## 🧪 Experimental Design

### Bias Evaluation Framework
- **Group Gap**: Performance difference between cultural groups
- **Statistical Testing**: t-tests with Bonferroni correction, effect size calculation
- **Fairness Metrics**: Equalized odds gap, demographic parity difference
- **Cultural Grouping**: Linguistically/musically/geographically motivated groups

### Model Training
- **Architecture**: Domain-specific neural networks optimized for each task
- **Training**: Adam optimizer, learning rate scheduling, early stopping
- **Validation**: Stratified 70/15/15 train/val/test splits
- **Reproducibility**: Fixed random seeds, deterministic operations

## 💻 Usage Examples

### Basic Experiment
```python
# Run experiments for specific configuration
python run_experiments.py \
    --domains speech \
    --frontends mel erb leaf \
    --epochs 25 \
    --batch_size 32
```

### Custom Analysis
```python
from src.bias_evaluation import BiasEvaluator
from src.datasets import SpeechDataset

# Load results and evaluate bias
evaluator = BiasEvaluator()
dataset = SpeechDataset('processed_data/speech')

# Calculate bias metrics
bias_metrics = evaluator.calculate_group_gaps(predictions, ground_truth, groups)
print(f"Group Gap: {bias_metrics['group_gap']:.3f}")
```

### Adding New Front-Ends
```python
from src.frontends import AudioFrontend

class CustomFrontend(AudioFrontend):
    def __init__(self, sample_rate=22050):
        super().__init__()
        self.sample_rate = sample_rate
        # Custom initialization
    
    def forward(self, audio):
        # Custom front-end implementation
        return features
```

## 🔧 Configuration

Experiments are configured in `config/experiment_config.py`:

```python
EXPERIMENT_CONFIG = {
    'sample_rate': 22050,
    'n_mels': 128,
    'hop_length': 512,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

## 🧪 Testing

```bash
# Run comprehensive tests
python tests/test_implementation.py

# Quick validation tests
python tests/test_quick.py

# Specific component tests
python -m pytest tests/ -v
```

## 📈 Expected Results

The framework reproduces key findings from the paper:

- **Speech**: 12-30% performance gaps between tonal/non-tonal languages with mel-scale
- **Music**: 15-25% bias against traditional music with standard front-ends
- **Scenes**: Geographic bias patterns in urban sound classification
- **Mitigation**: ERB, LEAF, and SincNet show significant bias reduction

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/new-frontend`)
3. **Add tests** for new functionality
4. **Run tests** to ensure compatibility (`python tests/test_implementation.py`)
5. **Submit** a pull request with clear description

### Extension Points
- New audio front-ends in `src/frontends.py`
- Additional domains/datasets in `src/datasets.py` 
- Custom bias metrics in `src/bias_evaluation.py`
- Model architectures in `src/models.py`

## 📄 Citation

```bibtex
@inproceedings{fairaudiobench2026,
  title={Cross-Cultural Bias in Mel-Scale Audio Front-Ends: Evidence from Speech and Music},
  author={Chauhan, Shivam and Pundhir, Ajay},
  booktitle={ICASSP 2026 - IEEE International Conference on Acoustics, Speech and Signal Processing},
  year={2026},
  organization={IEEE}
}
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Mozilla CommonVoice** for multilingual speech data
- **GTZAN and FMA** communities for music datasets  
- **TAU Acoustic Scenes** team for environmental sound data
- **Cultural music archives** for traditional music collections
- **PyTorch** and **librosa** communities for excellent audio tools

---

**Questions?** Open an issue on GitHub or refer to the [main project README](../README.md) for research context.

**Paper Link**: [Coming upon ICASSP 2026 publication]
