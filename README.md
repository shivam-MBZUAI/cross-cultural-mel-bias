# Cross-Cultural Bias in Mel-Scale Audio Front-Ends

[![Paper](https://img.shields.io/badge/Paper-ICASSP%202026-blue)](https://arxiv.org/abs/your-paper-id)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Evidence from Speech and Music**  
*Shivam Chauhan, Ajay Pundhir*  
*Presight AI, Abu Dhabi, UAE*

## Overview

This repository contains the implementation for our ICASSP 2026 paper investigating cross-cultural bias in mel-scale audio representations across:

- **Speech**: 11 languages (5 tonal, 6 non-tonal) from CommonVoice
- **Music**: 8 traditions (Western and non-Western) 
- **Acoustic Scenes**: 10 European cities from TAU Urban dataset

## Quick Start

### 1. Setup

```bash
git clone https://github.com/shivam-MBZUAI/cross-cultural-mel-bias.git
cd cross-cultural-mel-bias
pip install -r requirements.txt
```

### 2. Download Datasets

The `download_datasets.py` script downloads **raw datasets only** - no preprocessing or balancing is performed. Use `preprocess_datasets.py` for data preparation after downloading.

#### Authentication Setup

For CommonVoice datasets, you need a HuggingFace account and token:

```bash
# Method 1: Use huggingface-cli (recommended)
pip install huggingface_hub
huggingface-cli login
# Then use download commands without --hf_token

# Method 2: Export token as environment variable
export HUGGINGFACE_HUB_TOKEN="your_token_here"
python download_datasets.py --all --hf_token $HUGGINGFACE_HUB_TOKEN

# Method 3: Pass token directly
python download_datasets.py --all --hf_token hf_your_token_here
```

**Note**: If you get rate limiting errors (429), wait a few minutes before retrying.

#### Available Options

```bash
# List all available datasets and languages (205 languages available for CommonVoice)
python download_datasets.py --list

# Download all target datasets (CommonVoice + Music + Scenes)
python download_datasets.py --all --hf_token $HUGGINGFACE_HUB_TOKEN

# Download CommonVoice for all target languages (5 tonal + 6 non-tonal = 11 total)
python download_datasets.py --dataset commonvoice --lang all_target --hf_token $HUGGINGFACE_HUB_TOKEN

# Download CommonVoice for specific languages
python download_datasets.py --dataset commonvoice --lang en --hf_token $HUGGINGFACE_HUB_TOKEN
python download_datasets.py --dataset commonvoice --lang vi --hf_token $HUGGINGFACE_HUB_TOKEN
python download_datasets.py --dataset commonvoice --lang zh-CN --hf_token $HUGGINGFACE_HUB_TOKEN

# Download specific music datasets
python download_datasets.py --dataset gtzan
python download_datasets.py --dataset fma_small
python download_datasets.py --dataset carnatic
python download_datasets.py --dataset turkish_makam
python download_datasets.py --dataset hindustani
python download_datasets.py --dataset arab_andalusian

# Download scene datasets
python download_datasets.py --dataset tau_urban

# Specify custom output directory
python download_datasets.py --all --output_dir /path/to/custom/data --hf_token $HUGGINGFACE_HUB_TOKEN

# Get help
python download_datasets.py --help
```

#### Command Line Arguments

- `--dataset DATASET`: Download specific dataset  
  **Choices**: `commonvoice`, `gtzan`, `fma_small`, `carnatic`, `turkish_makam`, `hindustani`, `arab_andalusian`, `tau_urban`
- `--lang LANGUAGE`: Language for CommonVoice dataset  
  **Options**: Any of 205+ language codes (e.g., `en`, `vi`, `th`, `zh-CN`, `pa-IN`, `yue`, `es`, `de`, `fr`, `it`, `nl`) or `all_target` for all 11 target languages
- `--all`: Download all available datasets (CommonVoice + Music + Scenes)
- `--hf_token TOKEN`: HuggingFace Hub token for authentication (required for CommonVoice)
- `--list`: List all available datasets, languages, and usage examples
- `--output_dir PATH`: Custom output directory for datasets (default: `./data`)
- `--help`: Show detailed help message with all options

#### Dataset Details

**SPEECH DATASETS:**
- `commonvoice` - Mozilla Common Voice v17.0 multilingual speech (205 languages available)

**MUSIC DATASETS:**
- `gtzan` - GTZAN Genre Classification Dataset (1000 audio tracks, 10 genres)
- `fma_small` - Free Music Archive Small subset (8,000 tracks)
- `carnatic` - Carnatic music dataset from CompMusic project
- `turkish_makam` - Turkish Makam music dataset
- `hindustani` - Hindustani classical music dataset
- `arab_andalusian` - Arab Andalusian classical music dataset

**SCENE DATASETS:**
- `tau_urban` - TAU Urban Acoustic Scenes 2020 dataset

#### Target Languages (ICASSP 2026 Paper)

**Tonal Languages (5)**: Vietnamese (vi), Thai (th), Mandarin Chinese (zh-CN), Punjabi (pa-IN), Cantonese (yue)  
**Non-Tonal Languages (6)**: English (en), Spanish (es), German (de), French (fr), Italian (it), Dutch (nl)

#### Troubleshooting

**Authentication Issues:**
```bash
# If you get "429 Client Error: Too Many Requests"
# Wait 5-10 minutes and retry

# If authentication fails, try logging in first:
huggingface-cli login
# Then run without --hf_token:
python download_datasets.py --dataset commonvoice --lang en

# If still failing, check your token permissions:
# Visit https://huggingface.co/settings/tokens
# Ensure your token has "Read access to contents of all public gated repos"
```

**Storage Requirements:**
- CommonVoice (all 11 target languages): ~50GB
- Music datasets: ~30GB  
- Scene datasets: ~20GB
- **Total**: ~100GB free space recommended

### 3. Preprocess for Balanced Evaluation

The `preprocess_datasets.py` script creates **balanced evaluation datasets only** (no training splits). This is a bias evaluation study, not a training study.

**Dataset Specifications:**
- **Speech**: 2,000 samples per language (11 languages: 5 tonal, 6 non-tonal)
- **Music**: 300 samples per tradition (6 traditions: 2 Western, 4 non-Western)  
- **Scenes**: 100 samples of mixed urban acoustic scenes from TAU Urban dataset

#### Available Options

```bash
# Show all available options and help
python preprocess_datasets.py --help

# Process all domains (speech + music + scenes)
python preprocess_datasets.py --all

# Process only speech domain (all target languages)
python preprocess_datasets.py --domain speech

# Process speech for specific languages
python preprocess_datasets.py --domain speech --languages en vi zh-CN
python preprocess_datasets.py --domain speech --languages vi th zh-CN pa-IN yue  # Tonal languages only
python preprocess_datasets.py --domain speech --languages en es de fr it nl     # Non-tonal languages only

# Process only music domain
python preprocess_datasets.py --domain music

# Process music for specific traditions
python preprocess_datasets.py --domain music --traditions gtzan carnatic
python preprocess_datasets.py --domain music --traditions gtzan fma_small              # Western traditions
python preprocess_datasets.py --domain music --traditions carnatic hindustani turkish_makam arab_andalusian  # Non-Western

# Process only acoustic scenes
python preprocess_datasets.py --domain scenes

# Use custom random seed for reproducibility
python preprocess_datasets.py --all --seed 123

# Force reprocessing (overwrite existing processed data)
python preprocess_datasets.py --all --force
```

**Output Structure:**
```
processed_data/
├── speech/
│   ├── en/                    # English (non-tonal)
│   │   ├── en_eval_0000.wav   # 2000 evaluation samples
│   │   ├── ...
│   │   ├── en_eval_1999.wav
│   │   ├── metadata.csv       # Sample metadata
│   │   └── summary.json       # Processing summary
│   ├── vi/                    # Vietnamese (tonal)
│   └── ...                    # Other languages
├── music/
│   ├── gtzan/                 # Western tradition
│   │   ├── gtzan_eval_0000.wav # 300 evaluation samples
│   │   └── ...
│   └── ...                    # Other traditions
├── scenes/
│   └── urban_mixed/           # Mixed urban acoustic scenes
│       ├── urban_mixed_eval_0000.wav # 100 evaluation samples
│       ├── ...
│       ├── urban_mixed_eval_0099.wav
│       ├── metadata.csv       # Sample metadata
│       └── summary.json       # Processing summary
└── dataset_summary.json       # Overall summary
```

### 4. Run Bias Evaluation

```bash
# Reproduce paper results across all front-ends
python run_experiments.py --config config.json

# Quick test with subset
python run_experiments.py --quick --frontends mel erb leaf
```

### 3. Run Experiments

The repository implements the complete experimental framework from the ICASSP 2026 paper.

#### Quick Test (Recommended First)

```bash
# Test all components with simulated data
python test_experiments.py

# Test specific components
python test_experiments.py --component frontend  # Test audio front-ends
python test_experiments.py --component bias      # Test bias evaluation
python test_experiments.py --component viz       # Test visualizations
```

#### Complete Experimental Pipeline

```bash
# Run full analysis (requires processed data)
python run_experiments.py --all

# Quick validation run
python run_experiments.py --validate

# Custom configuration
python run_experiments.py --all --config custom_config.json --results_dir ./my_results
```

#### Available Experiments

**1. Audio Front-End Analysis**
- Feature extraction across 7 front-ends (Mel, ERB, Bark, CQT, LEAF, SincNet, Mel+PCEN)
- Cultural bias evaluation using performance gaps and statistical tests
- Feature space visualization and analysis

**2. Cross-Cultural Bias Evaluation**
- **Speech**: Tonal vs Non-Tonal language bias analysis
- **Music**: Western vs Non-Western tradition bias analysis  
- **Scenes**: Geographic bias analysis across European cities

**3. Statistical Significance Testing**
- Multiple hypothesis testing with Bonferroni correction
- Effect size analysis (Cohen's d, Hedge's g)
- Non-parametric tests (Mann-Whitney U, Kolmogorov-Smirnov)

**4. Visualization Generation**
- Performance gap comparison plots
- Feature space bias visualizations (t-SNE, PCA)
- Statistical significance heatmaps
- Domain-specific analysis charts
- Comprehensive bias summary tables

#### Command Line Options

```bash
python run_experiments.py [OPTIONS]

Options:
  --all                    Run complete experimental pipeline
  --validate              Run validation experiments only
  --quick                 Run abbreviated version for testing
  --data_dir PATH         Path to processed data directory (default: ./processed_data)
  --results_dir PATH      Path to save results (default: ./results)
  --config PATH           Path to configuration file (default: ./config.json)
  --help                  Show help message
```

#### Expected Outputs

After running experiments, you'll find in the results directory:

```
results/
├── comprehensive_results.json      # Complete experimental data
├── experiment_summary.json         # Structured summary
├── EXPERIMENTAL_SUMMARY.md         # Human-readable report
└── visualizations/
    ├── figure1_performance_gaps.png       # Main bias comparison
    ├── figure2_feature_space_bias.png     # Feature space analysis
    ├── figure3_statistical_significance.png # P-values and effect sizes
    ├── figure4_domain_analysis.png        # Domain-specific patterns
    └── table1_bias_summary.csv           # Comprehensive metrics table
```

## Results

Our study reveals significant cultural bias in mel-scale representations:

| Domain | Bias Metric | Mel-Scale | Best Alternative | Improvement |
|--------|-------------|-----------|------------------|-------------|
| **Speech** | CER Gap (Tonal vs Non-tonal) | 12.5% | ERB: 8.2% | **-34%** |
| **Music** | F1 Gap (Western vs Non-Western) | 15.7% | CQT: 9.1% | **-42%** |
| **Scenes** | Accuracy Gap (High vs Low HDI) | 8.3% | LEAF: 5.4% | **-35%** |

## Repository Structure

```
cross-cultural-mel-bias/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── download_datasets.py        # Dataset downloader
├── preprocess_datasets.py      # Balanced preprocessing
├── validate_datasets.py        # Dataset validation
├── run_experiments.py          # Main experiment runner
├── config.json                 # Experiment configuration
├── frontends.py                # Audio front-end implementations
├── bias_evaluation.py          # Bias metrics computation
├── .gitignore                  # Git ignore rules
├── data/                       # Raw datasets (auto-created)
└── processed_data/             # Processed datasets (auto-created)
```

## Datasets

### Speech (CommonVoice v17.0)
**Tonal Languages (5)**: Vietnamese (vi), Thai (th), Mandarin Chinese (zh-CN), Punjabi (pa-IN), Cantonese (yue)  
**Non-Tonal Languages (6)**: English (en), Spanish (es), German (de), French (fr), Italian (it), Dutch (nl)

### Music 
**Western**: GTZAN, FMA-small  
**Non-Western**: Carnatic, Hindustani, Turkish Makam, Arab-Andalusian

### Acoustic Scenes
**TAU Urban 2020**: Mixed urban acoustic scenes from 10 scene types recorded in 12 European cities

## Audio Front-Ends Evaluated

- **Mel-Scale**: Traditional mel-frequency features
- **ERB**: Equivalent Rectangular Bandwidth scale  
- **Bark**: Bark frequency scale
- **CQT**: Constant-Q Transform
- **LEAF**: Learnable Audio Front-End
- **SincNet**: Learnable sinc-based filters

## Key Findings

1. **Mel-scale features show systematic bias** against tonal languages and non-Western musical traditions
2. **ERB and CQT consistently reduce bias** with minimal computational overhead
3. **Learnable front-ends (LEAF, SincNet) effective** but require more training data
4. **Cultural bias correlates with linguistic/musical distance** from Western training data

## Requirements

- Python 3.8+
- ~100GB storage for datasets
- HuggingFace account for CommonVoice access
- GPU recommended for training (optional for evaluation)

## Citation

```bibtex
@inproceedings{chauhan2026crosscultural,
  title={Cross-Cultural Bias in Mel-Scale Audio Front-Ends: Evidence from Speech and Music},
  author={Chauhan, Shivam and Pundhir, Ajay},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
}
```

## License

MIT License

## Contact

**Shivam Chauhan**  
Presight AI, Abu Dhabi, UAE  
📧 [0shivam33@gmail.com](mailto:0shivam33@gmail.com)

---

*Building Fair Audio AI for Global Diversity*







# Cross-Cultural Bias in Mel-Scale Representations: Evidence and Alternatives

**ICASSP 2026 Paper Implementation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete implementation of our ICASSP 2026 paper: **"Cross-Cultural Bias in Mel-Scale Representations: Evidence and Alternatives from Speech and Music"**

**Authors**: Shivam Chauhan, Ajay Pundhir  
**Organization**: Presight AI, Abu Dhabi, United Arab Emirates

## 📄 Abstract

Modern audio systems universally employ mel-scale representations derived from 1940s Western psychoacoustic studies, potentially encoding cultural biases that create systematic performance disparities. We demonstrate that mel-scale features achieve **31.2% WER for tonal languages compared to 18.7% for non-tonal languages** (12.5% absolute gap), and show **15.7% F1 degradation between Western and non-Western music**. Alternative representations significantly reduce these disparities: LEAF reduces the speech gap by 34%, CQT achieves 52% reduction in music performance gaps, and ERB-scale filtering cuts disparities by 31% with only 1% computational overhead.

## 🎯 Key Contributions

1. **Systematic evaluation** of 7 front-ends across 11 languages, 8 musical traditions, and 10 European cities
2. **Demonstrating mel-scale bias**: 31.2% WER for tonal vs 18.7% for non-tonal languages (12.5% gap)
3. **Revealing critical frequencies**: 200-500 Hz where mel resolution is insufficient for tonal languages
4. **Showing alternatives work**: CQT (52% music gap reduction), LEAF (34% speech), ERB (31% across domains)
5. **Releasing FairAudioBench**: First benchmark for cross-cultural audio evaluation

## 📊 Paper Results Summary

### Performance Gaps (Figure 1)

| Domain | Mel Baseline Gap | Best Alternative | Reduction |
|--------|------------------|------------------|-----------|
| **Speech** (Tonal vs Non-tonal) | 12.5% WER | LEAF: 8.3% | 34% |
| **Music** (Western vs Non-Western) | 15.7% F1 | CQT: 7.6% | 52% |
| **Scenes** (Europe-1 vs Europe-2) | 5.6% Acc | ERB: 5.0% | 11% |

### Comprehensive Results (Table 1)

| Front-end | Speech WER/CER (%) |  | Music F1 (%) |  | Scenes Acc (%) |  | Overhead |
|-----------|---------|----------|--------|--------|----------|----------|----------|
|           | Tonal | Non-tonal | Non-West | West | Europe-1 | Europe-2 |          |
| **mel** | 31.2±1.2 | 18.7±0.8 | 56.7±2.1 | 72.4±1.5 | 71.2±1.4 | 76.8±1.2 | 1.00× |
| **ERB** | 26.4±1.0 | 17.8±0.7 | 62.8±2.0 | 73.1±1.4 | 72.6±1.3 | 77.2±1.1 | 1.01× |
| **Bark** | 27.2±1.0 | 18.1±0.8 | 61.9±2.1 | 72.8±1.5 | 72.2±1.3 | 76.9±1.2 | 1.01× |
| **CQT** | 28.8±1.1 | 19.2±0.9 | 65.3±1.9 | 72.9±1.4 | - | - | 1.15× |
| **LEAF** | 25.8±0.9 | 17.5±0.7 | 62.4±2.0 | 73.5±1.4 | 72.5±1.3 | 77.5±1.1 | 1.08× |
| **SincNet** | 30.8±1.1 | 18.5±0.8 | 58.3±2.1 | 72.5±1.5 | 71.4±1.3 | 76.9±1.2 | 1.06× |
| **mel+PCEN** | 28.9±1.1 | 18.2±0.7 | 59.2±2.2 | 72.6±1.5 | 72.3±1.3 | 77.1±1.1 | 1.04× |

### Fairness Metrics

| Metric | Formula | Speech | Music | Scenes |
|--------|---------|--------|-------|--------|
| **WGS** | min(Acc) | 68.8→74.2 | 56.7→65.3 | 71.2→72.5 |
| **Δ** | max-min | 12.5→8.3 | 15.7→7.6 | 5.6→5.0 |
| **ρ** | min/max | 0.85→0.90 | 0.78→0.90 | 0.93→0.94 |

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/shivam-MBZUAI/cross-cultural-mel-bias.git
cd cross-cultural-mel-bias

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📂 Dataset Configuration

### Speech Recognition (CommonVoice v17.0)
- **Tonal Languages (5)**: Mandarin Chinese (4 tones), Vietnamese (6 tones), Thai (5 tones), Punjabi (3 tones), Cantonese (6 tones)
- **Non-tonal Languages (6)**: English, Spanish, German, French, Italian, Dutch
- **Samples**: 2,000 test samples per language
- **Metrics**: CER for tonal, WER for non-tonal

### Music Analysis
- **Western Collections**: 
  - GTZAN (10 genres, 1000 tracks)
  - FMA-small (8 genres, 8000 tracks)
- **Non-Western Collections** (CompMusic):
  - Hindustani (1124 recordings, 195 ragas)
  - Carnatic (2380 recordings, 227 ragas)
  - Turkish makam (6500 recordings, 155 makams)
  - Arab-Andalusian (338 recordings, 11 mizans)
- **Samples**: 300 recordings per tradition

### Acoustic Scenes (TAU Urban Acoustic Scenes 2020 Mobile)
- **Europe-1 (Northern)**: Helsinki, Stockholm, Amsterdam, London, Vienna
- **Europe-2 (Southern)**: Barcelona, Lisbon, Paris, Lyon, Prague
- **Scene Types**: airport, bus, metro, park, public square, shopping mall, street pedestrian, street traffic, tram, metro station
- **Samples**: 100 recordings per city (10 per scene type)

## 🚀 Running Experiments

### Quick Start
```bash
# Download and prepare data
python prepare_data.py --task all

# Run complete evaluation pipeline
python evaluate_bias.py

# Results will be in ./results/ and ./plots/
```

### Individual Experiments
```bash
# Frequency Resolution Analysis (Section 3.4)
python evaluate_bias.py --experiment frequency_resolution

# Cross-Cultural Evaluation (Section 3.3)
python evaluate_bias.py --experiment cross_cultural

# Language-Specific Analysis (Section 3.5)
python evaluate_bias.py --experiment language_specific

# Feature-Level Analysis (Table 2)
python evaluate_bias.py --experiment feature_analysis

# Computational Efficiency (Figure 3)
python evaluate_bias.py --experiment efficiency
```

## 📈 Implemented Front-ends

All front-ends use identical CRNN backend (4 conv layers: 64-128-256-256 channels, 2-layer BiLSTM: 256 units, 5M total parameters):

1. **Mel**: 40 mel-spaced filters, 25ms windows, 10ms hop
2. **ERB**: 32 ERB-spaced filters (Glasberg & Moore 1990)
3. **Bark**: 24 critical bands (Zwicker 1961)
4. **SincNet**: 64 learnable sinc filters (Ravanelli & Bengio 2018)
5. **CQT**: 84 bins (7 octaves × 12 bins/octave) (Brown 1991)
6. **LEAF**: 64 learnable Gabor filters (Zeghidour et al. 2021)
7. **mel+PCEN**: Per-channel energy normalization (Wang et al. 2017)

## 🔬 Key Findings

### The Mechanism of Bias (Section 2.3)

The mel scale applies non-linear frequency warping:
```
ψ_mel(f) = 2595 log₁₀(1 + f/700)
```

At 300 Hz (critical for tones):
- **Mel resolution**: ~35 Hz
- **Required for tones**: ~3 Hz
- **Resolution deficit**: >10×

### LEAF's Adaptive Behavior (Figure 2)
- Allocates **42% of filters to 80-500 Hz** for tonal languages
- Only 23% for mel in same range
- Data-driven discovery validates theoretical analysis

### Language-Specific Improvements (Table 3)

| Language | Tones | mel WER | LEAF WER | Improvement |
|----------|-------|---------|----------|-------------|
| Vietnamese | 6 | 35.2% | 26.9% | -23.6% |
| Thai | 5 | 33.1% | 25.4% | -23.3% |
| Cantonese | 6 | 34.0% | 26.5% | -22.1% |
| Mandarin | 4 | 28.4% | 22.8% | -19.7% |
| Punjabi | 3 | 30.5% | 24.8% | -18.7% |

### Feature-Level Analysis (Table 2)

| Feature | mel | ERB | LEAF | Δ |
|---------|-----|-----|------|---|
| **Tones (F0)** | 71.2% | 82.4% | 83.7% | +12.5% |
| **Vowels** | 85.3% | 86.8% | 87.2% | +1.9% |
| **Consonants** | 88.1% | 88.4% | 88.9% | +0.8% |

## 📊 FairAudioBench

We introduce **FairAudioBench**, the first comprehensive benchmark for evaluating cross-cultural bias in audio systems:

- **Curated Datasets**: Balanced splits across 11 languages, 8 musical traditions, 10 European cities
- **Evaluation Suite**: Automated computation of WGS, Δ, ρ metrics with statistical significance testing
- **Reference Implementations**: All 7 front-ends with matched hyperparameters (5M params)

## 📁 Repository Structure

```
.
├── evaluate_bias.py          # Main evaluation script
├── prepare_data.py           # Data download and preprocessing
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/
│   ├── speech/
│   │   ├── tonal/           # 5 tonal languages
│   │   └── non_tonal/       # 6 non-tonal languages
│   ├── music/
│   │   ├── western/         # GTZAN, FMA
│   │   └── non_western/     # CompMusic collections
│   └── scenes/
│       ├── europe_1/        # Northern cities
│       └── europe_2/        # Southern cities
├── results/
│   ├── tables/              # Performance metrics
│   ├── fairness_metrics/    # WGS, Δ, ρ analysis
│   └── statistical_tests/   # Significance testing
└── plots/
    ├── figure1_gaps.png     # Performance gaps
    ├── figure2_leaf.png     # LEAF frequency allocation
    └── figure3_tradeoff.png # Fairness-efficiency
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

If you use this code or FairAudioBench in your research, please cite:

```bibtex
@inproceedings{chauhan2026crosscultural,
  title={Cross-Cultural Bias in Mel-Scale Representations: Evidence and Alternatives from Speech and Music},
  author={Chauhan, Shivam and Pundhir, Ajay},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
}
```

## 📧 Contact

- **Shivam Chauhan**: shivam.chauhan@presight.ai
- **Ajay Pundhir**: ajay.pundhir@presight.ai
- **Organization**: Presight AI, Abu Dhabi, UAE
- **GitHub**: https://github.com/shivam-MBZUAI/cross-cultural-mel-bias

## 🙏 Acknowledgments

We thank the CommonVoice contributors, CompMusic project, and TAU Urban Acoustic Scenes team for making their datasets available for research.

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

**Note**: This is an evaluation study examining bias in existing audio front-ends. We evaluate pre-trained models to measure cross-cultural fairness - no model training is performed.