# CROSS-CULTURAL BIAS IN MEL-SCALE REPRESENTATIONS: EVIDENCE AND ALTERNATIVES FROM SPEECH AND MUSIC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
Modern audio systems universally employ mel-scale representations derived from 1940s Western psychoacoustic studies, potentially encoding cultural biases that create systematic performance disparities. We demonstrate that mel-scale features achieve 31.2% WER for tonal languages compared to 18.7% for non-tonal languages (12.5% absolute gap), and show 15.7% F1 degradation between Western and non-Western music. Alternative representations significantly reduce these disparities: LEAF reduces the speech gap by 34%, CQT achieves 52% reduction in music performance gaps, and ERB-scale filtering cuts disparities by 31% with only 1% computational overhead.

### 1. Contributions

1. **Systematic evaluation** of 7 front-ends across 11 languages, 8 musical traditions, and 10 European cities
2. **Demonstrating mel-scale bias**: 31.2% WER for tonal vs 18.7% for non-tonal languages (12.5% gap)
3. **Revealing critical frequencies**: 200-500 Hz where mel resolution is insufficient for tonal languages
4. **Showing alternatives work**: CQT (52% music gap reduction), LEAF (34% speech), ERB (31% across domains)
5. **Releasing FairAudioBench**: First benchmark for cross-cultural audio evaluation

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

## Dataset Configuration

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


### 2. Quick Start

```bash
git clone https://github.com/shivam-MBZUAI/cross-cultural-mel-bias.git
cd cross-cultural-mel-bias

# Install dependencies
pip install -r requirements.txt

# For HuggingFace datasets (required for speech data)
pip install huggingface_hub
huggingface-cli login  # Login with your HF token
```


#### Dataset Preparation

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


### 3. Preprocess for Balanced Evaluation

The `preprocess_datasets.py` script creates **balanced evaluation datasets only** (no training splits). This is a bias evaluation study, not a training study.

**Dataset Specifications:**
- **Speech**: 2,000 samples per language (11 languages: 5 tonal, 6 non-tonal)
- **Music**: 300 samples per tradition (6 traditions: 2 Western, 4 non-Western)  
- **Scenes**: 100 samples of mixed urban acoustic scenes from TAU Urban dataset

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

### 4. Implemented Front-ends

All front-ends use identical CRNN backend (4 conv layers: 64-128-256-256 channels, 2-layer BiLSTM: 256 units, 5M total parameters):

1. **Mel**: 40 mel-spaced filters, 25ms windows, 10ms hop
2. **ERB**: 32 ERB-spaced filters (Glasberg & Moore 1990)
3. **Bark**: 24 critical bands (Zwicker 1961)
4. **SincNet**: 64 learnable sinc filters (Ravanelli & Bengio 2018)
5. **CQT**: 84 bins (7 octaves × 12 bins/octave) (Brown 1991)
6. **LEAF**: 64 learnable Gabor filters (Zeghidour et al. 2021)
7. **mel+PCEN**: Per-channel energy normalization (Wang et al. 2017)

#### Running Complete Evaluation

The complete evaluation pipeline is implemented in `frontends.py` and reproduces ALL experiments from the paper:

1. **Frequency resolution analysis** (Section 5.1)
2. **Cross-cultural evaluation** on 3 tasks (Section 5.2) 
3. **Computational efficiency analysis** (Section 5.3)
4. **Statistical significance testing** (Section 5.4)
5. **Confusion matrix analysis** for tonal languages (Section 5.5)
6. **Filter visualization and analysis** (Section 5.6)
7. **Ablation studies** (Section 5.7)

```bash
# Run all experiments (requires processed data)
python frontends.py

# This will:
# 1. Evaluate all 7 front-ends on all 3 tasks
# 2. Generate all figures from the paper
# 3. Compute statistical significance tests
# 4. Create fairness metrics comparison
# 5. Save results to results/ directory
```

### Expected Output Structure

```
results/
├── all_results.json                 # Complete results in JSON format
├── table2_complete_results.csv      # Main results table
└── plots/
    ├── figure2_fairness_metrics.png
    ├── figure3_frequency_resolution.png
    ├── figure5_tone_confusion.png
    ├── figure6_efficiency.png
    ├── figure7_filter_comparison.png
    ├── filters_mel.png
    ├── filters_erb.png
    ├── filters_bark.png
    ├── filters_cqt.png
    ├── filters_leaf.png
    └── filters_sincnet.png
```

### Individual Components

You can also import and run individual experiments:

```python
from frontends import ExperimentConfig, Experiment1_FrequencyResolution

# Initialize configuration
config = ExperimentConfig()

# Run individual experiment
exp1 = Experiment1_FrequencyResolution(config)
results = exp1.run()
```

## FairAudioBench

We introduce **FairAudioBench**, the first comprehensive benchmark for evaluating cross-cultural bias in audio systems:

- **Curated Datasets**: Balanced splits across 11 languages, 8 musical traditions, 10 European cities (`preprocess_datasets.py`)
- **Evaluation Suite**: Automated computation of WGS, Δ, ρ metrics with statistical significance testing
- **Reference Implementations**: All 7 front-ends with matched hyperparameters (5M params)
- **Reproducible Pipeline**: Complete evaluation in single script (`frontends.py`)