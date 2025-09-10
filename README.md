# Cross-Cultural Bias in Mel-Scale Audio Front-Ends

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Evidence and Alternatives from Speech and Music**  
*Shivam Chauhan, Ajay Pundhir*  
*Presight AI, Abu Dhabi, UAE*

## Paper Abstract

This paper presents a comprehensive analysis of cross-cultural bias in mel-scale audio representations across speech, music, and acoustic scene recognition tasks. We evaluate 7 audio front-ends on datasets spanning 11 languages, 8 musical traditions, and 10 european regions, revealing systematic biases favoring Western/English content and proposing fairer alternatives.

## Quick Start

### Setup Environment

```bash
git clone https://github.com/shivam-MBZUAI/cross-cultural-mel-bias.git
cd cross-cultural-mel-bias

# Install dependencies
pip install -r requirements.txt

# For HuggingFace datasets (required for speech data)
pip install huggingface_hub
huggingface-cli login  # Login with your HF token
```


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

#### Available Experiments

**1. Audio Front-End Analysis**
- Feature extraction across 7 front-ends (Mel, ERB, Bark, CQT, LEAF, SincNet, Mel+PCEN)
- Cultural bias evaluation using performance gaps and statistical tests
- Feature space visualization and analysis

**2. Cross-Cultural Bias Evaluation**
- **Speech**: Tonal vs Non-Tonal language bias analysis
- **Music**: Western vs Non-Western tradition bias analysis  
- **Scenes**: Geographic bias analysis across European cities


## Results

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