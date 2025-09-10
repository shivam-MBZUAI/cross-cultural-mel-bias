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
