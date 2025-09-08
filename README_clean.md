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

```bash
# Authenticate with HuggingFace
export HUGGINGFACE_HUB_TOKEN="your_token_here"

# Download target languages for balanced evaluation
python download_datasets.py --languages vi th zh-CN pa-IN yue en es de fr it nl --hf_token $HUGGINGFACE_HUB_TOKEN

# Or download all target datasets
python download_datasets.py --all --hf_token $HUGGINGFACE_HUB_TOKEN
```

### 3. Preprocess for Balanced Evaluation

```bash
# Process all datasets with balanced protocols (2000 speech samples/lang, 300 music samples/tradition)
python preprocess_datasets.py --all

# Validate processed datasets
python validate_datasets.py --domain all
```

### 4. Run Bias Evaluation

```bash
# Reproduce paper results across all front-ends
python run_experiments.py --config config.json

# Quick test with subset
python run_experiments.py --quick --frontends mel erb leaf
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
├── datasets.py                 # Dataset loading utilities
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
**TAU Urban 2020**: 10 European cities, 10 scene types

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
