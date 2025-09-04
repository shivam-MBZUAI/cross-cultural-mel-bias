# Cross-Cultural Bias in Mel-Scale Audio Front-Ends: Dataset Downloader

[![Paper](https://img.shields.io/badge/Paper-ICASSP%202026-blue)](https://arxiv.org/abs/your-paper-id)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Status](https://img.shields.io/badge/Status-Dataset%20Downloader%20Ready-green)](https://github.com/shivam-MBZUAI/cross-cultural-mel-bias)

**Evidence from Speech and Music**  
*Shivam Chauhan, Ajay Pundhir*  
*Presight AI, Abu Dhabi, United Arab Emirates*

> **📍 Current Status**: This repository contains the **dataset downloader** for our upcoming ICASSP 2026 paper. The complete analysis code and models will be released upon paper acceptance.

## 🎯 Overview

This repository provides automated tools to download and organize all datasets used in our research on cross-cultural bias in mel-scale audio representations. Our study investigates whether mel-scale features, derived from 1940s Western psychoacoustic studies, encode cultural biases across speech, music, and acoustic scene classification tasks.

### Research Scope (Full Paper)

1. **Cross-Cultural Evaluation**: Systematic study across speech (11 languages), music (8 traditions), and acoustic scenes (10 European cities)
2. **Bias Quantification**: Metrics for measuring cultural bias in mel-scale audio representations  
3. **Alternative Solutions**: Comparison of learnable (LEAF, SincNet) and psychoacoustic (ERB, Bark, CQT) alternatives  
4. **Mitigation Strategies**: Practical solutions for culturally-aware audio systems  

### Expected Findings (Preliminary Results)

- **Speech Recognition**: ~12.5% performance gap between tonal and non-tonal languages using mel-scale features  
- **Music Analysis**: ~15.7% F1 degradation between Western and non-Western musical traditions  
- **Promising Alternatives**: ERB and CQT show significant bias reduction with minimal computational overhead

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/shivam-MBZUAI/cross-cultural-mel-bias.git
cd cross-cultural-mel-bias

# Install dependencies
pip install -r requirements.txt
```

### Authentication Setup

Before downloading datasets, set up authentication:

```bash
# HuggingFace (for CommonVoice and GTZAN datasets)
huggingface-cli login
# OR export HUGGINGFACE_HUB_TOKEN="your_token"

# Kaggle (for Indian Classical Music datasets)  
# Download kaggle.json from Kaggle Account settings
# Place in ~/.kaggle/kaggle.json
```

### Dataset Download

Our comprehensive dataset downloader supports all research datasets:

# Download all target speech datasets (5 tonal + 6 non-tonal languages)
python download_datasets.py --dataset commonvoice --lang all_target --hf_token $HUGGINGFACE_HUB_TOKEN

# Download specific languages
python download_datasets.py --dataset commonvoice --lang vi --hf_token $HUGGINGFACE_HUB_TOKEN     # Vietnamese (tonal)
python download_datasets.py --dataset commonvoice --lang en --hf_token $HUGGINGFACE_HUB_TOKEN     # English (non-tonal)
python download_datasets.py --dataset commonvoice --lang zh-CN --hf_token $HUGGINGFACE_HUB_TOKEN  # Mandarin Chinese

# Download Western music datasets
python download_datasets.py --dataset gtzan --hf_token $HUGGINGFACE_HUB_TOKEN    # GTZAN: 10 genres, 1000 tracks
python download_datasets.py --dataset fma                                        # FMA-small: 8 genres, 8000 tracks

# Download non-Western music datasets
python download_datasets.py --dataset carnatic                                   # South Indian: 2380 recordings, 227 ragas
python download_datasets.py --dataset hindustani                                 # North Indian: 1124 recordings, 195 ragas
python download_datasets.py --dataset turkish_makam                              # Turkish: 6500 recordings, 155 makams
python download_datasets.py --dataset arab_andalusian                            # Maghrebi: 338 recordings, 11 mizans

# Download acoustic scene data  
python download_datasets.py --dataset tau_urban                                  # TAU Urban: 10 cities, 10 scenes, 64h

# List all available options and commands
python download_datasets.py --list

# Download everything at once (large download ~100GB)
python download_datasets.py --all --hf_token $HUGGINGFACE_HUB_TOKEN
```

**Supported Datasets:**
- **Speech:** CommonVoice v17.0 (11 balanced languages: 5 tonal, 6 non-tonal)
- **Music:** GTZAN, FMA-small (Western) + Carnatic, Hindustani, Turkish Makam, Arab-Andalusian (Non-Western)
- **Acoustic Scenes:** TAU Urban Acoustic Scenes 2020 (10 European cities, 10 scene types)

## 📊 Dataset Overview

Our study evaluates across three complementary domains using **balanced evaluation protocols** to ensure fair cross-cultural comparison and eliminate dataset size biases:

### Speech Recognition (CommonVoice v17.0)
**Linguistically stratified evaluation with standardized data volumes:**

- **Tonal Languages (5):** Mandarin Chinese (zh-CN, 4 tones), Vietnamese (vi, 6 tones), Thai (th, 5 tones), Punjabi (pa-IN, 3 tones), Cantonese (yue, 6 tones)
- **Non-Tonal Languages (6):** English (en), Spanish (es), German (de), French (fr), Italian (it), Dutch (nl)

**🎯 Balanced Evaluation Protocol:**
- **Standardized Sample Size:** Exactly 2,000 test samples per language, randomly sampled to eliminate data volume bias
- **Audio Specifications:** Average 4.2s duration per sample, 22kHz sample rate (CommonVoice standard)
- **Controlled Comparison:** Performance differences reflect representational bias, not dataset size disparities
- **Appropriate Metrics:** Character Error Rate (CER) for tonal languages, Word Error Rate (WER) for non-tonal languages (accounting for orthographic differences)

### Music Analysis
**Cross-cultural evaluation contrasting Western vs. Non-Western musical traditions:**

**Western Traditions:**
- **GTZAN:** 10 genres, 1000 tracks for genre classification
- **FMA-small:** 8 genres, 8000 tracks for balanced genre evaluation

**Non-Western Collections (CompMusic Project):**
- **Hindustani:** 1124 recordings, 195 ragas (North Indian classical)
- **Carnatic:** 2380 recordings, 227 ragas (South Indian classical) 
- **Turkish Makam:** 6500 recordings, 155 makams (Turkish classical)
- **Arab-Andalusian:** 338 recordings, 11 mizans (Maghrebi classical)

**🎯 Balanced Evaluation Protocol:**
- **Standardized Sample Size:** Exactly 300 recordings per tradition, randomly sampled to control for vastly different dataset sizes
- **Audio Specifications:** 30-second segments, standardized to 22kHz sample rate for consistent analysis
- **Modal Balance:** Equal representation across modal categories (ragas/makams/mizans) within each tradition
- **Fair Comparison:** Eliminates data volume bias between traditions with 300-6500+ recordings
- **Consistent Tasks:** Modal classification (non-Western) and genre classification (Western) using macro-F1 scores

### Acoustic Scene Classification
**Geographic diversity control for environmental audio analysis:**

**TAU Urban Acoustic Scenes 2020 Mobile Dataset:**
- **Coverage:** 10 European cities (Barcelona, Helsinki, London, Paris, Stockholm, Vienna, Amsterdam, Lisbon, Lyon, Prague)
- **Scenes:** 10 acoustic environments per city (airport, bus, metro, park, public square, shopping mall, street pedestrian, street traffic, tram, metro station)  
- **Volume:** 64 hours total recordings

**🎯 Balanced Evaluation Protocol:**
- **Geographic Balance:** Exactly 100 recordings per city (10 per scene type) to ensure equal geographic representation
- **Audio Specifications:** 10-second segments, 48kHz sample rate (TAU dataset standard)
- **Scene Diversity:** Equal representation across all 10 acoustic scene types
- **Bias Control:** Prevents bias toward cities with more extensive data collection
- **Standardized Evaluation:** 1,000 total samples with balanced urban diversity

### Evaluation Protocol Summary

**🔬 Why Balanced Evaluation Matters:**
- **Eliminates Confounding Variables:** Standardized sample sizes ensure performance differences reflect representational bias, not data availability
- **Fair Cross-Cultural Comparison:** Equal treatment across linguistic families, musical traditions, and geographic regions  
- **Statistical Validity:** Controlled comparisons enable meaningful conclusions about cultural bias in audio representations
- **Reproducible Research:** Clear protocols enable replication and extension of findings

## 🔧 Research Framework (Coming Soon)

Our complete framework will evaluate multiple audio front-ends:

- **Traditional**: Mel-scale, ERB, Bark scale representations
- **Perceptual**: Constant-Q Transform (CQT) with logarithmic frequency spacing  
- **Learnable**: LEAF, SincNet with data-driven filter learning
- **Evaluation**: Cross-cultural bias metrics and mitigation strategies

## 🔍 Expected Results

Based on preliminary experiments, our study reveals significant cultural bias in mel-scale representations:

- **Speech Recognition**: ~12.5% performance gap between tonal and non-tonal languages
- **Music Classification**: ~15.7% degradation for non-Western musical traditions  
- **Promising Solutions**: ERB and CQT showing significant bias reduction with minimal computational overhead

**Tonal Languages (Character Error Rate)**

| Language | Tones | Script | Mel CER | LEAF CER | ERB CER | Best Improvement |
|----------|-------|--------|---------|----------|----------|--------------------|
| Vietnamese | 6 | Latin | 31.2% | 23.8% | 21.9% | **-29.8%** (ERB) |
| Thai | 5 | Thai | 28.7% | 21.9% | 20.1% | **-30.0%** (ERB) |
| Mandarin | 4 | Hanzi | 33.4% | 26.8% | 24.3% | **-27.2%** (ERB) |
| Punjabi | 3 | Gurmukhi | 29.1% | 23.6% | 21.8% | **-25.1%** (ERB) |
| Cantonese | 6 | Hanzi | 35.6% | 27.8% | 26.1% | **-26.7%** (ERB) |

**Non-Tonal Languages (Word Error Rate)**

| Language | Family | Mel WER | LEAF WER | ERB WER | Best Improvement |
|----------|--------|---------|----------|----------|--------------------|
| English | Germanic | 18.7% | 17.2% | 17.5% | **-8.0%** (LEAF) |
| Spanish | Romance | 16.9% | 15.8% | 16.1% | **-6.5%** (LEAF) |
| German | Germanic | 21.3% | 19.7% | 19.9% | **-7.5%** (LEAF) |
| French | Romance | 19.8% | 18.4% | 18.6% | **-7.1%** (LEAF) |
| Italian | Romance | 17.4% | 16.1% | 16.3% | **-7.5%** (LEAF) |
| Dutch | Germanic | 20.1% | 18.9% | 19.1% | **-6.0%** (LEAF) |

## 🛠️ Current Implementation Status

✅ **Available Now:**
- Comprehensive dataset downloader for all research datasets
- Automated authentication for HuggingFace, Kaggle, and Zenodo
- Support for 205+ CommonVoice languages with tonal/non-tonal classification
- Complete music and acoustic scene dataset collection

🚧 **Coming Soon (Upon Paper Acceptance):**
- Audio front-end implementations (Mel, ERB, Bark, CQT, LEAF, SincNet)  
- Cross-cultural bias analysis framework
- Model training and evaluation pipelines
- Reproduction scripts for all paper experiments
- Interactive bias analysis tools

## 🗂️ Current Project Structure

```
cross-cultural-mel-bias/
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── download_datasets.py     # ✅ Dataset downloader (READY)
├── .gitignore              # Git configuration
└── data/                   # Downloaded datasets (gitignored)
    ├── commonvoice_*/      # Speech datasets by language (11 languages)
    ├── gtzan/              # Western: 10 genres, 1000 tracks
    ├── fma_small/          # Western: 8 genres, 8000 tracks  
    ├── carnatic/           # Indian classical: 2380 recordings, 227 ragas
    ├── hindustani/         # Indian classical: 1124 recordings, 195 ragas
    ├── turkish_makam/      # Turkish classical: 6500 recordings, 155 makams
    ├── arab_andalusian/    # Maghrebi classical: 338 recordings, 11 mizans
    └── tau_urban_2020/     # Acoustic scenes: 10 cities, 10 scenes, 64h
```

## 📋 Requirements

### System Requirements
- Python 3.8+ (tested on 3.8, 3.9, 3.10, 3.12)
- ~100GB storage for target datasets (11 languages + 6 music datasets + acoustic scenes)
- Internet connection for downloading datasets
- Git for cloning repository

### Authentication Required
- **HuggingFace Account**: For CommonVoice speech datasets and GTZAN music dataset
- **Kaggle Account**: For Indian classical music datasets (Carnatic, Hindustani)  
- **No registration needed**: For FMA, Turkish Makam, Arab-Andalusian, and TAU datasets

### Key Dependencies
```txt
# Audio processing
torchaudio>=0.12.0
librosa>=0.10.0  
soundfile>=0.12.1

# Dataset access
datasets>=2.0.0
huggingface-hub>=0.10.0
kaggle>=1.5.0

# Data handling
pandas>=1.4.0
numpy>=1.21.0
requests>=2.28.0

# Utilities
pathlib
zipfile
tarfile
```

## 🛠️ Current Implementation Status

✅ **Available Now:**
- Comprehensive dataset downloader for all research datasets
- Automated authentication for HuggingFace, Kaggle, and Zenodo
- Support for target languages with tonal/non-tonal classification
- Complete music and acoustic scene dataset collection
- Batch download support for all target languages (`--lang all_target`)

🚧 **Coming Soon (Upon Paper Acceptance):**
- Audio front-end implementations (Mel, ERB, Bark, CQT, LEAF, SincNet)  
- Cross-cultural bias analysis framework
- Model training and evaluation pipelines
- Reproduction scripts for all paper experiments
- Interactive bias analysis tools

## 🤝 Contributing

We welcome contributions to expand dataset support and improve the downloader! 

### Reporting Issues
Please use GitHub Issues for:
- Bug reports with dataset downloaders
- Feature requests for new datasets
- Authentication or setup problems

## 📄 License

This project is licensed under the MIT License.

## 📚 Citation (Preprint)

```bibtex
@inproceedings{chauhan2026crosscultural,
  title={Cross-Cultural Bias in Mel-Scale Audio Front-Ends: Evidence from Speech and Music},
  author={Chauhan, Shivam and Pundhir, Ajay},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
  year={2026},
  organization={IEEE},
  note={Paper under review}
}
```

## 🔗 Related Work

- [CommonVoice: A Massively-Multilingual Collection of Transcribed Speech](https://arxiv.org/abs/1912.06670)
- [LEAF: A Learnable Frontend for Audio Classification](https://arxiv.org/abs/2101.08596)  
- [SincNet: Interpretable 1D Convolutional Neural Networks](https://arxiv.org/abs/1808.00158)

## 📞 Contact

**Shivam Chauhan**  
Presight AI, Abu Dhabi, UAE  
📧 [0shivam33@gmail.com](mailto:0shivam33@gmail.com)  
🐙 [@shivam-MBZUAI](https://github.com/shivam-MBZUAI)

---

> 🌍 **Building Fair Audio AI for Global Diversity** - *Comprehensive dataset collection for cross-cultural bias research*
