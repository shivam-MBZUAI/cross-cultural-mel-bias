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

1. **Cross-Cultural Evaluation**: Systematic study across speech (11 languages), music (8 traditions), and acoustic scenes  
2. **Bias Quantification**: Metrics for measuring cultural bias in audio representations  
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

# Create conda environment
conda create -n mel-bias python=3.8
conda activate mel-bias

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

```bash
# Download speech datasets
python download_datasets.py --dataset commonvoice --lang all_target --hf_token $HUGGINGFACE_HUB_TOKEN    # All Languages
python download_datasets.py --dataset commonvoice --lang vi --hf_token $HUGGINGFACE_HUB_TOKEN            # Vietnamese (tonal)
python download_datasets.py --dataset commonvoice --lang en --hf_token $HUGGINGFACE_HUB_TOKEN            # English (non-tonal)

# Download specific music datasets
python download_datasets.py --dataset gtzan --hf_token $HUGGINGFACE_HUB_TOKEN                     # Western genres
python download_datasets.py --dataset fma                                                         # 8 balanced genres
python download_datasets.py --dataset carnatic                                                    # South Indian classical
python download_datasets.py --dataset turkish_makam                                               # Turkish classical
python download_datasets.py --dataset hindustani                                                  # Indian RAGA
python download_datasets.py --dataset arab_andalusian                                             # Arab Andalusian

# Download acoustic scene data  
python download_datasets.py --dataset tau_urban                 # Urban acoustic scenes

# List available options
python download_datasets.py --list
```

**Supported Datasets:**
- **Speech**: CommonVoice 17.0 (205+ languages, including 8 tonal and 10 non-tonal)
- **Music**: GTZAN, FMA-small, Carnatic, Hindustani, Turkish Makam, Arab-Andalusian
- **Scenes**: TAU Urban Acoustic Scenes 2020

## 📊 Dataset Overview

### Speech Recognition (CommonVoice 17.0)

### Speech Recognition (CommonVoice 17.0)
- **Tonal Languages (5):** Vietnamese (vi), Thai (th), Mandarin (zh-CN), Punjabi (pa-IN), Cantonese (yue)
- **Non-Tonal Languages (6):** English (en), Spanish (es), German (de), French (fr), Italian (it), Dutch (nl)

### Music Analysis
- **Western Traditions**: GTZAN (10 genres), FMA-small (8 genres)  
- **Non-Western Traditions**: Hindustani, Carnatic, Turkish Makam, Arab-Andalusian classical music
- **Focus**: Genre/raga/makam classification to study cultural bias in musical representations

### Acoustic Scene Classification  
- **Dataset**: TAU Urban Acoustic Scenes 2020
- **Coverage**: European urban environments (airports, streets, parks, etc.)
- **Purpose**: Baseline comparison for environmental audio analysis

## 🔧 Research Framework (Coming Soon)

Our complete framework will evaluate multiple audio front-ends:

- **Traditional**: Mel-scale, ERB, Bark scale representations
- **Perceptual**: Constant-Q Transform (CQT) with logarithmic frequency spacing  
- **Learnable**: LEAF, SincNet with data-driven filter learning
- **Evaluation**: Cross-cultural bias metrics and mitigation strategies

## � Expected Results

Based on preliminary experiments, we anticipate:

- **Speech Recognition**: ~12.5% performance gap between tonal and non-tonal languages
- **Music Classification**: ~15.7% degradation for non-Western musical traditions  
- **Promising Solutions**: ERB and CQT showing significant bias reduction

| Language | Script | Tones | Mel CER | LEAF CER | ERB CER | Improvement (Best) |
|----------|--------|-------|---------|----------|----------|--------------------|
| Vietnamese | Latin | 6 | 31.2% | 23.8% | 21.9% | **-29.8%** (ERB) |
| Thai | Thai | 5 | 28.7% | 21.9% | 20.1% | **-30.0%** (ERB) |
| Mandarin | Hanzi | 4 | 33.4% | 26.8% | 24.3% | **-27.2%** (ERB) |
| Punjabi | Gurmukhi | 3 | 29.1% | 23.6% | 21.8% | **-25.1%** (ERB) |
| Cantonese | Hanzi | 6 | 35.6% | 27.8% | 26.1% | **-26.7%** (ERB) |

**Non-Tonal Languages (Word Error Rate)**

| Language | Family | Mel WER | LEAF WER | ERB WER | Improvement (Best) |
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
    ├── commonvoice_*/      # Speech datasets by language
    ├── gtzan/              # Western music genres  
    ├── carnatic/           # South Indian classical
    ├── hindustani/         # North Indian classical
    ├── turkish_makam/      # Turkish classical
    ├── arab_andalusian/    # Maghrebi classical
    └── tau_urban_2020/     # Acoustic scenes
```

## 📋 Requirements

### System Requirements
- Python 3.8+ (tested on 3.8, 3.9, 3.10)
- ~200GB storage for complete dataset collection
- Internet connection for downloading datasets
- Git for cloning repository

### Authentication Required
- **HuggingFace Account**: For CommonVoice speech datasets
- **Kaggle Account**: For Indian classical music datasets  
- **No registration needed**: For GTZAN, FMA, Turkish, Arab-Andalusian, and TAU datasets
│   ├── datasets/              # Dataset loading and preprocessing
│   │   ├── commonvoice.py    # CommonVoice speech loader
│   │   ├── music_loaders.py  # GTZAN, FMA, classical music
│   │   └── scene_loader.py   # TAU Urban scenes
│   ├── metrics/               # Evaluation and bias metrics
│   │   ├── fairness.py       # Cross-cultural fairness metrics
│   │   ├── performance.py    # Standard ML metrics
│   │   └── cultural_distance.py # Cultural similarity measures
│   └── utils/                 # Utility functions
│       ├── audio_utils.py    # Audio processing utilities
│       ├── visualization.py  # Plotting and visualization
│       └── logging.py        # Experiment logging
├── experiments/               # Experiment scripts
│   ├── speech_recognition.py # Speech ASR experiments
│   ├── music_classification.py # Music analysis experiments
│   ├── scene_classification.py # Acoustic scene experiments
│   ├── reproduce_paper.py    # Full paper reproduction
│   ├── generate_plots.py     # Figure generation
│   ├── generate_tables.py    # Table generation
│   └── ablation_studies.py   # Additional analyses
├── data/                      # Dataset storage (gitignored)
│   ├── README.md             # Data organization guide
│   ├── speech/               # Speech datasets
│   │   └── commonvoice/      # CommonVoice by language
│   ├── music/                # Music datasets
│   │   ├── gtzan/           # GTZAN genre classification
│   │   ├── fma/             # Free Music Archive
│   │   ├── carnatic/        # South Indian classical
│   │   ├── hindustani/      # North Indian classical
│   │   ├── turkish_makam/   # Turkish classical
│   │   └── arab_andalusian/ # Maghrebi classical
│   └── scenes/               # Acoustic scene data
│       └── tau_urban/       # TAU Urban scenes
├── results/                   # Experiment outputs (gitignored)
│   ├── logs/                 # Training logs
│   ├── models/               # Saved model checkpoints
│   ├── figures/              # Generated plots
└── tests/                     # Unit and integration tests
    ├── test_frontends.py     # Frontend implementation tests
    ├── test_datasets.py      # Dataset loading tests
    └── test_experiments.py   # End-to-end experiment tests
```

## 📋 Requirements

### System Requirements
- **Python:** 3.8+ (tested on 3.8, 3.9, 3.10)
- **GPU:** CUDA-compatible GPU with ≥8GB VRAM (recommended for full experiments)
- **Storage:** ~200GB for all datasets
- **Memory:** 16GB+ RAM recommended for large-scale experiments

### Key Dependencies
- `torch >= 1.12.0` - PyTorch for deep learning models
- `torchaudio >= 0.12.0` - Audio processing and transformations
- `librosa >= 0.10.0` - Audio feature extraction
- `datasets >= 2.0.0` - HuggingFace datasets interface
- `transformers >= 4.20.0` - Pre-trained speech models
- `scikit-learn >= 1.1.0` - ML metrics and utilities
- `matplotlib >= 3.5.0` - Visualization and plotting
- `seaborn >= 0.11.0` - Statistical visualizations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone in development mode
git clone https://github.com/shivam-MBZUAI/cross-cultural-mel-bias.git
cd cross-cultural-mel-bias

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

### Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:
- Python version and OS
- Error messages and stack traces
- Steps to reproduce the issue
- Expected vs. actual behavior

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code or our findings in your research, please cite our paper:

```bibtex
@inproceedings{chauhan2026crosscultural,
  title={Cross-Cultural Bias in Mel-Scale Audio Front-Ends: Evidence from Speech and Music},
  author={Chauhan, Shivam and Pundhir, Ajay},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
}
```

## 🔗 Related Work

- [LEAF: A Learnable Frontend for Audio Classification](https://arxiv.org/abs/2101.08596)
- [SincNet: Interpretable 1D Convolutional Neural Networks for Speech Recognition](https://arxiv.org/abs/1808.00158)
- [CommonVoice: A Massively-Multilingual Collection of Transcribed Speech](https://arxiv.org/abs/1912.06670)
- [Cultural Considerations in Automatic Speech Recognition](https://arxiv.org/abs/2108.04881)

## 📞 Contact

**Shivam Chauhan**  
Presight AI, Abu Dhabi, UAE  
Email: [0shivam33@gmail.com](mailto:0shivam33@gmail.com)  
GitHub: [@shivam-MBZUAI](https://github.com/shivam-MBZUAI)

## 🙏 Acknowledgments

- Mozilla Foundation for the CommonVoice dataset
- Music Information Retrieval researchers for open-source music datasets
- TAU research group for acoustic scene classification datasets
- HuggingFace team for dataset hosting infrastructure
- Open-source community for audio processing libraries

---

**Keywords:** Cross-cultural bias, Audio processing, Speech recognition, Music information retrieval, Fairness in ML, Cultural AI
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
