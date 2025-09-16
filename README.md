## CROSS-CULTURAL BIAS IN MEL-SCALE REPRESENTATIONS: EVIDENCE AND ALTERNATIVES FROM SPEECH AND MUSIC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
Modern audio systems universally employ mel-scale representations derived from 1940s Western psychoacoustic studies, potentially encoding cultural biases that create systematic performance disparities. We demonstrate that mel-scale features achieve 31.2% WER for tonal languages compared to 18.7% for non-tonal languages (12.5% absolute gap), and show 15.7% F1 degradation between Western and non-Western music. Alternative representations significantly reduce these disparities: ERB-scale filtering cuts disparities by 31% with only 1% computational overhead, while CQT achieves 52% reduction in music performance gaps.

## 1. Contributions

1. **Systematic evaluation** of seven audio front-ends across 11 languages, 6 musical collections, and 10 European cities
2. **Demonstrating mel-scale bias**: 31.2% WER for tonal vs 18.7% for non-tonal languages (12.5% gap)
3. **Revealing critical frequencies**: 200-500 Hz where mel resolution is insufficient for tonal languages
4. **Showing alternatives work**: CQT (52% music gap reduction), ERB (31% across domains with minimal overhead)
5. **Releasing FairAudioBench**: Benchmark for cross-cultural audio evaluation

## 2. Results

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

## 3. Experimental Setup

### Datasets

#### Speech Recognition [CommonVoice v17.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
- **Tonal Languages (5)**: Mandarin Chinese (4 tones), Vietnamese (6 tones), Thai (5 tones), Punjabi (3 tones), Cantonese (6 tones)
- **Non-tonal Languages (6)**: English, Spanish, German, French, Italian, Dutch
- **Samples**: 2,000 test samples per language
- **Metrics**: CER for tonal, WER for non-tonal

#### Music Analysis
- **Western Collections**: 
  - [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) (10 genres, 1000 tracks)
  - [FMA-small](https://os.unil.cloud.switch.ch/fma/fma_small.zip) (8 genres, 8000 tracks)
- **Non-Western Collections** (CompMusic):
  - [Hindustani](https://www.kaggle.com/datasets/kcwaghmarewaghmare/indian-music-raga) (1124 recordings, 195 ragas)
  - [Carnatic](https://www.kaggle.com/datasets/desolationofsmaug/saraga-carnatic-music-dataset) (2380 recordings, 227 ragas)
  - [Turkish makam](https://zenodo.org/records/1283350/files/turkish_sarki_vocal_v2.0.zip?download=1) (6500 recordings, 155 makams)
  - [Arab-Andalusian](https://zenodo.org/records/1291776) (338 recordings, 11 mizans)
- **Samples**: 300 recordings per collection

#### Acoustic Scenes [TAU Urban Acoustic Scenes 2020 Mobile](https://zenodo.org/records/3819968)
- **Europe-1 (Northern)**: Helsinki, Stockholm, Amsterdam, London, Vienna
- **Europe-2 (Southern)**: Barcelona, Lisbon, Paris, Lyon, Prague
- **Scene Types**: 10 urban acoustic environments
- **Samples**: 100 recordings per city

## 4. Implemented Front-ends In Codebase

This codebase implements 5 audio front-ends with fixed transformations that can be used as drop-in replacements:

| Front-end | Type | Parameters | Description |
|-----------|------|------------|-------------|
| **Mel** | Fixed | 40 filters, 25ms window, 10ms hop | Standard mel-scale filterbank (baseline) |
| **ERB** | Fixed | 32 ERB-spaced filters | Equivalent Rectangular Bandwidth scale |
| **Bark** | Fixed | 24 critical bands | Psychoacoustic Bark scale |
| **CQT** | Fixed | 84 bins (7 octaves × 12 bins) | Constant-Q Transform for music |
| **Mel+PCEN** | Fixed | Mel + per-channel normalization | Adaptive gain normalization |

### Why LEAF and SincNet Are Not Included In The Codebase

While the paper reports LEAF and SincNet results, these **learnable front-ends** are not included in this codebase because:

1. **Different Architecture**: Unlike fixed filterbanks that process audio independently, LEAF and SincNet have learnable parameters that must be trained jointly with the downstream model. This requires a fundamentally different architecture where the front-end is part of the neural network, not a separate preprocessing step.

2. **FairAudioBench Design Philosophy**: FairAudioBench is designed as a plug-and-play evaluation framework where any fixed front-end can be swapped in using the same pre-trained CRNN models. This ensures performance differences come solely from the frequency decomposition method, not from different training procedures or model architectures.

3. **Reproducibility**: Fixed front-ends (Mel, ERB, Bark, CQT, PCEN) require no training and produce deterministic outputs, making results immediately reproducible. Learnable front-ends would require distributing multiple task-specific checkpoints and exact training configurations.

**Note**: We plan to release the LEAF and SincNet implementations along with their pre-trained models after paper acceptance. These will be available in the same repository with appropriate pipelines.

For researchers interested in learnable front-ends, we recommend training them end-to-end with your specific task following the original LEAF (Zeghidour et al., 2021) and SincNet (Ravanelli & Bengio, 2018) papers. The paper's Table 1 includes their full results for comparison.

## 5. Quick Start

```bash
git clone https://github.com/shivam-MBZUAI/cross-cultural-mel-bias.git
cd cross-cultural-mel-bias

# Install dependencies
pip install -r requirements.txt

# For HuggingFace datasets (required for speech data)
pip install huggingface_hub
huggingface-cli login  # Login with your HF token
```

### Dataset Preparation
Assuming the data files are already downloaded and are present inside data/ folder.

### Preprocessing

```bash
### Preprocessing

# Process specific domains
python preprocess_datasets.py --data_dir /path/to/data --output_dir processed_data --domain speech
python preprocess_datasets.py --data_dir /path/to/data --output_dir processed_data --domain music
python preprocess_datasets.py --data_dir /path/to/data --output_dir processed_data --domain scenes

# Parameters:
#   --data_dir: Directory containing raw data with speech/, music/, scenes/ subdirectories
#   --output_dir: Output directory for processed evaluation sets
#   --domain: Which domain to process [speech|music|scenes]

# This creates balanced evaluation sets:
#   - Speech: Max 2,000 samples per language (11 languages)
#   - Music: Max 300 samples per tradition (6 traditions)
#   - Scenes: Max 100 samples per region (2 regions)
```

### Running Experiments

```bash
# Run complete evaluation pipeline
python frontends_eval.py

# This will:
# 1. Load your processed audio data
# 2. Evaluate 5 implemented front-ends on all 3 tasks  
# 3. Calculate fairness metrics (WGS, Gap, DI)
```

## 5. FairAudioBench

We introduce **FairAudioBench**, the first comprehensive benchmark for evaluating cross-cultural bias in audio systems present in file 'preprocess_datasets.py' and 'frontends_eval.py':

### Components

- **Curated Datasets**: Balanced evaluation splits across 11 languages, 8 musical traditions, 10 cities ('preprocess_datasets.py')
- **Evaluation Suite**: Automated computation of fairness metrics (WGS, Δ, ρ) with statistical significance ('frontends_eval.py')
- **Reproducible Pipeline**: Complete evaluation in single script ('frontends_eval.py')
