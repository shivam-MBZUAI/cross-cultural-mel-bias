# CROSS-CULTURAL BIAS IN MEL-SCALE REPRESENTATIONS: EVIDENCE AND ALTERNATIVES FROM SPEECH AND MUSIC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
Modern audio systems universally employ mel-scale representations derived from 1940s Western psychoacoustic studies, potentially encoding cultural biases that create systematic performance disparities. We demonstrate that mel-scale features achieve 31.2% WER for tonal languages compared to 18.7% for non-tonal languages (12.5% absolute gap), and show 15.7% F1 degradation between Western and non-Western music. Alternative representations significantly reduce these disparities: ERB-scale filtering cuts disparities by 31% with only 1% computational overhead, while CQT achieves 52% reduction in music performance gaps.

## 1. Contributions

1. **Systematic evaluation** of audio front-ends across 11 languages, 8 musical traditions, and 10 European cities
2. **Demonstrating mel-scale bias**: 31.2% WER for tonal vs 18.7% for non-tonal languages (12.5% gap)
3. **Revealing critical frequencies**: 200-500 Hz where mel resolution is insufficient for tonal languages
4. **Showing alternatives work**: CQT (52% music gap reduction), ERB (31% across domains with minimal overhead)
5. **Releasing FairAudioBench**: First benchmark for cross-cultural audio evaluation

## 2. Performance Results

### Performance Gaps (Figure 1)

| Domain | Mel Baseline Gap | Best Alternative | Reduction |
|--------|------------------|------------------|-----------|
| **Speech** (Tonal vs Non-tonal) | 12.5% WER | ERB: 8.6% | 31% |
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
| **mel+PCEN** | 28.9±1.1 | 18.2±0.7 | 59.2±2.2 | 72.6±1.5 | 72.3±1.3 | 77.1±1.1 | 1.04× |

### Fairness Metrics

| Metric | Formula | Speech | Music | Scenes |
|--------|---------|--------|-------|--------|
| **WGS** | min(Acc) | 68.8→74.2 | 56.7→65.3 | 71.2→72.5 |
| **Δ** | max-min | 12.5→8.6 | 15.7→7.6 | 5.6→5.0 |
| **ρ** | min/max | 0.60→0.68 | 0.78→0.90 | 0.93→0.94 |

## 3. Experimental Setup

### Datasets

#### Speech Recognition (CommonVoice v17.0)
- **Tonal Languages (5)**: Mandarin Chinese (4 tones), Vietnamese (6 tones), Thai (5 tones), Punjabi (3 tones), Cantonese (6 tones)
- **Non-tonal Languages (6)**: English, Spanish, German, French, Italian, Dutch
- **Samples**: 2,000 test samples per language
- **Metrics**: CER for tonal, WER for non-tonal

#### Music Analysis
- **Western Collections**: 
  - GTZAN (10 genres, 1000 tracks)
  - FMA-small (8 genres, 8000 tracks)
- **Non-Western Collections** (CompMusic):
  - Hindustani (1124 recordings, 195 ragas)
  - Carnatic (2380 recordings, 227 ragas)
  - Turkish makam (6500 recordings, 155 makams)
  - Arab-Andalusian (338 recordings, 11 mizans)
- **Samples**: 300 recordings per tradition

#### Acoustic Scenes (TAU Urban Acoustic Scenes 2020 Mobile)
- **Europe-1 (Northern)**: Helsinki, Stockholm, Amsterdam, London, Vienna
- **Europe-2 (Southern)**: Barcelona, Lisbon, Paris, Lyon, Prague
- **Scene Types**: 10 urban acoustic environments
- **Samples**: 100 recordings per city

## 4. Implemented Front-ends

We evaluate 5 audio front-ends with fixed transformations that can be used as drop-in replacements:

| Front-end | Type | Parameters | Description |
|-----------|------|------------|-------------|
| **Mel** | Fixed | 40 filters, 25ms window, 10ms hop | Standard mel-scale filterbank (baseline) |
| **ERB** | Fixed | 32 ERB-spaced filters | Equivalent Rectangular Bandwidth scale |
| **Bark** | Fixed | 24 critical bands | Psychoacoustic Bark scale |
| **CQT** | Fixed | 84 bins (7 octaves × 12 bins) | Constant-Q Transform for music |
| **Mel+PCEN** | Fixed | Mel + per-channel normalization | Adaptive gain normalization |

### Why LEAF and SincNet Are Not Included

While the paper discusses LEAF and SincNet results, these are **learnable front-ends** that require fundamentally different evaluation:

1. **Task-Specific Training Required**: Unlike fixed filterbanks (Mel, ERB, Bark), LEAF and SincNet have learnable parameters that must be trained end-to-end with each task. They cannot be used as drop-in replacements.

2. **Different Evaluation Protocol**: These front-ends need separate training runs for each task (speech/music/scenes) to learn optimal filters, requiring significantly more computational resources and a different experimental setup.

3. **Fair Comparison Focus**: Our implementation focuses on fixed front-ends that can be fairly compared using the same pre-trained models, ensuring that performance differences come solely from the frequency decomposition method, not from different training procedures.

For researchers interested in learnable front-ends, we recommend training them end-to-end with your specific task following the original LEAF (Zeghidour et al., 2021) and SincNet (Ravanelli & Bengio, 2018) papers.

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

```bash
# Download all target datasets
python download_datasets.py --all --hf_token $HUGGINGFACE_HUB_TOKEN

# Download specific domains
python download_datasets.py --dataset commonvoice --lang all_target --hf_token $HF_TOKEN
python download_datasets.py --dataset gtzan
python download_datasets.py --dataset tau_urban
```

### Preprocessing

```bash
# Process all domains for evaluation
python preprocess_datasets.py --all

# Process specific domains
python preprocess_datasets.py --domain speech
python preprocess_datasets.py --domain music
python preprocess_datasets.py --domain scenes
```

### Running Experiments

```bash
# Train models (required first)
python train_models.py

# Run complete evaluation pipeline
python run_experiments.py

# This will:
# 1. Load your processed audio data
# 2. Evaluate 5 implemented front-ends on all 3 tasks  
# 3. Calculate fairness metrics (WGS, Gap, DI)
# 4. Generate visualizations
# 5. Save results to results/ directory
```

### Expected Output

```
results/
├── evaluation_results.json           # Numerical results for implemented front-ends
└── plots/
    ├── performance_fairness_tradeoff.png
    ├── fairness_metrics_comparison.png
    ├── groupwise_performance_music.png
    ├── groupwise_performance_scene.png
    └── groupwise_performance_speech.png
```

**Note**: Results will differ from paper as LEAF and SincNet are not yet implemented. The paper's complete results (Table 1) include all 7 front-ends.

## 6. FairAudioBench

We introduce **FairAudioBench**, the first comprehensive benchmark for evaluating cross-cultural bias in audio systems:

### Components

- **Curated Datasets**: Balanced evaluation splits across 11 languages, 8 musical traditions, 10 cities
- **Evaluation Suite**: Automated computation of fairness metrics (WGS, Δ, ρ) with statistical significance
- **Reference Implementations**: 5 fixed front-ends with matched architectures (5M params CRNN backend)
- **Reproducible Pipeline**: Complete evaluation in single script

### Key Features

1. **Standardized Evaluation Protocol**: Ensures fair comparison across front-ends
2. **Cross-Cultural Coverage**: Spans tonal/non-tonal languages, Western/non-Western music
3. **Statistical Rigor**: Bootstrap confidence intervals and significance testing
4. **Computational Efficiency**: Tracks inference overhead for practical deployment


## License

This project is licensed under the MIT License - see the LICENSE file for details.