"""
FairAudioBench: Cross-Cultural Bias Evaluation in Audio Front-Ends
ICASSP 2026 Paper Implementation

A comprehensive benchmark suite for evaluating cross-cultural bias in mel-scale 
audio front-ends across speech, music, and environmental sound classification tasks.

This package provides:
- Audio front-end implementations (Mel, ERB, Bark, CQT, LEAF, SincNet)
- Neural network models for speech, music, and scene classification
- Bias evaluation metrics and fairness analysis tools
- Dataset loading and preprocessing utilities
- Experiment orchestration and results analysis

Usage:
    from FairAudioBench.src.frontends import create_frontend
    from FairAudioBench.src.models import create_model
    from FairAudioBench.src.bias_evaluation import BiasMetrics

Author: Shivam Chauhan, Ajay Pundhir
Organization: Presight AI, Abu Dhabi, UAE
"""

__version__ = "1.0.0"
__author__ = "Shivam Chauhan, Ajay Pundhir"
__organization__ = "Presight AI, Abu Dhabi, UAE"
__license__ = "MIT"
__paper__ = "Cross-Cultural Bias in Mel-Scale Audio Front-Ends: Evidence from Speech and Music"
__conference__ = "ICASSP 2026"

# Package metadata
__all__ = [
    'frontends',
    'models', 
    'datasets',
    'bias_evaluation'
]
