"""
FairAudioBench Experiment Configuration
Centralized configuration for all benchmark experiments
"""

import torch

# Global experiment configuration
EXPERIMENT_CONFIG = {
    # Audio processing parameters
    'sample_rate': 16000,
    'segment_length': 3.0,  # seconds
    'hop_length': 512,
    'n_fft': 2048,
    'n_mels': 128,
    'n_erbs': 128,
    'n_barks': 128,
    'n_cqt': 128,
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'epochs': 50,
    'early_stopping_patience': 10,
    'validation_split': 0.15,
    'test_split': 0.15,
    
    # Model parameters
    'hidden_dim': 256,
    'num_layers': 3,
    'dropout': 0.3,
    
    # Data augmentation
    'use_augmentation': True,
    'noise_factor': 0.1,
    'time_shift_factor': 0.1,
    'pitch_shift_factor': 2,
    
    # Experimental setup
    'random_seed': 42,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': True,
    
    # Bias evaluation
    'fairness_metrics': [
        'group_gap',
        'equalized_odds_gap', 
        'demographic_parity',
        'individual_fairness'
    ],
    
    # Logging and saving
    'save_model_checkpoints': True,
    'save_predictions': True,
    'save_embeddings': False,
    'log_frequency': 10,
    
    # Dataset specific configurations
    'speech_config': {
        'languages': [
            'english', 'spanish', 'french', 'german', 'italian',
            'portuguese', 'russian', 'chinese', 'japanese', 'arabic'
        ],
        'min_duration': 1.0,
        'max_duration': 10.0,
        'task': 'language_id',
        'num_classes': 10
    },
    
    'music_config': {
        'traditions': [
            'gtzan', 'fma', 'carnatic', 'turkish_makam',
            'hindustani', 'arab_andalusian'
        ],
        'min_duration': 3.0,
        'max_duration': 30.0,
        'task': 'genre_classification',
        'num_classes': {
            'gtzan': 10,
            'fma': 16,
            'carnatic': 8,
            'turkish_makam': 6,
            'hindustani': 10,
            'arab_andalusian': 5
        }
    },
    
    'scenes_config': {
        'datasets': ['tau_urban'],
        'min_duration': 3.0,
        'max_duration': 10.0,
        'task': 'scene_classification',
        'num_classes': 10,
        'cities': [
            'barcelona', 'helsinki', 'london', 'lyon', 
            'milan', 'paris', 'prague', 'stockholm', 'vienna'
        ]
    }
}

# Frontend specific configurations
FRONTEND_CONFIGS = {
    'mel': {
        'n_mels': 128,
        'fmin': 0,
        'fmax': 8000,
        'power': 2.0
    },
    
    'erb': {
        'n_filters': 128,
        'fmin': 50,
        'fmax': 8000,
        'trainable': False
    },
    
    'bark': {
        'n_filters': 128, 
        'fmin': 50,
        'fmax': 8000,
        'trainable': False
    },
    
    'cqt': {
        'hop_length': 512,
        'fmin': 55,  # C2
        'n_bins': 128,
        'bins_per_octave': 24
    },
    
    'leaf': {
        'n_filters': 128,
        'sample_rate': 16000,
        'window_len': 25.0,  # ms
        'window_stride': 10.0,  # ms
        'compression_fn': 'pcen',
        'trainable': True
    },
    
    'sincnet': {
        'out_channels': 128,
        'kernel_size': 251,
        'stride': 1,
        'padding': 125,
        'sample_rate': 16000,
        'min_low_hz': 50,
        'min_band_hz': 50,
        'trainable': True
    }
}

# Model architectures for each domain
MODEL_CONFIGS = {
    'speech': {
        'type': 'transformer',
        'input_dim': 128,  # Will be set based on frontend
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1,
        'num_classes': 10  # Number of languages
    },
    
    'music': {
        'type': 'cnn_lstm',
        'input_dim': 128,  # Will be set based on frontend
        'conv_channels': [64, 128, 256],
        'conv_kernels': [3, 3, 3],
        'lstm_hidden': 256,
        'lstm_layers': 2,
        'dropout': 0.3,
        'num_classes': 10  # Will be set based on dataset
    },
    
    'scenes': {
        'type': 'resnet',
        'input_dim': 128,  # Will be set based on frontend
        'base_channels': 64,
        'num_blocks': [2, 2, 2, 2],
        'dropout': 0.2,
        'num_classes': 10  # Number of scene classes
    }
}

# Evaluation and analysis settings
EVALUATION_CONFIG = {
    'metrics': [
        'accuracy',
        'precision',
        'recall', 
        'f1_score',
        'confusion_matrix'
    ],
    
    'bias_analysis': {
        'protected_attributes': {
            'speech': 'language_family',
            'music': 'cultural_tradition',
            'scenes': 'recording_city'
        },
        'fairness_thresholds': {
            'group_gap': 0.05,
            'equalized_odds_gap': 0.1,
            'demographic_parity': 0.1
        }
    },
    
    'visualization': {
        'save_plots': True,
        'plot_formats': ['png', 'pdf'],
        'dpi': 300,
        'figsize': (10, 8)
    },
    
    'statistical_tests': {
        'significance_level': 0.05,
        'multiple_testing_correction': 'bonferroni',
        'effect_size_measures': ['cohens_d', 'eta_squared']
    }
}

# Results export settings
EXPORT_CONFIG = {
    'formats': ['csv', 'json', 'latex'],
    'tables': {
        'main_results': True,
        'bias_analysis': True,
        'statistical_tests': True,
        'ablation_studies': True
    },
    'figures': {
        'performance_comparison': True,
        'bias_visualization': True,
        'confusion_matrices': True,
        'embedding_visualization': False
    }
}

def get_config():
    """Get the complete configuration dictionary"""
    return {
        'experiment': EXPERIMENT_CONFIG,
        'frontends': FRONTEND_CONFIGS,
        'models': MODEL_CONFIGS,
        'evaluation': EVALUATION_CONFIG,
        'export': EXPORT_CONFIG
    }

def get_domain_config(domain):
    """Get configuration for a specific domain"""
    if domain not in ['speech', 'music', 'scenes']:
        raise ValueError(f"Unknown domain: {domain}")
    
    config = get_config()
    domain_key = f"{domain}_config"
    
    return {
        'general': config['experiment'],
        'domain_specific': config['experiment'][domain_key],
        'model': config['models'][domain],
        'evaluation': config['evaluation'],
        'export': config['export']
    }

def get_frontend_config(frontend_name):
    """Get configuration for a specific frontend"""
    config = get_config()
    if frontend_name not in config['frontends']:
        raise ValueError(f"Unknown frontend: {frontend_name}")
    
    return config['frontends'][frontend_name]
