"""
Experiment Configuration for Cross-Cultural Bias Research
ICASSP 2026 Paper

This file contains all configuration parameters for reproducing
the paper's experimental results.
"""

# Dataset configuration
DATASET_CONFIG = {
    "speech": {
        "tonal_languages": ["vi", "th", "yue", "pa-IN"],
        "non_tonal_languages": ["en", "es", "de", "fr", "it", "nl"],
        "target_samples_per_language": 2000,
        "sample_rate": 22050,
        "target_duration": 4.2,  # seconds
        "min_duration": 1.0,
        "max_duration": 8.0
    },
    "music": {
        "western_traditions": ["gtzan", "fma"],
        "non_western_traditions": ["carnatic", "hindustani", "turkish_makam", "arab_andalusian"],
        "target_samples_per_tradition": 300,
        "sample_rate": 22050,
        "segment_duration": 30.0,  # seconds
        "min_samples_per_class": 10
    },
    "scenes": {
        "dataset": "tau_urban",
        "cities": ["barcelona", "helsinki", "london", "paris", "stockholm", 
                  "vienna", "amsterdam", "lisbon", "lyon", "prague"],
        "scene_types": ["airport", "bus", "metro", "metro_station", "park",
                       "public_square", "shopping_mall", "street_pedestrian", 
                       "street_traffic", "tram"],
        "target_samples_per_city": 100,
        "sample_rate": 22050,  # Converted from original 48kHz
        "segment_duration": 10.0,  # seconds
        "samples_per_scene": 10
    }
}

# Audio frontend configuration
FRONTEND_CONFIG = {
    "mel": {
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
        "f_min": 0.0,
        "f_max": None,  # Will use Nyquist frequency
        "power": 2.0,
        "normalized": True
    },
    "erb": {
        "n_fft": 2048,
        "hop_length": 512,
        "n_filters": 128,
        "f_min": 50.0,
        "f_max": None,  # Will use Nyquist frequency
        "normalized": True
    },
    "bark": {
        "n_fft": 2048,
        "hop_length": 512,
        "n_filters": 128,
        "f_min": 50.0,
        "f_max": None,
        "normalized": True
    },
    "cqt": {
        "hop_length": 512,
        "n_bins": 128,
        "f_min": 32.7,  # C1 note
        "bins_per_octave": 12,
        "normalized": True
    },
    "leaf": {
        "n_filters": 128,
        "window_len": 401,
        "window_stride": 160,
        "learnable": True,
        "initialization": "mel_scale"
    },
    "sincnet": {
        "n_filters": 128,
        "kernel_size": 251,
        "stride": 160,
        "learnable_cutoffs": True,
        "initialization": "mel_scale"
    }
}

# Model architecture configuration
MODEL_CONFIG = {
    "speech": {
        "architecture": "CNN-RNN",
        "cnn_layers": [
            {"filters": 32, "kernel": (3, 3), "pool": (2, 2)},
            {"filters": 64, "kernel": (3, 3), "pool": (2, 2)},
            {"filters": 128, "kernel": (3, 3), "pool": (2, 2)},
            {"filters": 128, "kernel": (3, 3), "pool": (1, 1)}
        ],
        "rnn_type": "LSTM",
        "rnn_hidden_dim": 256,
        "rnn_num_layers": 3,
        "rnn_bidirectional": True,
        "dropout": 0.1,
        "loss_function": "CTC"
    },
    "music": {
        "architecture": "CNN-Attention",
        "cnn_blocks": [
            {"filters": 64, "layers": 2, "pool": (2, 2)},
            {"filters": 128, "layers": 2, "pool": (2, 2)},
            {"filters": 256, "layers": 2, "pool": (2, 2)},
            {"filters": 512, "layers": 2, "pool": "adaptive"}
        ],
        "attention_heads": 8,
        "attention_dim": 512,
        "classifier_hidden": [256, 128],
        "dropout": 0.1,
        "loss_function": "CrossEntropy"
    },
    "scenes": {
        "architecture": "CNN-SpatialAttention",
        "cnn_blocks": [
            {"filters": 32, "pool": (2, 2)},
            {"filters": 64, "pool": (2, 2)},
            {"filters": 128, "pool": (2, 2)},
            {"filters": 256, "pool": "adaptive"}
        ],
        "spatial_attention": True,
        "classifier_hidden": [128, 64],
        "dropout": 0.1,
        "loss_function": "CrossEntropy"
    }
}

# Training configuration
TRAINING_CONFIG = {
    "general": {
        "batch_size": 32,
        "num_epochs": 50,
        "early_stopping_patience": 10,
        "validation_split": 0.15,
        "test_split": 0.15,
        "random_seed": 42
    },
    "optimization": {
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience": 5,
        "scheduler_factor": 0.5,
        "gradient_clipping": 1.0
    },
    "regularization": {
        "dropout": 0.1,
        "batch_norm": True,
        "data_augmentation": {
            "speech": ["time_stretch", "pitch_shift", "add_noise"],
            "music": ["time_stretch", "pitch_shift", "add_noise", "spec_augment"],
            "scenes": ["time_shift", "add_noise", "spec_augment"]
        }
    }
}

# Evaluation metrics configuration
EVALUATION_CONFIG = {
    "speech": {
        "primary_metric": "Character Error Rate (CER)",
        "secondary_metrics": ["Word Error Rate (WER)", "BLEU Score"],
        "evaluation_method": "CTC_decode",
        "beam_width": 10
    },
    "music": {
        "primary_metric": "Macro F1 Score",
        "secondary_metrics": ["Accuracy", "Precision", "Recall", "Confusion Matrix"],
        "evaluation_method": "classification",
        "average": "macro"
    },
    "scenes": {
        "primary_metric": "Accuracy",
        "secondary_metrics": ["F1 Score", "Precision", "Recall", "Geographic Variance"],
        "evaluation_method": "classification",
        "analyze_by": ["city", "scene_type"]
    }
}

# Bias analysis configuration
BIAS_ANALYSIS_CONFIG = {
    "speech_bias": {
        "comparison_groups": {
            "tonal": ["vi", "th", "yue", "pa-IN"],
            "non_tonal": ["en", "es", "de", "fr", "it", "nl"]
        },
        "bias_metrics": [
            "performance_gap_absolute",
            "performance_gap_relative", 
            "statistical_significance",
            "effect_size_cohens_d"
        ],
        "significance_threshold": 0.05
    },
    "music_bias": {
        "comparison_groups": {
            "western": ["gtzan", "fma"],
            "non_western": ["carnatic", "hindustani", "turkish_makam", "arab_andalusian"]
        },
        "bias_metrics": [
            "f1_score_gap",
            "classification_accuracy_gap",
            "statistical_significance",
            "effect_size_cohens_d"
        ],
        "significance_threshold": 0.05
    },
    "geographic_bias": {
        "analysis_type": "variance_analysis",
        "metrics": [
            "inter_city_variance",
            "coefficient_of_variation",
            "performance_range",
            "outlier_detection"
        ]
    },
    "mitigation_analysis": {
        "baseline_frontend": "mel",
        "alternative_frontends": ["erb", "bark", "cqt", "leaf", "sincnet"],
        "effectiveness_thresholds": {
            "highly_effective": 0.50,  # >50% bias reduction
            "moderately_effective": 0.25,  # 25-50% bias reduction
            "mildly_effective": 0.10,  # 10-25% bias reduction
            "minimal": 0.01  # <10% bias reduction
        }
    }
}

# Expected results (for validation)
EXPECTED_RESULTS = {
    "speech_bias": {
        "mel_frontend": {
            "tonal_languages_avg_cer": 0.312,  # ~31.2% average CER
            "non_tonal_languages_avg_cer": 0.187,  # ~18.7% average CER
            "performance_gap_percent": 12.5  # ~12.5% relative gap
        },
        "erb_frontend": {
            "bias_reduction_percent": 29.8,  # ~30% bias reduction
            "tonal_languages_improvement": 0.093  # ~9.3 percentage points
        },
        "leaf_frontend": {
            "bias_reduction_percent": 25.0,  # ~25% bias reduction
            "non_tonal_languages_improvement": 0.015  # ~1.5 percentage points
        }
    },
    "music_bias": {
        "mel_frontend": {
            "western_traditions_avg_f1": 0.84,  # ~84% average F1
            "non_western_traditions_avg_f1": 0.71,  # ~71% average F1
            "performance_gap_percent": 15.7  # ~15.7% F1 degradation
        },
        "erb_frontend": {
            "bias_reduction_percent": 30.0,  # ~30% bias reduction
            "non_western_improvement": 0.07  # ~7 F1 points improvement
        },
        "leaf_frontend": {
            "bias_reduction_percent": 40.0,  # ~40% bias reduction
            "overall_improvement": 0.05  # ~5 F1 points overall
        }
    }
}

# Computational requirements
COMPUTATIONAL_CONFIG = {
    "hardware_requirements": {
        "minimum_gpu_memory": "8GB",
        "recommended_gpu_memory": "16GB", 
        "minimum_ram": "16GB",
        "recommended_ram": "32GB",
        "storage_requirements": "150GB"  # Including datasets and results
    },
    "estimated_runtime": {
        "data_download": "2-4 hours",
        "data_preprocessing": "1-2 hours",
        "single_frontend_speech": "4-6 hours",
        "single_frontend_music": "2-3 hours", 
        "single_frontend_scenes": "1-2 hours",
        "complete_experiment_suite": "24-36 hours",
        "bias_analysis": "30 minutes"
    },
    "parallelization": {
        "max_num_workers": 8,
        "gpu_utilization_target": 0.85,
        "batch_size_scaling": "auto",
        "distributed_training": False
    }
}

# Reproducibility configuration
REPRODUCIBILITY_CONFIG = {
    "random_seeds": {
        "data_splitting": 42,
        "model_initialization": 42,
        "training_shuffle": 42,
        "augmentation": 42
    },
    "deterministic_training": True,
    "version_requirements": {
        "python": ">=3.8",
        "torch": ">=1.12.0",
        "torchaudio": ">=0.12.0",
        "librosa": ">=0.10.0",
        "numpy": ">=1.21.0",
        "pandas": ">=1.4.0",
        "scikit_learn": ">=1.1.0"
    },
    "checkpoints": {
        "save_best_model": True,
        "save_last_model": True,
        "save_intermediate_epochs": [10, 25, 40],
        "checkpoint_format": "pytorch"
    }
}

# Output configuration
OUTPUT_CONFIG = {
    "results_directory": "results",
    "subdirectories": [
        "models",
        "logs", 
        "plots",
        "tables",
        "bias_analysis",
        "checkpoints"
    ],
    "file_formats": {
        "results_tables": ["csv", "latex", "json"],
        "plots": ["png", "pdf", "svg"],
        "logs": ["txt", "json"],
        "models": ["pth", "onnx"]
    },
    "paper_ready_outputs": {
        "table_1_speech_results": "tables/speech_results_by_language_and_frontend.tex",
        "table_2_music_results": "tables/music_results_by_tradition_and_frontend.tex", 
        "table_3_scene_results": "tables/scene_results_by_city_and_frontend.tex",
        "table_4_bias_summary": "tables/bias_analysis_summary.tex",
        "figure_1_bias_visualization": "plots/bias_comparison_frontends.pdf",
        "figure_2_performance_gaps": "plots/performance_gaps_by_domain.pdf",
        "figure_3_mitigation_effectiveness": "plots/mitigation_effectiveness_comparison.pdf"
    }
}

# Validation and quality assurance
VALIDATION_CONFIG = {
    "data_quality_checks": [
        "audio_file_integrity",
        "metadata_completeness",
        "label_distribution_balance",
        "duration_distribution_analysis",
        "sample_rate_consistency"
    ],
    "model_validation": [
        "training_convergence_check",
        "overfitting_detection",
        "gradient_flow_analysis",
        "activation_distribution_check"
    ],
    "result_validation": [
        "statistical_significance_testing",
        "confidence_interval_calculation",
        "cross_validation_consistency",
        "baseline_comparison_sanity_check"
    ]
}
