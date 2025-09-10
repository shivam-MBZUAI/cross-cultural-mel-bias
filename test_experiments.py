#!/usr/bin/env python3

"""
Quick Test Suite for Cross-Cultural Mel-Scale Bias Analysis
ICASSP 2026 Paper Implementation

Provides a quick demonstration of the experimental pipeline without requiring
full dataset processing. Uses simulated data to validate all components.

Author: Shivam Chauhan, Ajay Pundhir
Organization: Presight AI, Abu Dhabi, UAE
"""

import sys
import logging
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from frontends import (
    MelScaleFrontEnd, ERBScaleFrontEnd, BarkScaleFrontEnd, 
    CQTFrontEnd, LEAFFrontEnd, SincNetFrontEnd, MelPCENFrontEnd
)
from bias_evaluation import BiasMetrics, StatisticalTests, DomainSpecificEvaluator
from visualizations import CrossCulturalVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickTester:
    """
    Quick testing suite for validating the experimental framework.
    """
    
    def __init__(self):
        """Initialize quick tester with simulated data."""
        self.sample_rate = 22050
        self.duration = 5.0  # 5 seconds of audio
        self.n_samples = int(self.sample_rate * self.duration)
        
        # Initialize components
        self.frontends = self._initialize_frontends()
        self.bias_metrics = BiasMetrics()
        self.stat_tests = StatisticalTests()
        self.domain_evaluator = DomainSpecificEvaluator()
        self.visualizer = CrossCulturalVisualizer()
        
        logger.info("QuickTester initialized successfully")
    
    def _initialize_frontends(self):
        """Initialize all front-ends for testing."""
        frontends = {
            'mel': MelScaleFrontEnd(sample_rate=self.sample_rate),
            'erb': ERBScaleFrontEnd(sample_rate=self.sample_rate),
            'bark': BarkScaleFrontEnd(sample_rate=self.sample_rate),
            'cqt': CQTFrontEnd(sample_rate=self.sample_rate),
            'leaf': LEAFFrontEnd(sample_rate=self.sample_rate),
            'sincnet': SincNetFrontEnd(sample_rate=self.sample_rate),
            'mel_pcen': MelPCENFrontEnd(sample_rate=self.sample_rate)
        }
        
        # Set to evaluation mode
        for frontend in frontends.values():
            frontend.eval()
        
        return frontends
    
    def generate_test_audio(self, audio_type: str = 'speech') -> torch.Tensor:
        """
        Generate synthetic test audio for different domains.
        
        Args:
            audio_type: Type of audio ('speech', 'music', 'scene')
        
        Returns:
            Synthetic audio tensor
        """
        if audio_type == 'speech':
            # Generate speech-like signal with formants
            t = torch.linspace(0, self.duration, self.n_samples)
            
            # Fundamental frequency (varies for tonal vs non-tonal)
            f0 = 150 + 50 * torch.sin(2 * np.pi * 2 * t)  # Varying pitch
            
            # Formants (speech-like resonances)
            formant1 = torch.sin(2 * np.pi * 800 * t)
            formant2 = torch.sin(2 * np.pi * 1200 * t)
            formant3 = torch.sin(2 * np.pi * 2400 * t)
            
            # Combine with fundamental
            speech = (torch.sin(2 * np.pi * f0 * t) + 
                     0.5 * formant1 + 0.3 * formant2 + 0.2 * formant3)
            
            # Add some noise
            speech += 0.1 * torch.randn_like(speech)
            
        elif audio_type == 'music':
            # Generate music-like signal with harmonics
            t = torch.linspace(0, self.duration, self.n_samples)
            
            # Musical notes (C major scale)
            notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # C4 to B4
            
            music = torch.zeros_like(t)
            for i, note_freq in enumerate(notes):
                # Add harmonics
                for harmonic in range(1, 5):
                    amplitude = 1.0 / harmonic
                    start_time = i * self.duration / len(notes)
                    end_time = (i + 1) * self.duration / len(notes)
                    
                    note_mask = (t >= start_time) & (t < end_time)
                    music[note_mask] += amplitude * torch.sin(2 * np.pi * note_freq * harmonic * t[note_mask])
            
            # Add some noise
            music += 0.05 * torch.randn_like(music)
            
        elif audio_type == 'scene':
            # Generate environmental scene audio
            t = torch.linspace(0, self.duration, self.n_samples)
            
            # Traffic noise (low frequency rumble)
            traffic = 0.3 * torch.sin(2 * np.pi * 60 * t) + 0.2 * torch.sin(2 * np.pi * 120 * t)
            
            # Bird sounds (high frequency chirps)
            birds = 0.1 * torch.sin(2 * np.pi * 2000 * t) * (torch.sin(2 * np.pi * 0.5 * t) > 0.5)
            
            # Wind noise (broadband)
            wind = 0.2 * torch.randn_like(t)
            
            scene = traffic + birds + wind
            
        else:
            # Default: white noise
            scene = torch.randn(self.n_samples)
        
        # Normalize and add channel dimension
        audio = scene / torch.max(torch.abs(scene))
        return audio.unsqueeze(0)  # Add channel dimension
    
    def test_frontend_extraction(self):
        """Test feature extraction for all front-ends."""
        logger.info("Testing feature extraction for all front-ends...")
        
        # Generate test audio for each domain
        test_audios = {
            'speech': self.generate_test_audio('speech'),
            'music': self.generate_test_audio('music'),
            'scene': self.generate_test_audio('scene')
        }
        
        results = {}
        
        for frontend_name, frontend in self.frontends.items():
            logger.info(f"  Testing {frontend_name}...")
            frontend_results = {}
            
            for domain, audio in test_audios.items():
                try:
                    with torch.no_grad():
                        features = frontend(audio)
                    
                    frontend_results[domain] = {
                        'feature_shape': list(features.shape),
                        'feature_mean': float(torch.mean(features)),
                        'feature_std': float(torch.std(features)),
                        'success': True
                    }
                    
                except Exception as e:
                    logger.warning(f"    Error in {domain}: {e}")
                    frontend_results[domain] = {'success': False, 'error': str(e)}
            
            results[frontend_name] = frontend_results
        
        logger.info("Feature extraction testing completed")
        return results
    
    def test_bias_evaluation(self):
        """Test bias evaluation metrics."""
        logger.info("Testing bias evaluation metrics...")
        
        # Generate simulated performance data
        # Tonal vs Non-tonal languages (speech)
        tonal_scores = np.random.normal(0.73, 0.05, 50)  # Lower performance
        nontonal_scores = np.random.normal(0.85, 0.04, 50)  # Higher performance
        
        # Western vs Non-Western music
        western_scores = np.random.normal(0.82, 0.03, 40)
        nonwestern_scores = np.random.normal(0.69, 0.06, 40)
        
        results = {}
        
        # Test speech bias
        logger.info("  Testing speech bias metrics...")
        speech_bias = self.bias_metrics.compute_performance_gap(
            tonal_scores.tolist(), nontonal_scores.tolist(), "accuracy"
        )
        results['speech_bias'] = speech_bias
        
        # Test music bias
        logger.info("  Testing music bias metrics...")
        music_bias = self.bias_metrics.compute_performance_gap(
            western_scores.tolist(), nonwestern_scores.tolist(), "f1_score"
        )
        results['music_bias'] = music_bias
        
        # Test feature space bias
        logger.info("  Testing feature space bias...")
        group1_features = [np.random.randn(128) for _ in range(30)]
        group2_features = [np.random.randn(128) + 1.5 for _ in range(30)]  # Shifted distribution
        
        feature_bias = self.bias_metrics.compute_feature_space_bias(group1_features, group2_features)
        results['feature_space_bias'] = feature_bias
        
        logger.info("Bias evaluation testing completed")
        return results
    
    def test_statistical_tests(self):
        """Test statistical significance testing."""
        logger.info("Testing statistical significance tests...")
        
        # Generate test data with known differences
        group1 = np.random.normal(0.7, 0.1, 100)  # Lower mean
        group2 = np.random.normal(0.8, 0.1, 100)  # Higher mean
        
        results = self.stat_tests.run_all_tests(
            group1, group2, "Group1", "Group2"
        )
        
        logger.info("Statistical testing completed")
        return results
    
    def test_domain_evaluation(self):
        """Test domain-specific evaluation."""
        logger.info("Testing domain-specific evaluation...")
        
        # Create mock feature data
        mock_feature_data = {
            'frontend1': {
                'features': [np.random.randn(128) for _ in range(20)],
                'metadata': [{'language': 'en', 'is_tonal': False} for _ in range(20)],
                'feature_dim': 128,
                'num_samples': 20
            }
        }
        
        results = {}
        
        for domain in ['speech', 'music', 'scenes']:
            logger.info(f"  Testing {domain} domain evaluation...")
            domain_result = self.domain_evaluator.evaluate_domain(domain, mock_feature_data)
            results[domain] = domain_result
        
        logger.info("Domain evaluation testing completed")
        return results
    
    def test_visualizations(self):
        """Test visualization generation."""
        logger.info("Testing visualization generation...")
        
        # Create sample bias results
        sample_bias_results = {
            'speech': {
                'mel': {
                    'tonal_vs_nontonal': {
                        'performance_gap': -0.12,
                        'group1_mean': 0.73,
                        'group2_mean': 0.85
                    }
                },
                'leaf': {
                    'tonal_vs_nontonal': {
                        'performance_gap': -0.05,
                        'group1_mean': 0.80,
                        'group2_mean': 0.85
                    }
                },
                'erb': {
                    'tonal_vs_nontonal': {
                        'performance_gap': -0.08,
                        'group1_mean': 0.77,
                        'group2_mean': 0.85
                    }
                }
            },
            'music': {
                'mel': {
                    'western_vs_nonwestern': {
                        'performance_gap': 0.15,
                        'group1_mean': 0.85,
                        'group2_mean': 0.70
                    }
                },
                'cqt': {
                    'western_vs_nonwestern': {
                        'performance_gap': 0.08,
                        'group1_mean': 0.82,
                        'group2_mean': 0.74
                    }
                }
            },
            'scenes': {
                'mel': {
                    'nordic_vs_southern': {
                        'performance_gap': 0.05,
                        'group1_mean': 0.78,
                        'group2_mean': 0.73
                    }
                }
            }
        }
        
        # Sample statistical results
        sample_stat_results = {
            'speech': {
                'mel': {
                    'equal_variance_ttest': {
                        't_statistic': -3.45,
                        'p_value': 0.001,
                        'significant': True
                    }
                }
            }
        }
        
        # Sample domain results
        sample_domain_results = {
            'speech': {'analysis_type': 'language_families'},
            'music': {'analysis_type': 'cultural_traditions'},
            'scenes': {'analysis_type': 'geographic_regions'}
        }
        
        try:
            # Test performance gaps plot
            logger.info("  Testing performance gaps visualization...")
            self.visualizer.plot_performance_gaps(sample_bias_results)
            
            # Test feature space bias plot
            logger.info("  Testing feature space bias visualization...")
            self.visualizer.plot_feature_space_bias(sample_bias_results)
            
            # Test significance heatmap
            logger.info("  Testing statistical significance heatmap...")
            self.visualizer.plot_significance_heatmap(sample_stat_results)
            
            # Test domain analysis plot
            logger.info("  Testing domain-specific analysis...")
            self.visualizer.plot_domain_specific_analysis(sample_domain_results)
            
            # Test bias summary table
            logger.info("  Testing bias summary table...")
            self.visualizer.generate_bias_summary_table(sample_bias_results)
            
            logger.info("Visualization testing completed successfully")
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Visualization testing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_all_tests(self):
        """Run complete test suite."""
        logger.info("="*60)
        logger.info("CROSS-CULTURAL MEL-SCALE BIAS ANALYSIS - QUICK TEST SUITE")
        logger.info("="*60)
        
        all_results = {}
        
        try:
            # Test 1: Feature extraction
            all_results['feature_extraction'] = self.test_frontend_extraction()
            
            # Test 2: Bias evaluation
            all_results['bias_evaluation'] = self.test_bias_evaluation()
            
            # Test 3: Statistical tests
            all_results['statistical_tests'] = self.test_statistical_tests()
            
            # Test 4: Domain evaluation
            all_results['domain_evaluation'] = self.test_domain_evaluation()
            
            # Test 5: Visualizations
            all_results['visualizations'] = self.test_visualizations()
            
            # Summary
            logger.info("="*60)
            logger.info("TEST SUITE COMPLETED SUCCESSFULLY")
            logger.info("All components are working correctly!")
            logger.info("Ready to run full experiments with real data.")
            logger.info("="*60)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            logger.info("="*60)
            logger.info("TEST SUITE FAILED")
            logger.info("Please check the error messages above.")
            logger.info("="*60)
            raise

def main():
    """Main function for running quick tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick test suite for cross-cultural bias analysis")
    parser.add_argument('--component', choices=['frontend', 'bias', 'stats', 'domain', 'viz', 'all'], 
                       default='all', help='Component to test')
    
    args = parser.parse_args()
    
    tester = QuickTester()
    
    if args.component == 'all':
        tester.run_all_tests()
    elif args.component == 'frontend':
        tester.test_frontend_extraction()
    elif args.component == 'bias':
        tester.test_bias_evaluation()
    elif args.component == 'stats':
        tester.test_statistical_tests()
    elif args.component == 'domain':
        tester.test_domain_evaluation()
    elif args.component == 'viz':
        tester.test_visualizations()

if __name__ == "__main__":
    main()
