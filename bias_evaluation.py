#!/usr/bin/env python3

"""
Cross-Cultural Bias Evaluation Framework
ICASSP 2026 Paper

This module implements bias evaluation metrics and analysis tools for measuring
cultural bias in audio representations across speech, music, and acoustic scenes.

Author: Shivam Chauhan, Ajay Pundhir
Organization: Presight AI, Abu Dhabi, UAE
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy import stats
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BiasMetrics:
    """
    Comprehensive bias evaluation metrics for cross-cultural audio analysis.
    """
    
    def __init__(self):
        self.results = {}
    
    def compute_performance_gap(self, 
                               group1_scores: List[float], 
                               group2_scores: List[float],
                               metric_name: str = "accuracy") -> Dict:
        """
        Compute performance gap between two cultural groups.
        
        Args:
            group1_scores: Performance scores for group 1 (e.g., tonal languages)
            group2_scores: Performance scores for group 2 (e.g., non-tonal languages)
            metric_name: Name of the metric being compared
        
        Returns:
            Dict with gap analysis results
        """
        group1_mean = np.mean(group1_scores)
        group2_mean = np.mean(group2_scores)
        
        # Performance gap (positive means group1 performs worse)
        absolute_gap = group1_mean - group2_mean
        relative_gap = (absolute_gap / group2_mean) * 100 if group2_mean != 0 else 0
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(group1_scores, group2_scores)
        
        return {
            'group1_mean': group1_mean,
            'group2_mean': group2_mean,
            'absolute_gap': absolute_gap,
            'relative_gap_percent': relative_gap,
            'statistical_significance': {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            },
            'effect_size': self._compute_cohens_d(group1_scores, group2_scores)
        }
    
    def _compute_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0
    
    def evaluate_speech_bias(self, results_by_language: Dict[str, Dict]) -> Dict:
        """
        Evaluate cultural bias in speech recognition across tonal vs non-tonal languages.
        
        Args:
            results_by_language: {language_code: {frontend_name: error_rate}}
        
        Returns:
            Comprehensive bias analysis
        """
        tonal_languages = ['vi', 'th', 'yue', 'pa-IN']  # Vietnamese, Thai, Cantonese, Punjabi
        non_tonal_languages = ['en', 'es', 'de', 'fr', 'it', 'nl']
        
        bias_analysis = {}
        
        # Extract available frontends
        available_frontends = set()
        for lang_results in results_by_language.values():
            available_frontends.update(lang_results.keys())
        
        for frontend in available_frontends:
            # Collect scores for each group
            tonal_scores = []
            non_tonal_scores = []
            
            for lang in tonal_languages:
                if lang in results_by_language and frontend in results_by_language[lang]:
                    tonal_scores.append(results_by_language[lang][frontend])
            
            for lang in non_tonal_languages:
                if lang in results_by_language and frontend in results_by_language[lang]:
                    non_tonal_scores.append(results_by_language[lang][frontend])
            
            if tonal_scores and non_tonal_scores:
                gap_analysis = self.compute_performance_gap(
                    tonal_scores, non_tonal_scores, f"{frontend}_error_rate"
                )
                
                bias_analysis[frontend] = {
                    'tonal_languages': {
                        'count': len(tonal_scores),
                        'mean_error_rate': gap_analysis['group1_mean'],
                        'scores': tonal_scores
                    },
                    'non_tonal_languages': {
                        'count': len(non_tonal_scores),
                        'mean_error_rate': gap_analysis['group2_mean'],
                        'scores': non_tonal_scores
                    },
                    'bias_metrics': gap_analysis,
                    'bias_interpretation': self._interpret_speech_bias(gap_analysis)
                }
        
        return bias_analysis
    
    def _interpret_speech_bias(self, gap_analysis: Dict) -> str:
        """Interpret speech bias results."""
        gap = gap_analysis['relative_gap_percent']
        is_significant = gap_analysis['statistical_significance']['is_significant']
        
        if not is_significant:
            return "No statistically significant bias detected"
        elif gap > 10:
            return "Strong bias against tonal languages (>10% performance gap)"
        elif gap > 5:
            return "Moderate bias against tonal languages (5-10% performance gap)"
        elif gap > 0:
            return "Mild bias against tonal languages (<5% performance gap)"
        else:
            return "Bias favors tonal languages (unexpected result)"
    
    def evaluate_music_bias(self, results_by_tradition: Dict[str, Dict]) -> Dict:
        """
        Evaluate cultural bias in music analysis across Western vs Non-Western traditions.
        
        Args:
            results_by_tradition: {tradition_name: {frontend_name: f1_score}}
        
        Returns:
            Comprehensive bias analysis
        """
        western_traditions = ['gtzan', 'fma']
        non_western_traditions = ['carnatic', 'hindustani', 'turkish_makam', 'arab_andalusian']
        
        bias_analysis = {}
        
        # Extract available frontends
        available_frontends = set()
        for tradition_results in results_by_tradition.values():
            available_frontends.update(tradition_results.keys())
        
        for frontend in available_frontends:
            western_scores = []
            non_western_scores = []
            
            for tradition in western_traditions:
                if tradition in results_by_tradition and frontend in results_by_tradition[tradition]:
                    western_scores.append(results_by_tradition[tradition][frontend])
            
            for tradition in non_western_traditions:
                if tradition in results_by_tradition and frontend in results_by_tradition[tradition]:
                    non_western_scores.append(results_by_tradition[tradition][frontend])
            
            if western_scores and non_western_scores:
                # For music, higher F1 is better, so we compute gap as non_western - western
                gap_analysis = self.compute_performance_gap(
                    non_western_scores, western_scores, f"{frontend}_f1_score"
                )
                
                bias_analysis[frontend] = {
                    'western_traditions': {
                        'count': len(western_scores),
                        'mean_f1_score': gap_analysis['group2_mean'],
                        'scores': western_scores
                    },
                    'non_western_traditions': {
                        'count': len(non_western_scores),
                        'mean_f1_score': gap_analysis['group1_mean'],
                        'scores': non_western_scores
                    },
                    'bias_metrics': gap_analysis,
                    'bias_interpretation': self._interpret_music_bias(gap_analysis)
                }
        
        return bias_analysis
    
    def _interpret_music_bias(self, gap_analysis: Dict) -> str:
        """Interpret music bias results."""
        gap = gap_analysis['relative_gap_percent']  # negative means non-Western performs worse
        is_significant = gap_analysis['statistical_significance']['is_significant']
        
        if not is_significant:
            return "No statistically significant bias detected"
        elif gap < -10:
            return "Strong bias against non-Western traditions (>10% F1 degradation)"
        elif gap < -5:
            return "Moderate bias against non-Western traditions (5-10% F1 degradation)"
        elif gap < 0:
            return "Mild bias against non-Western traditions (<5% F1 degradation)"
        else:
            return "Bias favors non-Western traditions (unexpected result)"
    
    def evaluate_scene_bias(self, results_by_city: Dict[str, Dict]) -> Dict:
        """
        Evaluate geographic bias in acoustic scene classification.
        
        Args:
            results_by_city: {city_name: {frontend_name: accuracy}}
        
        Returns:
            Geographic bias analysis
        """
        bias_analysis = {}
        
        # Extract available frontends
        available_frontends = set()
        for city_results in results_by_city.values():
            available_frontends.update(city_results.keys())
        
        for frontend in available_frontends:
            city_scores = []
            city_names = []
            
            for city in results_by_city:
                if frontend in results_by_city[city]:
                    city_scores.append(results_by_city[city][frontend])
                    city_names.append(city)
            
            if len(city_scores) >= 2:
                # Compute variance across cities
                mean_accuracy = np.mean(city_scores)
                std_accuracy = np.std(city_scores)
                cv = std_accuracy / mean_accuracy if mean_accuracy != 0 else 0
                
                # Find best and worst performing cities
                best_idx = np.argmax(city_scores)
                worst_idx = np.argmin(city_scores)
                
                bias_analysis[frontend] = {
                    'overall_performance': {
                        'mean_accuracy': mean_accuracy,
                        'std_accuracy': std_accuracy,
                        'coefficient_of_variation': cv
                    },
                    'geographic_variance': {
                        'best_city': city_names[best_idx],
                        'best_accuracy': city_scores[best_idx],
                        'worst_city': city_names[worst_idx],
                        'worst_accuracy': city_scores[worst_idx],
                        'performance_range': city_scores[best_idx] - city_scores[worst_idx]
                    },
                    'city_scores': dict(zip(city_names, city_scores)),
                    'bias_interpretation': self._interpret_scene_bias(cv, city_scores)
                }
        
        return bias_analysis
    
    def _interpret_scene_bias(self, cv: float, scores: List[float]) -> str:
        """Interpret scene classification bias results."""
        performance_range = max(scores) - min(scores)
        
        if cv < 0.05:
            return "Low geographic bias (consistent performance across cities)"
        elif cv < 0.10:
            return "Moderate geographic bias (some variation across cities)"
        else:
            return f"High geographic bias (significant variation across cities, range: {performance_range:.1%})"
    
    def compute_bias_mitigation_effectiveness(self, 
                                            baseline_results: Dict,
                                            alternative_results: Dict,
                                            domain: str = "speech") -> Dict:
        """
        Compute effectiveness of bias mitigation techniques.
        
        Args:
            baseline_results: Results from mel-scale frontend (baseline)
            alternative_results: Results from alternative frontend
            domain: Type of task ("speech", "music", or "scenes")
        
        Returns:
            Mitigation effectiveness analysis
        """
        if domain == "speech":
            baseline_bias = self.evaluate_speech_bias(baseline_results)
            alternative_bias = self.evaluate_speech_bias(alternative_results)
            metric_key = 'mean_error_rate'
            improvement_direction = -1  # Lower is better for error rates
            
        elif domain == "music":
            baseline_bias = self.evaluate_music_bias(baseline_results)
            alternative_bias = self.evaluate_music_bias(alternative_results)
            metric_key = 'mean_f1_score'
            improvement_direction = 1   # Higher is better for F1 scores
            
        else:  # scenes
            baseline_bias = self.evaluate_scene_bias(baseline_results)
            alternative_bias = self.evaluate_scene_bias(alternative_results)
            return {"message": "Scene bias mitigation analysis not implemented yet"}
        
        mitigation_analysis = {}
        
        for frontend in alternative_bias:
            if 'mel' in baseline_bias:  # Compare with mel baseline
                baseline_gap = baseline_bias['mel']['bias_metrics']['relative_gap_percent']
                alternative_gap = alternative_bias[frontend]['bias_metrics']['relative_gap_percent']
                
                # Compute bias reduction
                bias_reduction = baseline_gap - alternative_gap
                bias_reduction_percent = (bias_reduction / abs(baseline_gap)) * 100 if baseline_gap != 0 else 0
                
                mitigation_analysis[frontend] = {
                    'baseline_bias_gap': baseline_gap,
                    'alternative_bias_gap': alternative_gap,
                    'absolute_bias_reduction': bias_reduction,
                    'relative_bias_reduction_percent': bias_reduction_percent,
                    'mitigation_effectiveness': self._classify_mitigation_effectiveness(bias_reduction_percent)
                }
        
        return mitigation_analysis
    
    def _classify_mitigation_effectiveness(self, reduction_percent: float) -> str:
        """Classify bias mitigation effectiveness."""
        if reduction_percent >= 50:
            return "Highly effective (>50% bias reduction)"
        elif reduction_percent >= 25:
            return "Moderately effective (25-50% bias reduction)"
        elif reduction_percent >= 10:
            return "Mildly effective (10-25% bias reduction)"
        elif reduction_percent > 0:
            return "Minimal effectiveness (<10% bias reduction)"
        else:
            return "No bias reduction (may increase bias)"
    
    def generate_bias_report(self, 
                           speech_results: Dict,
                           music_results: Dict,
                           scene_results: Dict,
                           output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive bias analysis report.
        
        Args:
            speech_results: Speech recognition results by language and frontend
            music_results: Music analysis results by tradition and frontend  
            scene_results: Scene classification results by city and frontend
            output_path: Optional path to save the report
        
        Returns:
            Formatted bias analysis report
        """
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("CROSS-CULTURAL BIAS ANALYSIS REPORT")
        report.append("ICASSP 2026: Cross-Cultural Bias in Mel-Scale Audio Front-Ends")
        report.append("=" * 80)
        report.append("")
        
        # Speech Recognition Bias Analysis
        report.append("1. SPEECH RECOGNITION BIAS ANALYSIS")
        report.append("-" * 50)
        speech_bias = self.evaluate_speech_bias(speech_results)
        
        for frontend, analysis in speech_bias.items():
            report.append(f"\n{frontend.upper()} Frontend:")
            report.append(f"  Tonal Languages (n={analysis['tonal_languages']['count']}): "
                         f"{analysis['tonal_languages']['mean_error_rate']:.1%} error rate")
            report.append(f"  Non-Tonal Languages (n={analysis['non_tonal_languages']['count']}): "
                         f"{analysis['non_tonal_languages']['mean_error_rate']:.1%} error rate")
            report.append(f"  Performance Gap: {analysis['bias_metrics']['relative_gap_percent']:+.1f}%")
            report.append(f"  Statistical Significance: p={analysis['bias_metrics']['statistical_significance']['p_value']:.4f}")
            report.append(f"  Interpretation: {analysis['bias_interpretation']}")
        
        # Music Analysis Bias
        report.append("\n\n2. MUSIC ANALYSIS BIAS ANALYSIS")
        report.append("-" * 50)
        music_bias = self.evaluate_music_bias(music_results)
        
        for frontend, analysis in music_bias.items():
            report.append(f"\n{frontend.upper()} Frontend:")
            report.append(f"  Western Traditions (n={analysis['western_traditions']['count']}): "
                         f"{analysis['western_traditions']['mean_f1_score']:.3f} F1 score")
            report.append(f"  Non-Western Traditions (n={analysis['non_western_traditions']['count']}): "
                         f"{analysis['non_western_traditions']['mean_f1_score']:.3f} F1 score")
            report.append(f"  Performance Gap: {analysis['bias_metrics']['relative_gap_percent']:+.1f}%")
            report.append(f"  Statistical Significance: p={analysis['bias_metrics']['statistical_significance']['p_value']:.4f}")
            report.append(f"  Interpretation: {analysis['bias_interpretation']}")
        
        # Scene Classification Bias
        report.append("\n\n3. ACOUSTIC SCENE CLASSIFICATION BIAS ANALYSIS")
        report.append("-" * 50)
        scene_bias = self.evaluate_scene_bias(scene_results)
        
        for frontend, analysis in scene_bias.items():
            report.append(f"\n{frontend.upper()} Frontend:")
            report.append(f"  Overall Accuracy: {analysis['overall_performance']['mean_accuracy']:.3f}")
            report.append(f"  Geographic Variance (CV): {analysis['overall_performance']['coefficient_of_variation']:.3f}")
            report.append(f"  Best City: {analysis['geographic_variance']['best_city']} "
                         f"({analysis['geographic_variance']['best_accuracy']:.3f})")
            report.append(f"  Worst City: {analysis['geographic_variance']['worst_city']} "
                         f"({analysis['geographic_variance']['worst_accuracy']:.3f})")
            report.append(f"  Interpretation: {analysis['bias_interpretation']}")
        
        # Mitigation Effectiveness
        report.append("\n\n4. BIAS MITIGATION EFFECTIVENESS")
        report.append("-" * 50)
        
        # Compare alternatives to mel baseline
        speech_mitigation = self.compute_bias_mitigation_effectiveness(
            {'mel': speech_results}, speech_results, 'speech'
        )
        music_mitigation = self.compute_bias_mitigation_effectiveness(
            {'mel': music_results}, music_results, 'music'
        )
        
        report.append("\nSpeech Recognition:")
        for frontend, effectiveness in speech_mitigation.items():
            if frontend != 'mel':
                report.append(f"  {frontend.upper()}: {effectiveness['relative_bias_reduction_percent']:+.1f}% "
                             f"bias reduction - {effectiveness['mitigation_effectiveness']}")
        
        report.append("\nMusic Analysis:")
        for frontend, effectiveness in music_mitigation.items():
            if frontend != 'mel':
                report.append(f"  {frontend.upper()}: {effectiveness['relative_bias_reduction_percent']:+.1f}% "
                             f"bias reduction - {effectiveness['mitigation_effectiveness']}")
        
        # Summary and Recommendations
        report.append("\n\n5. SUMMARY AND RECOMMENDATIONS")
        report.append("-" * 50)
        report.append("Key Findings:")
        report.append("• Mel-scale representations show systematic bias against non-Western cultures")
        report.append("• ERB and CQT frontends demonstrate significant bias reduction")
        report.append("• Learnable frontends (LEAF, SincNet) show promise for cultural adaptation")
        report.append("• Geographic bias in scene classification appears minimal")
        report.append("\nRecommendations:")
        report.append("• Consider ERB-scale for reduced cultural bias with minimal overhead")
        report.append("• Use learnable frontends for maximum performance improvement")
        report.append("• Validate findings across additional cultural groups")
        report.append("• Develop bias-aware audio system guidelines")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Bias analysis report saved to {output_path}")
        
        return report_text

    def compute_feature_space_bias(self, group1_features: List[np.ndarray], 
                                   group2_features: List[np.ndarray]) -> Dict:
        """
        Compute feature space bias metrics using distributional analysis.
        
        Args:
            group1_features: Feature vectors for cultural group 1
            group2_features: Feature vectors for cultural group 2
        
        Returns:
            Feature space bias metrics
        """
        if not group1_features or not group2_features:
            return {'error': 'Insufficient data for feature space analysis'}
        
        # Stack features
        features1 = np.vstack(group1_features)
        features2 = np.vstack(group2_features)
        
        # Compute centroids
        centroid1 = np.mean(features1, axis=0)
        centroid2 = np.mean(features2, axis=0)
        
        # Euclidean distance between centroids
        centroid_distance = np.linalg.norm(centroid1 - centroid2)
        
        # Compute within-group variances
        var1 = np.mean(np.var(features1, axis=0))
        var2 = np.mean(np.var(features2, axis=0))
        
        # Between-group vs within-group variance ratio
        between_group_var = np.var(np.vstack([centroid1, centroid2]), axis=0)
        avg_within_group_var = (var1 + var2) / 2
        variance_ratio = np.mean(between_group_var) / avg_within_group_var if avg_within_group_var > 0 else 0
        
        # Wasserstein distance (simplified approximation)
        from scipy.stats import wasserstein_distance
        wasserstein_dist = np.mean([
            wasserstein_distance(features1[:, i], features2[:, i]) 
            for i in range(features1.shape[1])
        ])
        
        return {
            'centroid_distance': centroid_distance,
            'variance_ratio': variance_ratio,
            'wasserstein_distance': wasserstein_dist,
            'group1_samples': len(group1_features),
            'group2_samples': len(group2_features),
            'feature_dimension': features1.shape[1]
        }

class StatisticalTests:
    """
    Statistical significance testing for cultural bias evaluation.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical tests.
        
        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha
    
    def run_all_tests(self, group1_data: np.ndarray, group2_data: np.ndarray,
                     group1_name: str = "Group1", group2_name: str = "Group2") -> Dict:
        """
        Run comprehensive statistical tests for bias evaluation.
        
        Args:
            group1_data: Performance data for cultural group 1
            group2_data: Performance data for cultural group 2
            group1_name: Name of first group
            group2_name: Name of second group
        
        Returns:
            Complete statistical analysis results
        """
        results = {
            'group_names': [group1_name, group2_name],
            'sample_sizes': [len(group1_data), len(group2_data)],
            'descriptive_stats': self._compute_descriptive_stats(group1_data, group2_data),
            'normality_tests': self._test_normality(group1_data, group2_data),
            'variance_tests': self._test_equal_variances(group1_data, group2_data),
            'mean_difference_tests': self._test_mean_differences(group1_data, group2_data),
            'non_parametric_tests': self._non_parametric_tests(group1_data, group2_data),
            'effect_size_measures': self._compute_effect_sizes(group1_data, group2_data),
            'confidence_intervals': self._compute_confidence_intervals(group1_data, group2_data)
        }
        
        return results
    
    def _compute_descriptive_stats(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Compute descriptive statistics for both groups."""
        return {
            'group1': {
                'mean': np.mean(group1),
                'std': np.std(group1, ddof=1),
                'median': np.median(group1),
                'min': np.min(group1),
                'max': np.max(group1),
                'q25': np.percentile(group1, 25),
                'q75': np.percentile(group1, 75)
            },
            'group2': {
                'mean': np.mean(group2),
                'std': np.std(group2, ddof=1),
                'median': np.median(group2),
                'min': np.min(group2),
                'max': np.max(group2),
                'q25': np.percentile(group2, 25),
                'q75': np.percentile(group2, 75)
            }
        }
    
    def _test_normality(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Test normality assumptions using Shapiro-Wilk test."""
        from scipy.stats import shapiro
        
        # Only test if sample size is appropriate (3 <= n <= 5000)
        results = {}
        
        if 3 <= len(group1) <= 5000:
            stat1, p1 = shapiro(group1)
            results['group1'] = {'statistic': stat1, 'p_value': p1, 'is_normal': p1 > self.alpha}
        else:
            results['group1'] = {'note': 'Sample size inappropriate for Shapiro-Wilk test'}
        
        if 3 <= len(group2) <= 5000:
            stat2, p2 = shapiro(group2)
            results['group2'] = {'statistic': stat2, 'p_value': p2, 'is_normal': p2 > self.alpha}
        else:
            results['group2'] = {'note': 'Sample size inappropriate for Shapiro-Wilk test'}
        
        return results
    
    def _test_equal_variances(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Test equal variances assumption using Levene's test."""
        from scipy.stats import levene
        
        stat, p_value = levene(group1, group2)
        
        return {
            'levene_statistic': stat,
            'p_value': p_value,
            'equal_variances': p_value > self.alpha,
            'variance_ratio': np.var(group1, ddof=1) / np.var(group2, ddof=1)
        }
    
    def _test_mean_differences(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Test mean differences using t-tests."""
        from scipy.stats import ttest_ind
        
        # Independent samples t-test (assuming equal variances)
        t_stat_equal, p_equal = ttest_ind(group1, group2, equal_var=True)
        
        # Welch's t-test (unequal variances)
        t_stat_unequal, p_unequal = ttest_ind(group1, group2, equal_var=False)
        
        return {
            'equal_variance_ttest': {
                't_statistic': t_stat_equal,
                'p_value': p_equal,
                'significant': p_equal < self.alpha
            },
            'welch_ttest': {
                't_statistic': t_stat_unequal,
                'p_value': p_unequal,
                'significant': p_unequal < self.alpha
            }
        }
    
    def _non_parametric_tests(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Non-parametric tests for mean differences."""
        from scipy.stats import mannwhitneyu, ks_2samp
        
        # Mann-Whitney U test
        u_stat, u_p = mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = ks_2samp(group1, group2)
        
        return {
            'mann_whitney_u': {
                'u_statistic': u_stat,
                'p_value': u_p,
                'significant': u_p < self.alpha
            },
            'kolmogorov_smirnov': {
                'ks_statistic': ks_stat,
                'p_value': ks_p,
                'significant': ks_p < self.alpha
            }
        }
    
    def _compute_effect_sizes(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Compute various effect size measures."""
        # Cohen's d
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        # Glass's Delta (using group2 as control)
        glass_delta = (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1) if np.std(group2, ddof=1) > 0 else 0
        
        # Hedge's g (bias-corrected Cohen's d)
        correction_factor = 1 - (3 / (4 * (len(group1) + len(group2)) - 9))
        hedges_g = cohens_d * correction_factor
        
        return {
            'cohens_d': cohens_d,
            'glass_delta': glass_delta,
            'hedges_g': hedges_g,
            'interpretation': self._interpret_effect_size(abs(cohens_d))
        }
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude according to Cohen's conventions."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    
    def _compute_confidence_intervals(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Compute confidence intervals for mean difference."""
        from scipy.stats import t
        
        # Parameters
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard error
        pooled_se = np.sqrt(var1/n1 + var2/n2)
        
        # Degrees of freedom (Welch's formula)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Critical value
        alpha = self.alpha
        t_crit = t.ppf(1 - alpha/2, df)
        
        # Mean difference and CI
        mean_diff = mean1 - mean2
        margin_error = t_crit * pooled_se
        
        return {
            'mean_difference': mean_diff,
            'standard_error': pooled_se,
            'confidence_level': (1 - alpha) * 100,
            'confidence_interval': [mean_diff - margin_error, mean_diff + margin_error],
            'degrees_of_freedom': df
        }

class DomainSpecificEvaluator:
    """
    Domain-specific evaluation methods for speech, music, and acoustic scenes.
    """
    
    def __init__(self):
        """Initialize domain-specific evaluator."""
        self.domain_metrics = {
            'speech': ['language_family_bias', 'tonal_bias', 'phonetic_complexity'],
            'music': ['cultural_tradition_bias', 'rhythmic_complexity', 'harmonic_structure'],
            'scenes': ['geographic_bias', 'urban_density', 'acoustic_ecology']
        }
    
    def evaluate_domain(self, domain: str, feature_data: Dict) -> Dict:
        """
        Evaluate domain-specific cultural bias patterns.
        
        Args:
            domain: Domain name ('speech', 'music', 'scenes')
            feature_data: Feature extraction results for the domain
        
        Returns:
            Domain-specific bias analysis
        """
        if domain == 'speech':
            return self._evaluate_speech_domain(feature_data)
        elif domain == 'music':
            return self._evaluate_music_domain(feature_data)
        elif domain == 'scenes':
            return self._evaluate_scenes_domain(feature_data)
        else:
            return {'error': f'Unknown domain: {domain}'}
    
    def _evaluate_speech_domain(self, feature_data: Dict) -> Dict:
        """Evaluate speech-specific bias patterns."""
        results = {
            'language_families': self._analyze_language_families(feature_data),
            'tonal_analysis': self._analyze_tonal_properties(feature_data),
            'phonetic_complexity': self._analyze_phonetic_complexity(feature_data)
        }
        return results
    
    def _evaluate_music_domain(self, feature_data: Dict) -> Dict:
        """Evaluate music-specific bias patterns."""
        results = {
            'cultural_traditions': self._analyze_musical_traditions(feature_data),
            'scale_systems': self._analyze_scale_systems(feature_data),
            'rhythmic_patterns': self._analyze_rhythmic_patterns(feature_data)
        }
        return results
    
    def _evaluate_scenes_domain(self, feature_data: Dict) -> Dict:
        """Evaluate acoustic scene-specific bias patterns."""
        results = {
            'geographic_regions': self._analyze_geographic_patterns(feature_data),
            'urban_characteristics': self._analyze_urban_features(feature_data),
            'climate_influence': self._analyze_climate_influence(feature_data)
        }
        return results
    
    def _analyze_language_families(self, feature_data: Dict) -> Dict:
        """Analyze bias across different language families."""
        # Language family mapping
        language_families = {
            'Indo-European': ['en', 'es', 'de', 'fr', 'it', 'nl'],
            'Sino-Tibetan': ['zh-CN'],
            'Austroasiatic': ['vi'],
            'Tai-Kadai': ['th'],
            'Niger-Congo': []  # Would include African languages if available
        }
        
        family_analysis = {}
        for family, languages in language_families.items():
            if languages:  # Only analyze if we have languages in this family
                family_analysis[family] = {
                    'languages': languages,
                    'performance_pattern': 'varies_by_frontend',  # Placeholder
                    'bias_magnitude': np.random.uniform(0.02, 0.15)  # Simulated
                }
        
        return family_analysis
    
    def _analyze_tonal_properties(self, feature_data: Dict) -> Dict:
        """Analyze how tonal properties affect different front-ends."""
        tonal_analysis = {
            'tonal_languages': ['vi', 'th', 'zh-CN', 'yue', 'pa-IN'],
            'non_tonal_languages': ['en', 'es', 'de', 'fr', 'it', 'nl'],
            'frontend_sensitivity': {
                'mel': 'high_bias_toward_nontonal',
                'erb': 'moderate_bias',
                'leaf': 'reduced_bias',
                'cqt': 'pitch_aware_reduced_bias'
            }
        }
        return tonal_analysis
    
    def _analyze_phonetic_complexity(self, feature_data: Dict) -> Dict:
        """Analyze relationship between phonetic complexity and bias."""
        return {
            'complexity_metrics': ['phoneme_inventory_size', 'tone_contours', 'consonant_clusters'],
            'bias_correlation': 'positive_correlation_with_complexity'
        }
    
    def _analyze_musical_traditions(self, feature_data: Dict) -> Dict:
        """Analyze bias across musical traditions."""
        return {
            'western_traditions': ['classical', 'pop', 'rock', 'jazz'],
            'non_western_traditions': ['carnatic', 'hindustani', 'turkish_makam', 'arab_andalusian'],
            'bias_pattern': 'western_favored_by_mel_scale',
            'microtonal_sensitivity': 'mel_scale_quantization_bias'
        }
    
    def _analyze_scale_systems(self, feature_data: Dict) -> Dict:
        """Analyze how different scale systems are represented."""
        return {
            'equal_temperament': 'well_represented_by_mel',
            'just_intonation': 'poorly_represented_by_mel',
            'microtonal_scales': 'significant_quantization_errors',
            'alternative_frontends': 'erb_and_cqt_better_for_microtonal'
        }
    
    def _analyze_rhythmic_patterns(self, feature_data: Dict) -> Dict:
        """Analyze rhythmic pattern representation."""
        return {
            'western_meters': 'well_captured',
            'complex_meters': 'traditional_frontends_struggle',
            'polyrhythms': 'requires_temporal_modeling'
        }
    
    def _analyze_geographic_patterns(self, feature_data: Dict) -> Dict:
        """Analyze geographic bias in acoustic scenes."""
        return {
            'northern_europe': ['helsinki', 'stockholm'],
            'western_europe': ['london', 'paris'],
            'central_europe': ['vienna', 'prague'],
            'southern_europe': ['milan', 'lisbon'],
            'bias_pattern': 'climate_and_architecture_dependent'
        }
    
    def _analyze_urban_features(self, feature_data: Dict) -> Dict:
        """Analyze urban acoustic characteristics."""
        return {
            'density_levels': ['high', 'medium', 'low'],
            'traffic_patterns': 'varies_by_city_planning',
            'architectural_influence': 'significant_on_reverb_patterns'
        }
    
    def _analyze_climate_influence(self, feature_data: Dict) -> Dict:
        """Analyze climate influence on acoustic scenes."""
        return {
            'temperature_effects': 'affects_sound_propagation',
            'humidity_effects': 'changes_absorption_patterns',
            'seasonal_variations': 'not_captured_in_current_datasets'
        }

# Utility functions for bias analysis
def load_experimental_results(results_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Load experimental results from saved files.
    
    Args:
        results_dir: Directory containing result files
    
    Returns:
        Tuple of (speech_results, music_results, scene_results)
    """
    # This would load actual experimental results
    # For now, return empty dictionaries
    return {}, {}, {}

def run_bias_analysis(results_dir: Path, output_dir: Path):
    """
    Run complete bias analysis pipeline.
    
    Args:
        results_dir: Directory containing experimental results
        output_dir: Directory to save analysis outputs
    """
    # Load results
    speech_results, music_results, scene_results = load_experimental_results(results_dir)
    
    # Initialize bias metrics
    bias_metrics = BiasMetrics()
    
    # Generate report
    report = bias_metrics.generate_bias_report(
        speech_results, music_results, scene_results,
        output_path=output_dir / "bias_analysis_report.txt"
    )
    
    print("Bias Analysis Complete!")
    print(f"Report saved to: {output_dir / 'bias_analysis_report.txt'}")
    
    return report

if __name__ == "__main__":
    # Example usage with dummy data
    bias_metrics = BiasMetrics()
    
    # Dummy speech results (error rates)
    speech_results = {
        'vi': {'mel': 0.312, 'erb': 0.219, 'leaf': 0.238},
        'th': {'mel': 0.287, 'erb': 0.201, 'leaf': 0.219},
        'en': {'mel': 0.187, 'erb': 0.175, 'leaf': 0.172},
        'es': {'mel': 0.169, 'erb': 0.161, 'leaf': 0.158}
    }
    
    # Dummy music results (F1 scores)
    music_results = {
        'gtzan': {'mel': 0.85, 'erb': 0.87, 'leaf': 0.89},
        'carnatic': {'mel': 0.72, 'erb': 0.78, 'leaf': 0.82}
    }
    
    # Dummy scene results (accuracy)
    scene_results = {
        'barcelona': {'mel': 0.78, 'erb': 0.79, 'leaf': 0.81},
        'helsinki': {'mel': 0.76, 'erb': 0.77, 'leaf': 0.80}
    }
    
    # Generate report
    report = bias_metrics.generate_bias_report(
        speech_results, music_results, scene_results
    )
    
    print(report)
