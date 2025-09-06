#!/usr/bin/env python3
"""
FairAudioBench Fairness Metrics Calculator
Computes WGS, Δ, ρ metrics with statistical significance testing.
Generates fairness reports comparing to the four-fifths rule threshold.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FairnessMetricsCalculator:
    """Calculates fairness metrics for cross-cultural bias evaluation."""
    
    def __init__(self, results_dir: str, output_dir: str = "./fairness_reports"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Four-fifths rule threshold
        self.four_fifths_threshold = 0.8
        
        # Statistical significance parameters
        self.alpha = 0.05
        self.confidence_level = 0.95
    
    def load_experiment_results(self, experiment_name: str) -> Optional[Dict]:
        """Load experiment results from JSON file."""
        results_file = self.results_dir / f"{experiment_name}_results.json"
        
        if not results_file.exists():
            logger.error(f"Results file not found: {results_file}")
            return None
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded results for {experiment_name}")
            return results
        except Exception as e:
            logger.error(f"Failed to load results for {experiment_name}: {e}")
            return None
    
    def calculate_wgs(self, group_performances: Dict[str, float]) -> float:
        """
        Calculate Worst Group Score (WGS).
        
        Args:
            group_performances: Dictionary mapping group names to performance scores
            
        Returns:
            Worst group performance score
        """
        if not group_performances:
            return 0.0
        
        wgs = min(group_performances.values())
        logger.debug(f"WGS: {wgs:.4f}")
        return wgs
    
    def calculate_delta(self, group_performances: Dict[str, float]) -> float:
        """
        Calculate Δ (Delta) - performance gap between best and worst groups.
        
        Args:
            group_performances: Dictionary mapping group names to performance scores
            
        Returns:
            Performance gap (max - min)
        """
        if not group_performances:
            return 0.0
        
        performances = list(group_performances.values())
        delta = max(performances) - min(performances)
        logger.debug(f"Δ: {delta:.4f}")
        return delta
    
    def calculate_rho(self, group_performances: Dict[str, float]) -> float:
        """
        Calculate ρ (Rho) - coefficient of variation of group performances.
        
        Args:
            group_performances: Dictionary mapping group names to performance scores
            
        Returns:
            Coefficient of variation (std/mean)
        """
        if not group_performances:
            return 0.0
        
        performances = np.array(list(group_performances.values()))
        
        if len(performances) < 2 or np.mean(performances) == 0:
            return 0.0
        
        rho = np.std(performances) / np.mean(performances)
        logger.debug(f"ρ: {rho:.4f}")
        return rho
    
    def calculate_four_fifths_ratio(self, group_performances: Dict[str, float]) -> float:
        """
        Calculate four-fifths ratio for fairness evaluation.
        
        Args:
            group_performances: Dictionary mapping group names to performance scores
            
        Returns:
            Four-fifths ratio (min_performance / max_performance)
        """
        if not group_performances:
            return 0.0
        
        performances = list(group_performances.values())
        max_perf = max(performances)
        min_perf = min(performances)
        
        if max_perf == 0:
            return 0.0
        
        ratio = min_perf / max_perf
        logger.debug(f"Four-fifths ratio: {ratio:.4f}")
        return ratio
    
    def statistical_significance_test(self, group_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Perform statistical significance testing between groups.
        
        Args:
            group_results: Dictionary mapping group names to lists of performance scores
            
        Returns:
            Dictionary containing test results
        """
        groups = list(group_results.keys())
        results = {
            "anova": None,
            "pairwise_tests": {},
            "significant_differences": []
        }
        
        # Prepare data for ANOVA
        group_data = [group_results[group] for group in groups if len(group_results[group]) > 0]
        
        if len(group_data) < 2:
            logger.warning("Not enough groups for statistical testing")
            return results
        
        # Perform one-way ANOVA
        try:
            f_stat, p_value = stats.f_oneway(*group_data)
            results["anova"] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha
            }
            
            logger.info(f"ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
            
            # If ANOVA is significant, perform pairwise t-tests
            if p_value < self.alpha:
                for i, group1 in enumerate(groups):
                    for j, group2 in enumerate(groups[i+1:], i+1):
                        if len(group_results[group1]) > 0 and len(group_results[group2]) > 0:
                            t_stat, t_p = stats.ttest_ind(
                                group_results[group1], 
                                group_results[group2]
                            )
                            
                            pair_key = f"{group1}_vs_{group2}"
                            results["pairwise_tests"][pair_key] = {
                                "t_statistic": float(t_stat),
                                "p_value": float(t_p),
                                "significant": t_p < self.alpha
                            }
                            
                            if t_p < self.alpha:
                                results["significant_differences"].append(pair_key)
            
        except Exception as e:
            logger.error(f"Statistical testing failed: {e}")
        
        return results
    
    def calculate_confidence_intervals(self, group_performances: Dict[str, List[float]]) -> Dict[str, Dict]:
        """Calculate confidence intervals for group performances."""
        confidence_intervals = {}
        
        for group, performances in group_performances.items():
            if len(performances) < 2:
                confidence_intervals[group] = {
                    "mean": np.mean(performances) if performances else 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "std_error": 0.0
                }
                continue
            
            mean_perf = np.mean(performances)
            std_err = stats.sem(performances)
            
            # Calculate confidence interval
            ci = stats.t.interval(
                self.confidence_level, 
                len(performances) - 1, 
                loc=mean_perf, 
                scale=std_err
            )
            
            confidence_intervals[group] = {
                "mean": float(mean_perf),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "std_error": float(std_err)
            }
        
        return confidence_intervals
    
    def analyze_speech_fairness(self, results: Dict) -> Dict[str, Any]:
        """Analyze fairness for speech recognition tasks."""
        logger.info("Analyzing speech fairness...")
        
        speech_analysis = {
            "tonal_vs_nontonal": {},
            "individual_languages": {},
            "overall_metrics": {}
        }
        
        # Extract speech results
        speech_results = results.get("speech", {})
        
        if not speech_results:
            logger.warning("No speech results found")
            return speech_analysis
        
        # Group by tonal property
        tonal_perfs = []
        nontonal_perfs = []
        language_perfs = {}
        
        for lang, lang_results in speech_results.items():
            if isinstance(lang_results, dict) and "performance" in lang_results:
                perf = lang_results["performance"]
                is_tonal = lang_results.get("is_tonal", False)
                
                language_perfs[lang] = perf
                
                if is_tonal:
                    tonal_perfs.append(perf)
                else:
                    nontonal_perfs.append(perf)
        
        # Calculate tonal vs non-tonal metrics
        tonal_nontonal_perfs = {
            "tonal": np.mean(tonal_perfs) if tonal_perfs else 0.0,
            "non_tonal": np.mean(nontonal_perfs) if nontonal_perfs else 0.0
        }
        
        speech_analysis["tonal_vs_nontonal"] = {
            "performances": tonal_nontonal_perfs,
            "wgs": self.calculate_wgs(tonal_nontonal_perfs),
            "delta": self.calculate_delta(tonal_nontonal_perfs),
            "rho": self.calculate_rho(tonal_nontonal_perfs),
            "four_fifths_ratio": self.calculate_four_fifths_ratio(tonal_nontonal_perfs),
            "passes_four_fifths": self.calculate_four_fifths_ratio(tonal_nontonal_perfs) >= self.four_fifths_threshold
        }
        
        # Calculate individual language metrics
        speech_analysis["individual_languages"] = {
            "performances": language_perfs,
            "wgs": self.calculate_wgs(language_perfs),
            "delta": self.calculate_delta(language_perfs),
            "rho": self.calculate_rho(language_perfs),
            "four_fifths_ratio": self.calculate_four_fifths_ratio(language_perfs),
            "passes_four_fifths": self.calculate_four_fifths_ratio(language_perfs) >= self.four_fifths_threshold
        }
        
        # Statistical testing
        group_results = {"tonal": tonal_perfs, "non_tonal": nontonal_perfs}
        speech_analysis["statistical_tests"] = self.statistical_significance_test(group_results)
        
        # Confidence intervals
        speech_analysis["confidence_intervals"] = self.calculate_confidence_intervals(group_results)
        
        return speech_analysis
    
    def analyze_music_fairness(self, results: Dict) -> Dict[str, Any]:
        """Analyze fairness for music classification tasks."""
        logger.info("Analyzing music fairness...")
        
        music_analysis = {
            "musical_traditions": {},
            "cultural_origins": {},
            "overall_metrics": {}
        }
        
        # Extract music results
        music_results = results.get("music", {})
        
        if not music_results:
            logger.warning("No music results found")
            return music_analysis
        
        # Group by tradition and cultural origin
        tradition_perfs = {}
        origin_perfs = defaultdict(list)
        
        for tradition, tradition_results in music_results.items():
            if isinstance(tradition_results, dict) and "performance" in tradition_results:
                perf = tradition_results["performance"]
                tradition_perfs[tradition] = perf
                
                # Map to cultural origin
                origin = self._get_cultural_origin(tradition)
                origin_perfs[origin].append(perf)
        
        # Average performances by cultural origin
        origin_avg_perfs = {
            origin: np.mean(perfs) for origin, perfs in origin_perfs.items()
        }
        
        # Calculate tradition-level metrics
        music_analysis["musical_traditions"] = {
            "performances": tradition_perfs,
            "wgs": self.calculate_wgs(tradition_perfs),
            "delta": self.calculate_delta(tradition_perfs),
            "rho": self.calculate_rho(tradition_perfs),
            "four_fifths_ratio": self.calculate_four_fifths_ratio(tradition_perfs),
            "passes_four_fifths": self.calculate_four_fifths_ratio(tradition_perfs) >= self.four_fifths_threshold
        }
        
        # Calculate cultural origin metrics
        music_analysis["cultural_origins"] = {
            "performances": origin_avg_perfs,
            "wgs": self.calculate_wgs(origin_avg_perfs),
            "delta": self.calculate_delta(origin_avg_perfs),
            "rho": self.calculate_rho(origin_avg_perfs),
            "four_fifths_ratio": self.calculate_four_fifths_ratio(origin_avg_perfs),
            "passes_four_fifths": self.calculate_four_fifths_ratio(origin_avg_perfs) >= self.four_fifths_threshold
        }
        
        # Statistical testing
        music_analysis["statistical_tests"] = self.statistical_significance_test(origin_perfs)
        
        # Confidence intervals
        music_analysis["confidence_intervals"] = self.calculate_confidence_intervals(origin_perfs)
        
        return music_analysis
    
    def analyze_urban_sounds_fairness(self, results: Dict) -> Dict[str, Any]:
        """Analyze fairness for urban sounds classification."""
        logger.info("Analyzing urban sounds fairness...")
        
        urban_analysis = {
            "individual_cities": {},
            "countries": {},
            "population_groups": {}
        }
        
        # Extract urban sounds results
        urban_results = results.get("urban_sounds", {})
        
        if not urban_results:
            logger.warning("No urban sounds results found")
            return urban_analysis
        
        # Group by city, country, and population
        city_perfs = {}
        country_perfs = defaultdict(list)
        population_groups = {"small": [], "medium": [], "large": []}
        
        for city, city_results in urban_results.items():
            if isinstance(city_results, dict) and "performance" in city_results:
                perf = city_results["performance"]
                city_perfs[city] = perf
                
                # Group by country
                country = city_results.get("country", "Unknown")
                country_perfs[country].append(perf)
                
                # Group by population size
                population = city_results.get("population", 0)
                if population < 1000000:
                    population_groups["small"].append(perf)
                elif population < 3000000:
                    population_groups["medium"].append(perf)
                else:
                    population_groups["large"].append(perf)
        
        # Average performances by country
        country_avg_perfs = {
            country: np.mean(perfs) for country, perfs in country_perfs.items()
        }
        
        # Average performances by population group
        pop_avg_perfs = {
            group: np.mean(perfs) if perfs else 0.0 
            for group, perfs in population_groups.items()
        }
        
        # Calculate city-level metrics
        urban_analysis["individual_cities"] = {
            "performances": city_perfs,
            "wgs": self.calculate_wgs(city_perfs),
            "delta": self.calculate_delta(city_perfs),
            "rho": self.calculate_rho(city_perfs),
            "four_fifths_ratio": self.calculate_four_fifths_ratio(city_perfs),
            "passes_four_fifths": self.calculate_four_fifths_ratio(city_perfs) >= self.four_fifths_threshold
        }
        
        # Calculate country-level metrics
        urban_analysis["countries"] = {
            "performances": country_avg_perfs,
            "wgs": self.calculate_wgs(country_avg_perfs),
            "delta": self.calculate_delta(country_avg_perfs),
            "rho": self.calculate_rho(country_avg_perfs),
            "four_fifths_ratio": self.calculate_four_fifths_ratio(country_avg_perfs),
            "passes_four_fifths": self.calculate_four_fifths_ratio(country_avg_perfs) >= self.four_fifths_threshold
        }
        
        # Calculate population group metrics
        urban_analysis["population_groups"] = {
            "performances": pop_avg_perfs,
            "wgs": self.calculate_wgs(pop_avg_perfs),
            "delta": self.calculate_delta(pop_avg_perfs),
            "rho": self.calculate_rho(pop_avg_perfs),
            "four_fifths_ratio": self.calculate_four_fifths_ratio(pop_avg_perfs),
            "passes_four_fifths": self.calculate_four_fifths_ratio(pop_avg_perfs) >= self.four_fifths_threshold
        }
        
        # Statistical testing
        urban_analysis["statistical_tests"] = self.statistical_significance_test(population_groups)
        
        # Confidence intervals
        urban_analysis["confidence_intervals"] = self.calculate_confidence_intervals(population_groups)
        
        return urban_analysis
    
    def _get_cultural_origin(self, tradition: str) -> str:
        """Map musical tradition to cultural origin."""
        origin_map = {
            "western_classical": "Western",
            "jazz": "African-American",
            "blues": "African-American", 
            "country": "American",
            "indian_classical": "South Asian",
            "middle_eastern": "Middle Eastern",
            "african": "African",
            "latin": "Latin American"
        }
        return origin_map.get(tradition, "Unknown")
    
    def generate_fairness_report(self, experiment_name: str, model_name: str) -> bool:
        """Generate comprehensive fairness report."""
        logger.info(f"Generating fairness report for {experiment_name} - {model_name}")
        
        # Load results
        results = self.load_experiment_results(experiment_name)
        if not results:
            return False
        
        # Extract model results
        model_results = results.get(model_name, {})
        if not model_results:
            logger.error(f"No results found for model {model_name}")
            return False
        
        # Analyze fairness for each domain
        speech_fairness = self.analyze_speech_fairness(model_results)
        music_fairness = self.analyze_music_fairness(model_results)
        urban_fairness = self.analyze_urban_sounds_fairness(model_results)
        
        # Compile comprehensive report
        fairness_report = {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "report_date": "2025-09-06",
            "fairness_threshold": self.four_fifths_threshold,
            "significance_level": self.alpha,
            "domains": {
                "speech": speech_fairness,
                "music": music_fairness,
                "urban_sounds": urban_fairness
            },
            "summary": self._generate_summary(speech_fairness, music_fairness, urban_fairness)
        }
        
        # Save report
        report_file = self.output_dir / f"{experiment_name}_{model_name}_fairness_report.json"
        with open(report_file, 'w') as f:
            json.dump(fairness_report, f, indent=2)
        
        # Generate visualizations
        self._generate_visualizations(fairness_report, experiment_name, model_name)
        
        logger.info(f"Fairness report saved to {report_file}")
        return True
    
    def _generate_summary(self, speech_fairness: Dict, music_fairness: Dict, 
                         urban_fairness: Dict) -> Dict[str, Any]:
        """Generate summary of fairness analysis."""
        summary = {
            "overall_fairness_score": 0.0,
            "domain_scores": {},
            "four_fifths_compliance": {},
            "significant_biases": [],
            "recommendations": []
        }
        
        domain_analyses = {
            "speech": speech_fairness,
            "music": music_fairness,
            "urban_sounds": urban_fairness
        }
        
        domain_scores = []
        
        for domain, analysis in domain_analyses.items():
            if not analysis:
                continue
            
            # Calculate domain fairness score (based on WGS and four-fifths compliance)
            domain_wgs_scores = []
            domain_compliance = []
            
            for level, metrics in analysis.items():
                if isinstance(metrics, dict) and "wgs" in metrics:
                    domain_wgs_scores.append(metrics["wgs"])
                    domain_compliance.append(metrics["passes_four_fifths"])
            
            if domain_wgs_scores:
                domain_score = np.mean(domain_wgs_scores)
                domain_scores.append(domain_score)
                summary["domain_scores"][domain] = domain_score
                summary["four_fifths_compliance"][domain] = np.mean(domain_compliance)
            
            # Check for significant biases
            stats_tests = analysis.get("statistical_tests", {})
            if stats_tests and stats_tests.get("anova", {}).get("significant", False):
                summary["significant_biases"].append(f"{domain}_bias_detected")
        
        # Calculate overall fairness score
        if domain_scores:
            summary["overall_fairness_score"] = np.mean(domain_scores)
        
        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations(summary)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict) -> List[str]:
        """Generate recommendations based on fairness analysis."""
        recommendations = []
        
        overall_score = summary["overall_fairness_score"]
        compliance = summary["four_fifths_compliance"]
        
        if overall_score < 0.6:
            recommendations.append("Overall fairness is concerning. Consider retraining with balanced data.")
        
        for domain, compliant in compliance.items():
            if not compliant:
                recommendations.append(f"Fairness issues detected in {domain}. Investigate data balance and feature engineering.")
        
        if "speech_bias_detected" in summary["significant_biases"]:
            recommendations.append("Significant bias between tonal and non-tonal languages. Consider language-specific preprocessing.")
        
        if "music_bias_detected" in summary["significant_biases"]:
            recommendations.append("Cultural bias in music classification. Expand training data for underrepresented traditions.")
        
        if "urban_sounds_bias_detected" in summary["significant_biases"]:
            recommendations.append("Geographic bias in urban sound classification. Include more diverse city recordings.")
        
        if not recommendations:
            recommendations.append("Model shows good fairness characteristics across cultural groups.")
        
        return recommendations
    
    def _generate_visualizations(self, report: Dict, experiment_name: str, model_name: str) -> None:
        """Generate fairness visualization plots."""
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Fairness Analysis: {experiment_name} - {model_name}', fontsize=16, fontweight='bold')
            
            # Plot 1: Speech fairness (tonal vs non-tonal)
            speech_data = report["domains"]["speech"].get("tonal_vs_nontonal", {})
            if speech_data and "performances" in speech_data:
                groups = list(speech_data["performances"].keys())
                perfs = list(speech_data["performances"].values())
                
                axes[0, 0].bar(groups, perfs, color=['skyblue', 'lightcoral'])
                axes[0, 0].axhline(y=self.four_fifths_threshold * max(perfs), 
                                  color='red', linestyle='--', label='Four-fifths threshold')
                axes[0, 0].set_title('Speech: Tonal vs Non-tonal')
                axes[0, 0].set_ylabel('Performance')
                axes[0, 0].legend()
            
            # Plot 2: Music fairness by cultural origin
            music_data = report["domains"]["music"].get("cultural_origins", {})
            if music_data and "performances" in music_data:
                origins = list(music_data["performances"].keys())
                perfs = list(music_data["performances"].values())
                
                axes[0, 1].bar(origins, perfs, color='lightgreen')
                axes[0, 1].axhline(y=self.four_fifths_threshold * max(perfs), 
                                  color='red', linestyle='--', label='Four-fifths threshold')
                axes[0, 1].set_title('Music: Cultural Origins')
                axes[0, 1].set_ylabel('Performance')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].legend()
            
            # Plot 3: Urban sounds by population groups
            urban_data = report["domains"]["urban_sounds"].get("population_groups", {})
            if urban_data and "performances" in urban_data:
                groups = list(urban_data["performances"].keys())
                perfs = list(urban_data["performances"].values())
                
                axes[0, 2].bar(groups, perfs, color='lightyellow')
                axes[0, 2].axhline(y=self.four_fifths_threshold * max(perfs), 
                                  color='red', linestyle='--', label='Four-fifths threshold')
                axes[0, 2].set_title('Urban Sounds: Population Groups')
                axes[0, 2].set_ylabel('Performance')
                axes[0, 2].legend()
            
            # Plot 4: WGS comparison across domains
            domains = []
            wgs_scores = []
            
            for domain, analysis in report["domains"].items():
                for level, metrics in analysis.items():
                    if isinstance(metrics, dict) and "wgs" in metrics:
                        domains.append(f"{domain}_{level}")
                        wgs_scores.append(metrics["wgs"])
            
            if domains and wgs_scores:
                axes[1, 0].bar(range(len(domains)), wgs_scores, color='orange')
                axes[1, 0].set_xticks(range(len(domains)))
                axes[1, 0].set_xticklabels(domains, rotation=45, ha='right')
                axes[1, 0].set_title('Worst Group Score (WGS) Comparison')
                axes[1, 0].set_ylabel('WGS')
            
            # Plot 5: Delta (performance gap) comparison
            deltas = []
            delta_labels = []
            
            for domain, analysis in report["domains"].items():
                for level, metrics in analysis.items():
                    if isinstance(metrics, dict) and "delta" in metrics:
                        delta_labels.append(f"{domain}_{level}")
                        deltas.append(metrics["delta"])
            
            if delta_labels and deltas:
                axes[1, 1].bar(range(len(delta_labels)), deltas, color='purple')
                axes[1, 1].set_xticks(range(len(delta_labels)))
                axes[1, 1].set_xticklabels(delta_labels, rotation=45, ha='right')
                axes[1, 1].set_title('Performance Gap (Δ) Comparison')
                axes[1, 1].set_ylabel('Δ')
            
            # Plot 6: Four-fifths compliance
            compliance_data = report["summary"]["four_fifths_compliance"]
            if compliance_data:
                domains = list(compliance_data.keys())
                compliance = list(compliance_data.values())
                
                colors = ['green' if c >= self.four_fifths_threshold else 'red' for c in compliance]
                axes[1, 2].bar(domains, compliance, color=colors)
                axes[1, 2].axhline(y=self.four_fifths_threshold, color='black', linestyle='--')
                axes[1, 2].set_title('Four-fifths Rule Compliance')
                axes[1, 2].set_ylabel('Compliance Ratio')
                axes[1, 2].set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"{experiment_name}_{model_name}_fairness_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Fairness plots saved to {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
    
    def compare_models(self, experiment_name: str, model_names: List[str]) -> bool:
        """Compare fairness metrics across multiple models."""
        logger.info(f"Comparing models: {model_names}")
        
        comparison_data = {}
        
        # Load results for each model
        for model_name in model_names:
            results = self.load_experiment_results(experiment_name)
            if results and model_name in results:
                # Analyze fairness
                model_results = results[model_name]
                speech_fairness = self.analyze_speech_fairness(model_results)
                music_fairness = self.analyze_music_fairness(model_results)
                urban_fairness = self.analyze_urban_sounds_fairness(model_results)
                
                comparison_data[model_name] = {
                    "speech": speech_fairness,
                    "music": music_fairness,
                    "urban_sounds": urban_fairness
                }
        
        if not comparison_data:
            logger.error("No valid model results found for comparison")
            return False
        
        # Generate comparison report
        comparison_report = {
            "experiment_name": experiment_name,
            "models_compared": model_names,
            "comparison_date": "2025-09-06",
            "models": comparison_data,
            "rankings": self._rank_models_by_fairness(comparison_data)
        }
        
        # Save comparison report
        comparison_file = self.output_dir / f"{experiment_name}_model_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        # Generate comparison visualizations
        self._generate_comparison_plots(comparison_report, experiment_name)
        
        logger.info(f"Model comparison saved to {comparison_file}")
        return True
    
    def _rank_models_by_fairness(self, comparison_data: Dict) -> Dict[str, List[str]]:
        """Rank models by different fairness criteria."""
        rankings = {
            "overall_wgs": [],
            "overall_delta": [],
            "four_fifths_compliance": []
        }
        
        model_scores = {}
        
        for model_name, model_data in comparison_data.items():
            wgs_scores = []
            delta_scores = []
            compliance_scores = []
            
            for domain, analysis in model_data.items():
                for level, metrics in analysis.items():
                    if isinstance(metrics, dict):
                        if "wgs" in metrics:
                            wgs_scores.append(metrics["wgs"])
                        if "delta" in metrics:
                            delta_scores.append(metrics["delta"])
                        if "passes_four_fifths" in metrics:
                            compliance_scores.append(1.0 if metrics["passes_four_fifths"] else 0.0)
            
            model_scores[model_name] = {
                "avg_wgs": np.mean(wgs_scores) if wgs_scores else 0.0,
                "avg_delta": np.mean(delta_scores) if delta_scores else 0.0,
                "avg_compliance": np.mean(compliance_scores) if compliance_scores else 0.0
            }
        
        # Rank by WGS (higher is better)
        rankings["overall_wgs"] = sorted(
            model_scores.keys(), 
            key=lambda x: model_scores[x]["avg_wgs"], 
            reverse=True
        )
        
        # Rank by Delta (lower is better)
        rankings["overall_delta"] = sorted(
            model_scores.keys(), 
            key=lambda x: model_scores[x]["avg_delta"]
        )
        
        # Rank by compliance (higher is better)
        rankings["four_fifths_compliance"] = sorted(
            model_scores.keys(), 
            key=lambda x: model_scores[x]["avg_compliance"], 
            reverse=True
        )
        
        return rankings
    
    def _generate_comparison_plots(self, comparison_report: Dict, experiment_name: str) -> None:
        """Generate comparison visualization plots."""
        try:
            models = list(comparison_report["models"].keys())
            
            # Extract metrics for plotting
            wgs_by_model = {model: [] for model in models}
            delta_by_model = {model: [] for model in models}
            
            for model, model_data in comparison_report["models"].items():
                for domain, analysis in model_data.items():
                    for level, metrics in analysis.items():
                        if isinstance(metrics, dict):
                            if "wgs" in metrics:
                                wgs_by_model[model].append(metrics["wgs"])
                            if "delta" in metrics:
                                delta_by_model[model].append(metrics["delta"])
            
            # Create comparison plots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Model Fairness Comparison: {experiment_name}', fontsize=16, fontweight='bold')
            
            # WGS comparison
            avg_wgs = [np.mean(wgs_by_model[model]) if wgs_by_model[model] else 0 for model in models]
            axes[0].bar(models, avg_wgs, color='skyblue')
            axes[0].set_title('Average Worst Group Score (WGS)')
            axes[0].set_ylabel('WGS')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Delta comparison
            avg_delta = [np.mean(delta_by_model[model]) if delta_by_model[model] else 0 for model in models]
            axes[1].bar(models, avg_delta, color='lightcoral')
            axes[1].set_title('Average Performance Gap (Δ)')
            axes[1].set_ylabel('Δ')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Rankings visualization
            rankings = comparison_report["rankings"]
            ranking_positions = {model: [] for model in models}
            
            for criterion, ranked_models in rankings.items():
                for i, model in enumerate(ranked_models):
                    ranking_positions[model].append(i + 1)  # 1-indexed
            
            avg_ranks = [np.mean(ranking_positions[model]) if ranking_positions[model] else len(models) 
                        for model in models]
            
            axes[2].bar(models, avg_ranks, color='lightgreen')
            axes[2].set_title('Average Ranking (lower is better)')
            axes[2].set_ylabel('Average Rank')
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].invert_yaxis()  # Lower rank should be at top
            
            plt.tight_layout()
            
            # Save comparison plot
            plot_file = self.output_dir / f"{experiment_name}_model_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comparison plots saved to {plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate comparison plots: {e}")

def main():
    """Main function to run fairness metrics calculation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate fairness metrics for FairAudioBench")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        default="./fairness_reports",
        help="Output directory for fairness reports"
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Name of the experiment to analyze"
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        required=True,
        help="Names of models to analyze"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Generate model comparison report"
    )
    
    args = parser.parse_args()
    
    calculator = FairnessMetricsCalculator(args.results_dir, args.output_dir)
    
    # Generate individual reports for each model
    for model_name in args.model_names:
        success = calculator.generate_fairness_report(args.experiment_name, model_name)
        if not success:
            logger.error(f"Failed to generate report for {model_name}")
    
    # Generate comparison report if requested
    if args.compare_models and len(args.model_names) > 1:
        calculator.compare_models(args.experiment_name, args.model_names)
    
    logger.info("Fairness analysis completed!")

if __name__ == "__main__":
    main()
