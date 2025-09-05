#!/usr/bin/env python3

"""
FairAudioBench Results Analysis and Report Generation
Comprehensive analysis of cross-cultural bias experiments
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from src.bias_evaluation import BiasMetrics

class FairAudioBenchAnalyzer:
    """
    Comprehensive analyzer for FairAudioBench results
    """
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.tables_dir = self.results_dir / "tables"
        self.figures_dir = self.results_dir / "figures"
        self.reports_dir = self.results_dir / "reports"
        
        for d in [self.tables_dir, self.figures_dir, self.reports_dir]:
            d.mkdir(exist_ok=True)
    
    def load_experiment_results(self, experiment_dir):
        """Load results from experiment directory"""
        exp_dir = Path(experiment_dir)
        results = {}
        
        # Load experiment metadata
        metadata_file = exp_dir / "experiment_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                results['metadata'] = json.load(f)
        
        # Load domain results
        for domain in ['speech', 'music', 'scenes']:
            domain_dir = exp_dir / domain
            if domain_dir.exists():
                results[domain] = self._load_domain_results(domain_dir)
        
        return results
    
    def _load_domain_results(self, domain_dir):
        """Load results for a specific domain"""
        domain_results = {}
        
        # Load performance metrics
        for frontend in ['mel', 'erb', 'bark', 'cqt', 'leaf', 'sincnet']:
            frontend_file = domain_dir / f"{frontend}_results.json"
            if frontend_file.exists():
                with open(frontend_file, 'r') as f:
                    domain_results[frontend] = json.load(f)
        
        return domain_results
    
    def analyze_performance_gaps(self, results):
        """Analyze performance gaps across cultural groups"""
        analysis = {}
        
        for domain, domain_results in results.items():
            if domain == 'metadata':
                continue
                
            domain_analysis = {}
            
            for frontend, frontend_results in domain_results.items():
                if 'group_performance' in frontend_results:
                    group_perf = frontend_results['group_performance']
                    
                    # Calculate group gaps
                    performances = list(group_perf.values())
                    max_perf = max(performances)
                    min_perf = min(performances)
                    group_gap = max_perf - min_perf
                    
                    # Calculate standard deviation
                    perf_std = np.std(performances)
                    
                    domain_analysis[frontend] = {
                        'group_gap': group_gap,
                        'performance_std': perf_std,
                        'max_performance': max_perf,
                        'min_performance': min_perf,
                        'group_performances': group_perf
                    }
            
            analysis[domain] = domain_analysis
        
        return analysis
    
    def generate_main_results_table(self, results):
        """Generate main results table for the paper"""
        print("Generating Main Results Table...")
        
        # Create comprehensive results table
        table_data = []
        
        domains = ['speech', 'music', 'scenes']
        frontends = ['mel', 'erb', 'bark', 'cqt', 'leaf', 'sincnet']
        
        for domain in domains:
            if domain in results:
                domain_results = results[domain]
                
                for frontend in frontends:
                    if frontend in domain_results:
                        frontend_results = domain_results[frontend]
                        
                        # Extract key metrics
                        accuracy = frontend_results.get('accuracy', 0.0)
                        f1_score = frontend_results.get('f1_score', 0.0)
                        group_gap = frontend_results.get('group_gap', 0.0)
                        
                        table_data.append({
                            'Domain': domain.capitalize(),
                            'Frontend': frontend.upper(),
                            'Accuracy': f"{accuracy:.3f}",
                            'F1-Score': f"{f1_score:.3f}",
                            'Group Gap': f"{group_gap:.3f}"
                        })
        
        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        csv_file = self.tables_dir / "main_results.csv"
        df.to_csv(csv_file, index=False)
        
        # Save as LaTeX
        latex_file = self.tables_dir / "main_results.tex"
        latex_table = df.to_latex(index=False, float_format="%.3f")
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        print(f"Main results table saved to: {csv_file}")
        return df
    
    def generate_bias_analysis_table(self, results):
        """Generate bias analysis table"""
        print("Generating Bias Analysis Table...")
        
        bias_data = []
        
        for domain, domain_results in results.items():
            if domain == 'metadata':
                continue
                
            for frontend, frontend_results in domain_results.items():
                if 'bias_metrics' in frontend_results:
                    bias_metrics = frontend_results['bias_metrics']
                    
                    bias_data.append({
                        'Domain': domain.capitalize(),
                        'Frontend': frontend.upper(),
                        'Group Gap': f"{bias_metrics.get('group_gap', 0):.3f}",
                        'Equalized Odds Gap': f"{bias_metrics.get('equalized_odds_gap', 0):.3f}",
                        'Demographic Parity': f"{bias_metrics.get('demographic_parity', 0):.3f}",
                        'Individual Fairness': f"{bias_metrics.get('individual_fairness', 0):.3f}"
                    })
        
        # Create DataFrame
        df = pd.DataFrame(bias_data)
        
        # Save files
        csv_file = self.tables_dir / "bias_analysis.csv"
        latex_file = self.tables_dir / "bias_analysis.tex"
        
        df.to_csv(csv_file, index=False)
        
        latex_table = df.to_latex(index=False, float_format="%.3f")
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        print(f"Bias analysis table saved to: {csv_file}")
        return df
    
    def create_performance_comparison_plot(self, results):
        """Create performance comparison visualization"""
        print("Creating performance comparison plot...")
        
        # Prepare data for plotting
        plot_data = []
        
        for domain, domain_results in results.items():
            if domain == 'metadata':
                continue
                
            for frontend, frontend_results in domain_results.items():
                accuracy = frontend_results.get('accuracy', 0.0)
                group_gap = frontend_results.get('group_gap', 0.0)
                
                plot_data.append({
                    'Domain': domain.capitalize(),
                    'Frontend': frontend.upper(),
                    'Accuracy': accuracy,
                    'Group Gap': group_gap
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create subplot for each domain
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        domains = ['Speech', 'Music', 'Scenes']
        
        for i, domain in enumerate(domains):
            domain_data = df[df['Domain'] == domain]
            
            if not domain_data.empty:
                axes[i].scatter(domain_data['Accuracy'], domain_data['Group Gap'], 
                              s=100, alpha=0.7)
                
                # Add frontend labels
                for idx, row in domain_data.iterrows():
                    axes[i].annotate(row['Frontend'], 
                                   (row['Accuracy'], row['Group Gap']),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=10)
                
                axes[i].set_title(f'{domain} Domain')
                axes[i].set_xlabel('Accuracy')
                axes[i].set_ylabel('Group Gap')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.figures_dir / "performance_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison plot saved to: {plot_file}")
    
    def create_bias_heatmap(self, results):
        """Create bias metrics heatmap"""
        print("Creating bias metrics heatmap...")
        
        # Prepare data
        domains = []
        frontends = []
        group_gaps = []
        
        for domain, domain_results in results.items():
            if domain == 'metadata':
                continue
                
            for frontend, frontend_results in domain_results.items():
                domains.append(domain.capitalize())
                frontends.append(frontend.upper())
                group_gaps.append(frontend_results.get('group_gap', 0.0))
        
        # Create pivot table
        df = pd.DataFrame({
            'Domain': domains,
            'Frontend': frontends, 
            'Group Gap': group_gaps
        })
        
        pivot_df = df.pivot(index='Frontend', columns='Domain', values='Group Gap')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='Reds', fmt='.3f',
                   cbar_kws={'label': 'Group Gap'})
        plt.title('Cultural Bias Across Domains and Front-ends')
        plt.tight_layout()
        
        # Save plot
        plot_file = self.figures_dir / "bias_heatmap.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Bias heatmap saved to: {plot_file}")
    
    def generate_statistical_analysis(self, results):
        """Generate statistical analysis report"""
        print("Generating statistical analysis...")
        
        # Collect all performance data
        all_data = []
        
        for domain, domain_results in results.items():
            if domain == 'metadata':
                continue
                
            for frontend, frontend_results in domain_results.items():
                if 'group_performance' in frontend_results:
                    group_perf = frontend_results['group_performance']
                    
                    for group, performance in group_perf.items():
                        all_data.append({
                            'domain': domain,
                            'frontend': frontend,
                            'group': group,
                            'performance': performance
                        })
        
        df = pd.DataFrame(all_data)
        
        # Statistical tests
        statistical_results = {}
        
        # ANOVA tests for each domain
        for domain in df['domain'].unique():
            domain_data = df[df['domain'] == domain]
            
            # Test for frontend effect
            frontend_groups = [group['performance'].values 
                             for name, group in domain_data.groupby('frontend')]
            
            if len(frontend_groups) > 1:
                f_stat, p_value = stats.f_oneway(*frontend_groups)
                statistical_results[f'{domain}_frontend_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # Save statistical results
        stats_file = self.reports_dir / "statistical_analysis.json"
        with open(stats_file, 'w') as f:
            json.dump(statistical_results, f, indent=2)
        
        print(f"Statistical analysis saved to: {stats_file}")
        return statistical_results
    
    def generate_comprehensive_report(self, results):
        """Generate comprehensive analysis report"""
        print("Generating comprehensive report...")
        
        report_lines = []
        report_lines.append("# FairAudioBench: Cross-Cultural Bias Analysis Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Executive summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        
        total_experiments = 0
        domains_tested = []
        frontends_tested = set()
        
        for domain, domain_results in results.items():
            if domain != 'metadata':
                domains_tested.append(domain)
                for frontend in domain_results.keys():
                    frontends_tested.add(frontend)
                    total_experiments += 1
        
        report_lines.append(f"- Total experiments conducted: {total_experiments}")
        report_lines.append(f"- Domains evaluated: {', '.join(domains_tested)}")
        report_lines.append(f"- Front-ends tested: {', '.join(sorted(frontends_tested))}")
        report_lines.append("")
        
        # Key findings
        report_lines.append("## Key Findings")
        report_lines.append("")
        
        # Analyze performance gaps
        gap_analysis = self.analyze_performance_gaps(results)
        
        for domain, domain_analysis in gap_analysis.items():
            report_lines.append(f"### {domain.capitalize()} Domain")
            
            # Find best and worst performing frontends
            frontend_gaps = {fe: data['group_gap'] for fe, data in domain_analysis.items()}
            
            if frontend_gaps:
                best_frontend = min(frontend_gaps, key=frontend_gaps.get)
                worst_frontend = max(frontend_gaps, key=frontend_gaps.get)
                
                report_lines.append(f"- Most fair frontend: {best_frontend.upper()} (gap: {frontend_gaps[best_frontend]:.3f})")
                report_lines.append(f"- Least fair frontend: {worst_frontend.upper()} (gap: {frontend_gaps[worst_frontend]:.3f})")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        report_lines.append("1. **Frontend Selection**: Consider cultural diversity when choosing audio front-ends")
        report_lines.append("2. **Bias Mitigation**: Implement bias-aware training strategies")
        report_lines.append("3. **Dataset Diversity**: Ensure balanced representation across cultural groups")
        report_lines.append("4. **Evaluation Protocols**: Include fairness metrics in model evaluation")
        report_lines.append("")
        
        # Technical details
        report_lines.append("## Technical Details")
        report_lines.append("")
        
        if 'metadata' in results:
            metadata = results['metadata']
            report_lines.append(f"- Experiment date: {metadata.get('timestamp', 'N/A')}")
            report_lines.append(f"- Configuration: {metadata.get('config_file', 'N/A')}")
            report_lines.append(f"- Random seed: {metadata.get('random_seed', 'N/A')}")
        
        # Save report
        report_file = self.reports_dir / "comprehensive_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Comprehensive report saved to: {report_file}")
        
        # Also save as text file
        txt_file = self.reports_dir / "comprehensive_report.txt"
        with open(txt_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return report_lines
    
    def run_complete_analysis(self, experiment_dir):
        """Run complete analysis pipeline"""
        print("Running complete FairAudioBench analysis...")
        print("=" * 60)
        
        # Load results
        results = self.load_experiment_results(experiment_dir)
        
        if not any(domain in results for domain in ['speech', 'music', 'scenes']):
            print("No valid experiment results found!")
            return None
        
        # Generate all analyses
        main_table = self.generate_main_results_table(results)
        bias_table = self.generate_bias_analysis_table(results)
        self.create_performance_comparison_plot(results)
        self.create_bias_heatmap(results)
        statistical_results = self.generate_statistical_analysis(results)
        report = self.generate_comprehensive_report(results)
        
        print("\n" + "=" * 60)
        print("Analysis complete! Results saved to:")
        print(f"  - Tables: {self.tables_dir}")
        print(f"  - Figures: {self.figures_dir}")
        print(f"  - Reports: {self.reports_dir}")
        print("=" * 60)
        
        return {
            'main_table': main_table,
            'bias_table': bias_table,
            'statistical_results': statistical_results,
            'report': report
        }

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze FairAudioBench experiment results")
    parser.add_argument("--experiment_dir", type=str, default="../experiments",
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="../results",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = FairAudioBenchAnalyzer(args.output_dir)
    
    # Run analysis
    results = analyzer.run_complete_analysis(args.experiment_dir)
    
    if results:
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed - no valid results found.")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
