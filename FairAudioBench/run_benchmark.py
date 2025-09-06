#!/usr/bin/env python3
"""
FairAudioBench: Complete Evaluation Pipeline
Runs the entire evaluation pipeline from data download to fairness analysis

EVALUATION vs TRAINING modes:
- evaluation-only: Focus on bias measurement (recommended for most users)
- training: Include model training (for research/reference)
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FairAudioBenchPipeline:
    """Complete pipeline for FairAudioBench evaluation."""
    
    def __init__(self, base_dir: str = ".", evaluation_only: bool = False):
        self.base_dir = Path(base_dir)
        self.evaluation_only = evaluation_only
        
        self.data_dir = self.base_dir / "data"
        self.processed_data_dir = self.base_dir / "processed_data"
        self.models_dir = self.base_dir / "trained_models"
        self.results_dir = self.base_dir / "results"
        self.reports_dir = self.base_dir / "fairness_reports"
        
        # Create directories
        for dir_path in [self.data_dir, self.processed_data_dir, self.models_dir, 
                        self.results_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        if evaluation_only:
            logger.info("🔍 Running in EVALUATION-ONLY mode (no training)")
        else:
            logger.info("🔬 Running in FULL mode (including training)")
    
    def run_command(self, command: str, description: str) -> bool:
        """Run a shell command and log results."""
        logger.info(f"Starting: {description}")
        logger.info(f"Command: {command}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=self.base_dir
            )
            logger.info(f"✓ Completed: {description}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed: {description}")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def step1_download_datasets(self) -> bool:
        """Step 1: Download datasets."""
        logger.info("="*50)
        logger.info("STEP 1: DOWNLOADING DATASETS")
        logger.info("="*50)
        
        command = f"python Data/download_datasets.py --output-dir {self.data_dir}"
        return self.run_command(command, "Dataset download and setup")
    
    def step2_preprocess_datasets(self) -> bool:
        """Step 2: Preprocess and balance datasets."""
        logger.info("="*50)
        logger.info("STEP 2: PREPROCESSING DATASETS")
        logger.info("="*50)
        
        command = (f"python Data/preprocess_datasets.py "
                  f"--input-dir {self.data_dir} "
                  f"--output-dir {self.processed_data_dir} "
                  f"--samples-per-group 1000")
        return self.run_command(command, "Dataset preprocessing and balancing")
    
    def step3_train_models(self, quick_test: bool = False) -> bool:
        """Step 3: Train all front-end models."""
        logger.info("="*50)
        logger.info("STEP 3: TRAINING MODELS")
        logger.info("="*50)
        
        quick_flag = "--quick-test" if quick_test else ""
        command = (f"python Implementations/train_models.py "
                  f"--data-dir {self.processed_data_dir} "
                  f"--output-dir {self.models_dir} "
                  f"--config config.json {quick_flag}")
        return self.run_command(command, "Model training")
    
    def step4_evaluate_models(self) -> bool:
        """Step 4: Evaluate trained models."""
        logger.info("="*50)
        logger.info("STEP 4: EVALUATING MODELS")
        logger.info("="*50)
        
        results_file = self.results_dir / "evaluation_results.json"
        command = (f"python Implementations/evaluate_models.py "
                  f"--model-dir {self.models_dir} "
                  f"--data-dir {self.processed_data_dir} "
                  f"--output-file {results_file} "
                  f"--report-dir {self.results_dir}")
        return self.run_command(command, "Model evaluation")
    
    def step5_fairness_analysis(self) -> bool:
        """Step 5: Compute fairness metrics."""
        logger.info("="*50)
        logger.info("STEP 5: FAIRNESS ANALYSIS")
        logger.info("="*50)
        
        frontends = ['standard_mel', 'erb_scale', 'gammatone', 'cochlear', 'bark_scale', 'learnable_mel']
        
        # Individual fairness reports
        for frontend in frontends:
            command = (f"python Scripts/fairness_metrics.py "
                      f"--results-dir {self.results_dir} "
                      f"--output-dir {self.reports_dir} "
                      f"--experiment-name evaluation_results "
                      f"--model-names {frontend}")
            
            success = self.run_command(command, f"Fairness analysis for {frontend}")
            if not success:
                logger.warning(f"Fairness analysis failed for {frontend}")
        
        # Comparative analysis
        models_str = " ".join(frontends)
        command = (f"python Scripts/fairness_metrics.py "
                  f"--results-dir {self.results_dir} "
                  f"--output-dir {self.reports_dir} "
                  f"--experiment-name evaluation_results "
                  f"--model-names {models_str} "
                  f"--compare-models")
        
        return self.run_command(command, "Comparative fairness analysis")
    
    def generate_final_report(self) -> bool:
        """Generate final benchmark report."""
        logger.info("="*50)
        logger.info("GENERATING FINAL REPORT")
        logger.info("="*50)
        
        try:
            # Load evaluation results
            results_file = self.results_dir / "evaluation_results.json"
            if not results_file.exists():
                logger.error("Evaluation results not found")
                return False
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Generate comprehensive report
            report = {
                "fairaudiobench_version": "1.0.0",
                "benchmark_date": "2025-09-06",
                "summary": {
                    "total_models": len([k for k in results.keys() if results[k]]),
                    "total_domains": 3,
                    "total_experiments": sum(len(v) for v in results.values()),
                },
                "model_performance": {},
                "fairness_summary": {},
                "recommendations": []
            }
            
            # Aggregate performance by frontend
            for frontend, domain_results in results.items():
                if not domain_results:
                    continue
                
                frontend_metrics = {
                    "overall_accuracy": [],
                    "domain_accuracies": domain_results
                }
                
                for domain, metrics in domain_results.items():
                    if metrics:
                        frontend_metrics["overall_accuracy"].append(metrics.get("overall_accuracy", 0))
                
                if frontend_metrics["overall_accuracy"]:
                    report["model_performance"][frontend] = {
                        "mean_accuracy": sum(frontend_metrics["overall_accuracy"]) / len(frontend_metrics["overall_accuracy"]),
                        "domain_accuracies": frontend_metrics["domain_accuracies"]
                    }
            
            # Add recommendations
            if report["model_performance"]:
                best_frontend = max(
                    report["model_performance"].items(),
                    key=lambda x: x[1]["mean_accuracy"]
                )[0]
                
                report["recommendations"] = [
                    f"Best overall performance: {best_frontend}",
                    "Check fairness reports for bias analysis",
                    "Consider ensemble methods for improved fairness",
                    "Validate on additional cultural groups for robustness"
                ]
            
            # Save final report
            final_report_file = self.base_dir / "FairAudioBench_Report.json"
            with open(final_report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Final report saved to {final_report_file}")
            
            # Print summary
            print("\n" + "="*60)
            print("FAIRAUDIOBENCH EVALUATION COMPLETE")
            print("="*60)
            print(f"Total Models Evaluated: {report['summary']['total_models']}")
            print(f"Total Experiments: {report['summary']['total_experiments']}")
            
            if report["model_performance"]:
                print("\nModel Performance Summary:")
                for frontend, metrics in report["model_performance"].items():
                    print(f"  {frontend}: {metrics['mean_accuracy']:.4f}")
            
            print(f"\nDetailed results: {self.results_dir}")
            print(f"Fairness reports: {self.reports_dir}")
            print(f"Final report: {final_report_file}")
            print("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")
            return False
    
    def run_full_pipeline(self, quick_test: bool = False, skip_data: bool = False) -> bool:
        """Run the complete FairAudioBench pipeline."""
        logger.info("Starting FairAudioBench Complete Pipeline")
        
        steps = []
        
        if not skip_data:
            steps.extend([
                (self.step1_download_datasets, "Dataset Download"),
                (self.step2_preprocess_datasets, "Dataset Preprocessing")
            ])
        
        steps.extend([
            (lambda: self.step3_train_models(quick_test), "Model Training"),
            (self.step4_evaluate_models, "Model Evaluation"),
            (self.step5_fairness_analysis, "Fairness Analysis"),
            (self.generate_final_report, "Final Report Generation")
        ])
        
        success_count = 0
        
        for step_func, step_name in steps:
            logger.info(f"\nExecuting: {step_name}")
            success = step_func()
            
            if success:
                success_count += 1
                logger.info(f"✓ {step_name} completed successfully")
            else:
                logger.error(f"✗ {step_name} failed")
                # Continue with remaining steps even if one fails
                continue
        
        # Final summary
        total_steps = len(steps)
        logger.info(f"\nPipeline Summary: {success_count}/{total_steps} steps completed successfully")
        
        if success_count == total_steps:
            logger.info("🎉 FairAudioBench pipeline completed successfully!")
            return True
        else:
            logger.warning(f"⚠️  Pipeline completed with {total_steps - success_count} failures")
            return False

def main():
    """Main function to run FairAudioBench pipeline."""
    parser = argparse.ArgumentParser(description="FairAudioBench Complete Pipeline")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for FairAudioBench (default: current directory)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with limited data"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data download and preprocessing (use existing processed data)"
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Run in evaluation-only mode (no training, focus on bias measurement)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["speech"],
        help="Domains to evaluate (speech, music, scenes)"
    )
    parser.add_argument(
        "--frontends",
        nargs="+",
        default=["mel", "erb", "bark"],
        help="Frontends to evaluate"
    )
    parser.add_argument(
        "--step",
        choices=['download', 'preprocess', 'train', 'evaluate', 'fairness', 'report'],
        help="Run only a specific step"
    )
    
    args = parser.parse_args()
    
    # Print mode information
    if args.evaluation_only:
        print("🔍 FairAudioBench - EVALUATION MODE")
        print("Focus: Measuring bias across audio front-ends")
        print("Domains:", ", ".join(args.domains))
        print("Frontends:", ", ".join(args.frontends))
        print()
    else:
        print("🔬 FairAudioBench - FULL MODE (including training)")
        print("Note: Use --evaluation-only for faster bias measurement")
        print()
    
    # Create pipeline
    pipeline = FairAudioBenchPipeline(args.base_dir, args.evaluation_only)
    
    if args.step:
        # Run specific step
        step_functions = {
            'download': pipeline.step1_download_datasets,
            'preprocess': pipeline.step2_preprocess_datasets,
            'train': lambda: pipeline.step3_train_models(args.quick_test) if not args.evaluation_only else True,
            'evaluate': pipeline.step4_evaluate_models,
            'fairness': pipeline.step5_fairness_analysis,
            'report': pipeline.generate_final_report
        }
        
        if args.step in step_functions:
            success = step_functions[args.step]()
            sys.exit(0 if success else 1)
        else:
            logger.error(f"Unknown step: {args.step}")
            sys.exit(1)
    else:
        # Run full pipeline
        success = pipeline.run_full_pipeline(args.quick_test, args.skip_data)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
