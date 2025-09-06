"""
Evaluation Script for FairAudioBench Models
Evaluates trained models and generates results for fairness analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import argparse
from collections import defaultdict

from models.frontends import create_frontend
from train_models import FairAudioDataset, ClassificationHead

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FairAudioBenchEvaluator:
    """Evaluator for FairAudioBench models with fairness analysis."""
    
    def __init__(self, model_dir: str, data_dir: str):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Results storage
        self.evaluation_results = {}
        
    def load_model(self, frontend_name: str, domain: str) -> Optional[nn.Module]:
        """Load trained model."""
        model_path = self.model_dir / f"{frontend_name}_{domain}" / "best_model.pth"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            config = checkpoint['config']
            
            # Create dataset to get number of classes
            splits_dir = self.data_dir / "splits" / domain
            test_dataset = FairAudioDataset(
                metadata_file=splits_dir / "test.csv",
                data_dir=self.data_dir,
                domain=domain,
                split="test"
            )
            num_classes = test_dataset.num_classes
            
            # Create model architecture
            frontend = create_frontend(
                frontend_name,
                sample_rate=config['sample_rate'],
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                n_mels=config['n_mels'],
                f_min=config['f_min'],
                f_max=config['f_max'],
                target_params=config['target_params']
            )
            
            classifier = ClassificationHead(
                input_dim=512,
                num_classes=num_classes,
                dropout=config['dropout']
            )
            
            model = nn.Sequential(frontend, classifier)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            logger.info(f"Loaded model: {frontend_name} for {domain}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {frontend_name}_{domain}: {e}")
            return None
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader, 
                      domain: str) -> Dict[str, Any]:
        """Evaluate model and return detailed results."""
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_metadata = []
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for waveforms, labels, metadata in tqdm(data_loader, desc="Evaluating"):
                waveforms = waveforms.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(waveforms)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_metadata.extend(metadata)
                
                # Accuracy calculation
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        overall_accuracy = total_correct / total_samples
        
        # Group-wise analysis
        group_results = self._analyze_group_performance(
            all_predictions, all_labels, all_metadata, domain
        )
        
        results = {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'group_results': group_results,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'metadata': all_metadata
        }
        
        return results
    
    def _analyze_group_performance(self, predictions: List[int], labels: List[int], 
                                  metadata: List[Dict], domain: str) -> Dict[str, Any]:
        """Analyze performance by different demographic groups."""
        group_analysis = {}
        
        # Convert to arrays for easier manipulation
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        if domain == "speech":
            # Group by language and tonal property
            language_results = defaultdict(list)
            tonal_results = defaultdict(list)
            
            for i, meta in enumerate(metadata):
                correct = predictions[i] == labels[i]
                language_results[meta['language']].append(correct)
                tonal_key = "tonal" if meta['is_tonal'] else "non_tonal"
                tonal_results[tonal_key].append(correct)
            
            # Calculate language-wise performance
            lang_performance = {}
            for lang, corrects in language_results.items():
                lang_performance[lang] = {
                    'accuracy': np.mean(corrects),
                    'samples': len(corrects),
                    'is_tonal': metadata[0]['is_tonal'] if metadata else False  # Get from first sample
                }
            
            # Calculate tonal vs non-tonal performance
            tonal_performance = {}
            for tonal_key, corrects in tonal_results.items():
                tonal_performance[tonal_key] = {
                    'accuracy': np.mean(corrects),
                    'samples': len(corrects)
                }
            
            group_analysis = {
                'by_language': lang_performance,
                'by_tonal_property': tonal_performance
            }
            
        elif domain == "music":
            # Group by tradition and cultural origin
            tradition_results = defaultdict(list)
            origin_results = defaultdict(list)
            
            for i, meta in enumerate(metadata):
                correct = predictions[i] == labels[i]
                tradition_results[meta['tradition']].append(correct)
                origin_results[meta['cultural_origin']].append(correct)
            
            # Calculate tradition-wise performance
            tradition_performance = {}
            for tradition, corrects in tradition_results.items():
                tradition_performance[tradition] = {
                    'accuracy': np.mean(corrects),
                    'samples': len(corrects)
                }
            
            # Calculate cultural origin performance
            origin_performance = {}
            for origin, corrects in origin_results.items():
                origin_performance[origin] = {
                    'accuracy': np.mean(corrects),
                    'samples': len(corrects)
                }
            
            group_analysis = {
                'by_tradition': tradition_performance,
                'by_cultural_origin': origin_performance
            }
            
        elif domain == "urban_sounds":
            # Group by city, country, and population size
            city_results = defaultdict(list)
            country_results = defaultdict(list)
            population_results = defaultdict(list)
            
            for i, meta in enumerate(metadata):
                correct = predictions[i] == labels[i]
                city_results[meta['city']].append(correct)
                country_results[meta['country']].append(correct)
                
                # Population grouping
                population = meta.get('population', 0)
                if population < 1000000:
                    pop_group = "small"
                elif population < 3000000:
                    pop_group = "medium"
                else:
                    pop_group = "large"
                population_results[pop_group].append(correct)
            
            # Calculate performances
            city_performance = {}
            for city, corrects in city_results.items():
                city_performance[city] = {
                    'accuracy': np.mean(corrects),
                    'samples': len(corrects)
                }
            
            country_performance = {}
            for country, corrects in country_results.items():
                country_performance[country] = {
                    'accuracy': np.mean(corrects),
                    'samples': len(corrects)
                }
            
            population_performance = {}
            for pop_group, corrects in population_results.items():
                population_performance[pop_group] = {
                    'accuracy': np.mean(corrects),
                    'samples': len(corrects)
                }
            
            group_analysis = {
                'by_city': city_performance,
                'by_country': country_performance,
                'by_population': population_performance
            }
        
        return group_analysis
    
    def evaluate_all_models(self, output_file: str) -> bool:
        """Evaluate all trained models."""
        logger.info("Starting comprehensive model evaluation...")
        
        frontends = ['standard_mel', 'erb_scale', 'gammatone', 'cochlear', 'bark_scale', 'learnable_mel']
        domains = ['speech', 'music', 'urban_sounds']
        
        # Initialize results structure
        experiment_results = {}
        
        for frontend in frontends:
            experiment_results[frontend] = {}
            
            for domain in domains:
                logger.info(f"Evaluating {frontend} on {domain}...")
                
                try:
                    # Load model
                    model = self.load_model(frontend, domain)
                    if model is None:
                        logger.warning(f"Skipping {frontend}_{domain} - model not found")
                        continue
                    
                    # Load test data
                    splits_dir = self.data_dir / "splits" / domain
                    test_dataset = FairAudioDataset(
                        metadata_file=splits_dir / "test.csv",
                        data_dir=self.data_dir,
                        domain=domain,
                        split="test"
                    )
                    
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=32,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True
                    )
                    
                    # Evaluate model
                    results = self.evaluate_model(model, test_loader, domain)
                    
                    # Store results (remove non-serializable items for JSON)
                    serializable_results = {
                        'overall_accuracy': results['overall_accuracy'],
                        'total_samples': results['total_samples'],
                        'group_results': results['group_results']
                    }
                    
                    experiment_results[frontend][domain] = serializable_results
                    
                    logger.info(f"✓ {frontend} on {domain}: {results['overall_accuracy']:.4f} accuracy")
                    
                except Exception as e:
                    logger.error(f"✗ Failed to evaluate {frontend} on {domain}: {e}")
                    continue
        
        # Save results
        try:
            with open(output_file, 'w') as f:
                json.dump(experiment_results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def generate_summary_report(self, results_file: str, output_dir: str) -> bool:
        """Generate summary report from evaluation results."""
        logger.info("Generating summary report...")
        
        try:
            # Load results
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create summary tables
            summary_data = []
            
            for frontend, domain_results in results.items():
                for domain, metrics in domain_results.items():
                    if not metrics:
                        continue
                    
                    row = {
                        'Frontend': frontend,
                        'Domain': domain,
                        'Overall_Accuracy': metrics['overall_accuracy'],
                        'Total_Samples': metrics['total_samples']
                    }
                    
                    # Add group-specific metrics
                    group_results = metrics.get('group_results', {})
                    
                    if domain == "speech":
                        tonal_data = group_results.get('by_tonal_property', {})
                        if 'tonal' in tonal_data and 'non_tonal' in tonal_data:
                            row['Tonal_Accuracy'] = tonal_data['tonal']['accuracy']
                            row['NonTonal_Accuracy'] = tonal_data['non_tonal']['accuracy']
                            row['Tonal_Gap'] = abs(tonal_data['tonal']['accuracy'] - tonal_data['non_tonal']['accuracy'])
                    
                    elif domain == "music":
                        origin_data = group_results.get('by_cultural_origin', {})
                        if origin_data:
                            accuracies = [group['accuracy'] for group in origin_data.values()]
                            row['Min_Cultural_Accuracy'] = min(accuracies)
                            row['Max_Cultural_Accuracy'] = max(accuracies)
                            row['Cultural_Gap'] = max(accuracies) - min(accuracies)
                    
                    elif domain == "urban_sounds":
                        pop_data = group_results.get('by_population', {})
                        if pop_data:
                            accuracies = [group['accuracy'] for group in pop_data.values()]
                            row['Min_Population_Accuracy'] = min(accuracies)
                            row['Max_Population_Accuracy'] = max(accuracies)
                            row['Population_Gap'] = max(accuracies) - min(accuracies)
                    
                    summary_data.append(row)
            
            # Create DataFrame and save
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_path / "evaluation_summary.csv", index=False)
            
            # Generate domain-specific reports
            for domain in ['speech', 'music', 'urban_sounds']:
                domain_df = summary_df[summary_df['Domain'] == domain]
                if not domain_df.empty:
                    domain_df.to_csv(output_path / f"{domain}_evaluation.csv", index=False)
            
            # Generate best model report
            best_models = {}
            for domain in ['speech', 'music', 'urban_sounds']:
                domain_df = summary_df[summary_df['Domain'] == domain]
                if not domain_df.empty:
                    best_idx = domain_df['Overall_Accuracy'].idxmax()
                    best_models[domain] = {
                        'frontend': domain_df.loc[best_idx, 'Frontend'],
                        'accuracy': domain_df.loc[best_idx, 'Overall_Accuracy']
                    }
            
            with open(output_path / "best_models.json", 'w') as f:
                json.dump(best_models, f, indent=2)
            
            logger.info(f"Summary report saved to {output_path}")
            
            # Print summary
            print("\n" + "="*50)
            print("EVALUATION SUMMARY")
            print("="*50)
            
            for domain in ['speech', 'music', 'urban_sounds']:
                domain_df = summary_df[summary_df['Domain'] == domain]
                if not domain_df.empty:
                    print(f"\n{domain.upper()}:")
                    for _, row in domain_df.iterrows():
                        print(f"  {row['Frontend']}: {row['Overall_Accuracy']:.4f}")
                    
                    if domain in best_models:
                        print(f"  Best: {best_models[domain]['frontend']} ({best_models[domain]['accuracy']:.4f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return False

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate FairAudioBench models")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing processed datasets"
    )
    parser.add_argument(
        "--output-file",
        default="./evaluation_results.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--report-dir",
        default="./evaluation_reports",
        help="Directory for summary reports"
    )
    parser.add_argument(
        "--frontend",
        choices=['standard_mel', 'erb_scale', 'gammatone', 'cochlear', 'bark_scale', 'learnable_mel'],
        help="Evaluate specific frontend only"
    )
    parser.add_argument(
        "--domain",
        choices=['speech', 'music', 'urban_sounds'],
        help="Evaluate specific domain only"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = FairAudioBenchEvaluator(args.model_dir, args.data_dir)
    
    # Run evaluation
    if args.frontend and args.domain:
        # Evaluate specific model
        logger.info(f"Evaluating {args.frontend} on {args.domain}")
        model = evaluator.load_model(args.frontend, args.domain)
        if model:
            # Load test data and evaluate
            splits_dir = Path(args.data_dir) / "splits" / args.domain
            test_dataset = FairAudioDataset(
                metadata_file=splits_dir / "test.csv",
                data_dir=args.data_dir,
                domain=args.domain,
                split="test"
            )
            
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
            results = evaluator.evaluate_model(model, test_loader, args.domain)
            
            print(f"Results for {args.frontend} on {args.domain}:")
            print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
            print(f"Total Samples: {results['total_samples']}")
    else:
        # Evaluate all models
        success = evaluator.evaluate_all_models(args.output_file)
        
        if success:
            # Generate summary report
            evaluator.generate_summary_report(args.output_file, args.report_dir)
        
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()
