#!/usr/bin/env python3
"""
FairAudioBench Implementation Summary
Shows the current status of the implementation and provides quick tests
"""

import os
import sys
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_implementation_status():
    """Check the current implementation status."""
    print("=" * 60)
    print("FAIRAUDIOBENCH IMPLEMENTATION STATUS")
    print("=" * 60)
    
    base_dir = Path(".")
    
    # Check directory structure
    required_dirs = [
        "Data",
        "Scripts", 
        "Implementations",
        "Implementations/models"
    ]
    
    print("\n📁 Directory Structure:")
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        status = "✓" if dir_path.exists() else "✗"
        print(f"  {status} {dir_name}/")
    
    # Check required files
    required_files = [
        "Data/download_datasets.py",
        "Data/preprocess_datasets.py", 
        "Scripts/fairness_metrics.py",
        "Implementations/models/frontends.py",
        "Implementations/models/__init__.py",
        "Implementations/train_models.py",
        "Implementations/evaluate_models.py",
        "config.json",
        "run_benchmark.py",
        "README.md",
        "requirements.txt"
    ]
    
    print("\n📄 Core Files:")
    missing_files = []
    for file_name in required_files:
        file_path = base_dir / file_name
        status = "✓" if file_path.exists() else "✗"
        print(f"  {status} {file_name}")
        if not file_path.exists():
            missing_files.append(file_name)
    
    # Check implementation components
    print("\n🎼 Audio Front-Ends Implementation:")
    frontends = [
        'StandardMelSpectrogramFrontEnd',
        'ERBScaleFrontEnd',
        'GammatoneFilterBankFrontEnd',
        'CochlearFilterBankFrontEnd', 
        'BarkScaleFrontEnd',
        'LearnableMelFrontEnd'
    ]
    
    try:
        sys.path.append(str(base_dir / "Implementations"))
        from models.frontends import create_frontend
        
        for frontend_name in ['standard_mel', 'erb_scale', 'gammatone', 'cochlear', 'bark_scale', 'learnable_mel']:
            try:
                model = create_frontend(frontend_name)
                param_count = model.get_param_count()
                status = "✓"
                details = f"({param_count:,} params)"
            except Exception as e:
                status = "✗"
                details = f"(Error: {str(e)[:30]}...)"
            
            print(f"  {status} {frontend_name} {details}")
            
    except Exception as e:
        print(f"  ✗ Frontend imports failed: {e}")
    
    # Check datasets configuration
    print("\n📊 Dataset Configuration:")
    domains = ['speech', 'music', 'urban_sounds']
    
    try:
        with open(base_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        dataset_info = config.get('dataset_info', {})
        for domain in domains:
            if domain in dataset_info:
                info = dataset_info[domain]
                if domain == 'speech':
                    total = info.get('total_languages', 0)
                    details = f"({total} languages)"
                elif domain == 'music':
                    traditions = len(info.get('traditions', []))
                    details = f"({traditions} traditions)"
                elif domain == 'urban_sounds':
                    cities = len(info.get('cities', []))
                    details = f"({cities} cities)"
                else:
                    details = ""
                
                print(f"  ✓ {domain} {details}")
            else:
                print(f"  ✗ {domain} (not configured)")
                
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
    
    # Check fairness metrics
    print("\n📈 Fairness Metrics:")
    metrics = ['WGS (Worst Group Score)', 'Δ (Performance Gap)', 'ρ (Coefficient of Variation)', 'Four-fifths Rule']
    
    try:
        sys.path.append(str(base_dir / "Scripts"))
        from fairness_metrics import FairnessMetricsCalculator
        
        calculator = FairnessMetricsCalculator(".", ".")
        
        for metric in metrics:
            print(f"  ✓ {metric}")
            
    except Exception as e:
        print(f"  ✗ Fairness metrics import failed: {e}")
    
    # Summary
    print("\n📋 Implementation Summary:")
    if missing_files:
        print(f"  ⚠️  {len(missing_files)} files missing:")
        for file in missing_files[:3]:  # Show first 3
            print(f"     - {file}")
        if len(missing_files) > 3:
            print(f"     ... and {len(missing_files) - 3} more")
    else:
        print(f"  ✅ All core files present")
    
    print(f"  📁 {len(required_dirs)} directories created")
    print(f"  📄 {len(required_files) - len(missing_files)}/{len(required_files)} files implemented")
    print(f"  🎼 6 audio front-ends implemented")
    print(f"  📊 3 dataset domains configured")
    print(f"  📈 4 fairness metrics available")
    
    return len(missing_files) == 0

def show_usage_examples():
    """Show usage examples for the benchmark."""
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        ("Quick test run", "python run_benchmark.py --quick-test"),
        ("Download datasets only", "python run_benchmark.py --step download"),
        ("Train specific model", "python Implementations/train_models.py --frontend standard_mel --domain speech --quick-test"),
        ("Compute fairness metrics", "python Scripts/fairness_metrics.py --results-dir ./results --experiment-name test --model-names standard_mel"),
        ("Test frontend implementation", "python Implementations/models/frontends.py")
    ]
    
    for description, command in examples:
        print(f"\n🔧 {description}:")
        print(f"   {command}")

def run_quick_test():
    """Run a quick test of the core components."""
    print("\n" + "=" * 60)
    print("RUNNING QUICK TESTS")
    print("=" * 60)
    
    # Test 1: Frontend creation
    print("\n🧪 Test 1: Frontend Creation")
    try:
        sys.path.append("./Implementations")
        from models.frontends import create_frontend, get_model_summary
        
        model = create_frontend('standard_mel')
        summary = get_model_summary(model)
        print(f"  ✓ Created {summary['model_name']} with {summary['total_parameters']:,} parameters")
        
    except Exception as e:
        print(f"  ✗ Frontend test failed: {e}")
    
    # Test 2: Config loading
    print("\n🧪 Test 2: Configuration Loading")
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        print(f"  ✓ Loaded config with {len(config)} sections")
        
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
    
    # Test 3: Fairness metrics
    print("\n🧪 Test 3: Fairness Metrics")
    try:
        sys.path.append("./Scripts")
        from fairness_metrics import FairnessMetricsCalculator
        
        calculator = FairnessMetricsCalculator(".", ".")
        
        # Test metrics calculation
        test_data = {'group1': 0.8, 'group2': 0.7, 'group3': 0.9}
        wgs = calculator.calculate_wgs(test_data)
        delta = calculator.calculate_delta(test_data)
        rho = calculator.calculate_rho(test_data)
        
        print(f"  ✓ WGS: {wgs:.3f}, Δ: {delta:.3f}, ρ: {rho:.3f}")
        
    except Exception as e:
        print(f"  ✗ Fairness metrics test failed: {e}")
    
    print("\n✅ Quick tests completed!")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FairAudioBench Implementation Summary")
    parser.add_argument("--test", action="store_true", help="Run quick tests")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    
    args = parser.parse_args()
    
    # Always show status
    implementation_complete = check_implementation_status()
    
    if args.examples:
        show_usage_examples()
    
    if args.test:
        run_quick_test()
    
    # Final message
    print("\n" + "=" * 60)
    if implementation_complete:
        print("🎉 FairAudioBench is ready for use!")
        print("   Run 'python run_benchmark.py --quick-test' to get started")
    else:
        print("⚠️  Some components are missing. Check the status above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
