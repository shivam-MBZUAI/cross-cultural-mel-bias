#!/usr/bin/env python3

"""
FairAudioBench: Complete Experiment Orchestration
Main entry point for all benchmark experiments
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def setup_logging():
    """Setup logging configuration"""
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"fairaudiobench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_data_availability(data_dir):
    """Check if processed data is available"""
    required_dirs = [
        "speech",
        "music", 
        "scenes"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not (data_dir / dir_name).exists():
            missing_dirs.append(dir_name)
    
    return missing_dirs

def run_data_download():
    """Run data download script"""
    logger.info("Starting data download...")
    download_script = current_dir / "scripts" / "download_datasets.py"
    
    import subprocess
    result = subprocess.run([
        sys.executable, str(download_script), 
        "--all", "--batch_size", "500"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Data download completed successfully")
        return True
    else:
        logger.error(f"Data download failed: {result.stderr}")
        return False

def run_data_preprocessing():
    """Run data preprocessing script"""
    logger.info("Starting data preprocessing...")
    preprocess_script = current_dir / "scripts" / "preprocess_datasets.py"
    
    import subprocess
    result = subprocess.run([
        sys.executable, str(preprocess_script),
        "--all", "--max_workers", "4"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Data preprocessing completed successfully")
        return True
    else:
        logger.error(f"Data preprocessing failed: {result.stderr}")
        return False

def run_experiments(domains, frontends, epochs):
    """Run main experiments"""
    logger.info(f"Starting experiments: domains={domains}, frontends={frontends}, epochs={epochs}")
    
    experiment_script = current_dir / "run_experiments.py"
    
    import subprocess
    cmd = [
        sys.executable, str(experiment_script),
        "--domains", " ".join(domains),
        "--frontends", " ".join(frontends), 
        "--epochs", str(epochs),
        "--data_dir", str(current_dir.parent / "processed_data")
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Experiments completed successfully")
        return True
    else:
        logger.error(f"Experiments failed: {result.stderr}")
        return False

def run_analysis():
    """Run results analysis and demonstration"""
    logger.info("Running results analysis...")
    
    analysis_script = current_dir / "demonstrate_experiments.py"
    
    import subprocess
    result = subprocess.run([
        sys.executable, str(analysis_script)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Analysis completed successfully")
        print(result.stdout)
        return True
    else:
        logger.error(f"Analysis failed: {result.stderr}")
        return False

def run_tests():
    """Run validation tests"""
    logger.info("Running validation tests...")
    
    test_script = current_dir / "tests" / "test_quick.py"
    
    import subprocess
    result = subprocess.run([
        sys.executable, str(test_script)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Tests passed successfully")
        return True
    else:
        logger.error(f"Tests failed: {result.stderr}")
        return False

def main():
    """Main orchestration function"""
    parser = argparse.ArgumentParser(description="FairAudioBench: Cross-Cultural Audio Bias Benchmark")
    
    parser.add_argument("--mode", choices=["setup", "download", "preprocess", "experiment", "analyze", "test", "full"],
                       default="full", help="Operation mode")
    parser.add_argument("--domains", nargs="+", choices=["speech", "music", "scenes", "all"], 
                       default=["all"], help="Domains to experiment with")
    parser.add_argument("--frontends", nargs="+", 
                       choices=["mel", "erb", "bark", "cqt", "leaf", "sincnet", "all"],
                       default=["all"], help="Front-ends to evaluate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--skip_download", action="store_true", help="Skip data download")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--data_dir", type=str, help="Path to processed data directory")
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging()
    
    logger.info("FairAudioBench: Cross-Cultural Audio Bias Benchmark")
    logger.info("=" * 60)
    
    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = current_dir.parent / "processed_data"
    
    # Setup mode
    if args.mode in ["setup", "full"]:
        logger.info("Setting up FairAudioBench environment...")
        
        # Create necessary directories
        for dir_name in ["data", "experiments", "results", "logs"]:
            (current_dir / dir_name).mkdir(exist_ok=True)
        
        # Run basic validation
        if not run_tests():
            logger.error("Setup validation failed")
            return 1
    
    # Download mode
    if args.mode in ["download", "full"] and not args.skip_download:
        missing_dirs = check_data_availability(data_dir)
        if missing_dirs:
            logger.info(f"Missing data directories: {missing_dirs}")
            if not run_data_download():
                logger.error("Data download failed")
                return 1
        else:
            logger.info("Data already available, skipping download")
    
    # Preprocessing mode
    if args.mode in ["preprocess", "full"] and not args.skip_preprocess:
        if not run_data_preprocessing():
            logger.error("Data preprocessing failed")
            return 1
    
    # Experiment mode
    if args.mode in ["experiment", "full"]:
        # Validate data availability
        missing_dirs = check_data_availability(data_dir)
        if missing_dirs:
            logger.error(f"Missing required data directories: {missing_dirs}")
            logger.error("Please run data download and preprocessing first")
            return 1
        
        domains = args.domains if "all" not in args.domains else ["speech", "music", "scenes"]
        frontends = args.frontends if "all" not in args.frontends else ["mel", "erb", "bark", "cqt", "leaf", "sincnet"]
        
        if not run_experiments(domains, frontends, args.epochs):
            logger.error("Experiments failed")
            return 1
    
    # Analysis mode
    if args.mode in ["analyze", "full"]:
        if not run_analysis():
            logger.error("Analysis failed")
            return 1
    
    # Test mode
    if args.mode in ["test"]:
        if not run_tests():
            logger.error("Tests failed")
            return 1
    
    logger.info("FairAudioBench operations completed successfully!")
    logger.info("\nResults are available in:")
    logger.info(f"  - Experiments: {current_dir / 'experiments'}")
    logger.info(f"  - Results: {current_dir / 'results'}")
    logger.info(f"  - Logs: {current_dir / 'logs'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
