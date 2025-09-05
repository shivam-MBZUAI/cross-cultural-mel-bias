# FairAudioBench Makefile
# Automates common tasks for the benchmark

.PHONY: help setup install download preprocess test experiment analyze clean all

# Default target
help:
	@echo "FairAudioBench: Cross-Cultural Audio Bias Benchmark"
	@echo "================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup       - Set up the environment and validate installation"
	@echo "  install     - Install Python dependencies"
	@echo "  download    - Download all datasets"
	@echo "  preprocess  - Preprocess all datasets"
	@echo "  validate    - Validate processed datasets"
	@echo "  test        - Run validation tests"
	@echo "  experiment  - Run all experiments (quick version)"
	@echo "  experiment-full - Run full experiments with all epochs"
	@echo "  analyze     - Analyze and generate results"
	@echo "  clean       - Clean up temporary files"
	@echo "  all         - Run complete pipeline (setup to analysis)"
	@echo ""
	@echo "Examples:"
	@echo "  make setup                 # Initial setup"
	@echo "  make download              # Download datasets"
	@echo "  make experiment            # Quick experiments"
	@echo "  make all                   # Complete pipeline"

# Setup environment
setup:
	@echo "Setting up FairAudioBench..."
	python setup.py

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

# Download datasets
download:
	@echo "Downloading datasets..."
	python scripts/download_datasets.py --all --batch_size 500

# Preprocess datasets
preprocess:
	@echo "Preprocessing datasets..."
	python scripts/preprocess_datasets.py --all --max_workers 4

# Validate datasets
validate:
	@echo "Validating datasets..."
	python scripts/validate_datasets.py --data_dir ../processed_data

# Run tests
test:
	@echo "Running validation tests..."
	python tests/test_quick.py

# Run quick experiments (10 epochs)
experiment:
	@echo "Running quick experiments..."
	python run_experiments.py --domains all --frontends mel erb --epochs 10 --data_dir ../processed_data

# Run full experiments (50 epochs)
experiment-full:
	@echo "Running full experiments..."
	python run_experiments.py --domains all --frontends all --epochs 50 --data_dir ../processed_data

# Analyze results
analyze:
	@echo "Analyzing results..."
	python analyze_results.py --experiment_dir experiments --output_dir results

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.log" -delete
	rm -rf logs/*.log

# Complete pipeline
all: setup install download preprocess validate test experiment analyze
	@echo "Complete FairAudioBench pipeline executed!"

# Development targets
dev-test:
	@echo "Running development tests..."
	python tests/test_implementation.py

dev-quick:
	@echo "Quick development test..."
	python fairaudiobench.py --mode test

# Paper reproduction targets
paper-results:
	@echo "Reproducing paper results..."
	python demonstrate_experiments.py

paper-tables:
	@echo "Generating paper tables..."
	python analyze_results.py --experiment_dir experiments --output_dir results
	@echo "LaTeX tables available in results/tables/"

# Data management
clean-data:
	@echo "Cleaning data directories..."
	rm -rf data/
	rm -rf ../processed_data/

reset-experiments:
	@echo "Resetting experiment results..."
	rm -rf experiments/
	rm -rf results/
	mkdir -p experiments results

# Installation checks
check-deps:
	@echo "Checking dependencies..."
	python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	python -c "import librosa; print(f'Librosa: {librosa.__version__}')"
	python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
	python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# GPU check
check-gpu:
	@echo "Checking GPU availability..."
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "README.md contains comprehensive documentation"
	@echo "Configuration details in config/fairaudiobench_config.py"

# Version info
version:
	@echo "FairAudioBench v1.0.0"
	@echo "ICASSP 2026 Paper Implementation"
	@echo "Cross-Cultural Bias in Mel-Scale Audio Front-Ends"
