#!/usr/bin/env python3
"""
FairAudioBench Data Preprocessing Script
Based on reference implementation for balanced evaluation protocols

This script implements the balanced evaluation protocols described in the ICASSP 2026 paper:
- Standardized sample sizes across all datasets to eliminate volume bias
- Consistent audio format and sample rates
- Stratified sampling for fair cross-cultural comparison
- Export only essential data for experiments

Author: Shivam Chauhan, Ajay Pundhir
Organization: Presight AI, Abu Dhabi, UAE
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import torchaudio
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import shutil

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [PROCESSED_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paper specifications for balanced evaluation
PAPER_CONFIG = {
    "speech": {
        "target_samples": 2000,  # Exactly 2,000 samples per language
        "target_sr": 22050,      # 22kHz sample rate (CommonVoice standard)
        "min_duration": 1.0,     # Minimum 1s duration
        "max_duration": 8.0,     # Maximum 8s duration
        "avg_duration": 4.2      # Target average 4.2s duration
    },
    "music": {
        "target_samples": 300,   # Exactly 300 recordings per tradition
        "target_sr": 22050,      # Standardized to 22kHz
        "segment_duration": 30.0, # 30-second segments
        "min_samples_per_class": 10  # Minimum samples per modal class
    },
    "scenes": {
        "target_samples": 100,   # Exactly 100 recordings per city
        "target_sr": 48000,      # 48kHz (TAU dataset standard)
        "segment_duration": 10.0, # 10-second segments
        "samples_per_scene": 10   # 10 per scene type per city
    }
}

# Target languages from paper
TONAL_LANGUAGES = ['vi', 'th', 'yue', 'pa-IN']  # 4 tonal languages
NON_TONAL_LANGUAGES = ['en', 'es', 'de', 'fr', 'it', 'nl']  # 6 non-tonal languages
ALL_TARGET_LANGUAGES = TONAL_LANGUAGES + NON_TONAL_LANGUAGES

# Music traditions
WESTERN_MUSIC = ['gtzan', 'fma']
NON_WESTERN_MUSIC = ['carnatic', 'hindustani', 'turkish_makam', 'arab_andalusian']

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed} for reproducible sampling")

def load_audio_file(file_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """Load and resample audio file to target sample rate."""
    try:
        # Try soundfile first (faster for most formats)
        audio, sr = sf.read(str(file_path))
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio, target_sr
    except Exception as e:
        # Fallback to librosa
        try:
            audio, sr = librosa.load(str(file_path), sr=target_sr)
            return audio, target_sr
        except Exception as e2:
            logger.error(f"Failed to load {file_path}: {e2}")
            return None, None

def balanced_sample_selection(samples: List[Dict], target_count: int, target_avg: float) -> List[Dict]:
    """
    Select balanced samples to match target count and average duration.
    This implements the paper's balanced sampling strategy.
    """
    if len(samples) <= target_count:
        return samples
    
    # Sort by duration for balanced selection
    samples = sorted(samples, key=lambda x: x['duration'])
    
    # Greedy selection to match target average duration
    selected_samples = []
    remaining_samples = samples.copy()
    
    for _ in range(target_count):
        if not remaining_samples:
            break
            
        # Calculate current average
        if selected_samples:
            current_avg = sum(s['duration'] for s in selected_samples) / len(selected_samples)
        else:
            current_avg = 0
        
        # Select sample to balance towards target average
        if current_avg < target_avg:
            # Need longer samples
            candidate = max(remaining_samples, key=lambda x: x['duration'])
        else:
            # Need shorter samples
            candidate = min(remaining_samples, key=lambda x: x['duration'])
        
        selected_samples.append(candidate)
        remaining_samples.remove(candidate)
    
    return selected_samples

def preprocess_speech_language(lang_code: str) -> bool:
    """Preprocess speech dataset for a specific language according to paper specifications."""
    logger.info(f"Preprocessing speech dataset for language: {lang_code}")
    
    input_dir = DATA_DIR / f"commonvoice_{lang_code}"
    output_dir = PROCESSED_DIR / "speech" / lang_code
    
    if not input_dir.exists():
        logger.warning(f"Input directory not found: {input_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    config = PAPER_CONFIG["speech"]
    
    # Load metadata
    metadata_path = input_dir / "metadata.csv"
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return False
    
    metadata_df = pd.read_csv(metadata_path)
    logger.info(f"Found {len(metadata_df)} samples for {lang_code}")
    
    # Process audio files and collect valid samples
    valid_samples = []
    
    for idx, row in metadata_df.iterrows():
        audio_path = Path(row['audio_path'])
        if not audio_path.exists():
            continue
        
        # Load and check audio
        audio, sr = load_audio_file(audio_path, config["target_sr"])
        if audio is None:
            continue
        
        duration = len(audio) / sr
        
        # Filter by duration criteria
        if duration < config["min_duration"] or duration > config["max_duration"]:
            continue
        
        valid_samples.append({
            'audio': audio,
            'text': row['text'],
            'duration': duration,
            'idx': idx
        })
    
    logger.info(f"Found {len(valid_samples)} valid samples for {lang_code}")
    
    if len(valid_samples) < config["target_samples"]:
        logger.warning(f"Only {len(valid_samples)} samples available (target: {config['target_samples']})")
    
    # Balanced selection
    selected_samples = balanced_sample_selection(
        valid_samples, 
        min(config["target_samples"], len(valid_samples)), 
        config["avg_duration"]
    )
    
    logger.info(f"Selected {len(selected_samples)} samples for {lang_code}")
    
    # Save processed samples
    processed_metadata = []
    
    for i, sample in enumerate(selected_samples):
        # Save audio file
        output_filename = f"{lang_code}_{i:06d}.wav"
        output_path = output_dir / output_filename
        
        try:
            sf.write(str(output_path), sample['audio'], config["target_sr"])
            
            # Add to metadata
            processed_metadata.append({
                'audio_path': str(output_path),
                'filename': output_filename,
                'text': sample['text'],
                'language': lang_code,
                'is_tonal': lang_code in TONAL_LANGUAGES,
                'duration': sample['duration'],
                'sample_rate': config["target_sr"],
                'original_idx': sample['idx']
            })
            
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {e}")
            continue
    
    # Save metadata
    if processed_metadata:
        metadata_df = pd.DataFrame(processed_metadata)
        metadata_df.to_csv(output_dir / "metadata.csv", index=False)
        
        # Save summary statistics
        avg_duration = metadata_df['duration'].mean()
        summary = {
            'language': lang_code,
            'is_tonal': lang_code in TONAL_LANGUAGES,
            'total_samples': len(metadata_df),
            'target_samples': config["target_samples"],
            'avg_duration': avg_duration,
            'target_avg_duration': config["avg_duration"],
            'sample_rate': config["target_sr"],
            'processing_date': datetime.now().isoformat()
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"SUCCESS: {lang_code} - {len(metadata_df)} samples, avg duration {avg_duration:.2f}s")
        return True
    else:
        logger.error(f"No valid processed samples for {lang_code}")
        return False

def create_balanced_splits(output_dir: Path, test_size: float = 0.2, val_size: float = 0.1):
    """Create balanced train/validation/test splits for all datasets."""
    logger.info("Creating balanced dataset splits...")
    
    splits_info = {}
    
    # Process speech datasets
    speech_dir = output_dir / "speech"
    if speech_dir.exists():
        for lang_dir in speech_dir.iterdir():
            if lang_dir.is_dir():
                metadata_path = lang_dir / "metadata.csv"
                if metadata_path.exists():
                    df = pd.read_csv(metadata_path)
                    
                    # Stratified split (if possible)
                    train_df, temp_df = train_test_split(
                        df, test_size=test_size + val_size, random_state=42, shuffle=True
                    )
                    val_df, test_df = train_test_split(
                        temp_df, test_size=test_size/(test_size + val_size), random_state=42, shuffle=True
                    )
                    
                    # Save splits
                    train_df.to_csv(lang_dir / "train.csv", index=False)
                    val_df.to_csv(lang_dir / "val.csv", index=False)
                    test_df.to_csv(lang_dir / "test.csv", index=False)
                    
                    splits_info[f"speech_{lang_dir.name}"] = {
                        'train': len(train_df),
                        'val': len(val_df), 
                        'test': len(test_df)
                    }
    
    # Save splits summary
    with open(output_dir / "splits_summary.json", 'w') as f:
        json.dump(splits_info, f, indent=2)
    
    logger.info(f"Created splits for {len(splits_info)} datasets")

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess datasets for FairAudioBench")
    parser.add_argument("--languages", nargs="+", default=ALL_TARGET_LANGUAGES,
                       help="Language codes to process")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="./processed_data",
                       help="Output directory for processed data")
    parser.add_argument("--create_splits", action="store_true",
                       help="Create train/val/test splits")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set global paths
    global DATA_DIR, PROCESSED_DIR
    DATA_DIR = Path(args.data_dir)
    PROCESSED_DIR = Path(args.output_dir)
    PROCESSED_DIR.mkdir(exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    print("=== FairAudioBench Data Preprocessing ===\n")
    
    # Process speech datasets
    speech_results = {}
    for lang in args.languages:
        if lang in ALL_TARGET_LANGUAGES:
            success = preprocess_speech_language(lang)
            speech_results[lang] = success
        else:
            logger.warning(f"Language {lang} not in target languages list")
    
    # Create balanced splits
    if args.create_splits:
        create_balanced_splits(PROCESSED_DIR)
    
    # Summary
    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Speech datasets: {sum(speech_results.values())}/{len(speech_results)} successful")
    
    successful_langs = [k for k, v in speech_results.items() if v]
    if successful_langs:
        print(f"Successfully processed: {', '.join(successful_langs)}")
    
    failed_langs = [k for k, v in speech_results.items() if not v]
    if failed_langs:
        print(f"Failed to process: {', '.join(failed_langs)}")
        print("Check that datasets were downloaded and are in the correct format")
    
    print(f"\nProcessed data saved to: {PROCESSED_DIR}")
    
    if args.create_splits:
        print("✓ Train/validation/test splits created")
    else:
        print("Note: Use --create_splits to create train/val/test splits")

if __name__ == "__main__":
    main()
