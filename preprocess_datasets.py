#!/usr/bin/env python3
"""
Cross-Cultural Mel-Scale Audio Frontend Bias Research
Dataset Preprocessing Script for ICASSP 2026 Paper

Creates balanced evaluation datasets exactly as specified in the paper:
- Speech: 2000 samples per language (11 languages: 5 tonal, 6 non-tonal)
- Music: 300 samples per tradition (8 traditions: 4 Western, 4 non-Western)  
- Scenes: 100 samples per city (10 European cities from TAU Urban)

Authors: Shivam Chauhan, Ajay Pundhir
Organization: Presight AI, Abu Dhabi, UAE
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split
import shutil

# Project Configuration
PROJECT_ROOT = Path(__file__).parent
RAW_DATA_DIR = Path("/soot/shivam.chauhan/Sample/data")  # Reference to the raw data
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [PROCESSED_DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Setup logging
def setup_logging():
    """Setup logging configuration."""
    log_file = LOGS_DIR / f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Paper Configuration - Exact specifications from ICASSP 2026
# This is a BIAS EVALUATION study - no training, only balanced evaluation datasets
PAPER_CONFIG = {
    "speech": {
        "target_samples_per_lang": 2000,  # Exactly 2,000 samples per language for evaluation
        "target_sr": 22050,               # 22kHz sample rate
        "min_duration": 1.0,              # Minimum 1s duration
        "max_duration": 10.0,             # Maximum 10s duration
    },
    "music": {
        "target_samples_per_tradition": 300,  # Exactly 300 samples per tradition for evaluation
        "target_sr": 22050,                   # 22kHz sample rate
        "segment_duration": 30.0,             # 30-second segments
    },
    "scenes": {
        "target_samples_per_city": 100,  # Exactly 100 samples per city for evaluation
        "target_sr": 48000,              # 48kHz (TAU standard)
        "segment_duration": 10.0,        # 10-second segments
    }
}

# Target Languages (from ICASSP 2026 paper)
TONAL_LANGUAGES = ['vi', 'th', 'zh-CN', 'pa-IN', 'yue']           # 5 tonal languages
NON_TONAL_LANGUAGES = ['en', 'es', 'de', 'fr', 'it', 'nl']       # 6 non-tonal languages
ALL_TARGET_LANGUAGES = TONAL_LANGUAGES + NON_TONAL_LANGUAGES     # 11 total

# Music Traditions
WESTERN_MUSIC = ['gtzan', 'fma_small']                                               # 2 Western
NON_WESTERN_MUSIC = ['carnatic', 'hindustani', 'turkish_makam', 'arab_andalusian']  # 4 non-Western
ALL_MUSIC_TRADITIONS = WESTERN_MUSIC + NON_WESTERN_MUSIC                            # 6 total (paper mentions 8, we have 6)

# Scene Datasets (TAU Urban 2020 - 10 European cities)
TAU_CITIES = [
    'amsterdam', 'barcelona', 'helsinki', 'lisbon', 'london', 
    'lyon', 'milan', 'prague', 'paris', 'stockholm'
]

def set_seed(seed: int = 42):
    """Set random seed for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed} for reproducible sampling")

def load_audio_safely(file_path: Path, target_sr: int) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Safely load audio file with fallback methods and better error handling."""
    
    # Skip obviously bad files
    if file_path.stat().st_size < 1024:  # Less than 1KB
        logger.debug(f"Skipping tiny file: {file_path} ({file_path.stat().st_size} bytes)")
        return None, None
    
    try:
        # Primary method: soundfile (fastest for most formats)
        audio, sr = sf.read(str(file_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        
        # Check for valid audio data
        if len(audio) == 0:
            logger.debug(f"Empty audio file: {file_path}")
            return None, None
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio, target_sr
    
    except Exception as e1:
        # Common soundfile errors - try librosa
        logger.debug(f"soundfile failed for {file_path}, trying librosa: {str(e1)[:100]}")
        try:
            # Fallback: librosa with more robust loading
            audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True, res_type='kaiser_fast')
            
            # Check for valid audio data
            if len(audio) == 0:
                logger.debug(f"Empty audio after librosa load: {file_path}")
                return None, None
                
            return audio, sr
            
        except Exception as e2:
            # Try one more method for MP3 files
            if file_path.suffix.lower() == '.mp3':
                try:
                    # For problematic MP3s, try loading with different parameters
                    audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True, offset=0.1, duration=None)
                    if len(audio) > 0:
                        return audio, sr
                except:
                    pass
            
            # Log error but don't spam the logs with every failed file
            if "corrupted" not in str(e1).lower() and "illegal" not in str(e1).lower():
                logger.warning(f"Failed to load {file_path}: {str(e2)[:100]}")
            else:
                logger.debug(f"Skipped corrupted file: {file_path}")
            
            return None, None

def validate_audio(audio: np.ndarray, sr: int, min_duration: float, max_duration: float) -> bool:
    """Validate audio meets duration and quality requirements."""
    if audio is None or len(audio) == 0:
        return False
    
    duration = len(audio) / sr
    
    # Check duration constraints
    if duration < min_duration or duration > max_duration:
        return False
    
    # Check for silence (more than 95% zeros)
    if np.sum(np.abs(audio) < 1e-6) / len(audio) > 0.95:
        return False
    
    # Check for clipping (more than 1% at max amplitude)
    if np.sum(np.abs(audio) > 0.99) / len(audio) > 0.01:
        return False
    
    return True

def segment_audio(audio: np.ndarray, sr: int, duration: float, random_segment: bool = True) -> np.ndarray:
    """Extract audio segment of specified duration."""
    target_length = int(duration * sr)
    
    if len(audio) >= target_length:
        if random_segment:
            # Random segment from longer audio
            max_start = len(audio) - target_length
            start_idx = random.randint(0, max_start)
        else:
            # Center segment
            start_idx = (len(audio) - target_length) // 2
        
        return audio[start_idx:start_idx + target_length]
    
    else:
        # Pad shorter audio with silence
        padding = target_length - len(audio)
        return np.pad(audio, (0, padding), mode='constant', constant_values=0)

def create_evaluation_dataset(data_list: List, target_count: int, dataset_name: str = "") -> List:
    """Create balanced evaluation dataset (no train/val/test splits - evaluation only)."""
    random.shuffle(data_list)
    
    if len(data_list) < target_count:
        logger.warning(f"Only {len(data_list)} samples available for {dataset_name}, need {target_count}")
        if len(data_list) < target_count * 0.5:  # Less than 50% of target
            logger.error(f"Insufficient samples for {dataset_name}: {len(data_list)} < {target_count * 0.5:.0f} (50% of target)")
            logger.info(f"Consider downloading more data or reducing target count for {dataset_name}")
        return data_list
    else:
        # Select exactly target_count samples for evaluation
        selected = random.sample(data_list, target_count)
        logger.info(f"Selected {len(selected)} samples for evaluation from {len(data_list)} available")
        return selected

def preprocess_commonvoice_language(lang_code: str) -> bool:
    """Preprocess CommonVoice dataset for a specific language."""
    logger.info(f"Processing CommonVoice {lang_code}...")
    
    # Directories
    input_dir = RAW_DATA_DIR / f"commonvoice_{lang_code}"
    output_dir = PROCESSED_DATA_DIR / "speech" / lang_code
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files
    audio_files = list(input_dir.glob("*.wav"))
    if not audio_files:
        logger.error(f"No audio files found in {input_dir}")
        return False
    
    logger.info(f"Found {len(audio_files)} audio files for {lang_code}")
    
    # Load and validate audio files
    config = PAPER_CONFIG["speech"]
    valid_samples = []
    processed_count = 0
    
    for audio_file in audio_files:
        if processed_count >= config["target_samples_per_lang"] * 2:  # Process more than needed for selection
            break
            
        audio, sr = load_audio_safely(audio_file, config["target_sr"])
        if audio is None:
            continue
        
        # Validate audio quality and duration
        if validate_audio(audio, sr, config["min_duration"], config["max_duration"]):
            valid_samples.append({
                "file_path": audio_file,
                "audio": audio,
                "duration": len(audio) / sr,
                "language": lang_code,
                "tonal": lang_code in TONAL_LANGUAGES
            })
            processed_count += 1
    
    logger.info(f"Found {len(valid_samples)} valid samples for {lang_code}")
    
    # Select exactly target_samples_per_lang samples for evaluation
    selected_samples = create_evaluation_dataset(valid_samples, config["target_samples_per_lang"], f"{lang_code} speech")
    
    # Save processed audio and metadata (all for evaluation)
    metadata_all = []
    
    for idx, sample in enumerate(selected_samples):
        # Save audio file
        output_filename = f"{lang_code}_eval_{idx:04d}.wav"
        output_path = output_dir / output_filename
        sf.write(str(output_path), sample["audio"], config["target_sr"])
        
        # Add metadata
        metadata_all.append({
            "file_path": str(output_path.relative_to(PROCESSED_DATA_DIR)),
            "language": lang_code,
            "tonal": sample["tonal"],
            "purpose": "evaluation",  # All samples for evaluation
            "duration": sample["duration"],
            "sample_rate": config["target_sr"],
            "original_file": str(sample["file_path"])
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_all)
    metadata_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    
    # Save summary
    summary = {
        "language": lang_code,
        "tonal": lang_code in TONAL_LANGUAGES,
        "total_samples": len(metadata_all),
        "purpose": "evaluation",  # All samples for bias evaluation
        "target_sr": config["target_sr"],
        "avg_duration": metadata_df["duration"].mean(),
        "processed_at": datetime.now().isoformat()
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Successfully processed {lang_code}: {len(metadata_all)} samples")
    return True

def find_audio_files_recursive(directory: Path, extensions: List[str] = ['.wav', '.mp3', '.flac']) -> List[Path]:
    """Recursively find audio files in directory and subdirectories."""
    audio_files = []
    
    if not directory.exists():
        return audio_files
    
    for ext in extensions:
        # Search recursively for audio files
        audio_files.extend(directory.rglob(f"*{ext}"))
    
    return audio_files

def preprocess_music_tradition(tradition: str) -> bool:
    """Preprocess music dataset for a specific tradition."""
    logger.info(f"Processing music tradition: {tradition}...")
    
    # Directories
    input_dir = RAW_DATA_DIR / tradition
    output_dir = PROCESSED_DATA_DIR / "music" / tradition
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files recursively (handles nested structures)
    audio_files = find_audio_files_recursive(input_dir)
    
    if not audio_files:
        logger.error(f"No audio files found in {input_dir} (searched recursively)")
        return False
    
    logger.info(f"Found {len(audio_files)} audio files for {tradition} (including nested directories)")
    
    # Load and process audio files
    config = PAPER_CONFIG["music"]
    valid_samples = []
    processed_files = 0
    skipped_files = 0
    
    for audio_file in audio_files:
        processed_files += 1
        
        # Progress logging for large datasets
        if processed_files % 50 == 0:
            logger.info(f"  Processed {processed_files}/{len(audio_files)} files, found {len(valid_samples)} valid segments")
        
        audio, sr = load_audio_safely(audio_file, config["target_sr"])
        if audio is None:
            skipped_files += 1
            continue
        
        # For music, we create 30-second segments
        duration = len(audio) / sr
        if duration >= config["segment_duration"]:
            # Can create at least one segment
            segment = segment_audio(audio, sr, config["segment_duration"], random_segment=True)
            
            # Additional validation for segment quality
            if validate_audio(segment, sr, config["segment_duration"] * 0.9, config["segment_duration"] * 1.1):
                valid_samples.append({
                    "file_path": audio_file,
                    "audio": segment,
                    "tradition": tradition,
                    "western": tradition in WESTERN_MUSIC,
                    "duration": config["segment_duration"]
                })
        else:
            logger.debug(f"Audio too short ({duration:.1f}s < {config['segment_duration']}s): {audio_file.name}")
        
        # If we have enough samples, stop early
        if len(valid_samples) >= config["target_samples_per_tradition"] * 2:
            logger.info(f"  Reached target sample count, stopping early at file {processed_files}")
            break
    
    logger.info(f"Created {len(valid_samples)} segments for {tradition} (processed {processed_files} files, skipped {skipped_files})")
    
    # Select exactly target_samples_per_tradition samples for evaluation
    selected_samples = create_evaluation_dataset(valid_samples, config["target_samples_per_tradition"], tradition)
    
    # Save processed audio and metadata (all for evaluation)
    metadata_all = []
    
    for idx, sample in enumerate(selected_samples):
        # Save audio file
        output_filename = f"{tradition}_eval_{idx:04d}.wav"
        output_path = output_dir / output_filename
        sf.write(str(output_path), sample["audio"], config["target_sr"])
        
        # Add metadata
        metadata_all.append({
            "file_path": str(output_path.relative_to(PROCESSED_DATA_DIR)),
            "tradition": tradition,
            "western": sample["western"],
            "purpose": "evaluation",  # All samples for evaluation
            "duration": sample["duration"],
            "sample_rate": config["target_sr"],
            "original_file": str(sample["file_path"])
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_all)
    metadata_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    
    # Save summary
    summary = {
        "tradition": tradition,
        "western": tradition in WESTERN_MUSIC,
        "total_samples": len(metadata_all),
        "purpose": "evaluation",  # All samples for bias evaluation
        "target_sr": config["target_sr"],
        "segment_duration": config["segment_duration"],
        "processed_at": datetime.now().isoformat()
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Successfully processed {tradition}: {len(metadata_all)} samples")
    return True

def preprocess_tau_urban_scenes() -> bool:
    """Preprocess TAU Urban Acoustic Scenes dataset."""
    logger.info("Processing TAU Urban Acoustic Scenes...")
    
    # Directories - check for nested structure
    input_dir = RAW_DATA_DIR / "tau_urban_2020"
    nested_audio_dir = input_dir / "TAU-urban-acoustic-scenes-2020-mobile-evaluation" / "audio"
    
    # Try nested structure first
    if nested_audio_dir.exists():
        audio_dir = nested_audio_dir
        logger.info(f"Using nested audio directory: {audio_dir}")
    elif input_dir.exists():
        audio_dir = input_dir
        logger.info(f"Using top-level directory: {audio_dir}")
    else:
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    output_dir = PROCESSED_DATA_DIR / "scenes"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files recursively
    audio_files = find_audio_files_recursive(audio_dir)
    
    if not audio_files:
        logger.error(f"No audio files found in {audio_dir} (searched recursively)")
        return False
    
    logger.info(f"Found {len(audio_files)} audio files for TAU Urban")
    
    # Process scenes by city
    config = PAPER_CONFIG["scenes"]
    all_metadata = []
    
    # Group files by city (extract from filename)
    city_files = defaultdict(list)
    for audio_file in audio_files:
        filename = audio_file.name
        # TAU Urban files typically have city names in them
        for city in TAU_CITIES:
            if city.lower() in filename.lower():
                city_files[city].append(audio_file)
                break
    
    if not city_files:
        logger.warning("Could not identify cities from filenames, processing all files as 'urban'")
        city_files['urban'] = audio_files
    
    for city, files in city_files.items():
        logger.info(f"Processing city: {city} ({len(files)} files)")
        
        city_output_dir = output_dir / city
        city_output_dir.mkdir(exist_ok=True)
        
        valid_samples = []
        
        for audio_file in files:
            audio, sr = load_audio_safely(audio_file, config["target_sr"])
            if audio is None:
                continue
            
            # Create 10-second segments
            duration = len(audio) / sr
            if duration >= config["segment_duration"]:
                segment = segment_audio(audio, sr, config["segment_duration"], random_segment=True)
                
                valid_samples.append({
                    "file_path": audio_file,
                    "audio": segment,
                    "city": city,
                    "duration": config["segment_duration"]
                })
            
            # If we have enough samples for this city, stop
            if len(valid_samples) >= config["target_samples_per_city"] * 2:
                break
        
        logger.info(f"Created {len(valid_samples)} segments for {city}")
        
        # Select exactly target_samples_per_city samples for evaluation
        selected_samples = create_evaluation_dataset(valid_samples, config["target_samples_per_city"], f"{city} scenes")
        
        # Save processed audio and metadata (all for evaluation)
        for idx, sample in enumerate(selected_samples):
            # Save audio file
            output_filename = f"{city}_eval_{idx:04d}.wav"
            output_path = city_output_dir / output_filename
            sf.write(str(output_path), sample["audio"], config["target_sr"])
            
            # Add metadata
            all_metadata.append({
                "file_path": str(output_path.relative_to(PROCESSED_DATA_DIR)),
                "city": city,
                "purpose": "evaluation",  # All samples for evaluation
                "duration": sample["duration"],
                "sample_rate": config["target_sr"],
                "original_file": str(sample["file_path"])
            })
        
        # Save city-specific metadata
        city_metadata = [m for m in all_metadata if m["city"] == city]
        city_metadata_df = pd.DataFrame(city_metadata)
        city_metadata_path = city_output_dir / "metadata.csv"
        city_metadata_df.to_csv(city_metadata_path, index=False)
        
        # Save city summary
        city_summary = {
            "city": city,
            "total_samples": len(city_metadata),
            "purpose": "evaluation",  # All samples for bias evaluation
            "target_sr": config["target_sr"],
            "segment_duration": config["segment_duration"],
            "processed_at": datetime.now().isoformat()
        }
        
        city_summary_path = city_output_dir / "summary.json"
        with open(city_summary_path, 'w') as f:
            json.dump(city_summary, f, indent=2)
    
    # Save overall scenes metadata
    overall_metadata_df = pd.DataFrame(all_metadata)
    overall_metadata_path = output_dir / "metadata.csv"
    overall_metadata_df.to_csv(overall_metadata_path, index=False)
    
    logger.info(f"Successfully processed TAU Urban: {len(all_metadata)} samples across {len(city_files)} cities")
    return True

def create_dataset_summary():
    """Create overall dataset summary."""
    logger.info("Creating dataset summary...")
    
    summary = {
        "dataset_name": "Cross-Cultural Mel-Scale Audio Frontend Bias Evaluation",
        "paper": "ICASSP 2026",
        "processed_at": datetime.now().isoformat(),
        "domains": {}
    }
    
    # Speech summary
    speech_dir = PROCESSED_DATA_DIR / "speech"
    if speech_dir.exists():
        speech_langs = []
        total_speech_samples = 0
        
        for lang_dir in speech_dir.iterdir():
            if lang_dir.is_dir():
                summary_file = lang_dir / "summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        lang_summary = json.load(f)
                        speech_langs.append(lang_summary)
                        total_speech_samples += lang_summary["total_samples"]
        
        summary["domains"]["speech"] = {
            "languages": len(speech_langs),
            "tonal_languages": len([l for l in speech_langs if l["tonal"]]),
            "non_tonal_languages": len([l for l in speech_langs if not l["tonal"]]),
            "total_samples": total_speech_samples,
            "target_samples_per_lang": PAPER_CONFIG["speech"]["target_samples_per_lang"],
            "language_details": speech_langs
        }
    
    # Music summary
    music_dir = PROCESSED_DATA_DIR / "music"
    if music_dir.exists():
        music_traditions = []
        total_music_samples = 0
        
        for tradition_dir in music_dir.iterdir():
            if tradition_dir.is_dir():
                summary_file = tradition_dir / "summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        tradition_summary = json.load(f)
                        music_traditions.append(tradition_summary)
                        total_music_samples += tradition_summary["total_samples"]
        
        summary["domains"]["music"] = {
            "traditions": len(music_traditions),
            "western_traditions": len([t for t in music_traditions if t["western"]]),
            "non_western_traditions": len([t for t in music_traditions if not t["western"]]),
            "total_samples": total_music_samples,
            "target_samples_per_tradition": PAPER_CONFIG["music"]["target_samples_per_tradition"],
            "tradition_details": music_traditions
        }
    
    # Scenes summary
    scenes_dir = PROCESSED_DATA_DIR / "scenes"
    if scenes_dir.exists():
        cities = []
        total_scene_samples = 0
        
        for city_dir in scenes_dir.iterdir():
            if city_dir.is_dir():
                summary_file = city_dir / "summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        city_summary = json.load(f)
                        cities.append(city_summary)
                        total_scene_samples += city_summary["total_samples"]
        
        summary["domains"]["scenes"] = {
            "cities": len(cities),
            "total_samples": total_scene_samples,
            "target_samples_per_city": PAPER_CONFIG["scenes"]["target_samples_per_city"],
            "city_details": cities
        }
    
    # Save overall summary
    summary_path = PROCESSED_DATA_DIR / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Dataset summary saved to dataset_summary.json")
    return summary

def check_dataset_availability() -> Dict[str, bool]:
    """Check which datasets are available in the raw data directory."""
    availability = {}
    
    logger.info("\n--- CHECKING DATASET AVAILABILITY ---")
    
    # Check speech datasets
    logger.info("Speech datasets (CommonVoice):")
    for lang in ALL_TARGET_LANGUAGES:
        lang_dir = RAW_DATA_DIR / f"commonvoice_{lang}"
        available = lang_dir.exists() and any(lang_dir.glob("*.wav"))
        availability[f"speech_{lang}"] = available
        status = "✓" if available else "✗"
        logger.info(f"  {status} {lang}: {lang_dir}")
        
        if not available and lang_dir.exists():
            # Check if directory exists but no wav files
            logger.warning(f"    Directory exists but no .wav files found")
    
    # Check music datasets  
    logger.info("\nMusic datasets:")
    for tradition in ALL_MUSIC_TRADITIONS:
        tradition_dir = RAW_DATA_DIR / tradition
        audio_files = find_audio_files_recursive(tradition_dir) if tradition_dir.exists() else []
        available = len(audio_files) > 0
        availability[f"music_{tradition}"] = available
        status = "✓" if available else "✗"
        logger.info(f"  {status} {tradition}: {tradition_dir} ({len(audio_files)} audio files)")
        
        if not available and tradition_dir.exists():
            logger.warning(f"    Directory exists but no audio files found")
    
    # Check scene datasets
    tau_dir = RAW_DATA_DIR / "tau_urban_2020"
    nested_audio_dir = tau_dir / "TAU-urban-acoustic-scenes-2020-mobile-evaluation" / "audio"
    
    if nested_audio_dir.exists():
        audio_files = find_audio_files_recursive(nested_audio_dir)
        available = len(audio_files) > 0
        availability["scenes_tau"] = available
        status = "✓" if available else "✗"
        logger.info(f"\nScene datasets:")
        logger.info(f"  {status} TAU Urban: {nested_audio_dir} ({len(audio_files)} audio files)")
    elif tau_dir.exists():
        audio_files = find_audio_files_recursive(tau_dir)
        available = len(audio_files) > 0
        availability["scenes_tau"] = available
        status = "✓" if available else "✗"
        logger.info(f"\nScene datasets:")
        logger.info(f"  {status} TAU Urban: {tau_dir} ({len(audio_files)} audio files)")
    else:
        availability["scenes_tau"] = False
        logger.info(f"\nScene datasets:")
        logger.info(f"  ✗ TAU Urban: {tau_dir} (not found)")
    
    # Summary
    total_datasets = len(availability)
    available_datasets = sum(availability.values())
    logger.info(f"\nDataset availability: {available_datasets}/{total_datasets} datasets found")
    
    if available_datasets < total_datasets:
        logger.warning("\nSome datasets are missing. To download:")
        logger.warning("python download_datasets.py --all")
    
    return availability

# Add this function call to main()
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for Cross-Cultural Mel-Scale Audio Frontend Bias Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python preprocess_datasets.py --all
    python preprocess_datasets.py --domain speech
    python preprocess_datasets.py --domain speech --languages en vi zh-CN
    python preprocess_datasets.py --domain music --traditions gtzan carnatic
    python preprocess_datasets.py --domain scenes
        """
    )
    
    parser.add_argument(
        "--domain",
        choices=["speech", "music", "scenes"],
        help="Specific domain to preprocess"
    )
    
    parser.add_argument(
        "--languages", "--langs",
        nargs="+",
        default=ALL_TARGET_LANGUAGES,
        help="Languages to process for speech domain"
    )
    
    parser.add_argument(
        "--traditions",
        nargs="+", 
        default=ALL_MUSIC_TRADITIONS,
        help="Music traditions to process"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all domains"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output exists"
    )
    
    return parser.parse_args()

def main():
    """Main preprocessing function."""
    args = parse_arguments()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    logger.info("=== Cross-Cultural Mel-Scale Audio Frontend Bias - Dataset Preprocessing ===")
    logger.info(f"Raw data directory: {RAW_DATA_DIR}")
    logger.info(f"Processed data directory: {PROCESSED_DATA_DIR}")
    
    # Check dataset availability first
    availability = check_dataset_availability()
    
    success_results = []
    
    # Check dataset availability before processing
    availability = check_dataset_availability()
    
    if args.all or args.domain == "speech":
        logger.info("\n--- PROCESSING SPEECH DATASETS ---")
        for lang in args.languages:
            if lang in ALL_TARGET_LANGUAGES:
                success = preprocess_commonvoice_language(lang)
                success_results.append(("Speech", lang, success))
            else:
                logger.warning(f"Language {lang} not in target languages: {ALL_TARGET_LANGUAGES}")
    
    if args.all or args.domain == "music":
        logger.info("\n--- PROCESSING MUSIC DATASETS ---")
        for tradition in args.traditions:
            if tradition in ALL_MUSIC_TRADITIONS:
                success = preprocess_music_tradition(tradition)
                success_results.append(("Music", tradition, success))
            else:
                logger.warning(f"Tradition {tradition} not in available traditions: {ALL_MUSIC_TRADITIONS}")
    
    if args.all or args.domain == "scenes":
        logger.info("\n--- PROCESSING SCENE DATASETS ---")
        success = preprocess_tau_urban_scenes()
        success_results.append(("Scenes", "TAU Urban", success))
    
    # Create dataset summary
    if success_results:
        logger.info("\n--- CREATING DATASET SUMMARY ---")
        summary = create_dataset_summary()
        
        # Print final results
        logger.info("\n" + "="*60)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*60)
        
        for domain, dataset, success in success_results:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"{status}: {domain} - {dataset}")
        
        successful_count = sum(1 for _, _, success in success_results if success)
        total_count = len(success_results)
        
        logger.info(f"\nOverall: {successful_count}/{total_count} datasets processed successfully")
        
        if successful_count > 0:
            logger.info(f"\nProcessed data available in: {PROCESSED_DATA_DIR}")
            logger.info("Next steps:")
            logger.info("1. Validate datasets: python validate_datasets.py --all")
            logger.info("2. Run experiments: python run_experiments.py")
        
        # Print dataset statistics
        if "speech" in summary["domains"]:
            speech = summary["domains"]["speech"]
            logger.info(f"\nSpeech: {speech['languages']} languages, {speech['total_samples']} samples")
            logger.info(f"  Tonal: {speech['tonal_languages']}, Non-tonal: {speech['non_tonal_languages']}")
        
        if "music" in summary["domains"]:
            music = summary["domains"]["music"]
            logger.info(f"Music: {music['traditions']} traditions, {music['total_samples']} samples")
            logger.info(f"  Western: {music['western_traditions']}, Non-Western: {music['non_western_traditions']}")
        
        if "scenes" in summary["domains"]:
            scenes = summary["domains"]["scenes"]
            logger.info(f"Scenes: {scenes['cities']} cities, {scenes['total_samples']} samples")
    
    else:
        logger.error("No datasets to process. Use --help for usage information.")

if __name__ == "__main__":
    main()
