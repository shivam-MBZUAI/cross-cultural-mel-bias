#!/usr/bin/env python3
"""
Dataset Preprocessing for Cross-Cultural Audio Bias Evaluation
Creates balanced evaluation datasets as specified in ICASSP 2026 paper

Data structure expected:
- speech/{language}/*.wav
- music/{tradition}/*.wav or *.mp3
- scenes/{region}/*.wav

Creates evaluation sets:
- Speech: 2000 samples per language (11 languages)
- Music: 300 samples per tradition (6 traditions)
- Scenes: 100 samples per region (2 regions)
"""

import os
import json
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import soundfile as sf
import librosa
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from paper
PAPER_CONFIG = {
    "speech": {
        "samples_per_language": 2000,
        "target_sr": 16000,
        "max_duration": 10.0,  # seconds
        "languages": {
            "tonal": ['vi', 'th', 'zh-CN', 'pa-IN', 'yue'],
            "non_tonal": ['en', 'es', 'de', 'fr', 'it', 'nl']
        }
    },
    "music": {
        "samples_per_tradition": 300,
        "target_sr": 22050,
        "segment_duration": 30.0,  # seconds
        "traditions": {
            "western": ['gtzan', 'fma_small'],
            "non_western": ['carnatic', 'hindustani', 'turkish_makam', 'arab_andalusian']
        }
    },
    "scenes": {
        "samples_per_region": 100,
        "target_sr": 16000,
        "segment_duration": 10.0,  # seconds
        "regions": ['european-1', 'european-2']
    }
}

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")

def load_audio(file_path, target_sr):
    """Load and resample audio file."""
    try:
        # Try soundfile first (faster for WAV)
        if file_path.suffix.lower() in ['.wav', '.flac']:
            audio, sr = sf.read(str(file_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            # Use librosa for other formats (MP3, etc.)
            audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
        
        return audio, target_sr
    except Exception as e:
        logger.debug(f"Failed to load {file_path}: {e}")
        return None, None

def segment_audio(audio, sr, duration, random_offset=True):
    """Extract a segment of specified duration from audio."""
    target_samples = int(duration * sr)
    
    if len(audio) >= target_samples:
        if random_offset:
            max_start = len(audio) - target_samples
            start = random.randint(0, max_start) if max_start > 0 else 0
        else:
            start = 0
        return audio[start:start + target_samples]
    else:
        # Pad if shorter
        return np.pad(audio, (0, target_samples - len(audio)), mode='constant')

def preprocess_speech(data_dir, output_dir):
    """Preprocess speech data for all languages."""
    logger.info("Processing speech data...")
    
    config = PAPER_CONFIG['speech']
    all_languages = config['languages']['tonal'] + config['languages']['non_tonal']
    
    speech_dir = Path(data_dir) / 'speech'
    if not speech_dir.exists():
        logger.error(f"Speech directory not found: {speech_dir}")
        return False
    
    results = {}
    
    for lang in all_languages:
        lang_dir = speech_dir / lang
        if not lang_dir.exists():
            logger.warning(f"Language directory not found: {lang_dir}")
            continue
        
        # Get all audio files
        audio_files = list(lang_dir.glob('*.wav')) + list(lang_dir.glob('*.mp3')) + list(lang_dir.glob('*.flac'))
        
        if not audio_files:
            logger.warning(f"No audio files found for {lang}")
            continue
        
        logger.info(f"Found {len(audio_files)} files for {lang}")
        
        # Process files
        valid_samples = []
        for audio_file in audio_files:
            audio, sr = load_audio(audio_file, config['target_sr'])
            if audio is not None:
                # Trim to max duration
                duration = len(audio) / sr
                if duration > config['max_duration']:
                    audio = segment_audio(audio, sr, config['max_duration'], random_offset=True)
                
                valid_samples.append({
                    'file': audio_file,
                    'audio': audio,
                    'duration': len(audio) / sr
                })
            
            # Stop if we have enough samples
            if len(valid_samples) >= config['samples_per_language'] * 1.5:
                break
        
        # Select required number of samples
        if len(valid_samples) < config['samples_per_language']:
            logger.warning(f"Only {len(valid_samples)} valid samples for {lang}, need {config['samples_per_language']}")
        
        selected = random.sample(valid_samples, min(len(valid_samples), config['samples_per_language']))
        
        # Save processed samples
        lang_output_dir = Path(output_dir) / 'speech' / lang
        lang_output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, sample in enumerate(selected):
            output_file = lang_output_dir / f"{lang}_{idx:04d}.wav"
            sf.write(str(output_file), sample['audio'], config['target_sr'])
        
        results[lang] = {
            'processed': len(selected),
            'is_tonal': lang in config['languages']['tonal']
        }
        
        logger.info(f"Processed {len(selected)} samples for {lang}")
    
    # Save summary
    summary_file = Path(output_dir) / 'speech' / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return True

def preprocess_music(data_dir, output_dir):
    """Preprocess music data for all traditions."""
    logger.info("Processing music data...")
    
    config = PAPER_CONFIG['music']
    all_traditions = config['traditions']['western'] + config['traditions']['non_western']
    
    music_dir = Path(data_dir) / 'music'
    if not music_dir.exists():
        logger.error(f"Music directory not found: {music_dir}")
        return False
    
    results = {}
    
    for tradition in all_traditions:
        tradition_dir = music_dir / tradition
        if not tradition_dir.exists():
            logger.warning(f"Tradition directory not found: {tradition_dir}")
            continue
        
        # Get all audio files (including subdirectories)
        audio_files = list(tradition_dir.rglob('*.wav')) + list(tradition_dir.rglob('*.mp3'))
        
        if not audio_files:
            logger.warning(f"No audio files found for {tradition}")
            continue
        
        logger.info(f"Found {len(audio_files)} files for {tradition}")
        
        # Process files and create segments
        segments = []
        for audio_file in audio_files:
            audio, sr = load_audio(audio_file, config['target_sr'])
            if audio is not None:
                duration = len(audio) / sr
                
                # Create 30-second segments
                if duration >= config['segment_duration']:
                    # Can create at least one full segment
                    num_segments = int(duration / config['segment_duration'])
                    for i in range(min(num_segments, 3)):  # Max 3 segments per file
                        segment = segment_audio(audio, sr, config['segment_duration'], random_offset=True)
                        segments.append({
                            'file': audio_file,
                            'audio': segment,
                            'segment_idx': i
                        })
                elif duration >= config['segment_duration'] * 0.8:  # Accept if at least 80% of target
                    segments.append({
                        'file': audio_file,
                        'audio': audio,
                        'segment_idx': 0
                    })
            
            # Stop if we have enough segments
            if len(segments) >= config['samples_per_tradition'] * 1.5:
                break
        
        # Select required number of segments
        if len(segments) < config['samples_per_tradition']:
            logger.warning(f"Only {len(segments)} segments for {tradition}, need {config['samples_per_tradition']}")
        
        selected = random.sample(segments, min(len(segments), config['samples_per_tradition']))
        
        # Save processed segments
        tradition_output_dir = Path(output_dir) / 'music' / tradition
        tradition_output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, segment in enumerate(selected):
            output_file = tradition_output_dir / f"{tradition}_{idx:04d}.wav"
            sf.write(str(output_file), segment['audio'], config['target_sr'])
        
        results[tradition] = {
            'processed': len(selected),
            'is_western': tradition in config['traditions']['western']
        }
        
        logger.info(f"Processed {len(selected)} segments for {tradition}")
    
    # Save summary
    summary_file = Path(output_dir) / 'music' / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return True

def preprocess_scenes(data_dir, output_dir):
    """Preprocess scene data for both regions."""
    logger.info("Processing scene data...")
    
    config = PAPER_CONFIG['scenes']
    
    scenes_dir = Path(data_dir) / 'scenes'
    if not scenes_dir.exists():
        logger.error(f"Scenes directory not found: {scenes_dir}")
        return False
    
    results = {}
    
    for region in config['regions']:
        region_dir = scenes_dir / region
        if not region_dir.exists():
            logger.warning(f"Region directory not found: {region_dir}")
            continue
        
        # Get all audio files
        audio_files = list(region_dir.glob('*.wav')) + list(region_dir.glob('*.mp3'))
        
        if not audio_files:
            logger.warning(f"No audio files found for {region}")
            continue
        
        logger.info(f"Found {len(audio_files)} files for {region}")
        
        # Process files
        segments = []
        for audio_file in audio_files:
            audio, sr = load_audio(audio_file, config['target_sr'])
            if audio is not None:
                # Create 10-second segment
                segment = segment_audio(audio, sr, config['segment_duration'], random_offset=True)
                segments.append({
                    'file': audio_file,
                    'audio': segment
                })
            
            # Stop if we have enough segments
            if len(segments) >= config['samples_per_region'] * 1.5:
                break
        
        # Select required number of segments
        if len(segments) < config['samples_per_region']:
            logger.warning(f"Only {len(segments)} segments for {region}, need {config['samples_per_region']}")
        
        selected = random.sample(segments, min(len(segments), config['samples_per_region']))
        
        # Save processed segments
        region_output_dir = Path(output_dir) / 'scenes' / region
        region_output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, segment in enumerate(selected):
            output_file = region_output_dir / f"{region}_{idx:04d}.wav"
            sf.write(str(output_file), segment['audio'], config['target_sr'])
        
        results[region] = {
            'processed': len(selected)
        }
        
        logger.info(f"Processed {len(selected)} segments for {region}")
    
    # Save summary
    summary_file = Path(output_dir) / 'scenes' / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return True

def create_dataset_summary(output_dir):
    """Create overall dataset summary."""
    logger.info("Creating dataset summary...")
    
    summary = {
        'dataset': 'Cross-Cultural Audio Bias Evaluation Dataset',
        'paper': 'ICASSP 2026',
        'created': datetime.now().isoformat(),
        'domains': {}
    }
    
    # Check each domain
    for domain in ['speech', 'music', 'scenes']:
        summary_file = Path(output_dir) / domain / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                domain_data = json.load(f)
                
                if domain == 'speech':
                    tonal_count = sum(1 for v in domain_data.values() if isinstance(v, dict) and v.get('is_tonal'))
                    non_tonal_count = sum(1 for v in domain_data.values() if isinstance(v, dict) and not v.get('is_tonal'))
                    total_samples = sum(v['processed'] for v in domain_data.values() if isinstance(v, dict))
                    
                    summary['domains']['speech'] = {
                        'languages': len(domain_data),
                        'tonal': tonal_count,
                        'non_tonal': non_tonal_count,
                        'total_samples': total_samples,
                        'target_per_language': PAPER_CONFIG['speech']['samples_per_language']
                    }
                
                elif domain == 'music':
                    western_count = sum(1 for v in domain_data.values() if isinstance(v, dict) and v.get('is_western'))
                    non_western_count = sum(1 for v in domain_data.values() if isinstance(v, dict) and not v.get('is_western'))
                    total_samples = sum(v['processed'] for v in domain_data.values() if isinstance(v, dict))
                    
                    summary['domains']['music'] = {
                        'traditions': len(domain_data),
                        'western': western_count,
                        'non_western': non_western_count,
                        'total_samples': total_samples,
                        'target_per_tradition': PAPER_CONFIG['music']['samples_per_tradition']
                    }
                
                elif domain == 'scenes':
                    total_samples = sum(v['processed'] for v in domain_data.values() if isinstance(v, dict))
                    
                    summary['domains']['scenes'] = {
                        'regions': len(domain_data),
                        'total_samples': total_samples,
                        'target_per_region': PAPER_CONFIG['scenes']['samples_per_region']
                    }
    
    # Save overall summary
    summary_file = Path(output_dir) / 'dataset_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DATASET SUMMARY")
    logger.info("="*60)
    
    for domain, info in summary['domains'].items():
        logger.info(f"\n{domain.upper()}:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
    
    return summary

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess datasets for audio bias evaluation')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Directory containing raw data (with speech/, music/, scenes/ subdirs)')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--domain', type=str, choices=['speech', 'music', 'scenes', 'all'],
                       default='all', help='Which domain to process')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("CROSS-CULTURAL AUDIO BIAS DATASET PREPROCESSING")
    logger.info("="*60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Domain: {args.domain}")
    
    # Process domains
    success = True
    
    if args.domain in ['speech', 'all']:
        success = success and preprocess_speech(args.data_dir, args.output_dir)
    
    if args.domain in ['music', 'all']:
        success = success and preprocess_music(args.data_dir, args.output_dir)
    
    if args.domain in ['scenes', 'all']:
        success = success and preprocess_scenes(args.data_dir, args.output_dir)
    
    # Create summary
    if success:
        create_dataset_summary(args.output_dir)
        logger.info(f"\nProcessing complete! Data saved to {args.output_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Run evaluation: python frontends_eval.py")
    else:
        logger.error("Some preprocessing steps failed. Check logs for details.")

if __name__ == "__main__":
    main()