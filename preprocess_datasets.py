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

# Configuration from paper and frontends_eval.py specifications
PAPER_CONFIG = {
    "speech": {
        "samples_per_language": 2000,  # Paper specification
        "target_sr": 16000,            # From frontends_eval.py  
        "max_duration": 10.0,          # seconds (from frontends_eval.py)
        "n_fft": 512,                  # From frontends_eval.py
        "hop_length": 160,             # From frontends_eval.py (10ms at 16kHz)
        "n_mels": 80,                  # From frontends_eval.py
        "languages": {
            # From paper Table 1
            "tonal": ['vi', 'th', 'zh-CN', 'pa-IN', 'yue'],
            "non_tonal": ['en', 'es', 'de', 'fr', 'it', 'nl']
        }
    },
    "music": {
        "samples_per_tradition": 300,  # Paper specification  
        "target_sr": 22050,            # Standard for music analysis
        "segment_duration": 30.0,      # seconds (from paper)
        "n_fft": 1024,                 # Higher resolution for music
        "hop_length": 441,             # ~20ms at 22.05kHz
        "n_mels": 80,                  # Consistent with speech
        "traditions": {
            # From paper Table 1
            "western": ['gtzan', 'fma_small'],
            "non_western": ['carnatic', 'hindustani', 'turkish_makam', 'arab_andalusian']
        }
    },
    "scenes": {
        "samples_per_region": 100,     # Paper specification
        "target_sr": 16000,            # From frontends_eval.py (10 seconds)
        "segment_duration": 10.0,      # seconds (from frontends_eval.py)
        "n_fft": 512,                  # From frontends_eval.py
        "hop_length": 160,             # From frontends_eval.py
        "n_mels": 80,                  # From frontends_eval.py
        "regions": ['european-1', 'european-2']  # From paper (TAU Urban divided)
    }
}

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")

def load_audio_with_frontends_specs(file_path, target_sr, target_duration=None):
    """Load and preprocess audio exactly as in frontends_eval.py"""
    try:
        # Load audio file with robust error handling for corrupted files
        if file_path.suffix.lower() in ['.wav']:
            # Use soundfile for WAV (more robust than torchaudio for corrupted files)
            try:
                audio, sr = sf.read(str(file_path), always_2d=False)
                if audio.ndim > 1:  # Convert to mono if stereo
                    audio = audio.mean(axis=1)
            except Exception as wav_error:
                # Fallback to librosa for problematic WAV files
                audio, sr = librosa.load(str(file_path), sr=None, mono=True)
        else:
            # Use librosa for other formats with enhanced error handling
            try:
                # Try with librosa first (handles MP3 corruption better)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio, sr = librosa.load(str(file_path), sr=None, mono=True)
                    
                # Check if audio was loaded successfully
                if audio is None or len(audio) == 0:
                    raise ValueError("Empty audio loaded")
                    
            except Exception as mp3_error:
                # Try with soundfile as fallback
                try:
                    audio, sr = sf.read(str(file_path), always_2d=False)
                    if audio.ndim > 1:  # Convert to mono if stereo
                        audio = audio.mean(axis=1)
                except Exception as sf_error:
                    logger.debug(f"Failed to load {file_path} with both librosa and soundfile: {mp3_error}, {sf_error}")
                    return None, None
        
        # Check if audio is valid
        if audio is None or len(audio) == 0:
            logger.debug(f"Empty audio loaded from {file_path}")
            return None, None
            
        # Convert to float32 and normalize if needed
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Trim or pad to target duration (as in frontends_eval.py)
        if target_duration is not None:
            target_samples = int(target_duration * target_sr)
            if len(audio) > target_samples:
                # Random offset for training variety (as in preprocessing)
                max_start = len(audio) - target_samples
                start = random.randint(0, max_start) if max_start > 0 else 0
                audio = audio[start:start + target_samples]
            else:
                # Pad with zeros (as in frontends_eval.py)
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio, target_sr
        
    except Exception as e:
        logger.debug(f"Failed to load {file_path}: {e}")
        return None, None

def validate_audio_quality(audio, sr, min_duration=1.0):
    """Validate audio quality and remove problematic samples"""
    if audio is None or len(audio) == 0:
        return False
    
    # Check for NaN or infinite values first
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        return False
    
    # Check minimum duration
    duration = len(audio) / sr
    if duration < min_duration:
        return False
    
    # Check for silence (all zeros or very low energy)
    rms_energy = np.sqrt(np.mean(audio**2))
    if rms_energy < 1e-6:
        return False
    
    # Check for clipping (too many samples at max values)
    clipped_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
    if clipped_ratio > 0.1:  # More than 10% clipped
        return False
    
    # Check for extremely noisy audio (very high variance)
    if np.std(audio) > 2.0:  # Extremely high variance suggests corruption
        return False
    
    # Check for digital artifacts (too many repeated values)
    unique_ratio = len(np.unique(audio)) / len(audio)
    if unique_ratio < 0.01:  # Less than 1% unique values suggests digital artifacts
        return False
    
    return True

def preprocess_speech(data_dir, output_dir):
    """Preprocess speech data for all languages."""
    logger.info("Processing speech data...")
    
    config = PAPER_CONFIG['speech']
    
    # Language mapping to actual directory names we have
    language_mapping = {
        'vi': 'vi',          # Vietnamese (tonal)
        'th': 'th',          # Thai (tonal) 
        'zh-CN': 'zh-CN',    # Mandarin Chinese (tonal)
        'pa-IN': 'pa-IN',    # Punjabi (tonal)
        'yue': 'yue',        # Cantonese (tonal)
        'en': 'en',          # English (non-tonal)
        'es': 'es',          # Spanish (non-tonal)
        'de': 'de',          # German (non-tonal)
        'fr': 'fr',          # French (non-tonal)
        'it': 'it',          # Italian (non-tonal)
        'nl': 'nl'           # Dutch (non-tonal)
    }
    
    speech_dir = Path(data_dir) / 'speech'
    if not speech_dir.exists():
        logger.error(f"Speech directory not found: {speech_dir}")
        return False
    
    results = {}
    
    # Get available languages from directory
    available_langs = [d.name for d in speech_dir.iterdir() if d.is_dir()]
    logger.info(f"Available languages: {available_langs}")
    
    for lang_code, dir_name in language_mapping.items():
        lang_dir = speech_dir / dir_name
        if not lang_dir.exists():
            logger.warning(f"Language directory not found: {lang_dir}")
            continue
        
        # Get all audio files (already in _eval_XXXX.wav format)
        audio_files = list(lang_dir.glob('*.wav'))
        
        if not audio_files:
            logger.warning(f"No audio files found for {lang_code}")
            continue
        
        logger.info(f"Found {len(audio_files)} files for {lang_code} ({dir_name})")
        
        # Process files
        valid_samples = []
        for audio_file in audio_files:
            audio, sr = load_audio_with_frontends_specs(audio_file, config['target_sr'], config['max_duration'])
            if audio is not None and validate_audio_quality(audio, sr):
                valid_samples.append({
                    'file': audio_file,
                    'audio': audio,
                    'duration': len(audio) / sr
                })
        
        # Select required number of samples
        target_samples = config['samples_per_language']
        if len(valid_samples) < target_samples:
            logger.warning(f"Only {len(valid_samples)} valid samples for {lang_code}, need {target_samples}")
            target_samples = len(valid_samples)
        
        selected = random.sample(valid_samples, target_samples)
        
        # Save processed samples to processed_data
        lang_output_dir = Path(output_dir) / 'speech' / lang_code
        lang_output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, sample in enumerate(selected):
            output_file = lang_output_dir / f"{lang_code}_eval_{idx:04d}.wav"
            sf.write(str(output_file), sample['audio'], config['target_sr'])
        
        is_tonal = lang_code in config['languages']['tonal']
        results[lang_code] = {
            'processed': len(selected),
            'available': len(valid_samples),
            'is_tonal': is_tonal,
            'category': 'tonal' if is_tonal else 'non-tonal'
        }
        
        logger.info(f"Processed {len(selected)} samples for {lang_code} ({'tonal' if is_tonal else 'non-tonal'})")
    
    # Save summary
    summary_file = Path(output_dir) / 'speech' / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return True

def preprocess_music(data_dir, output_dir):
    """Preprocess music data for all traditions."""
    logger.info("Processing music data...")
    
    config = PAPER_CONFIG['music']
    
    # Tradition mapping to actual directory names we have
    tradition_mapping = {
        'gtzan': 'gtzan',                    # Western
        'fma_small': 'fma_small',           # Western
        'carnatic': 'carnatic',             # Non-Western (South Indian)
        'hindustani': 'hindustani',         # Non-Western (North Indian)
        'turkish_makam': 'turkish_makam',   # Non-Western (Turkish)
        'arab_andalusian': 'arab_andalusian' # Non-Western (Arab-Andalusian)
    }
    
    music_dir = Path(data_dir) / 'music'
    if not music_dir.exists():
        logger.error(f"Music directory not found: {music_dir}")
        return False
    
    results = {}
    
    # Get available traditions from directory
    available_traditions = [d.name for d in music_dir.iterdir() if d.is_dir()]
    logger.info(f"Available traditions: {available_traditions}")
    
    for tradition_code, dir_name in tradition_mapping.items():
        tradition_dir = music_dir / dir_name
        if not tradition_dir.exists():
            logger.warning(f"Tradition directory not found: {tradition_dir}")
            continue
        
        # Get all audio files (already in tradition_eval_XXXX.wav format)
        audio_files = list(tradition_dir.glob('*.wav'))
        
        if not audio_files:
            logger.warning(f"No audio files found for {tradition_code}")
            continue
        
        logger.info(f"Found {len(audio_files)} files for {tradition_code} ({dir_name})")
        
        # Process files and create segments
        segments = []
        failed_files = 0
        
        logger.info(f"Processing {len(audio_files)} files for {tradition_code}...")
        for i, audio_file in enumerate(audio_files):
            if i % 500 == 0:  # Progress update every 500 files
                logger.info(f"  Progress: {i}/{len(audio_files)} files processed...")
                
            try:
                audio, sr = load_audio_with_frontends_specs(audio_file, config['target_sr'])
                if audio is not None and validate_audio_quality(audio, sr):
                    duration = len(audio) / sr
                    
                    # Create 30-second segments (as specified in paper)
                    if duration >= config['segment_duration']:
                        # Can create at least one full segment
                        num_segments = int(duration / config['segment_duration'])
                        for i in range(min(num_segments, 2)):  # Max 2 segments per file for variety
                            segment_audio, _ = load_audio_with_frontends_specs(audio_file, config['target_sr'], config['segment_duration'])
                            if segment_audio is not None:
                                segments.append({
                                    'file': audio_file,
                                    'audio': segment_audio,
                                    'segment_idx': i
                                })
                    elif duration >= config['segment_duration'] * 0.7:  # Accept if at least 70% of target
                        # Pad shorter files to target duration
                        segment_audio, _ = load_audio_with_frontends_specs(audio_file, config['target_sr'], config['segment_duration'])
                        if segment_audio is not None:
                            segments.append({
                                'file': audio_file,
                                'audio': segment_audio,
                                'segment_idx': 0
                            })
                else:
                    failed_files += 1
                    if failed_files <= 5:  # Only log first few failures to avoid spam
                        logger.debug(f"Failed validation for {audio_file.name}")
            except Exception as e:
                failed_files += 1
                if failed_files <= 5:  # Only log first few failures to avoid spam
                    logger.debug(f"Error processing {audio_file.name}: {e}")
        
        if failed_files > 0:
            logger.info(f"Skipped {failed_files} corrupted/invalid files for {tradition_code}")
        
        # Select required number of segments
        target_samples = config['samples_per_tradition']
        if len(segments) < target_samples:
            logger.warning(f"Only {len(segments)} segments for {tradition_code}, need {target_samples}")
            target_samples = len(segments)
        
        selected = random.sample(segments, target_samples)
        
        # Save processed segments to processed_data
        tradition_output_dir = Path(output_dir) / 'music' / tradition_code
        tradition_output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, segment in enumerate(selected):
            output_file = tradition_output_dir / f"{tradition_code}_eval_{idx:04d}.wav"
            sf.write(str(output_file), segment['audio'], config['target_sr'])
        
        is_western = tradition_code in config['traditions']['western']
        results[tradition_code] = {
            'processed': len(selected),
            'available': len(segments),
            'is_western': is_western,
            'category': 'western' if is_western else 'non-western'
        }
        
        logger.info(f"Processed {len(selected)} segments for {tradition_code} ({'western' if is_western else 'non-western'})")
    
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
    
    # Check what regions we have available
    available_regions = [d.name for d in scenes_dir.iterdir() if d.is_dir()]
    logger.info(f"Available regions: {available_regions}")
    
    for region in config['regions']:
        region_dir = scenes_dir / region
        if not region_dir.exists():
            logger.warning(f"Region directory not found: {region_dir}")
            continue
        
        # Get all audio files (already in region_eval_XXXX.wav format)
        audio_files = list(region_dir.glob('*.wav'))
        
        if not audio_files:
            logger.warning(f"No audio files found for {region}")
            continue
        
        logger.info(f"Found {len(audio_files)} files for {region}")
        
        # Process files
        segments = []
        for audio_file in audio_files:
            audio, sr = load_audio_with_frontends_specs(audio_file, config['target_sr'], config['segment_duration'])
            if audio is not None and validate_audio_quality(audio, sr):
                segments.append({
                    'file': audio_file,
                    'audio': audio
                })
        
        # Select required number of segments
        target_samples = config['samples_per_region'] 
        if len(segments) < target_samples:
            logger.warning(f"Only {len(segments)} segments for {region}, need {target_samples}")
            target_samples = len(segments)
        
        selected = random.sample(segments, target_samples)
        
        # Save processed segments to processed_data
        region_output_dir = Path(output_dir) / 'scenes' / region
        region_output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, segment in enumerate(selected):
            output_file = region_output_dir / f"{region}_eval_{idx:04d}.wav"
            sf.write(str(output_file), segment['audio'], config['target_sr'])
        
        results[region] = {
            'processed': len(selected),
            'available': len(segments)
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
    
    logger.info("\n" + "="*60)
    logger.info("DATASET SUMMARY")
    logger.info("="*60)
    
    for domain, info in summary['domains'].items():
        logger.info(f"\n{domain.upper()}:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
    
    # Print paper compliance
    logger.info("\n" + "="*60)
    logger.info("PAPER COMPLIANCE CHECK")
    logger.info("="*60)
    
    if 'speech' in summary['domains']:
        speech = summary['domains']['speech']
        logger.info(f"Speech: {speech['total_samples']} samples")
        logger.info(f"  Target: {speech['tonal']} tonal + {speech['non_tonal']} non-tonal = {speech['languages']} languages")
        logger.info(f"  Each language: {speech['target_per_language']} samples")
        logger.info(f"  Audio specs: 16kHz, 10s max, 80 mel features")
    
    if 'music' in summary['domains']:
        music = summary['domains']['music']
        logger.info(f"Music: {music['total_samples']} samples") 
        logger.info(f"  Target: {music['western']} Western + {music['non_western']} non-Western = {music['traditions']} traditions")
        logger.info(f"  Each tradition: {music['target_per_tradition']} samples")
        logger.info(f"  Audio specs: 22.05kHz, 30s segments, 80 mel features")
    
    if 'scenes' in summary['domains']:
        scenes = summary['domains']['scenes']
        logger.info(f"Scenes: {scenes['total_samples']} samples")
        logger.info(f"  Target: {scenes['regions']} European regions")
        logger.info(f"  Each region: {scenes['target_per_region']} samples")
        logger.info(f"  Audio specs: 16kHz, 10s segments, 80 mel features")
    
    return summary

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess datasets for audio bias evaluation')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing organized data (with speech/, music/, scenes/ subdirs)')
    parser.add_argument('--output_dir', type=str, default='./processed_data',
                       help='Output directory for final evaluation datasets')
    parser.add_argument('--domain', type=str, choices=['speech', 'music', 'scenes', 'all'],
                       default='all', help='Which domain to process')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if output exists')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("CROSS-CULTURAL AUDIO BIAS DATASET PREPROCESSING")
    logger.info("Paper: Evidence and Alternatives from Speech and Music (ICASSP 2026)")
    logger.info("="*60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Random seed: {args.seed}")
    
    # Check if data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return
    
    # Process domains
    success = True
    
    if args.domain in ['speech', 'all']:
        speech_output = output_dir / 'speech'
        if args.force or not speech_output.exists() or not any(speech_output.iterdir()):
            success = success and preprocess_speech(args.data_dir, args.output_dir)
        else:
            logger.info("Speech data already processed. Use --force to reprocess.")
    
    if args.domain in ['music', 'all']:
        music_output = output_dir / 'music'
        if args.force or not music_output.exists() or not any(music_output.iterdir()):
            success = success and preprocess_music(args.data_dir, args.output_dir)
        else:
            logger.info("Music data already processed. Use --force to reprocess.")
    
    if args.domain in ['scenes', 'all']:
        scenes_output = output_dir / 'scenes'
        if args.force or not scenes_output.exists() or not any(scenes_output.iterdir()):
            success = success and preprocess_scenes(args.data_dir, args.output_dir)
        else:
            logger.info("Scenes data already processed. Use --force to reprocess.")
    
    # Create summary
    if success:
        summary = create_dataset_summary(args.output_dir)
        logger.info(f"\nProcessing complete! Evaluation dataset saved to {args.output_dir}")
        logger.info("\nNext steps:")
        logger.info("Run evaluation: python frontends_eval.py")
    else:
        logger.error("Some preprocessing steps failed. Check logs for details.")

if __name__ == "__main__":
    main()