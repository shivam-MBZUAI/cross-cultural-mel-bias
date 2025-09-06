#!/usr/bin/env python3
"""
FairAudioBench Dataset Preprocessor
Creates balanced datasets for cross-cultural bias evaluation:
- Balances samples across languages, musical traditions, and cities
- Standardizes audio format and sample rates
- Creates train/validation/test splits
- Generates metadata with demographic information
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Handles audio preprocessing and format standardization."""
    
    def __init__(self, target_sr: int = 16000, target_duration: float = 4.0):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)
    
    def load_and_standardize_audio(self, audio_path: Path) -> Optional[np.ndarray]:
        """Load and standardize audio to target format."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            # Handle duration
            if len(audio) > self.target_samples:
                # Random crop for longer audio
                start_idx = random.randint(0, len(audio) - self.target_samples)
                audio = audio[start_idx:start_idx + self.target_samples]
            elif len(audio) < self.target_samples:
                # Pad shorter audio
                pad_length = self.target_samples - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant')
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            return None
    
    def save_audio(self, audio: np.ndarray, output_path: Path) -> bool:
        """Save audio to standardized format."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, self.target_sr)
            return True
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {e}")
            return False

class DatasetBalancer:
    """Creates balanced datasets across cultural groups."""
    
    def __init__(self, base_dir: str, output_dir: str, samples_per_group: int = 1000):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.samples_per_group = samples_per_group
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = AudioPreprocessor()
        
        # Initialize metadata storage
        self.metadata = {
            "speech": [],
            "music": [],
            "urban_sounds": []
        }
    
    def balance_speech_dataset(self) -> bool:
        """Balance speech dataset across languages and tonal properties."""
        logger.info("Balancing speech dataset...")
        
        speech_dir = self.base_dir / "speech" / "common_voice"
        output_speech_dir = self.output_dir / "speech"
        
        if not speech_dir.exists():
            logger.warning("Speech dataset not found. Skipping...")
            return False
        
        # Load language metadata
        try:
            with open(speech_dir / "languages_metadata.json", 'r') as f:
                languages_meta = json.load(f)
        except FileNotFoundError:
            logger.error("Language metadata not found")
            return False
        
        # Group languages by tonal property
        tonal_langs = [lang for lang, info in languages_meta.items() if info.get("tonal", False)]
        non_tonal_langs = [lang for lang, info in languages_meta.items() if not info.get("tonal", False)]
        
        logger.info(f"Tonal languages: {tonal_langs}")
        logger.info(f"Non-tonal languages: {non_tonal_langs}")
        
        # Balance samples for each language
        for lang_code, lang_info in languages_meta.items():
            lang_dir = speech_dir / lang_code
            output_lang_dir = output_speech_dir / lang_code
            
            if not lang_dir.exists():
                logger.warning(f"Language directory {lang_dir} not found")
                continue
            
            # Process language samples
            self._process_language_samples(
                lang_dir, output_lang_dir, lang_code, lang_info
            )
        
        # Create balanced splits
        self._create_speech_splits(tonal_langs, non_tonal_langs)
        
        return True
    
    def _process_language_samples(self, lang_dir: Path, output_dir: Path, 
                                 lang_code: str, lang_info: Dict) -> None:
        """Process samples for a specific language."""
        
        # Look for audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(list(lang_dir.rglob(f"*{ext}")))
        
        if not audio_files:
            logger.warning(f"No audio files found for {lang_code}")
            return
        
        # Sample files if we have more than needed
        if len(audio_files) > self.samples_per_group:
            audio_files = random.sample(audio_files, self.samples_per_group)
        
        logger.info(f"Processing {len(audio_files)} files for {lang_code}")
        
        # Process files
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_count = 0
        
        for audio_file in tqdm(audio_files, desc=f"Processing {lang_code}"):
            try:
                # Load and standardize audio
                audio = self.preprocessor.load_and_standardize_audio(audio_file)
                if audio is None:
                    continue
                
                # Generate output filename
                output_filename = f"{lang_code}_{processed_count:06d}.wav"
                output_path = output_dir / output_filename
                
                # Save audio
                if self.preprocessor.save_audio(audio, output_path):
                    # Add to metadata
                    self.metadata["speech"].append({
                        "file_path": str(output_path.relative_to(self.output_dir)),
                        "language": lang_code,
                        "language_name": lang_info["name"],
                        "is_tonal": lang_info["tonal"],
                        "original_file": str(audio_file),
                        "sample_rate": self.preprocessor.target_sr,
                        "duration": self.preprocessor.target_duration
                    })
                    processed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                continue
        
        logger.info(f"Successfully processed {processed_count} samples for {lang_code}")
    
    def balance_music_dataset(self) -> bool:
        """Balance music dataset across musical traditions."""
        logger.info("Balancing music dataset...")
        
        music_dir = self.base_dir / "music"
        output_music_dir = self.output_dir / "music"
        
        if not music_dir.exists():
            logger.warning("Music dataset not found. Skipping...")
            return False
        
        # Define musical traditions
        traditions = [
            "western_classical", "jazz", "blues", "country",
            "indian_classical", "middle_eastern", "african", "latin"
        ]
        
        # Process each tradition
        for tradition in traditions:
            tradition_dir = music_dir / "mtg_jamendo" / tradition
            output_tradition_dir = output_music_dir / tradition
            
            if not tradition_dir.exists():
                logger.warning(f"Tradition directory {tradition_dir} not found")
                continue
            
            self._process_music_tradition(tradition_dir, output_tradition_dir, tradition)
        
        return True
    
    def _process_music_tradition(self, tradition_dir: Path, output_dir: Path, tradition: str) -> None:
        """Process samples for a musical tradition."""
        
        # Find audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(list(tradition_dir.rglob(f"*{ext}")))
        
        if not audio_files:
            logger.warning(f"No audio files found for {tradition}")
            return
        
        # Sample files if needed
        if len(audio_files) > self.samples_per_group:
            audio_files = random.sample(audio_files, self.samples_per_group)
        
        logger.info(f"Processing {len(audio_files)} files for {tradition}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_count = 0
        
        for audio_file in tqdm(audio_files, desc=f"Processing {tradition}"):
            try:
                audio = self.preprocessor.load_and_standardize_audio(audio_file)
                if audio is None:
                    continue
                
                output_filename = f"{tradition}_{processed_count:06d}.wav"
                output_path = output_dir / output_filename
                
                if self.preprocessor.save_audio(audio, output_path):
                    self.metadata["music"].append({
                        "file_path": str(output_path.relative_to(self.output_dir)),
                        "tradition": tradition,
                        "cultural_origin": self._get_cultural_origin(tradition),
                        "original_file": str(audio_file),
                        "sample_rate": self.preprocessor.target_sr,
                        "duration": self.preprocessor.target_duration
                    })
                    processed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                continue
        
        logger.info(f"Successfully processed {processed_count} samples for {tradition}")
    
    def balance_urban_sounds_dataset(self) -> bool:
        """Balance urban sounds dataset across cities."""
        logger.info("Balancing urban sounds dataset...")
        
        urban_dir = self.base_dir / "urban_sounds" / "urbansound8k"
        output_urban_dir = self.output_dir / "urban_sounds"
        
        if not urban_dir.exists():
            logger.warning("Urban sounds dataset not found. Skipping...")
            return False
        
        # Load city metadata
        try:
            with open(urban_dir / "cities_metadata.json", 'r') as f:
                cities_meta = json.load(f)
        except FileNotFoundError:
            logger.error("Cities metadata not found")
            return False
        
        # Process each city
        for city, city_info in cities_meta.items():
            city_dir = urban_dir / city
            output_city_dir = output_urban_dir / city
            
            if not city_dir.exists():
                logger.warning(f"City directory {city_dir} not found")
                continue
            
            self._process_city_samples(city_dir, output_city_dir, city, city_info)
        
        return True
    
    def _process_city_samples(self, city_dir: Path, output_dir: Path, 
                             city: str, city_info: Dict) -> None:
        """Process samples for a specific city."""
        
        # Find audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(list(city_dir.rglob(f"*{ext}")))
        
        if not audio_files:
            logger.warning(f"No audio files found for {city}")
            return
        
        # Sample files if needed
        if len(audio_files) > self.samples_per_group:
            audio_files = random.sample(audio_files, self.samples_per_group)
        
        logger.info(f"Processing {len(audio_files)} files for {city}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_count = 0
        
        for audio_file in tqdm(audio_files, desc=f"Processing {city}"):
            try:
                audio = self.preprocessor.load_and_standardize_audio(audio_file)
                if audio is None:
                    continue
                
                output_filename = f"{city}_{processed_count:06d}.wav"
                output_path = output_dir / output_filename
                
                if self.preprocessor.save_audio(audio, output_path):
                    self.metadata["urban_sounds"].append({
                        "file_path": str(output_path.relative_to(self.output_dir)),
                        "city": city,
                        "country": city_info["country"],
                        "population": city_info["population"],
                        "original_file": str(audio_file),
                        "sample_rate": self.preprocessor.target_sr,
                        "duration": self.preprocessor.target_duration
                    })
                    processed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                continue
        
        logger.info(f"Successfully processed {processed_count} samples for {city}")
    
    def _get_cultural_origin(self, tradition: str) -> str:
        """Map musical tradition to cultural origin."""
        origin_map = {
            "western_classical": "Western",
            "jazz": "African-American",
            "blues": "African-American", 
            "country": "American",
            "indian_classical": "South Asian",
            "middle_eastern": "Middle Eastern",
            "african": "African",
            "latin": "Latin American"
        }
        return origin_map.get(tradition, "Unknown")
    
    def _create_speech_splits(self, tonal_langs: List[str], non_tonal_langs: List[str]) -> None:
        """Create balanced train/val/test splits for speech data."""
        logger.info("Creating speech dataset splits...")
        
        speech_metadata = pd.DataFrame(self.metadata["speech"])
        
        if speech_metadata.empty:
            logger.warning("No speech metadata available for splitting")
            return
        
        # Create stratified splits maintaining tonal/non-tonal balance
        train_data, temp_data = train_test_split(
            speech_metadata, 
            test_size=0.4, 
            stratify=speech_metadata['is_tonal'],
            random_state=42
        )
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,
            stratify=temp_data['is_tonal'],
            random_state=42
        )
        
        # Save splits
        splits_dir = self.output_dir / "splits" / "speech"
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        train_data.to_csv(splits_dir / "train.csv", index=False)
        val_data.to_csv(splits_dir / "val.csv", index=False)
        test_data.to_csv(splits_dir / "test.csv", index=False)
        
        # Log split statistics
        logger.info(f"Speech splits created:")
        logger.info(f"  Train: {len(train_data)} samples")
        logger.info(f"  Val: {len(val_data)} samples") 
        logger.info(f"  Test: {len(test_data)} samples")
        
        # Log tonal distribution
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            tonal_dist = split_data['is_tonal'].value_counts()
            logger.info(f"  {split_name} - Tonal: {tonal_dist.get(True, 0)}, Non-tonal: {tonal_dist.get(False, 0)}")
    
    def create_all_splits(self) -> None:
        """Create train/val/test splits for all datasets."""
        logger.info("Creating dataset splits...")
        
        # Create splits for each domain
        for domain in ["music", "urban_sounds"]:
            if self.metadata[domain]:
                df = pd.DataFrame(self.metadata[domain])
                
                # Determine stratification column
                if domain == "music":
                    stratify_col = "tradition"
                elif domain == "urban_sounds":
                    stratify_col = "country"
                
                # Create splits
                train_data, temp_data = train_test_split(
                    df, test_size=0.4, stratify=df[stratify_col], random_state=42
                )
                val_data, test_data = train_test_split(
                    temp_data, test_size=0.5, stratify=temp_data[stratify_col], random_state=42
                )
                
                # Save splits
                splits_dir = self.output_dir / "splits" / domain
                splits_dir.mkdir(parents=True, exist_ok=True)
                
                train_data.to_csv(splits_dir / "train.csv", index=False)
                val_data.to_csv(splits_dir / "val.csv", index=False)
                test_data.to_csv(splits_dir / "test.csv", index=False)
                
                logger.info(f"{domain} splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    def save_metadata(self) -> None:
        """Save comprehensive metadata."""
        logger.info("Saving metadata...")
        
        # Create metadata summary
        metadata_summary = {
            "fairaudiobench_version": "1.0.0",
            "preprocessing_date": "2025-09-06",
            "audio_format": {
                "sample_rate": self.preprocessor.target_sr,
                "duration": self.preprocessor.target_duration,
                "channels": 1,
                "format": "wav"
            },
            "balancing": {
                "samples_per_group": self.samples_per_group,
                "strategy": "random_sampling"
            },
            "statistics": {
                "speech": {
                    "total_samples": len(self.metadata["speech"]),
                    "languages": len(set([s["language"] for s in self.metadata["speech"]])),
                    "tonal_samples": len([s for s in self.metadata["speech"] if s["is_tonal"]]),
                    "non_tonal_samples": len([s for s in self.metadata["speech"] if not s["is_tonal"]])
                },
                "music": {
                    "total_samples": len(self.metadata["music"]),
                    "traditions": len(set([s["tradition"] for s in self.metadata["music"]]))
                },
                "urban_sounds": {
                    "total_samples": len(self.metadata["urban_sounds"]),
                    "cities": len(set([s["city"] for s in self.metadata["urban_sounds"]]))
                }
            }
        }
        
        # Save metadata
        with open(self.output_dir / "metadata_summary.json", 'w') as f:
            json.dump(metadata_summary, f, indent=2)
        
        # Save detailed metadata for each domain
        for domain, data in self.metadata.items():
            if data:
                df = pd.DataFrame(data)
                df.to_csv(self.output_dir / f"{domain}_metadata.csv", index=False)
        
        logger.info("Metadata saved successfully")
    
    def process_all(self) -> bool:
        """Process all datasets."""
        logger.info("Starting dataset preprocessing and balancing...")
        
        success = True
        success &= self.balance_speech_dataset()
        success &= self.balance_music_dataset()
        success &= self.balance_urban_sounds_dataset()
        
        if success:
            self.create_all_splits()
            self.save_metadata()
            logger.info("Dataset preprocessing completed successfully!")
        else:
            logger.error("Some datasets failed to process.")
        
        return success

def main():
    """Main function to run the dataset preprocessor."""
    parser = argparse.ArgumentParser(description="Preprocess and balance FairAudioBench datasets")
    parser.add_argument(
        "--input-dir",
        default="./datasets",
        help="Input directory containing raw datasets"
    )
    parser.add_argument(
        "--output-dir",
        default="./processed_datasets",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=1000,
        help="Number of samples per group (default: 1000)"
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000)"
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=4.0,
        help="Target duration in seconds (default: 4.0)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    balancer = DatasetBalancer(
        base_dir=args.input_dir,
        output_dir=args.output_dir,
        samples_per_group=args.samples_per_group
    )
    
    # Update preprocessor settings
    balancer.preprocessor.target_sr = args.target_sr
    balancer.preprocessor.target_duration = args.target_duration
    balancer.preprocessor.target_samples = int(args.target_sr * args.target_duration)
    
    success = balancer.process_all()
    
    if success:
        print(f"\nDataset preprocessing completed!")
        print(f"Processed datasets saved to: {args.output_dir}")
        print(f"Check metadata_summary.json for statistics")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
