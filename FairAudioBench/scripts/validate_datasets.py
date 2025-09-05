#!/usr/bin/env python3

"""
FairAudioBench Dataset Validation Script
Validates all processed datasets and their metadata
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Validates FairAudioBench datasets"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
    
    def validate_speech_datasets(self):
        """Validate speech recognition datasets"""
        logger.info("Validating speech datasets...")
        
        speech_dir = self.data_dir / "speech"
        if not speech_dir.exists():
            logger.error(f"Speech directory not found: {speech_dir}")
            return False
        
        # Expected languages from CommonVoice
        expected_languages = [
            "english", "spanish", "french", "german", "italian", 
            "portuguese", "russian", "chinese", "japanese", "arabic"
        ]
        
        speech_results = {}
        total_files = 0
        
        for lang in expected_languages:
            lang_dir = speech_dir / lang
            if lang_dir.exists():
                metadata_file = lang_dir / "metadata.csv"
                if metadata_file.exists():
                    df = pd.read_csv(metadata_file)
                    file_count = len(df)
                    speech_results[lang] = {
                        "files": file_count,
                        "metadata_valid": True,
                        "audio_dir_exists": (lang_dir / "audio").exists()
                    }
                    total_files += file_count
                    logger.info(f"  {lang}: {file_count} files")
                else:
                    speech_results[lang] = {
                        "files": 0,
                        "metadata_valid": False,
                        "audio_dir_exists": False
                    }
                    logger.warning(f"  {lang}: Missing metadata.csv")
            else:
                logger.warning(f"  {lang}: Directory not found")
        
        self.validation_results["speech"] = {
            "total_files": total_files,
            "languages": speech_results,
            "expected_languages": len(expected_languages),
            "found_languages": len([l for l in speech_results if speech_results[l]["metadata_valid"]])
        }
        
        logger.info(f"Speech validation: {total_files} total files across {len(speech_results)} languages")
        return len(speech_results) > 0
    
    def validate_music_datasets(self):
        """Validate music classification datasets"""
        logger.info("Validating music datasets...")
        
        music_dir = self.data_dir / "music"
        if not music_dir.exists():
            logger.error(f"Music directory not found: {music_dir}")
            return False
        
        # Expected music traditions
        expected_traditions = [
            "gtzan", "fma", "carnatic", "turkish_makam", 
            "hindustani", "arab_andalusian"
        ]
        
        music_results = {}
        total_files = 0
        
        for tradition in expected_traditions:
            tradition_dir = music_dir / tradition
            if tradition_dir.exists():
                metadata_file = tradition_dir / "metadata.csv"
                if metadata_file.exists():
                    df = pd.read_csv(metadata_file)
                    file_count = len(df)
                    music_results[tradition] = {
                        "files": file_count,
                        "metadata_valid": True,
                        "audio_dir_exists": (tradition_dir / "audio").exists()
                    }
                    total_files += file_count
                    
                    # Check for genre/class distribution
                    if 'genre' in df.columns:
                        genres = df['genre'].value_counts()
                        music_results[tradition]["genres"] = len(genres)
                        logger.info(f"  {tradition}: {file_count} files, {len(genres)} genres")
                    else:
                        logger.info(f"  {tradition}: {file_count} files")
                else:
                    music_results[tradition] = {
                        "files": 0,
                        "metadata_valid": False,
                        "audio_dir_exists": False
                    }
                    logger.warning(f"  {tradition}: Missing metadata.csv")
            else:
                logger.warning(f"  {tradition}: Directory not found")
        
        self.validation_results["music"] = {
            "total_files": total_files,
            "traditions": music_results,
            "expected_traditions": len(expected_traditions),
            "found_traditions": len([t for t in music_results if music_results[t]["metadata_valid"]])
        }
        
        logger.info(f"Music validation: {total_files} total files across {len(music_results)} traditions")
        return len(music_results) > 0
    
    def validate_scene_datasets(self):
        """Validate environmental sound datasets"""
        logger.info("Validating scene datasets...")
        
        scenes_dir = self.data_dir / "scenes"
        if not scenes_dir.exists():
            logger.error(f"Scenes directory not found: {scenes_dir}")
            return False
        
        # TAU Urban Acoustic Scenes
        tau_dir = scenes_dir / "tau_urban"
        scene_results = {}
        total_files = 0
        
        if tau_dir.exists():
            metadata_file = tau_dir / "metadata.csv"
            if metadata_file.exists():
                df = pd.read_csv(metadata_file)
                file_count = len(df)
                scene_results["tau_urban"] = {
                    "files": file_count,
                    "metadata_valid": True,
                    "audio_dir_exists": (tau_dir / "audio").exists()
                }
                total_files += file_count
                
                # Check city distribution
                if 'city' in df.columns:
                    cities = df['city'].value_counts()
                    scene_results["tau_urban"]["cities"] = len(cities)
                    logger.info(f"  tau_urban: {file_count} files, {len(cities)} cities")
                
                # Check scene distribution  
                if 'scene_label' in df.columns:
                    scenes = df['scene_label'].value_counts()
                    scene_results["tau_urban"]["scenes"] = len(scenes)
                    logger.info(f"    Scenes: {list(scenes.index)}")
            else:
                scene_results["tau_urban"] = {
                    "files": 0,
                    "metadata_valid": False,
                    "audio_dir_exists": False
                }
                logger.warning("  tau_urban: Missing metadata.csv")
        else:
            logger.warning("  tau_urban: Directory not found")
        
        self.validation_results["scenes"] = {
            "total_files": total_files,
            "datasets": scene_results,
            "expected_datasets": 1,
            "found_datasets": len([d for d in scene_results if scene_results[d]["metadata_valid"]])
        }
        
        logger.info(f"Scenes validation: {total_files} total files")
        return len(scene_results) > 0
    
    def validate_summaries(self):
        """Validate summary files"""
        logger.info("Validating summary files...")
        
        # Check for master summary
        master_summary = self.data_dir / "master_summary.json"
        if master_summary.exists():
            with open(master_summary, 'r') as f:
                summary_data = json.load(f)
            logger.info(f"Master summary found with {len(summary_data)} entries")
            self.validation_results["master_summary"] = True
        else:
            logger.warning("Master summary not found")
            self.validation_results["master_summary"] = False
        
        # Check individual domain summaries
        for domain in ["speech", "music", "scenes"]:
            summary_file = self.data_dir / domain / "summary.json"
            if summary_file.exists():
                logger.info(f"Domain summary found: {domain}")
                self.validation_results[f"{domain}_summary"] = True
            else:
                logger.warning(f"Domain summary missing: {domain}")
                self.validation_results[f"{domain}_summary"] = False
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        logger.info("\n" + "="*60)
        logger.info("FAIRAUDIOBENCH DATASET VALIDATION REPORT")
        logger.info("="*60)
        
        # Overall statistics
        total_files = 0
        if "speech" in self.validation_results:
            total_files += self.validation_results["speech"]["total_files"]
        if "music" in self.validation_results:
            total_files += self.validation_results["music"]["total_files"]
        if "scenes" in self.validation_results:
            total_files += self.validation_results["scenes"]["total_files"]
        
        logger.info(f"Total processed files: {total_files:,}")
        logger.info("")
        
        # Speech report
        if "speech" in self.validation_results:
            speech_data = self.validation_results["speech"]
            logger.info(f"SPEECH RECOGNITION:")
            logger.info(f"  Total files: {speech_data['total_files']:,}")
            logger.info(f"  Languages: {speech_data['found_languages']}/{speech_data['expected_languages']}")
            
            for lang, data in speech_data["languages"].items():
                if data["metadata_valid"]:
                    status = "✓" if data["audio_dir_exists"] else "⚠"
                    logger.info(f"    {status} {lang}: {data['files']} files")
            logger.info("")
        
        # Music report
        if "music" in self.validation_results:
            music_data = self.validation_results["music"]
            logger.info(f"MUSIC CLASSIFICATION:")
            logger.info(f"  Total files: {music_data['total_files']:,}")
            logger.info(f"  Traditions: {music_data['found_traditions']}/{music_data['expected_traditions']}")
            
            for tradition, data in music_data["traditions"].items():
                if data["metadata_valid"]:
                    status = "✓" if data["audio_dir_exists"] else "⚠"
                    genre_info = f" ({data.get('genres', 0)} genres)" if "genres" in data else ""
                    logger.info(f"    {status} {tradition}: {data['files']} files{genre_info}")
            logger.info("")
        
        # Scenes report
        if "scenes" in self.validation_results:
            scenes_data = self.validation_results["scenes"]
            logger.info(f"ENVIRONMENTAL SOUNDS:")
            logger.info(f"  Total files: {scenes_data['total_files']:,}")
            logger.info(f"  Datasets: {scenes_data['found_datasets']}/{scenes_data['expected_datasets']}")
            
            for dataset, data in scenes_data["datasets"].items():
                if data["metadata_valid"]:
                    status = "✓" if data["audio_dir_exists"] else "⚠"
                    city_info = f" ({data.get('cities', 0)} cities)" if "cities" in data else ""
                    scene_info = f" ({data.get('scenes', 0)} scenes)" if "scenes" in data else ""
                    logger.info(f"    {status} {dataset}: {data['files']} files{city_info}{scene_info}")
            logger.info("")
        
        # Summary status
        logger.info("METADATA SUMMARIES:")
        summary_items = [
            ("master_summary", "Master Summary"),
            ("speech_summary", "Speech Summary"),
            ("music_summary", "Music Summary"),
            ("scenes_summary", "Scenes Summary")
        ]
        
        for key, name in summary_items:
            if key in self.validation_results:
                status = "✓" if self.validation_results[key] else "✗"
                logger.info(f"  {status} {name}")
        
        logger.info("\n" + "="*60)
        logger.info("Validation complete!")
        
        return self.validation_results

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate FairAudioBench datasets")
    parser.add_argument("--data_dir", type=str, 
                       default="../processed_data",
                       help="Path to processed data directory")
    parser.add_argument("--output", type=str,
                       help="Output file for validation report (JSON)")
    
    args = parser.parse_args()
    
    # Determine data directory
    if Path(args.data_dir).is_absolute():
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent / args.data_dir
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Run validation
    validator = DatasetValidator(data_dir)
    
    # Validate each domain
    speech_ok = validator.validate_speech_datasets()
    music_ok = validator.validate_music_datasets()
    scenes_ok = validator.validate_scene_datasets()
    validator.validate_summaries()
    
    # Generate report
    results = validator.generate_report()
    
    # Save results if requested
    if args.output:
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Validation results saved to: {output_file}")
    
    # Return status
    if speech_ok or music_ok or scenes_ok:
        logger.info("Dataset validation completed successfully!")
        return 0
    else:
        logger.error("Dataset validation failed - no valid datasets found")
        return 1

if __name__ == "__main__":
    sys.exit(main())
