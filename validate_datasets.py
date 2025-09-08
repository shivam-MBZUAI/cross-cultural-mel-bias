#!/usr/bin/env python3
"""
Cross-Cultural Mel-Scale Audio Frontend Bias Research
Dataset Validation Script

Validates downloaded and processed datasets for quality and compliance
with balanced evaluation protocols.

Authors: Shivam Chauhan, Ajay Pundhir
Organization: Presight AI, Abu Dhabi, UAE
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
VALIDATION_DIR = PROJECT_ROOT / "validation"
VALIDATION_DIR.mkdir(exist_ok=True)

# Expected sample counts for balanced evaluation
EXPECTED_SPEECH_SAMPLES = 2000  # per language
EXPECTED_MUSIC_SAMPLES = 300    # per tradition
EXPECTED_SCENE_SAMPLES = 100    # per city

# Target languages and datasets
TONAL_LANGUAGES = ['vi', 'th', 'zh-CN', 'pa-IN', 'yue']
NON_TONAL_LANGUAGES = ['en', 'es', 'de', 'fr', 'it', 'nl']
MUSIC_TRADITIONS = ['gtzan', 'fma', 'carnatic', 'turkish_makam', 'hindustani', 'arab_andalusian']
SCENE_CITIES = ['barcelona', 'helsinki', 'london', 'paris', 'stockholm', 'vienna', 'amsterdam', 'lisbon', 'lyon', 'prague']

class DatasetValidator:
    """Validates datasets for quality and compliance with paper protocols."""
    
    def __init__(self):
        self.validation_results = {
            "speech": {},
            "music": {},
            "scenes": {},
            "summary": {}
        }
    
    def validate_speech_datasets(self) -> Dict:
        """Validate CommonVoice speech datasets."""
        print("Validating speech datasets...")
        speech_results = {}
        
        all_languages = TONAL_LANGUAGES + NON_TONAL_LANGUAGES
        
        for lang in all_languages:
            print(f"  Validating {lang}...")
            
            # Check raw data
            raw_dir = DATA_DIR / f"commonvoice_{lang}"
            processed_dir = PROCESSED_DIR / "speech" / lang
            
            lang_results = {
                "raw_exists": raw_dir.exists(),
                "processed_exists": processed_dir.exists(),
                "raw_sample_count": 0,
                "processed_sample_count": 0,
                "metadata_valid": False,
                "audio_quality": {},
                "compliance": {}
            }
            
            # Validate raw data
            if raw_dir.exists():
                raw_metadata = raw_dir / "metadata.csv"
                if raw_metadata.exists():
                    try:
                        df = pd.read_csv(raw_metadata)
                        lang_results["raw_sample_count"] = len(df)
                        lang_results["metadata_valid"] = True
                    except Exception as e:
                        print(f"    Error reading raw metadata: {e}")
            
            # Validate processed data
            if processed_dir.exists():
                processed_metadata = processed_dir / "metadata.csv"
                if processed_metadata.exists():
                    try:
                        df = pd.read_csv(processed_metadata)
                        lang_results["processed_sample_count"] = len(df)
                        
                        # Check sample count compliance
                        expected = EXPECTED_SPEECH_SAMPLES
                        actual = len(df)
                        lang_results["compliance"]["sample_count"] = {
                            "expected": expected,
                            "actual": actual,
                            "compliant": abs(actual - expected) <= expected * 0.05  # 5% tolerance
                        }
                        
                        # Validate audio quality (sample a few files)
                        audio_dir = processed_dir / "audio"
                        if audio_dir.exists():
                            lang_results["audio_quality"] = self._validate_audio_quality(
                                audio_dir, df.sample(min(10, len(df)))
                            )
                        
                    except Exception as e:
                        print(f"    Error validating processed data: {e}")
            
            speech_results[lang] = lang_results
        
        self.validation_results["speech"] = speech_results
        return speech_results
    
    def validate_music_datasets(self) -> Dict:
        """Validate music datasets."""
        print("Validating music datasets...")
        music_results = {}
        
        for tradition in MUSIC_TRADITIONS:
            print(f"  Validating {tradition}...")
            
            # Check processed data
            processed_dir = PROCESSED_DIR / "music" / tradition
            
            tradition_results = {
                "processed_exists": processed_dir.exists(),
                "processed_sample_count": 0,
                "metadata_valid": False,
                "audio_quality": {},
                "compliance": {}
            }
            
            if processed_dir.exists():
                processed_metadata = processed_dir / "metadata.csv"
                if processed_metadata.exists():
                    try:
                        df = pd.read_csv(processed_metadata)
                        tradition_results["processed_sample_count"] = len(df)
                        tradition_results["metadata_valid"] = True
                        
                        # Check sample count compliance
                        expected = EXPECTED_MUSIC_SAMPLES
                        actual = len(df)
                        tradition_results["compliance"]["sample_count"] = {
                            "expected": expected,
                            "actual": actual,
                            "compliant": abs(actual - expected) <= expected * 0.05
                        }
                        
                        # Validate audio quality
                        audio_dir = processed_dir / "audio"
                        if audio_dir.exists():
                            tradition_results["audio_quality"] = self._validate_audio_quality(
                                audio_dir, df.sample(min(5, len(df))), expected_duration=30.0
                            )
                        
                    except Exception as e:
                        print(f"    Error validating {tradition}: {e}")
            
            music_results[tradition] = tradition_results
        
        self.validation_results["music"] = music_results
        return music_results
    
    def validate_scene_datasets(self) -> Dict:
        """Validate acoustic scene datasets."""
        print("Validating acoustic scene datasets...")
        scene_results = {}
        
        for city in SCENE_CITIES:
            print(f"  Validating {city}...")
            
            processed_dir = PROCESSED_DIR / "scenes" / city
            
            city_results = {
                "processed_exists": processed_dir.exists(),
                "processed_sample_count": 0,
                "metadata_valid": False,
                "audio_quality": {},
                "compliance": {}
            }
            
            if processed_dir.exists():
                processed_metadata = processed_dir / "metadata.csv"
                if processed_metadata.exists():
                    try:
                        df = pd.read_csv(processed_metadata)
                        city_results["processed_sample_count"] = len(df)
                        city_results["metadata_valid"] = True
                        
                        # Check sample count compliance
                        expected = EXPECTED_SCENE_SAMPLES
                        actual = len(df)
                        city_results["compliance"]["sample_count"] = {
                            "expected": expected,
                            "actual": actual,
                            "compliant": abs(actual - expected) <= expected * 0.1  # 10% tolerance
                        }
                        
                        # Validate audio quality
                        audio_dir = processed_dir / "audio"
                        if audio_dir.exists():
                            city_results["audio_quality"] = self._validate_audio_quality(
                                audio_dir, df.sample(min(5, len(df))), expected_duration=10.0, expected_sr=48000
                            )
                        
                    except Exception as e:
                        print(f"    Error validating {city}: {e}")
            
            scene_results[city] = city_results
        
        self.validation_results["scenes"] = scene_results
        return scene_results
    
    def _validate_audio_quality(self, audio_dir: Path, sample_df: pd.DataFrame, 
                               expected_duration: Optional[float] = None,
                               expected_sr: int = 22050) -> Dict:
        """Validate audio file quality for a sample of files."""
        quality_results = {
            "files_checked": 0,
            "files_valid": 0,
            "sample_rate_correct": 0,
            "duration_correct": 0,
            "amplitude_normalized": 0,
            "no_corruption": 0,
            "average_duration": 0.0,
            "sample_rates": []
        }
        
        durations = []
        
        for _, row in sample_df.iterrows():
            try:
                # Get audio file path
                if 'audio_path' in row:
                    audio_path = Path(row['audio_path'])
                    if not audio_path.is_absolute():
                        audio_path = audio_dir / audio_path.name
                else:
                    # Try to find file by pattern
                    audio_files = list(audio_dir.glob("*.wav"))
                    if not audio_files:
                        continue
                    audio_path = audio_files[0]
                
                if not audio_path.exists():
                    continue
                
                quality_results["files_checked"] += 1
                
                # Load audio
                audio, sr = sf.read(audio_path)
                duration = len(audio) / sr
                
                durations.append(duration)
                quality_results["sample_rates"].append(sr)
                
                # Check sample rate
                if sr == expected_sr:
                    quality_results["sample_rate_correct"] += 1
                
                # Check duration
                if expected_duration is None or abs(duration - expected_duration) <= 2.0:
                    quality_results["duration_correct"] += 1
                
                # Check amplitude normalization
                if np.max(np.abs(audio)) <= 1.0 and np.max(np.abs(audio)) > 0.01:
                    quality_results["amplitude_normalized"] += 1
                
                # Check for corruption (silence, clipping)
                if not (np.all(audio == 0) or np.any(np.abs(audio) >= 0.99)):
                    quality_results["no_corruption"] += 1
                
                quality_results["files_valid"] += 1
                
            except Exception as e:
                print(f"      Error checking audio file: {e}")
                continue
        
        if durations:
            quality_results["average_duration"] = float(np.mean(durations))
        
        return quality_results
    
    def generate_summary(self) -> Dict:
        """Generate validation summary."""
        summary = {
            "speech": {
                "total_languages": len(TONAL_LANGUAGES + NON_TONAL_LANGUAGES),
                "languages_processed": 0,
                "total_samples": 0,
                "compliant_languages": 0
            },
            "music": {
                "total_traditions": len(MUSIC_TRADITIONS),
                "traditions_processed": 0,
                "total_samples": 0,
                "compliant_traditions": 0
            },
            "scenes": {
                "total_cities": len(SCENE_CITIES),
                "cities_processed": 0,
                "total_samples": 0,
                "compliant_cities": 0
            },
            "overall_compliance": False
        }
        
        # Speech summary
        speech_data = self.validation_results["speech"]
        for lang, results in speech_data.items():
            if results["processed_exists"]:
                summary["speech"]["languages_processed"] += 1
                summary["speech"]["total_samples"] += results["processed_sample_count"]
                
                if results.get("compliance", {}).get("sample_count", {}).get("compliant", False):
                    summary["speech"]["compliant_languages"] += 1
        
        # Music summary
        music_data = self.validation_results["music"]
        for tradition, results in music_data.items():
            if results["processed_exists"]:
                summary["music"]["traditions_processed"] += 1
                summary["music"]["total_samples"] += results["processed_sample_count"]
                
                if results.get("compliance", {}).get("sample_count", {}).get("compliant", False):
                    summary["music"]["compliant_traditions"] += 1
        
        # Scene summary
        scene_data = self.validation_results["scenes"]
        for city, results in scene_data.items():
            if results["processed_exists"]:
                summary["scenes"]["cities_processed"] += 1
                summary["scenes"]["total_samples"] += results["processed_sample_count"]
                
                if results.get("compliance", {}).get("sample_count", {}).get("compliant", False):
                    summary["scenes"]["compliant_cities"] += 1
        
        # Overall compliance
        speech_compliant = summary["speech"]["compliant_languages"] >= len(TONAL_LANGUAGES + NON_TONAL_LANGUAGES) * 0.8
        music_compliant = summary["music"]["compliant_traditions"] >= len(MUSIC_TRADITIONS) * 0.5
        scene_compliant = summary["scenes"]["compliant_cities"] >= len(SCENE_CITIES) * 0.5
        
        summary["overall_compliance"] = speech_compliant and (music_compliant or scene_compliant)
        
        self.validation_results["summary"] = summary
        return summary
    
    def save_results(self, output_path: Path):
        """Save validation results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        print(f"Validation results saved to: {output_path}")
    
    def generate_report(self, output_path: Path):
        """Generate human-readable validation report."""
        summary = self.validation_results["summary"]
        
        report = f"""# Dataset Validation Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

### Speech Datasets (CommonVoice)
- **Languages Processed**: {summary['speech']['languages_processed']}/{summary['speech']['total_languages']}
- **Total Samples**: {summary['speech']['total_samples']:,}
- **Compliant Languages**: {summary['speech']['compliant_languages']}/{summary['speech']['total_languages']}
- **Target per Language**: {EXPECTED_SPEECH_SAMPLES:,} samples

### Music Datasets
- **Traditions Processed**: {summary['music']['traditions_processed']}/{summary['music']['total_traditions']}
- **Total Samples**: {summary['music']['total_samples']:,}
- **Compliant Traditions**: {summary['music']['compliant_traditions']}/{summary['music']['total_traditions']}
- **Target per Tradition**: {EXPECTED_MUSIC_SAMPLES:,} samples

### Acoustic Scene Datasets
- **Cities Processed**: {summary['scenes']['cities_processed']}/{summary['scenes']['total_cities']}
- **Total Samples**: {summary['scenes']['total_samples']:,}
- **Compliant Cities**: {summary['scenes']['compliant_cities']}/{summary['scenes']['total_cities']}
- **Target per City**: {EXPECTED_SCENE_SAMPLES:,} samples

## Overall Compliance: {'✅ PASS' if summary['overall_compliance'] else '❌ FAIL'}

## Detailed Results

### Speech Languages
"""
        
        speech_data = self.validation_results["speech"]
        for lang in TONAL_LANGUAGES + NON_TONAL_LANGUAGES:
            results = speech_data.get(lang, {})
            status = "✅" if results.get("processed_exists", False) else "❌"
            sample_count = results.get("processed_sample_count", 0)
            compliant = "✅" if results.get("compliance", {}).get("sample_count", {}).get("compliant", False) else "❌"
            
            report += f"- **{lang}**: {status} Processed: {sample_count:,} samples {compliant}\n"
        
        report += "\n### Music Traditions\n"
        music_data = self.validation_results["music"]
        for tradition in MUSIC_TRADITIONS:
            results = music_data.get(tradition, {})
            status = "✅" if results.get("processed_exists", False) else "❌"
            sample_count = results.get("processed_sample_count", 0)
            compliant = "✅" if results.get("compliance", {}).get("sample_count", {}).get("compliant", False) else "❌"
            
            report += f"- **{tradition}**: {status} Processed: {sample_count:,} samples {compliant}\n"
        
        report += "\n### Acoustic Scene Cities\n"
        scene_data = self.validation_results["scenes"]
        for city in SCENE_CITIES:
            results = scene_data.get(city, {})
            status = "✅" if results.get("processed_exists", False) else "❌"
            sample_count = results.get("processed_sample_count", 0)
            compliant = "✅" if results.get("compliance", {}).get("sample_count", {}).get("compliant", False) else "❌"
            
            report += f"- **{city}**: {status} Processed: {sample_count:,} samples {compliant}\n"
        
        report += f"""

## Recommendations

{'✅ Datasets are ready for experiments!' if summary['overall_compliance'] else '⚠️ Some datasets need attention:'}

"""
        
        if not summary['overall_compliance']:
            if summary['speech']['compliant_languages'] < len(TONAL_LANGUAGES + NON_TONAL_LANGUAGES):
                report += "1. **Speech**: Run `python preprocess_datasets.py --domain speech` to process remaining languages\n"
            
            if summary['music']['compliant_traditions'] < len(MUSIC_TRADITIONS):
                report += "2. **Music**: Download missing datasets and run `python preprocess_datasets.py --domain music`\n"
            
            if summary['scenes']['compliant_cities'] < len(SCENE_CITIES):
                report += "3. **Scenes**: Download TAU dataset and run `python preprocess_datasets.py --domain scenes`\n"
        
        report += """
## Next Steps

1. **If validation passes**: Start experiments with `python experiments/reproduce_paper.py`
2. **If validation fails**: Follow recommendations above and re-run validation
3. **For detailed results**: Check `validation_results.json` for technical details

---
*This report validates compliance with balanced evaluation protocols described in the ICASSP 2026 paper.*
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Validation report saved to: {output_path}")

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate datasets for Cross-Cultural Mel-Scale Audio Frontend Bias Research")
    parser.add_argument("--domain", choices=["all", "speech", "music", "scenes"], default="all",
                       help="Domain to validate (default: all)")
    parser.add_argument("--output", type=str, default="validation",
                       help="Output directory for validation results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("=== Cross-Cultural Mel-Scale Audio Frontend Bias - Dataset Validation ===\n")
    
    validator = DatasetValidator()
    
    # Run validation
    if args.domain in ["all", "speech"]:
        validator.validate_speech_datasets()
    
    if args.domain in ["all", "music"]:
        validator.validate_music_datasets()
    
    if args.domain in ["all", "scenes"]:
        validator.validate_scene_datasets()
    
    # Generate summary and save results
    summary = validator.generate_summary()
    
    # Save detailed results
    results_path = output_dir / "validation_results.json"
    validator.save_results(results_path)
    
    # Generate human-readable report
    report_path = output_dir / "validation_report.md"
    validator.generate_report(report_path)
    
    # Print summary
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Overall Compliance: {'✅ PASS' if summary['overall_compliance'] else '❌ FAIL'}")
    print(f"Speech: {summary['speech']['compliant_languages']}/{summary['speech']['total_languages']} languages")
    print(f"Music: {summary['music']['compliant_traditions']}/{summary['music']['total_traditions']} traditions")
    print(f"Scenes: {summary['scenes']['compliant_cities']}/{summary['scenes']['total_cities']} cities")
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
