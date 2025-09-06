#!/usr/bin/env python3
"""
FairAudioBench Dataset Downloader
Downloads all required datasets for cross-cultural bias evaluation based on reference implementation
"""

import os
import sys
from pathlib import Path

# Set cache directories relative to project or user home
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "hf_cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Use environment variable if set, otherwise use project-relative cache
HF_CACHE_BASE = os.environ.get("HF_CACHE_BASE", str(DEFAULT_CACHE_DIR))

os.environ["HF_HOME"] = HF_CACHE_BASE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_BASE, "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_BASE, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_BASE, "datasets")

import argparse
from pathlib import Path
import requests
from datasets import load_dataset, Dataset
import soundfile as sf
import pandas as pd
import zipfile
import tarfile
from typing import List, Dict, Optional, Tuple

# Configuration
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Create HF cache directory in our data folder
HF_CACHE_DIR = DATA_DIR / "hf_cache"
HF_CACHE_DIR.mkdir(exist_ok=True)

# Available datasets and configurations - from reference implementation
COMMONVOICE_LANGUAGES = {
    'ab': 'Abkhaz', 'ace': 'Acehnese', 'ady': 'Adyghe', 'af': 'Afrikaans', 'am': 'Amharic', 
    'an': 'Aragonese', 'ar': 'Arabic', 'arn': 'Mapudungun', 'as': 'Assamese', 'ast': 'Asturian', 
    'az': 'Azerbaijani', 'ba': 'Bashkir', 'bas': 'Basaa', 'be': 'Belarusian', 'bg': 'Bulgarian', 
    'bm': 'Bambara', 'bn': 'Bengali', 'bo': 'Tibetan', 'br': 'Breton', 'bs': 'Bosnian', 
    'bxr': 'Buryat', 'byv': 'Medumba', 'ca': 'Catalan', 'cak': 'Kaqchikel', 'ckb': 'Central Kurdish', 
    'cnh': 'Hakha Chin', 'co': 'Corsican', 'crh': 'Crimean Tatar', 'cs': 'Czech', 'cv': 'Chuvash', 
    'cy': 'Welsh', 'da': 'Danish', 'dag': 'Dagbani', 'de': 'German', 'dsb': 'Sorbian, Lower', 
    'dv': 'Dhivehi', 'dyu': 'Dioula', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 
    'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'ewo': 'Ewondo', 'fa': 'Persian', 
    'ff': 'Fulah', 'fi': 'Finnish', 'fo': 'Faroese', 'fr': 'French', 'fuf': 'Pular Guinea', 
    'fy-NL': 'Frisian', 'ga-IE': 'Irish', 'gl': 'Galician', 'gn': 'Guarani', 'gom': 'Goan Konkani', 
    'gu-IN': 'Gujarati', 'guc': 'Wayuunaiki', 'ha': 'Hausa', 'he': 'Hebrew', 'hi': 'Hindi', 
    'hil': 'Hiligaynon', 'hr': 'Croatian', 'hsb': 'Sorbian, Upper', 'ht': 'Haitian', 'hu': 'Hungarian', 
    'hy-AM': 'Armenian', 'hyw': 'Armenian Western', 'ia': 'Interlingua', 'id': 'Indonesian', 
    'ie': 'Interlingue', 'ig': 'Igbo', 'is': 'Icelandic', 'it': 'Italian', 'izh': 'Izhorian', 
    'ja': 'Japanese', 'jbo': 'Lojban', 'jv': 'Javanese', 'ka': 'Georgian', 'kaa': 'Karakalpak', 
    'kab': 'Kabyle', 'kbd': 'Kabardian', 'ki': 'Kikuyu', 'kk': 'Kazakh', 'km': 'Khmer', 
    'kmr': 'Kurmanji Kurdish', 'kn': 'Kannada', 'knn': 'Konkani (Devanagari)', 'ko': 'Korean', 
    'kpv': 'Komi-Zyrian', 'kw': 'Cornish', 'ky': 'Kyrgyz', 'lb': 'Luxembourgish', 'lg': 'Luganda', 
    'lij': 'Ligurian', 'ln': 'Lingala', 'lo': 'Lao', 'lt': 'Lithuanian', 'ltg': 'Latgalian', 
    'lv': 'Latvian', 'lzz': 'Laz', 'mai': 'Maithili', 'mdf': 'Moksha', 'mg': 'Malagasy', 
    'mhr': 'Meadow Mari', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mni': 'Meetei Lon', 
    'mos': 'Mossi', 'mr': 'Marathi', 'mrj': 'Hill Mari', 'ms': 'Malay', 'mt': 'Maltese', 
    'my': 'Burmese', 'myv': 'Erzya', 'nan-tw': 'Taiwanese (Minnan)', 'nb-NO': 'Norwegian Bokmål', 
    'nd': 'IsiNdebele (North)', 'ne-NP': 'Nepali', 'nhe': 'Eastern Huasteca Nahuatl', 
    'nhi': 'Western Sierra Puebla Nahuatl', 'nia': 'Nias', 'nl': 'Dutch', 'nn-NO': 'Norwegian Nynorsk', 
    'nr': 'IsiNdebele (South)', 'nso': 'Northern Sotho', 'ny': 'Chinyanja', 'nyn': 'Runyankole', 
    'oc': 'Occitan', 'om': 'Afaan Oromo', 'or': 'Odia', 'os': 'Ossetian', 'pa-IN': 'Punjabi', 
    'pap-AW': 'Papiamento (Aruba)', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese', 
    'quc': "K'iche'", 'quy': 'Quechua Chanka', 'qvi': 'Kichwa', 'rm-sursilv': 'Romansh Sursilvan', 
    'rm-vallader': 'Romansh Vallader', 'ro': 'Romanian', 'ru': 'Russian', 'rw': 'Kinyarwanda', 
    'sah': 'Sakha', 'sat': 'Santali (Ol Chiki)', 'sc': 'Sardinian', 'scn': 'Sicilian', 'sco': 'Scots', 
    'sd': 'Sindhi', 'sdh': 'Southern Kurdish', 'shi': 'Shilha', 'si': 'Sinhala', 'sk': 'Slovak', 
    'skr': 'Saraiki', 'sl': 'Slovenian', 'snk': 'Soninke', 'so': 'Somali', 'sq': 'Albanian', 
    'sr': 'Serbian', 'ss': 'Siswati', 'st': 'Southern Sotho', 'sv-SE': 'Swedish', 'sw': 'Swahili', 
    'syr': 'Syriac', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 
    'tig': 'Tigre', 'tk': 'Turkmen', 'tl': 'Tagalog', 'tn': 'Setswana', 'tok': 'Toki Pona', 
    'tr': 'Turkish', 'ts': 'Xitsonga', 'tt': 'Tatar', 'tw': 'Twi', 'ty': 'Tahitian', 'tyv': 'Tuvan', 
    'uby': 'Ubykh', 'udm': 'Udmurt', 'ug': 'Uyghur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 
    've': 'Tshivenda', 'vec': 'Venetian', 'vi': 'Vietnamese', 'vmw': 'Emakhuwa', 'vot': 'Votic', 
    'wep': 'Westphalian', 'wo': 'Wolof', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 
    'yue': 'Cantonese', 'zgh': 'Tamazight', 'zh-CN': 'Chinese (China)', 'zh-HK': 'Chinese (Hong Kong)', 
    'zh-TW': 'Chinese (Taiwan)', 'zu': 'Zulu', 'zza': 'Zaza'
}

# Tonal vs non-tonal classification for research purposes
TONAL_LANGUAGES = ['vi', 'th', 'zh-HK', 'zh-TW', 'yue', 'nan-tw', 'pa-IN']  # Vietnamese, Thai, Chinese variants, Cantonese, Punjabi
NON_TONAL_LANGUAGES = ['en', 'es', 'de', 'fr', 'it', 'nl', 'pt', 'ru', 'pl', 'sv-SE']  # Major European languages

MUSIC_DATASETS = ["gtzan", "fma", "carnatic", "turkish_makam", "hindustani", "arab_andalusian"]
SCENE_DATASETS = ["tau_urban"]

ALL_DATASETS = ["commonvoice"] + MUSIC_DATASETS + SCENE_DATASETS

def setup_huggingface_auth(hf_token: Optional[str] = None) -> bool:
    """Setup Hugging Face authentication."""
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print(f"Using provided HF token")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"Authenticated as: {user.get('name', 'Unknown User')}")
        return True
    except Exception as e:
        print(f"Authentication failed: {e}")
        return False

class DatasetDownloader:
    """Handles downloading datasets for FairAudioBench - based on reference implementation."""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def download_commonvoice_hf(self, lang_code: str, max_samples: int = 2000, hf_token: Optional[str] = None) -> bool:
        """Download CommonVoice dataset from Hugging Face - based on reference implementation."""
        print(f"Downloading CommonVoice {COMMONVOICE_LANGUAGES.get(lang_code, lang_code)} ({lang_code}) from Hugging Face...")
        
        if lang_code not in COMMONVOICE_LANGUAGES:
            print(f"ERROR: Language code '{lang_code}' not supported")
            print(f"Available languages: {', '.join(list(COMMONVOICE_LANGUAGES.keys())[:20])}... (and {len(COMMONVOICE_LANGUAGES)-20} more)")
            return False
        
        output_dir = self.base_dir / f"commonvoice_{lang_code}"
        output_dir.mkdir(exist_ok=True)
        
        # Check authentication first
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            user_info = api.whoami()
            print(f"  Authenticated as: {user_info.get('name', 'Unknown')}")
        except Exception:
            print("  ERROR: Not authenticated with Hugging Face")
            return False
        
        try:
            print(f"  Loading CommonVoice 17.0 dataset for language: {lang_code}")
            
            # Use reference approach - direct load_dataset call with explicit cache_dir
            dataset = load_dataset(
                "mozilla-foundation/common_voice_17_0", 
                lang_code, 
                split="test",
                cache_dir=str(HF_CACHE_DIR)
            )
            
            print(f"  Processing up to {max_samples} samples...")
            
            audio_files = []
            texts = []
            
            # Process the dataset samples
            total_samples = len(dataset)
            samples_to_process = min(max_samples, total_samples)
            
            print(f"  Dataset loaded: {total_samples} total samples, processing {samples_to_process}")
            
            for i in range(samples_to_process):
                try:
                    sample = dataset[i]
                    
                    # Extract audio data
                    audio_data = sample["audio"]
                    audio_array = audio_data["array"]
                    sample_rate = audio_data["sampling_rate"]
                    
                    # Extract text
                    sentence = sample["sentence"].strip()
                    
                    # Skip if no valid content
                    if len(sentence) < 3 or len(audio_array) < 1000:  # Skip very short samples
                        continue
                    
                    # Save audio file
                    audio_path = output_dir / f"cv17_{lang_code}_{i:06d}.wav"
                    sf.write(str(audio_path), audio_array, sample_rate)
                    
                    # Clean text
                    if sentence.startswith('"') and sentence.endswith('"'):
                        sentence = sentence[1:-1]
                    if sentence and sentence[-1] not in [".", "?", "!"]:
                        sentence = sentence + "."
                    
                    audio_files.append(str(audio_path))
                    texts.append(sentence)
                    
                    if (i + 1) % 100 == 0:
                        print(f"    Processed {i + 1}/{samples_to_process} samples...")
                        
                except Exception as e:
                    print(f"    Warning: Failed to process sample {i}: {str(e)[:50]}...")
                    continue
            
            if audio_files:
                # Save metadata
                metadata = pd.DataFrame({
                    "audio_path": audio_files,
                    "text": texts,
                    "language": lang_code,
                    "language_name": COMMONVOICE_LANGUAGES[lang_code],
                    "source": "commonvoice_17.0",
                    "is_tonal": lang_code in TONAL_LANGUAGES
                })
                metadata.to_csv(output_dir / "metadata.csv", index=False)
                
                print(f"  SUCCESS: Downloaded {len(audio_files)} samples for {lang_code}")
                return True
            else:
                print(f"  ERROR: No valid samples processed for {lang_code}")
                return False
                
        except Exception as e:
            print(f"  ERROR downloading {lang_code}: {e}")
            return False

    def download_music_datasets(self) -> Dict[str, bool]:
        """Download music datasets with manual instructions."""
        results = {}
        
        print("\n=== MUSIC DATASETS ===")
        print("Due to licensing restrictions, music datasets require manual download.")
        print("Please follow these instructions:\n")
        
        # GTZAN
        print("1. GTZAN Dataset:")
        print("   - Download from: http://marsyas.info/downloads/datasets.html")
        print("   - Extract to: data/gtzan/")
        print("   - Structure: data/gtzan/genres/[genre]/[audio_files].wav")
        
        # FMA
        print("\n2. Free Music Archive (FMA):")
        print("   - Download fma_small.zip from: https://github.com/mdeff/fma")
        print("   - Extract to: data/fma_small/")
        print("   - Also download metadata: fma_metadata.zip")
        
        # Classical music
        print("\n3. Indian Classical Music:")
        print("   - Carnatic: Contact CompMusic project or use Saraga dataset")
        print("   - Hindustani: Contact CompMusic project or use Saraga dataset")
        print("   - Extract to: data/carnatic/ and data/hindustani/")
        
        # Turkish Makam
        print("\n4. Turkish Makam:")
        print("   - Contact CompMusic project")
        print("   - Extract to: data/turkish_makam/")
        
        # Arab-Andalusian
        print("\n5. Arab-Andalusian:")
        print("   - Contact CompMusic project")
        print("   - Extract to: data/arab_andalusian/")
        
        print("\nOnce downloaded, run the preprocessing script to standardize all datasets.")
        
        # Create placeholder directories and README files
        for dataset in MUSIC_DATASETS:
            dataset_dir = self.base_dir / dataset
            dataset_dir.mkdir(exist_ok=True)
            
            readme_path = dataset_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(f"# {dataset.upper()} Dataset\n\n")
                f.write("This directory should contain the manually downloaded dataset.\n")
                f.write("Please refer to the main download script for instructions.\n")
            
            results[dataset] = False  # Manual download required
        
        return results

    def download_scene_datasets(self) -> Dict[str, bool]:
        """Download acoustic scene datasets."""
        results = {}
        
        print("\n=== ACOUSTIC SCENE DATASETS ===")
        print("TAU Urban Acoustic Scenes requires registration.")
        print("Please follow these instructions:\n")
        
        print("1. TAU Urban Acoustic Scenes 2020:")
        print("   - Register at: https://dcase.community/challenge2020/")
        print("   - Download development dataset")
        print("   - Extract to: data/tau_urban/")
        print("   - Structure: data/tau_urban/audio/[city]/[scene]/[audio_files].wav")
        
        # Create placeholder directory
        tau_dir = self.base_dir / "tau_urban"
        tau_dir.mkdir(exist_ok=True)
        
        readme_path = tau_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write("# TAU Urban Acoustic Scenes Dataset\n\n")
            f.write("This directory should contain the manually downloaded dataset.\n")
            f.write("Please register at DCASE community for access.\n")
        
        results["tau_urban"] = False  # Manual download required
        return results

def main():
    """Main download function."""
    parser = argparse.ArgumentParser(description="Download datasets for FairAudioBench")
    parser.add_argument("--languages", nargs="+", default=TONAL_LANGUAGES + NON_TONAL_LANGUAGES[:6],
                       help="Language codes to download")
    parser.add_argument("--max_samples", type=int, default=2000,
                       help="Maximum samples per language")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Hugging Face token")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Data directory")
    parser.add_argument("--all", action="store_true",
                       help="Download all available datasets")
    
    args = parser.parse_args()
    
    # Setup authentication
    if args.hf_token:
        setup_huggingface_auth(args.hf_token)
    
    downloader = DatasetDownloader(args.data_dir)
    
    print("=== FairAudioBench Dataset Downloader ===\n")
    
    # Download speech datasets
    speech_results = {}
    for lang in args.languages:
        if lang in COMMONVOICE_LANGUAGES:
            success = downloader.download_commonvoice_hf(lang, args.max_samples, args.hf_token)
            speech_results[lang] = success
        else:
            print(f"Skipping unsupported language: {lang}")
    
    # Download music and scene datasets (manual instructions)
    music_results = downloader.download_music_datasets()
    scene_results = downloader.download_scene_datasets()
    
    # Summary
    print("\n=== DOWNLOAD SUMMARY ===")
    print(f"Speech datasets: {sum(speech_results.values())}/{len(speech_results)} successful")
    print(f"Music datasets: {sum(music_results.values())}/{len(music_results)} (manual download required)")
    print(f"Scene datasets: {sum(scene_results.values())}/{len(scene_results)} (manual download required)")
    
    if all(speech_results.values()):
        print("\n✓ All speech datasets downloaded successfully!")
        print("Next: Run the preprocessing script to standardize all datasets")
    else:
        failed = [k for k, v in speech_results.items() if not v]
        print(f"\n⚠ Failed downloads: {failed}")
        print("Check your Hugging Face authentication and try again")

if __name__ == "__main__":
    main()
                        "de": {"name": "German", "tonal": False},
                        "it": {"name": "Italian", "tonal": False},
                        "pt": {"name": "Portuguese", "tonal": False}
                    }
                },
                "librispeech": {
                    "url": "http://www.openslr.org/resources/12/",
                    "subsets": ["dev-clean", "test-clean", "train-clean-100"]
                }
            },
            "music": {
                "mtg_jamendo": {
                    "url": "https://mtg.github.io/mtg-jamendo-dataset/",
                    "traditions": [
                        "western_classical", "jazz", "blues", "country",
                        "indian_classical", "middle_eastern", "african", "latin"
                    ]
                },
                "musicnet": {
                    "url": "https://www.kaggle.com/datasets/imsparsh/musicnet-dataset"
                }
            },
            "urban_sounds": {
                "urbansound8k": {
                    "url": "https://urbansounddataset.weebly.com/urbansound8k.html",
                    "cities": [
                        "london", "paris", "berlin", "madrid", "rome",
                        "amsterdam", "vienna", "prague", "stockholm", "dublin"
                    ]
                },
                "freesound": {
                    "url": "https://freesound.org/api/sounds/",
                    "api_required": True
                }
            }
        }
    
    def verify_checksum(self, filepath: Path, expected_hash: str) -> bool:
        """Verify file integrity using SHA256 checksum."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_hash
    
    def download_file(self, url: str, filepath: Path, expected_hash: str = None) -> bool:
        """Download a file with progress bar and optional checksum verification."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=f"Downloading {filepath.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            if expected_hash and not self.verify_checksum(filepath, expected_hash):
                logger.error(f"Checksum verification failed for {filepath}")
                return False
            
            logger.info(f"Successfully downloaded {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def extract_archive(self, filepath: Path, extract_dir: Path) -> bool:
        """Extract various archive formats."""
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif filepath.suffix in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(filepath, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            elif filepath.suffix == '.gz':
                with gzip.open(filepath, 'rb') as gz_file:
                    with open(extract_dir / filepath.stem, 'wb') as out_file:
                        out_file.write(gz_file.read())
            else:
                logger.warning(f"Unsupported archive format: {filepath.suffix}")
                return False
            
            logger.info(f"Successfully extracted {filepath} to {extract_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {filepath}: {e}")
            return False
    
    def download_common_voice(self) -> bool:
        """Download Common Voice dataset for multiple languages."""
        logger.info("Downloading Common Voice datasets...")
        cv_dir = self.base_dir / "speech" / "common_voice"
        cv_dir.mkdir(parents=True, exist_ok=True)
        
        # Create language metadata
        languages_meta = {}
        
        for lang_code, lang_info in self.datasets_config["speech"]["common_voice"]["languages"].items():
            lang_dir = cv_dir / lang_code
            lang_dir.mkdir(exist_ok=True)
            
            # Store language metadata
            languages_meta[lang_code] = {
                "name": lang_info["name"],
                "tonal": lang_info["tonal"],
                "downloaded": False,
                "samples_count": 0
            }
            
            logger.info(f"Setting up {lang_info['name']} ({lang_code})")
            
            # Note: Actual Common Voice requires registration and manual download
            # This creates placeholder structure and documentation
            readme_content = f"""
# Common Voice - {lang_info['name']} ({lang_code})

## Download Instructions:
1. Visit https://commonvoice.mozilla.org/
2. Create an account and agree to terms
3. Download the {lang_info['name']} dataset
4. Extract to this directory: {lang_dir}

## Expected Structure:
```
{lang_code}/
├── clips/          # Audio files (.mp3)
├── train.tsv       # Training metadata
├── dev.tsv         # Development metadata
├── test.tsv        # Test metadata
└── validated.tsv   # All validated samples
```

## Language Properties:
- Tonal: {lang_info['tonal']}
- ISO Code: {lang_code}
"""
            with open(lang_dir / "README.md", 'w') as f:
                f.write(readme_content)
        
        # Save languages metadata
        with open(cv_dir / "languages_metadata.json", 'w') as f:
            json.dump(languages_meta, f, indent=2)
        
        return True
    
    def download_music_datasets(self) -> bool:
        """Download music datasets for cross-cultural analysis."""
        logger.info("Downloading music datasets...")
        music_dir = self.base_dir / "music"
        music_dir.mkdir(parents=True, exist_ok=True)
        
        # MTG-Jamendo dataset placeholder
        jamendo_dir = music_dir / "mtg_jamendo"
        jamendo_dir.mkdir(exist_ok=True)
        
        readme_content = """
# MTG-Jamendo Dataset

## Download Instructions:
1. Visit https://mtg.github.io/mtg-jamendo-dataset/
2. Download the dataset splits
3. Extract to this directory

## Musical Traditions Included:
- Western Classical
- Jazz
- Blues
- Country
- Indian Classical
- Middle Eastern
- African
- Latin

## Expected Structure:
```
mtg_jamendo/
├── audio/           # Audio files organized by tradition
├── metadata/        # Metadata files
└── splits/          # Train/val/test splits
```
"""
        with open(jamendo_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        return True
    
    def download_urban_sounds(self) -> bool:
        """Download urban sounds datasets."""
        logger.info("Downloading urban sounds datasets...")
        urban_dir = self.base_dir / "urban_sounds"
        urban_dir.mkdir(parents=True, exist_ok=True)
        
        # UrbanSound8K placeholder
        us8k_dir = urban_dir / "urbansound8k"
        us8k_dir.mkdir(exist_ok=True)
        
        # Create city metadata
        cities_meta = {}
        for city in self.datasets_config["urban_sounds"]["urbansound8k"]["cities"]:
            cities_meta[city] = {
                "country": self._get_country_for_city(city),
                "population": self._get_population_for_city(city),
                "samples_count": 0
            }
        
        with open(us8k_dir / "cities_metadata.json", 'w') as f:
            json.dump(cities_meta, f, indent=2)
        
        readme_content = """
# UrbanSound8K Dataset

## Download Instructions:
1. Visit https://urbansounddataset.weebly.com/urbansound8k.html
2. Register and download the dataset
3. Extract to this directory

## Cities Included:
- London, UK
- Paris, France
- Berlin, Germany
- Madrid, Spain
- Rome, Italy
- Amsterdam, Netherlands
- Vienna, Austria
- Prague, Czech Republic
- Stockholm, Sweden
- Dublin, Ireland

## Expected Structure:
```
urbansound8k/
├── audio/           # Audio files organized by city
├── metadata/        # City and demographic metadata
└── UrbanSound8K.csv # Original annotations
```
"""
        with open(us8k_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        return True
    
    def _get_country_for_city(self, city: str) -> str:
        """Get country for a given city."""
        city_country_map = {
            "london": "UK", "paris": "France", "berlin": "Germany",
            "madrid": "Spain", "rome": "Italy", "amsterdam": "Netherlands",
            "vienna": "Austria", "prague": "Czech Republic",
            "stockholm": "Sweden", "dublin": "Ireland"
        }
        return city_country_map.get(city.lower(), "Unknown")
    
    def _get_population_for_city(self, city: str) -> int:
        """Get approximate population for a given city."""
        population_map = {
            "london": 9000000, "paris": 2100000, "berlin": 3700000,
            "madrid": 3200000, "rome": 2800000, "amsterdam": 870000,
            "vienna": 1900000, "prague": 1300000,
            "stockholm": 975000, "dublin": 550000
        }
        return population_map.get(city.lower(), 0)
    
    def create_dataset_index(self) -> bool:
        """Create an index of all available datasets."""
        logger.info("Creating dataset index...")
        
        index = {
            "fairaudibench_version": "1.0.0",
            "created_date": "2025-09-06",
            "datasets": {
                "speech": {
                    "total_languages": 11,
                    "tonal_languages": 5,
                    "non_tonal_languages": 6,
                    "sources": ["common_voice", "librispeech"]
                },
                "music": {
                    "total_traditions": 8,
                    "sources": ["mtg_jamendo", "musicnet"]
                },
                "urban_sounds": {
                    "total_cities": 10,
                    "sources": ["urbansound8k", "freesound"]
                }
            },
            "download_instructions": {
                "automated": [
                    "Some datasets require manual registration",
                    "Run preprocessing script after manual downloads"
                ],
                "manual_required": [
                    "Common Voice (registration required)",
                    "MTG-Jamendo (academic license)",
                    "UrbanSound8K (registration required)"
                ]
            }
        }
        
        with open(self.base_dir / "dataset_index.json", 'w') as f:
            json.dump(index, f, indent=2)
        
        return True
    
    def download_all(self) -> bool:
        """Download all datasets."""
        logger.info("Starting FairAudioBench dataset download...")
        
        success = True
        success &= self.download_common_voice()
        success &= self.download_music_datasets()
        success &= self.download_urban_sounds()
        success &= self.create_dataset_index()
        
        if success:
            logger.info("Dataset download setup completed successfully!")
            logger.info(f"Datasets prepared in: {self.base_dir}")
            logger.info("Note: Some datasets require manual download due to licensing.")
            logger.info("Please check the README files in each dataset directory.")
        else:
            logger.error("Some datasets failed to download.")
        
        return success

def main():
    """Main function to run the dataset downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download FairAudioBench datasets")
    parser.add_argument(
        "--output-dir", 
        default="./datasets",
        help="Output directory for datasets (default: ./datasets)"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["all", "speech", "music", "urban"],
        default="all",
        help="Type of dataset to download (default: all)"
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.output_dir)
    
    if args.dataset_type == "all":
        success = downloader.download_all()
    elif args.dataset_type == "speech":
        success = downloader.download_common_voice()
    elif args.dataset_type == "music":
        success = downloader.download_music_datasets()
    elif args.dataset_type == "urban":
        success = downloader.download_urban_sounds()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
