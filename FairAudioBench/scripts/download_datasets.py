#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Set cache directories relative to project or user home
PROJECT_ROOT = Path(__file__).parent
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
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Create HF cache directory in our data folder
HF_CACHE_DIR = DATA_DIR / "hf_cache"
HF_CACHE_DIR.mkdir(exist_ok=True)

# Available datasets and configurations
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

def download_commonvoice_hf(lang_code: str, max_samples: int = 2000, hf_token: Optional[str] = None) -> bool:
    """Download CommonVoice dataset from Hugging Face - simplified version."""
    print(f"Downloading CommonVoice {COMMONVOICE_LANGUAGES.get(lang_code, lang_code)} ({lang_code}) from Hugging Face...")
    
    if lang_code not in COMMONVOICE_LANGUAGES:
        print(f"ERROR: Language code '{lang_code}' not supported")
        print(f"Available languages: {', '.join(list(COMMONVOICE_LANGUAGES.keys())[:20])}... (and {len(COMMONVOICE_LANGUAGES)-20} more)")
        return False
    
    output_dir = DATA_DIR / f"commonvoice_{lang_code}"
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
        
        # Use your suggested approach - direct load_dataset call with explicit cache_dir
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
            
            # Create README
            readme_path = output_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write(f"# CommonVoice Dataset - {COMMONVOICE_LANGUAGES[lang_code]} ({lang_code})\n\n")
                f.write(f"- **Source**: Mozilla Common Voice 17.0\n")
                f.write(f"- **Language**: {COMMONVOICE_LANGUAGES[lang_code]} ({lang_code})\n")
                f.write(f"- **Samples**: {len(audio_files)}\n")
                f.write(f"- **Tonal Language**: {'Yes' if lang_code in TONAL_LANGUAGES else 'No'}\n")
                f.write(f"- **Downloaded**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Files\n")
                f.write("- `metadata.csv`: Audio file paths and transcriptions\n")
                f.write("- `cv17_*.wav`: Audio files (16kHz WAV format)\n")
            
            print(f"SUCCESS: CommonVoice {lang_code}: {len(audio_files)} samples saved")
            return True
        else:
            print(f"ERROR: No valid samples processed for {lang_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to download CommonVoice {lang_code}: {str(e)}")
        return False

def download_gtzan(hf_token: Optional[str] = None):
    """Download GTZAN dataset from Hugging Face."""
    print("Downloading GTZAN music dataset from Hugging Face...")
    
    output_dir = DATA_DIR / "gtzan"
    output_dir.mkdir(exist_ok=True)
    
    # Setup authentication if token provided
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    # Check authentication first
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"  Authenticated as: {user_info.get('name', 'Unknown')}")
    except Exception:
        print("  ERROR: Not authenticated with Hugging Face")
        print("  Please run: huggingface-cli login")
        return False
    
    try:
        print("  Loading GTZAN dataset...")
        # Use your specified approach with explicit cache_dir
        ds = load_dataset("confit/gtzan-parquet", cache_dir=str(HF_CACHE_DIR))
        
        # Get the train split (or first available split)
        if "train" in ds:
            dataset = ds["train"]
        else:
            # Use first available split
            split_name = list(ds.keys())[0]
            dataset = ds[split_name]
            print(f"  Using split: {split_name}")
        
        audio_files = []
        labels = []
        genres = []
        
        total_samples = len(dataset)
        samples_to_process = total_samples  # Process all samples
        
        print(f"  Dataset loaded: {total_samples} total samples, processing {samples_to_process}")
        
        for i in range(samples_to_process):
            try:
                sample = dataset[i]
                
                # Extract audio data
                audio_data = sample["audio"]
                audio_array = audio_data["array"]
                sample_rate = audio_data["sampling_rate"]
                
                # Save audio file
                audio_path = output_dir / f"gtzan_{i:04d}.wav"
                sf.write(str(audio_path), audio_array, sample_rate)
                
                # Extract metadata
                audio_files.append(str(audio_path))
                labels.append(sample.get("label", sample.get("class", "unknown")))
                genres.append(sample.get("genre", sample.get("class", "unknown")))
                
                if (i + 1) % 100 == 0:
                    print(f"    Processed {i + 1}/{samples_to_process} samples...")
                    
            except Exception as e:
                print(f"    Warning: Failed to process sample {i}: {str(e)[:50]}...")
                continue
        
        if audio_files:
            # Save metadata
            metadata = pd.DataFrame({
                "audio_path": audio_files,
                "label": labels,
                "genre": genres,
                "source": "gtzan",
                "dataset": "confit/gtzan-parquet"
            })
            metadata.to_csv(output_dir / "metadata.csv", index=False)
            
            # Create README
            readme_path = output_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write("# GTZAN Music Genre Dataset\n\n")
                f.write("- **Source**: GTZAN Music Genre Classification Dataset\n")
                f.write("- **Repository**: confit/gtzan-parquet\n")
                f.write(f"- **Samples**: {len(audio_files)}\n")
                f.write(f"- **Genres**: {len(set(genres))}\n")
                f.write(f"- **Downloaded**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Files\n")
                f.write("- `metadata.csv`: Audio file paths and genre labels\n")
                f.write("- `gtzan_*.wav`: Audio files\n\n")
                f.write("## Genre Distribution\n")
                genre_counts = pd.Series(genres).value_counts()
                for genre, count in genre_counts.items():
                    f.write(f"- {genre}: {count} samples\n")
            
            print(f"SUCCESS: GTZAN: {len(audio_files)} samples saved")
            return True
        else:
            print("ERROR: No GTZAN samples downloaded")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to download GTZAN: {e}")
        download_gtzan_manual()
        return False

def download_gtzan_manual():
    """Fallback manual download for GTZAN."""
    url = "https://huggingface.co/datasets/confit/gtzan-parquet"
    print(f"MANUAL: Download GTZAN music dataset from: {url}")
    print(f"        Place extracted files in: {DATA_DIR}/gtzan/")

def download_fma():
    """Download FMA dataset from direct download link."""
    print("Downloading FMA Small dataset from official source...")
    
    output_dir = DATA_DIR / "fma_small"
    output_dir.mkdir(exist_ok=True)
    
    # Direct download URL
    fma_url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    
    try:
        print("  Downloading FMA Small archive...")
        
        # Download the zip file
        zip_path = output_dir / "fma_small.zip"
        
        print(f"  Fetching from: {fma_url}")
        with requests.get(fma_url, stream=True) as response:
            response.raise_for_status()
            
            # Get file size for progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Simple progress indicator
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024 * 50) == 0:  # Every 50MB
                            print(f"    Downloaded: {downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB ({progress:.1f}%)")
        
        print(f"  Archive downloaded: {zip_path}")
        
        # Extract the zip file
        print("  Extracting FMA Small archive...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print("  Archive extracted successfully")
        
        # Optional: Remove the zip file to save space
        # zip_path.unlink()
        
        # Create README
        readme_path = output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write("# Free Music Archive (FMA) Small Dataset\n\n")
            f.write("- **Source**: Free Music Archive\n")
            f.write("- **Dataset**: FMA Small (8,000 tracks, 30-second clips)\n")
            f.write("- **URL**: https://os.unil.cloud.switch.ch/fma/fma_small.zip\n")
            f.write("- **Genres**: 8 genres\n")
            f.write(f"- **Downloaded**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Dataset Structure\n")
            f.write("- Audio files are in nested directories by track ID\n")
            f.write("- Metadata is provided in CSV format\n")
            f.write("- All audio files are 30-second clips in MP3 format\n\n")
            f.write("## Usage\n")
            f.write("- Use the provided metadata files to map track IDs to genre labels\n")
            f.write("- Convert MP3 files to WAV format if needed for your analysis\n")
        
        print(f"SUCCESS: FMA Small dataset downloaded to: {output_dir}")
        
        # List contents for verification
        extracted_files = list(output_dir.rglob("*"))
        audio_files = [f for f in extracted_files if f.suffix.lower() in ['.mp3', '.wav']]
        csv_files = [f for f in extracted_files if f.suffix.lower() == '.csv']
        
        print(f"  Extracted files: {len(extracted_files)} total")
        print(f"  Audio files: {len(audio_files)}")
        print(f"  Metadata files: {len(csv_files)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to download FMA dataset: {e}")
        download_fma_manual()
        return False

def download_fma_manual():
    """Fallback manual download for FMA."""
    print("MANUAL: Download FMA Small dataset from official source:")
    print("  1. Visit: https://github.com/mdeff/fma")
    print("  2. Or download directly: https://os.unil.cloud.switch.ch/fma/fma_small.zip")
    print(f"  3. Extract to: {DATA_DIR}/fma_small/")
    print("  4. The dataset contains 8,000 tracks of 30s, 8 balanced genres (GTZAN-like)")
    print("  5. Metadata files are included for genre classification")

def download_turkish_makam():
    """Download Turkish Makam dataset from Zenodo direct link."""
    print("Downloading Turkish Sarki Vocal v2.0 dataset from Zenodo...")
    
    output_dir = DATA_DIR / "turkish_makam"
    output_dir.mkdir(exist_ok=True)
    
    # Direct download URL for Turkish Sarki vocal dataset v2.0
    download_url = "https://zenodo.org/records/1283350/files/turkish_sarki_vocal_v2.0.zip?download=1"
    
    try:
        print("  Downloading Turkish Sarki vocal dataset...")
        
        # Download the zip file
        zip_path = output_dir / "turkish_sarki_vocal_v2.0.zip"
        
        print(f"  Fetching from: {download_url}")
        with requests.get(download_url, stream=True) as response:
            response.raise_for_status()
            
            # Get file size for progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Simple progress indicator
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024 * 10) == 0:  # Every 10MB
                            print(f"    Downloaded: {downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB ({progress:.1f}%)")
        
        print(f"  Archive downloaded: {zip_path}")
        
        # Extract the zip file
        print("  Extracting Turkish Sarki vocal archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print("  Archive extracted successfully")
        
        # Create README
        readme_path = output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write("# Turkish Sarki Vocal Dataset v2.0\n\n")
            f.write("- **Source**: Turkish Music Information Retrieval Research\n")
            f.write("- **Dataset**: Turkish Sarki Vocal v2.0\n")
            f.write("- **URL**: https://zenodo.org/records/1283350\n")
            f.write("- **Description**: Turkish classical music (Sarki) vocal recordings\n")
            f.write("- **Format**: Audio files with makam (modal) annotations\n")
            f.write(f"- **Downloaded**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Dataset Structure\n")
            f.write("- Audio files in various formats (WAV, MP3)\n")
            f.write("- Makam (mode) annotations and metadata\n")
            f.write("- Musical analysis data for Turkish classical music\n\n")
            f.write("## Usage\n")
            f.write("- Use for Turkish makam music analysis and classification\n")
            f.write("- Contains makam-based musical content for cultural bias studies\n")
            f.write("- Suitable for cross-cultural music analysis research\n")
        
        print(f"SUCCESS: Turkish Makam dataset downloaded to: {output_dir}")
        
        # List contents for verification
        extracted_files = list(output_dir.rglob("*"))
        audio_files = [f for f in extracted_files if f.suffix.lower() in ['.wav', '.mp3', '.flac']]
        metadata_files = [f for f in extracted_files if f.suffix.lower() in ['.csv', '.json', '.txt']]
        
        print(f"  Extracted files: {len(extracted_files)} total")
        print(f"  Audio files: {len(audio_files)}")
        print(f"  Metadata files: {len(metadata_files)}")
        
        # Optional: Remove the zip file to save space
        # zip_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to download Turkish Makam dataset: {e}")
        print("   Falling back to manual download...")
        download_turkish_makam_manual()
        return False

def download_turkish_makam_manual():
    """Fallback manual download for Turkish Makam."""
    url = "https://zenodo.org/records/1287656"
    print(f"[Manual] Download Turkish Makam dataset from: {url}")
    print(f"Place extracted files in: {DATA_DIR}/turkish_makam/")

def download_hindustani():
    """Download Hindustani/Indian Raga dataset from Kaggle."""
    print("Downloading Hindustani Indian Raga dataset from Kaggle...")
    
    output_dir = DATA_DIR / "hindustani"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Try to use Kaggle API
        import kaggle
        
        print("  Downloading dataset using Kaggle API...")
        kaggle.api.dataset_download_files(
            'kcwaghmarewaghmare/indian-music-raga', 
            path=str(output_dir),
            unzip=True
        )
        
        print(f"SUCCESS: Hindustani dataset downloaded to: {output_dir}")
        return True
        
    except ImportError:
        print("ERROR: Kaggle API not installed. Install with: pip install kaggle")
        print("   Falling back to manual download...")
        download_hindustani_manual()
        return False
    except Exception as e:
        print(f"ERROR: Failed to download Hindustani dataset: {e}")
        print("   This might be due to:")
        print("   1. Kaggle API not configured - run: kaggle --version")
        print("   2. Need to accept dataset terms on Kaggle website")
        print("   3. Need API credentials in ~/.kaggle/kaggle.json")
        print("   Falling back to manual download...")
        download_hindustani_manual()
        return False

def download_hindustani_manual():
    """Fallback manual download for Hindustani."""
    url = "https://www.kaggle.com/datasets/kcwaghmarewaghmare/indian-music-raga"
    print(f"[Manual] Download Hindustani Indian Raga dataset from: {url}")
    print(f"Steps:")
    print(f"1. Create Kaggle account and accept dataset terms")
    print(f"2. Install Kaggle API: pip install kaggle")
    print(f"3. Setup credentials: https://www.kaggle.com/docs/api")
    print(f"4. Download with: kaggle datasets download -d kcwaghmarewaghmare/indian-music-raga")
    print(f"5. Extract to: {DATA_DIR}/hindustani/")

def download_carnatic():
    """Download Carnatic dataset from Kaggle."""
    print("Downloading Saraga Carnatic music dataset from Kaggle...")
    
    output_dir = DATA_DIR / "carnatic"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Try to use Kaggle API
        import kaggle
        
        print("  Downloading dataset using Kaggle API...")
        kaggle.api.dataset_download_files(
            'desolationofsmaug/saraga-carnatic-music-dataset', 
            path=str(output_dir),
            unzip=True
        )
        
        # Create README
        readme_path = output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write("# Saraga Carnatic Music Dataset\n\n")
            f.write("- **Source**: Saraga Research Collection\n")
            f.write("- **Dataset**: Carnatic music recordings\n")
            f.write("- **URL**: https://www.kaggle.com/datasets/desolationofsmaug/saraga-carnatic-music-dataset\n")
            f.write("- **Description**: Collection of Carnatic classical music recordings\n")
            f.write(f"- **Downloaded**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Dataset Structure\n")
            f.write("- Audio files in various formats (WAV, MP3)\n")
            f.write("- Metadata files with raga, tala, and artist information\n")
            f.write("- Annotations for musical analysis\n\n")
            f.write("## Usage\n")
            f.write("- Use for Carnatic music analysis and classification\n")
            f.write("- Contains raga-based musical content for cultural bias studies\n")
        
        print(f"SUCCESS: Carnatic dataset downloaded to: {output_dir}")
        
        # List contents for verification
        extracted_files = list(output_dir.rglob("*"))
        audio_files = [f for f in extracted_files if f.suffix.lower() in ['.wav', '.mp3', '.flac']]
        metadata_files = [f for f in extracted_files if f.suffix.lower() in ['.csv', '.json', '.txt']]
        
        print(f"  Extracted files: {len(extracted_files)} total")
        print(f"  Audio files: {len(audio_files)}")
        print(f"  Metadata files: {len(metadata_files)}")
        
        return True
        
    except ImportError:
        print("ERROR: Kaggle API not installed. Install with: pip install kaggle")
        print("   Falling back to manual download...")
        download_carnatic_manual()
        return False
    except Exception as e:
        print(f"ERROR: Failed to download Carnatic dataset: {e}")
        print("   This might be due to:")
        print("   1. Kaggle API not configured - run: kaggle --version")
        print("   2. Need to accept dataset terms on Kaggle website")
        print("   3. Need API credentials in ~/.kaggle/kaggle.json")
        print("   Falling back to manual download...")
        download_carnatic_manual()
        return False

def download_carnatic_manual():
    """Fallback manual download for Carnatic."""
    output_dir = DATA_DIR / "carnatic"
    output_dir.mkdir(exist_ok=True)
    
    # Create manual download instructions
    readme_path = output_dir / "MANUAL_DOWNLOAD.md"
    with open(readme_path, "w") as f:
        f.write("# Manual Download: Saraga Carnatic Music Dataset\n\n")
        f.write("## Kaggle Download (Recommended)\n")
        f.write("1. Visit: https://www.kaggle.com/datasets/desolationofsmaug/saraga-carnatic-music-dataset\n")
        f.write("2. Click 'Download' to get the dataset\n")
        f.write("3. Extract the downloaded ZIP file to this directory\n\n")
        f.write("## Kaggle API Setup\n")
        f.write("1. Install Kaggle API: pip install kaggle\n")
        f.write("2. Get API credentials from Kaggle Account settings\n")
        f.write("3. Place kaggle.json in ~/.kaggle/\n")
        f.write("4. Run: kaggle datasets download -d desolationofsmaug/saraga-carnatic-music-dataset\n\n")
        f.write("## Alternative Sources\n")
        f.write("- CompMusic project: https://compmusic.upf.edu/\n")
        f.write("- Saraga collection: http://saraga.upf.edu/\n")
        f.write("- Original Zenodo: https://zenodo.org/record/4301737\n\n")
        f.write(f"Target directory: {output_dir}\n")
    
    print(f"Manual download instructions created: {readme_path}")
    print(f"[Manual] Download Carnatic dataset from: https://www.kaggle.com/datasets/desolationofsmaug/saraga-carnatic-music-dataset")
    print(f"Place extracted files in: {DATA_DIR}/carnatic/")

def download_arab_andalusian():
    """Download Arab Andalusian dataset from Zenodo."""
    print("Downloading Arab Andalusian dataset from Zenodo...")
    
    output_dir = DATA_DIR / "arab_andalusian"
    output_dir.mkdir(exist_ok=True)
    
    # Zenodo record URL and API
    zenodo_record_id = "1291776"
    zenodo_api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"
    
    try:
        print("  Fetching dataset info from Zenodo...")
        response = requests.get(zenodo_api_url)
        response.raise_for_status()
        
        data = response.json()
        files = data.get("files", [])
        
        if not files:
            print("ERROR: No files found in Zenodo record")
            return False
            
        # Download all relevant files
        downloaded_files = 0
        for file_info in files:
            filename = file_info["key"]
            file_url = file_info["links"]["self"]
            file_size_mb = file_info["size"] / (1024 * 1024)
                
            print(f"  Downloading: {filename} ({file_size_mb:.1f} MB)")
            
            file_response = requests.get(file_url, stream=True)
            file_response.raise_for_status()
            
            file_path = output_dir / filename
            
            # Show progress for large files
            downloaded = 0
            with open(file_path, "wb") as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress indicator for large files
                    if file_size_mb > 100 and downloaded % (1024 * 1024 * 50) == 0:  # Every 50MB
                        progress = (downloaded / (file_size_mb * 1024 * 1024)) * 100
                        print(f"    Progress: {downloaded // (1024*1024)} MB / {file_size_mb:.1f} MB ({progress:.1f}%)")
            
            # Extract if it's an archive
            if filename.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                print(f"    Extracted {filename}")
                
            elif filename.endswith('.tar.gz'):
                import tarfile
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(output_dir)
                print(f"    Extracted {filename}")
            else:
                print(f"    Downloaded: {filename}")
            
            downloaded_files += 1
        
        # Create README
        readme_path = output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write("# Arab Andalusian Music Dataset\n\n")
            f.write("- **Source**: Arab Andalusian Music Collection\n")
            f.write("- **URL**: https://zenodo.org/records/1291776\n")
            f.write("- **Description**: Collection of Arab Andalusian classical music recordings\n")
            f.write("- **Format**: Audio files with musical annotations\n")
            f.write(f"- **Downloaded**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Dataset Structure\n")
            f.write("- Audio files in various formats (WAV, MP3)\n")
            f.write("- Musical annotations and metadata\n")
            f.write("- Analysis data for Arab Andalusian classical music\n\n")
            f.write("## Usage\n")
            f.write("- Use for Arab Andalusian music analysis and classification\n")
            f.write("- Contains modal-based musical content for cultural bias studies\n")
            f.write("- Suitable for cross-cultural music analysis research\n")
        
        print(f"SUCCESS: Arab Andalusian dataset downloaded to: {output_dir}")
        
        # List contents for verification
        extracted_files = list(output_dir.rglob("*"))
        audio_files = [f for f in extracted_files if f.suffix.lower() in ['.wav', '.mp3', '.flac']]
        metadata_files = [f for f in extracted_files if f.suffix.lower() in ['.csv', '.json', '.txt']]
        
        print(f"  Extracted files: {len(extracted_files)} total")
        print(f"  Audio files: {len(audio_files)}")
        print(f"  Metadata files: {len(metadata_files)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to download Arab Andalusian dataset: {e}")
        print("   Falling back to manual download...")
        download_arab_andalusian_manual()
        return False

def download_arab_andalusian_manual():
    """Fallback manual download for Arab Andalusian."""
    url = "https://zenodo.org/records/1291776"
    print(f"[Manual] Download Arab Andalusian dataset from: {url}")
    print(f"Place extracted files in: {DATA_DIR}/arab_andalusian/")

def download_tau_urban_auto():
    """Automatically download TAU Urban Acoustic Scenes 2020 from Zenodo."""
    print("Downloading TAU Urban Acoustic Scenes 2020 from Zenodo...")
    
    output_dir = DATA_DIR / "tau_urban_2020"
    output_dir.mkdir(exist_ok=True)
    
    # Zenodo record URL and file info
    zenodo_record_id = "3685828"
    zenodo_api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"
    
    try:
        # Get file metadata from Zenodo API
        response = requests.get(zenodo_api_url)
        response.raise_for_status()
        record_data = response.json()
        
        # Find the main dataset files
        files_to_download = []
        for file_info in record_data['files']:
            filename = file_info['key']
            # Download development and evaluation sets
            if any(keyword in filename.lower() for keyword in ['development', 'evaluation', 'dev', 'eval']):
                if filename.endswith(('.zip', '.tar.gz')):
                    files_to_download.append(file_info)
        
        if not files_to_download:
            # Fallback: download first few zip/tar files
            files_to_download = [f for f in record_data['files'] if f['key'].endswith(('.zip', '.tar.gz'))][:2]
        
        print(f"  Found {len(files_to_download)} files to download")
        
        for file_info in files_to_download:
            filename = file_info['key']
            file_url = file_info['links']['self']
            file_size_mb = file_info['size'] / (1024 * 1024)
            
            print(f"  Downloading {filename} ({file_size_mb:.1f} MB)...")
            
            # Download file
            file_path = output_dir / filename
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Extract if it's an archive
            if filename.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                print(f"    SUCCESS: Extracted {filename}")
                # Optionally remove the zip file to save space
                # file_path.unlink()
                
            elif filename.endswith('.tar.gz'):
                import tarfile
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(output_dir)
                print(f"    SUCCESS: Extracted {filename}")
                # Optionally remove the tar file to save space
                # file_path.unlink()
        
        print(f"SUCCESS: TAU Urban Acoustic Scenes: Downloaded to {output_dir}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to auto-download TAU Urban: {e}")
        print("   Falling back to manual download instructions...")
        download_tau_urban_manual()
        return False

def download_tau_urban_manual():
    """Fallback manual download instructions for TAU Urban."""
    url = "https://zenodo.org/records/3685828"
    print(f"[Manual] Download TAU Urban Acoustic Scenes 2020 from: {url}")
    print(f"Place extracted files in: {DATA_DIR}/tau_urban_2020/")

def download_tau_urban():
    """Download TAU Urban dataset (try automatic first, fall back to manual)."""
    return download_tau_urban_auto()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download datasets for Cross-Cultural Mel-Scale Audio Frontend Bias Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_datasets.py --all --hf_token YOUR_TOKEN
    python download_datasets.py --dataset commonvoice --lang english
    python download_datasets.py --dataset gtzan
    python download_datasets.py --list
        """
    )
    
    parser.add_argument(
        "--dataset", 
        choices=ALL_DATASETS,
        help="Specific dataset to download"
    )
    
    parser.add_argument(
        "--lang", "--language",
        help="Language for CommonVoice dataset (e.g., en, hi, de, fr, es, zh-CN, vi, th) or 'all_target' for all tonal+non-tonal languages"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets"
    )
    
    parser.add_argument(
        "--hf_token", "--hf-token",
        help="Hugging Face token for authentication"
    )
    
    parser.add_argument(
        "--max_samples", "--max-samples",
        type=int,
        default=2000,
        help="Maximum samples to download per language (default: 2000 for balanced evaluation)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and languages"
    )
    
    parser.add_argument(
        "--output_dir", "--output-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Output directory for datasets (default: {DATA_DIR})"
    )
    
    return parser.parse_args()

def list_available_datasets():
    """List all available datasets and their details."""
    print("\n=== AVAILABLE DATASETS ===\n")
    
    print("SPEECH DATASETS:")
    print("  commonvoice - Mozilla Common Voice multilingual speech")
    print("    Target tonal languages:", ", ".join(TONAL_LANGUAGES))
    print("    Target non-tonal languages:", ", ".join(NON_TONAL_LANGUAGES))
    print(f"    All available languages: {len(COMMONVOICE_LANGUAGES)} total")
    print("    Use language codes (e.g., 'en' for English, 'hi' for Hindi)")
    print("    Or use 'all_target' to download all tonal + non-tonal target languages")
    print()
    
    print("MUSIC DATASETS:")
    for dataset in MUSIC_DATASETS:
        descriptions = {
            "gtzan": "GTZAN Genre Classification Dataset - 1000 audio tracks, 10 genres",
            "fma": "Free Music Archive - Large collection of music tracks",
            "carnatic": "Carnatic music dataset from CompMusic project",
            "turkish_makam": "Turkish Makam music dataset",
            "hindustani": "Hindustani classical music dataset",
            "arab_andalusian": "Arab Andalusian classical music dataset"
        }
        print(f"  {dataset} - {descriptions.get(dataset, 'Music dataset')}")
    print()
    
    print("SCENE DATASETS:")
    print("  tau_urban - TAU Urban Acoustic Scenes 2020 dataset")
    print()
    
    print("EXAMPLE COMMANDS:")
    print("  python download_datasets.py --list")
    print("  python download_datasets.py --all --hf_token YOUR_TOKEN")
    print("  python download_datasets.py --dataset commonvoice --lang en --hf_token YOUR_TOKEN")
    print("  python download_datasets.py --dataset commonvoice --lang all_target --hf_token YOUR_TOKEN")
    print("  python download_datasets.py --dataset gtzan")
    print()

def download_dataset(dataset_name: str, **kwargs) -> bool:
    """Download a specific dataset based on name."""
    success = False
    
    if dataset_name == "commonvoice":
        lang = kwargs.get("lang")
        if not lang:
            print("ERROR: --lang required for CommonVoice dataset")
            print(f"Available languages: {', '.join(list(COMMONVOICE_LANGUAGES.keys())[:20])}... (and {len(COMMONVOICE_LANGUAGES)-20} more)")
            print("Use language codes like: en, hi, de, fr, es, zh-CN, vi, th")
            print("Or use 'all_target' to download all tonal and non-tonal target languages")
            return False
        
        # Handle batch download for all target languages
        if lang == "all_target":
            all_target_languages = TONAL_LANGUAGES + NON_TONAL_LANGUAGES
            print(f"Downloading CommonVoice for all target languages ({len(all_target_languages)} languages)")
            success_count = 0
            failed_languages = []
            
            for lang_code in all_target_languages:
                print(f"\n--- Downloading CommonVoice for language: {lang_code} ---")
                try:
                    lang_success = download_commonvoice_hf(
                        lang_code,
                        max_samples=kwargs.get("max_samples", 2000),
                        hf_token=kwargs.get("hf_token")
                    )
                    if lang_success:
                        success_count += 1
                    else:
                        failed_languages.append(lang_code)
                except Exception as e:
                    print(f"Failed to download {lang_code}: {str(e)}")
                    failed_languages.append(lang_code)
            
            print(f"\n=== BATCH DOWNLOAD SUMMARY ===")
            print(f"Successfully downloaded: {success_count}/{len(all_target_languages)} languages")
            if failed_languages:
                print(f"Failed languages: {', '.join(failed_languages)}")
            
            return success_count > 0
        
        # Handle single language download
        if lang not in COMMONVOICE_LANGUAGES:
            print(f"ERROR: Unsupported language '{lang}'")
            print(f"Available languages: {', '.join(list(COMMONVOICE_LANGUAGES.keys())[:20])}... (and {len(COMMONVOICE_LANGUAGES)-20} more)")
            print("Use language codes like: en, hi, de, fr, es, zh-CN, vi, th")
            print("Or use 'all_target' to download all tonal and non-tonal target languages")
            return False
        
        success = download_commonvoice_hf(
            lang, 
            max_samples=kwargs.get("max_samples", 2000),
            hf_token=kwargs.get("hf_token")
        )
    
    elif dataset_name == "gtzan":
        success = download_gtzan(hf_token=kwargs.get("hf_token"))
    elif dataset_name == "fma":
        success = download_fma()
    elif dataset_name == "carnatic":
        success = download_carnatic()
    elif dataset_name == "turkish_makam":
        success = download_turkish_makam()
    elif dataset_name == "hindustani":
        success = download_hindustani()
    elif dataset_name == "arab_andalusian":
        success = download_arab_andalusian()
    elif dataset_name == "tau_urban":
        success = download_tau_urban_auto()
        if not success:
            download_tau_urban_manual()
    else:
        print(f"ERROR: Unknown dataset '{dataset_name}'")
        return False
    
    return success

def test_download_function():
    """Test the download functionality with a small sample."""
    print("Testing download functionality...")
    
    # Test with English as it's most likely to be available
    test_lang = "english"
    test_samples = 5
    
    print(f"Testing CommonVoice download for {test_lang} with {test_samples} samples...")
    
    try:
        success = download_commonvoice_hf(test_lang, test_samples)
        if success:
            print("✓ Test passed: CommonVoice download working")
            
            # Check if files were created
            test_dir = DATA_DIR / f"commonvoice_{test_lang}"
            if test_dir.exists():
                audio_files = list(test_dir.glob("*.wav"))
                metadata_file = test_dir / "metadata.csv"
                
                print(f"  - Audio files created: {len(audio_files)}")
                print(f"  - Metadata file exists: {metadata_file.exists()}")
                
                if metadata_file.exists():
                    import pandas as pd
                    df = pd.read_csv(metadata_file)
                    print(f"  - Metadata rows: {len(df)}")
                    print(f"  - Sample text: {df['text'].iloc[0] if len(df) > 0 else 'None'}")
                
                return True
            else:
                print("✗ Test failed: No output directory created")
                return False
        else:
            print("✗ Test failed: Download function returned False")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False

def main():
    """Main entry point with CLI argument parsing."""
    args = parse_arguments()
    
    # Set global DATA_DIR if specified
    global DATA_DIR
    DATA_DIR = args.output_dir
    DATA_DIR.mkdir(exist_ok=True)
    
    print(f"\nCross-Cultural Mel-Scale Audio Frontend Bias - Dataset Downloader")
    print(f"Target directory: {DATA_DIR}")
    
    # List datasets if requested
    if args.list:
        list_available_datasets()
        return
    
    # Validate arguments
    if not args.all and not args.dataset:
        print("ERROR: Must specify either --dataset DATASET or --all")
        print("Run with --list to see available datasets")
        return
    
    if args.dataset == "commonvoice" and not args.lang:
        print("ERROR: CommonVoice requires --lang parameter")
        print("Run with --list to see available languages")
        return
    
    # Setup authentication if needed
    auth_success = True
    if args.all or args.dataset == "commonvoice":
        auth_success = setup_huggingface_auth(args.hf_token)
        if not auth_success:
            print("WARNING: Authentication failed. Some downloads may not work.")
    
    print("\n" + "="*50)
    
    # Download datasets
    if args.all:
        print("DOWNLOADING ALL DATASETS...")
        
        # Download CommonVoice for key languages (subset for testing)
        print("\nSPEECH DATASETS (CommonVoice):")
        # Use a subset for --all to avoid downloading 200+ languages
        key_languages = TONAL_LANGUAGES + NON_TONAL_LANGUAGES
        all_languages = key_languages
        successful_languages = 0
        
        for lang in all_languages:
            print(f"\n  Downloading CommonVoice {lang}...")
            success = download_commonvoice_hf(lang, args.max_samples, args.hf_token)
            if success:
                successful_languages += 1
        
        print(f"\nCommonVoice Summary: {successful_languages}/{len(all_languages)} languages downloaded")
        
        # Download all music datasets
        print("\nMUSIC DATASETS:")
        music_successful = 0
        for dataset in MUSIC_DATASETS:
            print(f"\n  Downloading {dataset}...")
            success = download_dataset(dataset, hf_token=args.hf_token)
            if success:
                music_successful += 1
        
        print(f"\nMusic Summary: {music_successful}/{len(MUSIC_DATASETS)} datasets downloaded")
        
        # Download scene datasets
        print("\nSCENE DATASETS:")
        print("\n  Downloading TAU Urban...")
        download_dataset("tau_urban")
    
    else:
        # Download specific dataset
        print(f"DOWNLOADING {args.dataset.upper()}...")
        success = download_dataset(
            args.dataset,
            lang=args.lang,
            max_samples=args.max_samples,
            hf_token=args.hf_token
        )
        
        if success:
            print(f"\nSUCCESS: {args.dataset} downloaded successfully")
        else:
            print(f"\nFAILED: Could not download {args.dataset}")
    
    print("\n" + "="*50)
    print("Download process completed.")
    print("Next steps:")
    print("1. Verify downloaded data in:", DATA_DIR)
    print("2. Run validation: python validate.py")
    print("3. Start experiments: python experiments/reproduce_paper.py")

if __name__ == "__main__":
    main()
