"""
Dataset Preparation Script for Jinsaryko/Elise
Downloads and prepares the dataset for CSM LoRA training
"""

import os
import json
import logging
from datasets import load_dataset, Audio
from tqdm import tqdm
import soundfile as sf

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
DATASET_NAME = "Jinsaryko/Elise"
OUTPUT_DIR = "audio_data"
CACHE_DIR = "./hf_cache"  # Hugging Face cache directory
METADATA_FILE = "dataset_metadata.json"


def download_and_prepare_dataset():
    """
    Download Elise dataset from Hugging Face and prepare it for training
    """
    logger.info(f"Loading dataset: {DATASET_NAME}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    try:
        # Load dataset from Hugging Face
        # The dataset has audio samples with text transcriptions
        logger.info("Downloading dataset from Hugging Face Hub...")
        dataset = load_dataset(DATASET_NAME, split="train", cache_dir=CACHE_DIR)

        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Total samples: {len(dataset)}")
        logger.info(f"Dataset features: {dataset.features}")

        # Metadata storage
        metadata = {
            "total_samples": len(dataset),
            "dataset_name": DATASET_NAME,
            "speaker": "Ceylia",  # Single speaker from dataset info
            "samples": [],
        }

        # Process each sample
        logger.info("Processing audio samples...")
        successful = 0
        failed = 0

        for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            try:
                # Extract audio and metadata
                audio_data = sample["audio"]
                text = sample["text"]

                # Audio data structure: {'path': str, 'array': np.array, 'sampling_rate': int}
                audio_array = audio_data["array"]
                sampling_rate = audio_data["sampling_rate"]

                # Save audio file as WAV
                output_filename = f"elise_{idx:04d}.wav"
                output_path = os.path.join(OUTPUT_DIR, output_filename)

                # Write audio file
                sf.write(output_path, audio_array, sampling_rate)

                # Store metadata
                sample_meta = {
                    "audio_file": output_filename,
                    "text": text,
                    "sample_rate": sampling_rate,
                    "duration": len(audio_array) / sampling_rate,
                    "speaker_id": 0,  # Single speaker, ID 0
                }

                # Include acoustic metrics if available
                if "utterance_pitch_mean" in sample:
                    sample_meta["pitch_mean"] = float(sample["utterance_pitch_mean"])
                if "snr" in sample:
                    sample_meta["snr"] = float(sample["snr"])
                if "speaking_rate" in sample:
                    sample_meta["speaking_rate"] = sample["speaking_rate"]

                metadata["samples"].append(sample_meta)
                successful += 1

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                failed += 1
                continue

        # Save metadata
        metadata_path = os.path.join(OUTPUT_DIR, METADATA_FILE)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset preparation complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Successful: {successful} samples")
        logger.info(f"Failed: {failed} samples")
        logger.info(f"Audio files saved to: {OUTPUT_DIR}")
        logger.info(f"Metadata saved to: {metadata_path}")

        # Print dataset statistics
        logger.info(f"\nDataset Statistics:")
        logger.info(
            f"  Total duration: {sum(s['duration'] for s in metadata['samples']):.2f} seconds"
        )
        logger.info(
            f"  Average duration: {sum(s['duration'] for s in metadata['samples']) / len(metadata['samples']):.2f} seconds"
        )
        logger.info(f"  Total audio files: {len(metadata['samples'])}")

        # Print sample
        if metadata["samples"]:
            logger.info(f"\nSample entry:")
            logger.info(f"  File: {metadata['samples'][0]['audio_file']}")
            logger.info(f"  Text: {metadata['samples'][0]['text']}")
            logger.info(f"  Duration: {metadata['samples'][0]['duration']:.2f}s")

        return True

    except Exception as e:
        logger.error(f"Fatal error during dataset preparation: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def verify_dataset():
    """
    Verify the prepared dataset is ready for training
    """
    logger.info("\nVerifying dataset...")

    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        logger.error(f"Output directory not found: {OUTPUT_DIR}")
        return False

    # Check if metadata exists
    metadata_path = os.path.join(OUTPUT_DIR, METADATA_FILE)
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return False

    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Verify audio files exist
    missing_files = []
    for sample in metadata["samples"]:
        audio_path = os.path.join(OUTPUT_DIR, sample["audio_file"])
        if not os.path.exists(audio_path):
            missing_files.append(sample["audio_file"])

    if missing_files:
        logger.warning(f"Missing {len(missing_files)} audio files:")
        for f in missing_files[:10]:  # Show first 10
            logger.warning(f"  - {f}")
        if len(missing_files) > 10:
            logger.warning(f"  ... and {len(missing_files) - 10} more")
        return False

    logger.info(f" Dataset verified successfully!")
    logger.info(f" {len(metadata['samples'])} audio files found")
    logger.info(f" All files accounted for")

    return True


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Elise Dataset Preparation for CSM LoRA Training")
    logger.info("=" * 60)

    # Download and prepare
    success = download_and_prepare_dataset()

    if success:
        # Verify
        verify_dataset()

        logger.info("\n" + "=" * 60)
        logger.info("NEXT STEPS:")
        logger.info("=" * 60)
        logger.info("1. Review the prepared dataset in the 'audio_data' directory")
        logger.info("2. Adjust training parameters in lora.py if needed")
        logger.info("3. Run training with: python lora.py")
        logger.info("=" * 60)
    else:
        logger.error("\nDataset preparation failed. Please check the errors above.")
