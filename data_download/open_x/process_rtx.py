"""
RTX Dataset Video Extraction Tool

This script processes robotics datasets from the RTX collection, extracting image sequences
from various camera viewpoints and converting them into MP4 video files. It handles multiple
dataset formats and camera configurations commonly found in robotics datasets.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Suppress TensorFlow verbose logging to reduce console output during processing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(Path(__file__).parents[2].as_posix())

import imageio  # For video file creation and image processing
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds  # For loading and processing TensorFlow datasets
from sklearn.model_selection import train_test_split
from tqdm.auto import trange  # For progress bars during batch processing

from data_download.open_x.constants import (DISPLAY_KEYS,
                                            EPISODE_COUNT_PER_DATASET)
from data_download.utils import split_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process RTX datasets to extract videos"
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=Path(__file__).parents[2] / "data" / "open_x",
        help="Directory to save processed video files",
    )
    parser.add_argument(
        "--rtx_root",
        type=str,
        default=Path(__file__).parents[2] / "data" / "rtx",
        help="Root directory where original RTX datasets are stored",
    )
    parser.add_argument(
        "--download_and_process",
        action="store_true",
        help="If set, download and process datasets; otherwise, only process existing data",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="If set, process only a small sample of the data for testing",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples to process if --sample is set",
    )
    parser.add_argument(
        "--remove_raw_videos",
        action="store_true",
        help="If set, remove raw video files after processing to save space",
    )
    return parser.parse_args()


def get_data_split(n_samples: int) -> pd.DataFrame:

    # Create index for dataset
    indices = None
    categories = None
    for dataset, episode_count in EPISODE_COUNT_PER_DATASET.items():
        dataset_indices = np.arange(episode_count)
        dataset_categories = pd.Series([dataset] * episode_count)
        if indices is None:
            indices = dataset_indices
            categories = dataset_categories
        else:
            indices = np.concatenate((indices, dataset_indices))
            categories = pd.concat((categories, dataset_categories))

    df = pd.DataFrame({"indices": indices, "categories": categories})

    sample_indices, _ = split_data(
        df, train_size=(n_samples / len(df)), strata=["categories"], random_state=42
    )

    return sample_indices


def download_dataset(dataset_name: str, rtx_root: Path) -> None:
    """
    Download the specified RTX dataset using TensorFlow Datasets.

    Args:
        dataset_name (str): Name of the dataset to download
        rtx_root (Path): Root directory where datasets should be stored
    """

    # Check if dataset already exists
    dataset_path = rtx_root / dataset_name
    if dataset_path.exists():
        logger.info(f"  ⏭️  Dataset {dataset_name} already exists, skipping download.")
        return

    try:
        logger.info(f"  ⬇️  Downloading dataset: {dataset_name}")

        logger.info(f"     Running download script for {dataset_name}...")
        logger.info(
            f"bash {Path(__file__).parent}/download_specific_dataset.sh {dataset_name} {rtx_root}"
        )
        ret = os.system(
            f"bash {Path(__file__).parent}/download_specific_dataset.sh {dataset_name} {rtx_root}"
        )

        if ret != 0:
            raise RuntimeError(f"Download script failed with exit code {ret}")
    except Exception as e:
        logger.error(f"  ❌  Failed to download dataset {dataset_name}: {e}")


def dataset2path(rtx_root: Path, dataset_name: str) -> str:
    """
    Construct the path to the latest version of a specified dataset.

    RTX datasets are organized with version directories (e.g., '1.0.0', '2.1.0').
    This function automatically selects the latest version based on directory naming.

    Args:
        dataset_name (str): Name of the dataset to locate

    Returns:
        str: Full path to the latest version of the dataset
    """
    logger.info(f"  📁 Looking for dataset: {dataset_name}")

    # List all version directories for the given dataset
    dataset_path = rtx_root / dataset_name
    versions = os.listdir(dataset_path)
    versions.sort()
    logger.info(f"     Found versions: {versions}")

    # Filter for version directories (typically have 5-character names like '1.0.0')
    versions = [version for version in versions if len(version) == 5]

    # Select the latest version (last in sorted order)
    version = versions[-1]
    final_path = rtx_root / dataset_name / version
    logger.info(f"     Using latest version: {version} -> {final_path}")

    return final_path


def save_images_to_video(images: list, output_file: str, fps: int = 10) -> None:
    """
    Convert a sequence of images into an MP4 video file.

    Args:
        images (list): List of image arrays (numpy arrays or similar)
        output_file (str): Path where the output video file will be saved
        fps (int, optional): Frames per second for the output video. Defaults to 10.
    """
    logger.info(
        f"      🎬 Creating video: {output_file} ({len(images)} frames @ {fps}fps)"
    )

    # Create video writer with specified fps
    writer = imageio.get_writer(output_file, fps=fps)

    # Add each image frame to the video
    for image in images:
        writer.append_data(image)

    # Close the writer to finalize the video file
    writer.close()
    logger.info(f"      ✅ Video saved successfully")


def extract_sample(
    tfds_builder: tfds.core.DatasetBuilder,
    obs_key: str,
    dataset_name: str,
    save_dir: str,
    split: str,
    extra: str = None,
    sample_indices: pd.DataFrame = None,
) -> None:
    """
    Extract image sequences from a dataset and save them as MP4 videos.

    This function processes episodes from robotics datasets, extracting image observations
    from specified camera viewpoints and converting them to video files.

    Args:
        tfds_builder: TensorFlow dataset builder object
        obs_key (str): Key identifying the observation type (e.g., 'image', 'wrist_image')
        dataset_name (str): Name of the dataset being processed
        save_dir (str): Directory where videos will be saved
        split (str): Dataset split to process ('train', 'test', 'validation')
        extra (str, optional): Additional key for nested observation structures
    """
    logger.info(f"    🔄 Processing {split} split for observation key: {obs_key}")
    if extra:
        logger.info(f"       Using nested key: {extra}")

    # Try to load dataset as TensorFlow dataset, fallback to data source if needed
    try:
        ds = tfds_builder.as_dataset(split=split)
        logger.info(f"       ✅ Loaded as TensorFlow dataset")
    except Exception as e:
        # Some datasets require data source loading instead of dataset loading
        logger.warning(
            f"       ⚠️  Failed to load as dataset, trying data source: {str(e)[:100]}..."
        )
        ds = tfds_builder.as_data_source(split=split)
        logger.info(f"       ✅ Loaded as data source")

    ds_iter = iter(ds)
    episodes_processed = 0
    episodes_failed = 0

    # Process each episode in the dataset with progress tracking
    for episode_idx in trange(
        len(ds), desc=f"Extracting {dataset_name.upper()} {split}"
    ):
        try:
            episode = next(ds_iter)

            if sample_indices is not None and episode_idx not in sample_indices:
                continue

            # Handle different dataset structures for image extraction
            if dataset_name == "robot_vqa":
                # Special handling for robot_vqa dataset which has nested image lists
                images = []
                for step in episode["steps"]:
                    images.extend([img for img in step["observation"][obs_key]])
            elif extra is None:
                # Standard case: extract images directly from observation key
                images = [step["observation"][obs_key] for step in episode["steps"]]
            else:
                # Handle nested observation structures with additional key
                images = [
                    step["observation"][obs_key][extra] for step in episode["steps"]
                ]

            # Convert TensorFlow tensors to numpy arrays if needed
            try:
                images = [image.numpy() for image in images]
            except:
                # Images are already in correct format (numpy arrays)
                images = images

            # Create output path with zero-padded episode index
            save_path = save_dir / split / f"{episode_idx:08}.mp4"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert image sequence to video file
            save_images_to_video(images, save_path)
            episodes_processed += 1

        except Exception as e:
            # Skip episodes that fail to process (corrupted data, missing keys, etc.)
            episodes_failed += 1
            if episodes_failed <= 5:  # Only show first few errors to avoid spam
                logger.warning(
                    f"       ⚠️  Episode {episode_idx} failed: {str(e)[:100]}..."
                )
            elif episodes_failed == 6:
                logger.info(f"       ⚠️  Suppressing further error messages...")

    logger.info(
        f"    📊 Split {split} complete: {episodes_processed} processed, {episodes_failed} failed"
    )


def get_compatible_keys(
    builder: tfds.core.DatasetBuilder, infeasible_datasets: list, dataset: str
) -> tuple:
    # Check each possible image key to see if it exists in this dataset
    compatible_keys = []
    for display_key in DISPLAY_KEYS:
        if display_key in builder.info.features["steps"]["observation"]:
            compatible_keys.append(display_key)

        if compatible_keys:
            logger.info(
                f"  🎯 Found {len(compatible_keys)} compatible image keys: {compatible_keys}"
            )
        else:
            logger.warning(f"  ⚠️  No compatible image keys found")
            infeasible_datasets.append(dataset)
            continue

    return compatible_keys, infeasible_datasets


def main():
    """Main function to process all RTX datasets and extract videos from image observations."""

    # Parse command line arguments
    args = parse_args()

    # Track processing statistics
    feasible_datasets = 0  # Count of datasets that have compatible image observations
    infeasible_datasets = []  # List of datasets that lack compatible image keys

    # Comprehensive list of possible image observation keys found across RTX datasets
    # These represent different camera viewpoints and image formats used in robotics datasets
    logger.info("🚀 Starting RTX dataset processing...")
    logger.info(f"📁 Source directory: {args.rtx_root}")
    logger.info(f"💾 Output directory: {args.save_root}")
    logger.info(
        f"📋 Datasets to process: {[d for d in EPISODE_COUNT_PER_DATASET.keys()]}"
    )
    logger.info("=" * 80)

    sample_indices = None
    if args.sample:
        logger.info(
            f"🔍 Sampling enabled: processing {args.n_samples} samples per dataset"
        )
        sample_indices = get_data_split(args.n_samples)
        logger.info(f"   Sample indices length: {len(sample_indices)}")

    # Process each dataset in the lists
    for dataset_idx, dataset in enumerate(EPISODE_COUNT_PER_DATASET.keys(), 1):
        logger.info(
            f"\n🗂️  Processing dataset {dataset_idx}/{len(EPISODE_COUNT_PER_DATASET)}: {dataset}"
        )
        is_feasible = False

        if args.download_and_process:
            download_dataset(dataset, args.rtx_root)

        try:
            # Load the TensorFlow dataset builder for this dataset
            builder = tfds.builder_from_directory(
                builder_dir=dataset2path(args.rtx_root, dataset)
            )
            logger.info(f"  ✅ Dataset builder loaded successfully")
            logger.info(f"  📊 Available splits: {list(builder.info.splits.keys())}")

            # Show available observation keys
            obs_keys = list(builder.info.features["steps"]["observation"].keys())
            logger.info(
                f"  🔍 Available observation keys: {obs_keys[:10]}{'...' if len(obs_keys) > 10 else ''}"
            )

        except Exception as e:
            logger.error(f"  ❌ Failed to load dataset builder: {str(e)}")
            infeasible_datasets.append(dataset)
            continue

        # Determine which image keys are compatible with this dataset
        compatible_keys, infeasible_datasets = get_compatible_keys(
            builder, infeasible_datasets, dataset
        )

        if sample_indices is not None:
            if dataset not in sample_indices["categories"].values:
                logger.info(
                    f"  ⏭️  Skipping {dataset} (no samples selected, class is too small for sample size)"
                )
                continue

            dataset_sample_indices = sample_indices[
                sample_indices["categories"] == dataset
            ]

        for display_key in compatible_keys:
            logger.info(f"\n  🖼️  Processing image key: {display_key}")

            # Special handling for mimic_play dataset which has nested image structure
            if dataset == "mimic_play":
                if display_key == "image":
                    # Extract front_image_1 from the nested image structure
                    folder = args.save_root / f"{dataset}-front_image_1"
                    if not os.path.exists(folder):
                        logger.info(f"    📁 Creating output folder: {folder}")
                        for split_name in builder.info.splits.keys():
                            extract_sample(
                                builder,
                                display_key,
                                dataset,
                                folder,
                                split_name,
                                "front_image_1",
                                sample_indices=(
                                    set(dataset_sample_indices["indices"].values)
                                    if sample_indices is not None
                                    else None
                                ),
                            )
                    else:
                        logger.info(f"    ⏭️  Skipping {folder} (already exists)")

                    # Extract front_image_2 from the nested image structure
                    folder = args.save_root / f"{dataset}-front_image_2"
                    if not os.path.exists(folder):
                        logger.info(f"    📁 Creating output folder: {folder}")
                        for split_name in builder.info.splits.keys():
                            extract_sample(
                                builder,
                                display_key,
                                dataset,
                                folder,
                                split_name,
                                "front_image_2",
                                sample_indices=(
                                    set(dataset_sample_indices["indices"].values)
                                    if sample_indices is not None
                                    else None
                                ),
                            )
                    else:
                        logger.info(f"    ⏭️  Skipping {folder} (already exists)")
                else:
                    # Handle other image keys in mimic_play with nested structure
                    folder = args.save_root / f"{dataset}-{display_key}"
                    if not os.path.exists(folder):
                        logger.info(f"    📁 Creating output folder: {folder}")
                        for split_name in builder.info.splits.keys():
                            extract_sample(
                                builder,
                                display_key,
                                dataset,
                                folder,
                                split_name,
                                display_key,
                                sample_indices=(
                                    set(dataset_sample_indices["indices"].values)
                                    if sample_indices is not None
                                    else None
                                ),
                            )
                    else:
                        logger.info(f"    ⏭️  Skipping {folder} (already exists)")
            else:
                # Standard processing for most datasets
                folder = args.save_root / f"{dataset}-{display_key}"
                if not os.path.exists(folder):  # Skip if already processed
                    logger.info(f"    📁 Creating output folder: {folder}")
                    # Process all available splits (train, test, validation, etc.)
                    for split_name in builder.info.splits.keys():
                        extract_sample(
                            builder,
                            display_key,
                            dataset,
                            folder,
                            split_name,
                            sample_indices=(
                                set(dataset_sample_indices["indices"].values)
                                if sample_indices is not None
                                else None
                            ),
                        )
                else:
                    logger.info(f"    ⏭️  Skipping {folder} (already exists)")

            # Mark dataset as feasible since it has at least one compatible image key
            is_feasible = True

        # Update processing statistics
        if is_feasible:
            feasible_datasets += 1
            logger.info(f"  ✅ Dataset {dataset} processed successfully")
        else:
            infeasible_datasets.append(dataset)
            logger.info(f"  ❌ Dataset {dataset} marked as infeasible")

        if args.remove_raw_videos:
            os.system(f"rm -rf {args.rtx_root}/{dataset}")
            logger.info("Raw videos have been removed after clipping.")

    # Print final processing summary
    logger.info("\n" + "=" * 80)
    logger.info("🏁 PROCESSING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"✅ Feasible datasets processed: {feasible_datasets}")
    logger.info(f"❌ Infeasible datasets: {len(infeasible_datasets)}")

    if infeasible_datasets:
        logger.info(f"\n📋 Infeasible datasets list:")
        for i, dataset in enumerate(infeasible_datasets, 1):
            logger.info(f"   {i}. {dataset}")

    logger.info(f"\n📁 All output videos saved to: {args.save_root}")
    logger.info("🎉 Done!")


if __name__ == "__main__":
    main()
