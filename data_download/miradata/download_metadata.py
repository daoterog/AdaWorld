"""This module contains code to extract stratified samples from datasets."""

import logging
import os
import unicodedata

import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_strings(
    series: pd.Series,
    form: str = "NFKC",
    remove_invisibles: bool = True,
    casefold: bool = False,
) -> pd.Series:
    # work on string representation but keep everything as strings
    s = series.astype(str)

    # 1) Unicode normalize with pandas (vectorized, fast)
    s = s.str.normalize(form)

    # 2) remove common zero-width / BOM chars with a vectorized replace
    if remove_invisibles:
        s = s.str.replace(r"[\u200B\u200C\u200D\uFEFF]", "", regex=True)

        # If you still suspect other invisible "format" chars, do a small python-level fallback
        # only for rows that still contain characters in Unicode category 'Cf'.
        # This avoids calling Python on every element.
        maybe = s.str.contains(r"[\u200B\u200C\u200D\uFEFF]", regex=True) == False
        # detect any remaining Cf chars using a cheap per-string check
        mask = s[maybe].apply(
            lambda x: any(unicodedata.category(ch) == "Cf" for ch in x)
        )
        if mask.any():

            def remove_format_chars(x):
                return "".join(ch for ch in x if unicodedata.category(ch) != "Cf")

            s.loc[mask.index[mask]] = [
                remove_format_chars(x) for x in s.loc[mask.index[mask]]
            ]

    # 3) trim whitespace
    s = s.str.strip()

    # 4) optional casefold for case-insensitive equivalence (keeps alnum)
    if casefold:
        s = s.str.casefold()

    return s


def download_metadata_if_needed(metadata_dir, split_name="train"):
    """
    Download Miradata metadata CSV files if they don't exist.

    Args:
        metadata_dir (str): Directory where metadata CSV files should be stored
        split_name (str): Which dataset split to use (train, validation, test)

    Returns:
        str: Path to the downloaded/existing CSV file
    """
    csv_path = os.path.join(metadata_dir, f"{split_name}.csv")

    # Check if metadata CSV already exists
    if os.path.exists(csv_path):
        logger.info(f"Metadata CSV already exists at {csv_path}")
        return csv_path

    # Create metadata directory if it doesn't exist
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
        logger.info(f"Created metadata directory: {metadata_dir}")

    # Download the dataset metadata
    logger.info("Metadata CSV not found. Downloading Miradata dataset metadata...")
    try:
        logger.info("Downloading Miradata dataset metadata...")
        dataset = load_dataset("TencentARC/MiraData")

        # Drop duplicates if any in each split
        for split in dataset.keys():
            logger.info(f"Removing duplicates of '{split}' split...")
            split_ds = dataset[split].to_pandas()
            split_ds["video_id"] = normalize_strings(split_ds["video_id"])
            dataset[split] = split_ds.drop_duplicates()

        # Save the requested split to CSV
        if split_name in dataset.keys():
            logger.info(f"Saving {split_name} split to {csv_path}...")
            dataset[split_name].to_csv(csv_path)
            logger.info(f"Successfully downloaded metadata to {csv_path}")
        else:
            # If specific split not found, save all available splits
            logger.warning(
                f"Split '{split_name}' not found. Downloading all available splits: {list(dataset.keys())}"
            )
            for split in dataset.keys():
                split_path = os.path.join(metadata_dir, f"{split}.csv")
                logger.info(f"Saving {split} split to {split_path}...")
                dataset[split].to_csv(split_path)

            # Default to first available split if requested split doesn't exist
            csv_path = os.path.join(metadata_dir, f"{list(dataset.keys())[0]}.csv")
            logger.info(f"Using {list(dataset.keys())[0]} split as default")

    except Exception as error:
        logger.error(f"Error downloading metadata: {error}")
        raise

    return csv_path
