#!/usr/bin/env python3
"""
Miradata Video Downloader and Clipper

This script downloads and processes video data from the Miradata dataset.
It performs three main operations:
1. Downloads metadata CSV files from the Miradata dataset (if not already present)
2. Downloads full videos from YouTube or direct URLs at 360p resolution
3. Clips the downloaded videos to specific timestamps as specified in the metadata CSV

The script supports resuming downloads - it will skip files that already exist.
"""

import argparse
import os
from datetime import datetime

import pandas as pd
import requests
import tqdm
from datasets import load_dataset  # For downloading Miradata metadata


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
        print(f"Metadata CSV already exists at {csv_path}")
        return csv_path

    # Create metadata directory if it doesn't exist
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
        print(f"Created metadata directory: {metadata_dir}")

    # Download the dataset metadata
    print("Metadata CSV not found. Downloading Miradata dataset metadata...")
    try:
        dataset = load_dataset("TencentARC/MiraData")

        # Save the requested split to CSV
        if split_name in dataset.keys():
            print(f"Saving {split_name} split to {csv_path}...")
            dataset[split_name].to_csv(csv_path)
            print(f"Successfully downloaded metadata to {csv_path}")
        else:
            # If specific split not found, save all available splits
            print(
                f"Split '{split_name}' not found. Downloading all available splits: {list(dataset.keys())}"
            )
            for split in dataset.keys():
                split_path = os.path.join(metadata_dir, f"{split}.csv")
                print(f"Saving {split} split to {split_path}...")
                dataset[split].to_csv(split_path)

            # Default to first available split if requested split doesn't exist
            csv_path = os.path.join(metadata_dir, f"{list(dataset.keys())[0]}.csv")
            print(f"Using {list(dataset.keys())[0]} split as default")

    except Exception as error:
        print(f"Error downloading metadata: {error}")
        raise

    return csv_path


# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Download and clip Miradata videos")
parser.add_argument(
    "--split",
    type=str,
    default="train",
    help="Dataset split to use (train, validation, test)",
)
parser.add_argument(
    "--meta_csv",
    type=str,
    default=None,
    help="Path to the metadata CSV file (if not provided, will auto-download)",
)
parser.add_argument(
    "--raw_video_save_dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "../../data/miradata/raw_video"),
    help="Directory to save downloaded raw videos",
)
parser.add_argument(
    "--clip_video_save_dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "../../data/miradata/"),
    help="Directory to save clipped video segments",
)
args = parser.parse_args()

# Handle metadata CSV - either use provided path or auto-download
if args.meta_csv is None:
    # Auto-download metadata if no CSV path provided
    metadata_dir = os.path.dirname(__file__)
    metadata_csv_path = download_metadata_if_needed(metadata_dir, args.split)
else:
    # Use provided CSV path, but check if it exists
    metadata_csv_path = args.meta_csv
    if not os.path.exists(metadata_csv_path):
        print(f"Specified metadata CSV not found: {metadata_csv_path}")
        print("Attempting to auto-download metadata...")
        metadata_dir = os.path.dirname(metadata_csv_path)
        metadata_csv_path = download_metadata_if_needed(metadata_dir, args.split)

# Load the metadata CSV file containing video information
df = pd.read_csv(metadata_csv_path, encoding="utf-8")
print(
    f"Successfully loaded the CSV file with {len(df)} entries from {metadata_csv_path}"
)

# Process each video entry in the dataset
for i, row in tqdm.tqdm(df.iterrows(), desc="Processing videos"):
    # Only process YouTube videos (skip other sources for now)
    if "youtube" in row["source"]:
        # Extract clip ID and create standardized filename (12-digit zero-padded)
        download_id = int(row["clip_id"])
        raw_video_download_path = os.path.join(
            args.raw_video_save_dir, str(download_id).zfill(12) + ".mp4"
        )

        # Skip download if video already exists (resume functionality)
        if not os.path.exists(raw_video_download_path):
            # Create directory structure if it doesn't exist
            if not os.path.exists(os.path.dirname(raw_video_download_path)):
                os.makedirs(os.path.dirname(raw_video_download_path))

            # Download the video
            try:
                if "youtube" in row["source"]:
                    # Use yt-dlp to download YouTube video at 360p with H.264/AAC encoding
                    # Extract video ID from YouTube URL and download
                    video_id = row["video_url"].split("watch?v=")[-1]
                    ret = os.system(
                        f"yt-dlp -S vcodec:h264,res:360,acodec:aac -o '{raw_video_download_path}' -- {video_id}"
                    )
                else:
                    # For non-YouTube videos, download directly via HTTP stream
                    res = requests.get(row["video_url"], stream=True)
                    # Clean up any existing temporary file
                    if os.path.exists(raw_video_download_path + ".tmp"):
                        os.remove(raw_video_download_path + ".tmp")
                    # Download in chunks to handle large files efficiently
                    with open(raw_video_download_path + ".tmp", "wb") as f:
                        for chunk in res.iter_content(chunk_size=10240):  # 10KB chunks
                            f.write(chunk)
                    # Atomically rename temp file to final name once download completes
                    os.rename(raw_video_download_path + ".tmp", raw_video_download_path)
            except Exception as error:
                print(f"Error downloading video {download_id}: {error}")

        # Clip the downloaded video to the specified timestamp range
        try:
            # Build output path for the clipped video segment
            clip_video_path = os.path.join(args.clip_video_save_dir, row["file_path"])

            # Skip if clipped video already exists (resume functionality)
            if os.path.exists(clip_video_path):
                continue

            # Parse timestamp information from the CSV (stored as string representation of list)
            # Format: ["HH:MM:SS.fff", "HH:MM:SS.fff"] for [start_time, end_time]
            timestamps = eval(row["timestamp"])
            start_time = timestamps[0]  # Start timestamp for clipping
            end_time = timestamps[1]  # End timestamp for clipping

            # Calculate duration by subtracting start time from end time
            duration = str(
                datetime.strptime(end_time, "%H:%M:%S.%f")
                - datetime.strptime(start_time, "%H:%M:%S.%f")
            )

            # Build ffmpeg command to extract video segment
            # -ss: start time, -t: duration, -c copy: avoid re-encoding (faster), -y: overwrite output
            run_command = f"ffmpeg -ss {start_time} -t {duration} -i {raw_video_download_path} -c copy -y {clip_video_path}"

            # Create output directory if it doesn't exist
            if not os.path.exists(os.path.dirname(clip_video_path)):
                os.makedirs(os.path.dirname(clip_video_path))

            # Execute the ffmpeg command to create the video clip
            os.system(run_command)

            # TODO: Remove this debug breakpoint in production
            # breakpoint()
        except Exception as error:
            print(f"Error clipping video {download_id}: {error}")

print("Processing complete! All videos have been downloaded and clipped.")
print(f"Metadata used: {metadata_csv_path}")
print(f"Raw videos saved to: {args.raw_video_save_dir}")
print(f"Clipped videos saved to: {args.clip_video_save_dir}")
