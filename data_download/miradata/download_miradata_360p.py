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
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import tqdm

from .utils import (download_metadata_if_needed,
                                          split_data)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set up command line argument parsing
def parse_args() -> argparse.Namespace:
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
        "--sample",
        action="store_true",
        help="If set, we download only a small sample for reproducibility testing",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=300,
        help="Number of samples to download if --sample is set",
    )
    parser.add_argument(
        "--raw_video_save_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "/miradata" / "raw_video",
        help="Directory to save downloaded raw videos",
    )
    parser.add_argument(
        "--clip_video_save_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "miradata" / "clip_video",
        help="Directory to save clipped video segments",
    )
    args = parser.parse_args()
    return args


def download_ytb_video(
    video_url: str, raw_video_download_path: str, download_id: int
) -> None:
    """Download YouTube video at 360p resolution using yt-dlp."""
    try:
        # Extract video ID from YouTube URL and download
        video_id = video_url.split("watch?v=")[-1]

        # Use yt-dlp to download YouTube video at 360p with H.264/AAC encoding
        ret = os.system(
            f"yt-dlp -S vcodec:h264,res:360,acodec:aac -o '{raw_video_download_path}' -- {video_id}"
        )

        if ret != 0:
            raise Exception(f"yt-dlp failed with return code {ret}")
    except Exception as error:
        print(f"Error downloading YouTube video {download_id}: {error}")


def download_stream_video(
    video_url: str, raw_video_download_path: str, download_id: int
) -> None:
    """Download video from direct URL using requests."""
    try:
        res = requests.get(video_url, stream=True)

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
        print(f"Error downloading stream video {download_id}: {error}")


def clip_video(
    download_id: str,
    clip_video_save_dir: str,
    file_path: str,
    raw_video_download_path: str,
    clip_video_path: str,
    start_time: str,
    end_time: str,
    timestamp: str,
) -> None:
    """Clip the downloaded video to the specified timestamp range."""
    try:
        # Build output path for the clipped video segment
        clip_video_path = os.path.join(clip_video_save_dir, file_path)

        # Skip if clipped video already exists (resume functionality)
        if os.path.exists(clip_video_path):
            return

        # Parse timestamp information from the CSV (stored as string representation of list)
        # Format: ["HH:MM:SS.fff", "HH:MM:SS.fff"] for [start_time, end_time]
        timestamps = eval(timestamp)
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

    except Exception as error:
        print(f"Error clipping video {download_id}: {error}")


def process_videos(
    df: pd.DataFrame, raw_video_save_dir: str, clip_video_save_dir: str
) -> None:
    """Process and download videos from the Miradata dataset."""
    for _, row in tqdm.tqdm(df.iterrows(), desc="Processing videos"):

        # Define paths for raw and clipped videos
        download_id = int(row["clip_id"])
        raw_video_download_path = os.path.join(
            raw_video_save_dir, str(download_id).zfill(12) + ".mp4"
        )

        # Skip download if video already exists (resume functionality)
        if not os.path.exists(raw_video_download_path):
            # Create directory structure if it doesn't exist
            os.makedirs(os.path.dirname(raw_video_download_path), exist_ok=True)

            if "youtube" in row["source"]:
                download_ytb_video(
                    row["video_url"], raw_video_download_path, download_id
                )
            else:
                download_stream_video(
                    row["video_url"], raw_video_download_path, download_id
                )

        clip_video(
            download_id=download_id,
            clip_video_save_dir=clip_video_save_dir,
            file_path=row["file_path"],
            raw_video_download_path=raw_video_download_path,
            clip_video_path=clip_video_save_dir,
            start_time=row["start_time"],
            end_time=row["end_time"],
            timestamp=row["timestamp"],
        )


def main() -> None:
    # Parse command line arguments
    args = parse_args()

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

    # If --sample flag is set, reduce to a small subset for testing
    if args.sample:
        df, _ = split_data(
            df, train_size=args.n_samples / len(df), strata=["source"], random_state=42
        )
        print(
            f"Sample mode enabled: reduced dataset to {args.n_samples} random entries for testing"
        )

    process_videos(df, args.raw_video_save_dir, args.clip_video_save_dir)

    print("Processing complete! All videos have been downloaded and clipped.")
    print(f"Metadata used: {metadata_csv_path}")
    print(f"Raw videos saved to: {args.raw_video_save_dir}")
    print(f"Clipped videos saved to: {args.clip_video_save_dir}")


if __name__ == "__main__":
    main()
