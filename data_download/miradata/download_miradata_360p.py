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
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

sys.path.append(Path(__file__).parents[2].as_posix())


import pandas as pd
import requests
import tqdm

from data_download.miradata.download_metadata import \
    download_metadata_if_needed
from data_download.utils import split_data

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
        default=Path(__file__).resolve().parents[2] / "data" / "miradata" / "raw_video",
        help="Directory to save downloaded raw videos",
    )
    parser.add_argument(
        "--clip_video_save_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "miradata",
        help="Directory to save clipped video segments",
    )
    parser.add_argument(
        "--remove_raw_videos",
        action="store_true",
        help="If set, remove raw videos after clipping",
    )
    args = parser.parse_args()
    return args


def download_ytb_video(
    video_url: str, raw_video_download_path: Path, download_id: int
) -> None:
    """Download YouTube video at 360p resolution using yt-dlp."""
    try:
        # Extract video ID from YouTube URL and download
        video_id = video_url.split("watch?v=")[-1]

        if os.path.exists(raw_video_download_path.with_suffix(".part")):
            # remove any existing partial download file
            logger.info(
                f"Removing existing partial download file: {str(raw_video_download_path) + '.part'}"
            )
            os.remove(raw_video_download_path.with_suffix(".part"))

        if os.path.exists(raw_video_download_path):
            # skip if video already exists (resume functionality)
            logger.info(f"Video already exists, skipping: {raw_video_download_path}")
            return

        # Use yt-dlp to download YouTube video at 360p with H.264/AAC encoding
        ret = os.system(
            f"yt-dlp -S vcodec:h264,res:360,acodec:aac -o '{raw_video_download_path}' -- {video_id}"
        )

        if ret != 0:
            raise Exception(f"yt-dlp failed with return code {ret}")
    except Exception as error:
        logger.error(f"Error downloading YouTube video {download_id}: {error}")


def download_stream_video(
    video_url: str, raw_video_download_path: str, download_id: int
) -> None:
    """Download video from direct URL using requests."""
    try:
        res = requests.get(video_url, stream=True)

        if os.path.exists(raw_video_download_path.with_suffix(".tmp")):
            # Clean up any existing temporary file
            logger.info(
                f"Removing temporary file: {raw_video_download_path.with_suffix('.tmp')}"
            )
            os.remove(raw_video_download_path.with_suffix(".tmp"))

        if os.path.exists(raw_video_download_path):
            # Skip if video already exists (resume functionality)
            logger.info(f"Video already exists, skipping: {raw_video_download_path}")
            return

        # Download in chunks to handle large files efficiently
        logger.info(f"Downloading stream video to {raw_video_download_path}")
        with open(raw_video_download_path.with_suffix(".tmp"), "wb") as f:
            for chunk in tqdm.tqdm(
                res.iter_content(chunk_size=10240),
                total=int(res.headers.get("Content-Length", 0)) // 10240,
            ):
                f.write(chunk)
        logger.info(f"Download complete: {raw_video_download_path}")

        # Atomically rename temp file to final name once download completes
        os.rename(raw_video_download_path.with_suffix(".tmp"), raw_video_download_path)

    except Exception as error:
        logger.error(f"Error downloading stream video {download_id}: {error}")


def clip_video(
    download_id: str,
    raw_video_download_path: str,
    clip_video_path: str,
    timestamp: str,
) -> None:
    """Clip the downloaded video to the specified timestamp range."""
    try:

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
        clip_video_path.parent.mkdir(parents=True, exist_ok=True)

        # Execute the ffmpeg command to create the video clip
        os.system(run_command)

    except Exception as error:
        logger.error(f"Error clipping video {download_id}: {error}")


def process_videos(
    df: pd.DataFrame,
    raw_video_save_dir: Path,
    clip_video_save_dir: str,
    remove_raw_videos: bool = False,
) -> None:
    """Process and download videos from the Miradata dataset."""

    # Count occurrences of each video URL for removal later.
    video_url_counts = df["video_url"].value_counts()
    video_url_counter = {}

    for _, row in tqdm.tqdm(df.iterrows(), desc="Processing videos"):

        # Define path for clipped video
        clip_video_path = clip_video_save_dir / row["file_path"]

        # Skip if clipped video already exists (resume functionality)
        if os.path.exists(clip_video_path):
            logger.info(f"Clipped video already exists, skipping: {clip_video_path}")
            continue

        # Define paths for raw
        raw_download_id = quote(row["video_url"], safe="")
        raw_video_download_path = raw_video_save_dir / (raw_download_id + ".mp4")

        # Skip download if video already exists (resume functionality)
        if not os.path.exists(raw_video_download_path):
            # Create directory structure if it doesn't exist
            raw_video_download_path.parent.mkdir(parents=True, exist_ok=True)

            if "youtube" in row["source"]:
                logger.info(f"Downloading YouTube video ID {raw_download_id}")
                download_ytb_video(
                    row["video_url"], raw_video_download_path, raw_download_id
                )
            else:
                logger.info(f"Downloading stream video ID {raw_download_id}")
                download_stream_video(
                    row["video_url"], raw_video_download_path, raw_download_id
                )

        clip_video(
            download_id=raw_download_id,
            raw_video_download_path=raw_video_download_path,
            clip_video_path=clip_video_path,
            timestamp=row["timestamp"],
        )

        if remove_raw_videos:
            if row["video_url"] in video_url_counter:
                video_url_counter[row["video_url"]] += 1
            else:
                video_url_counter[row["video_url"]] = 1

            if (
                video_url_counter[row["video_url"]]
                == video_url_counts[row["video_url"]]
            ):
                if os.path.exists(raw_video_download_path):
                    os.remove(raw_video_download_path)
                    logger.info(f"Removed raw video: {raw_video_download_path}")


def main() -> None:
    try:
        # Parse command line arguments
        args = parse_args()

        # Handle metadata CSV - either use provided path or auto-download
        metadata_dir = Path(__file__).parent.resolve()
        if args.meta_csv is None:
            # Auto-download metadata if no CSV path provided
            metadata_csv_path = download_metadata_if_needed(metadata_dir, args.split)
        else:
            # Use provided CSV path, but check if it exists
            metadata_csv_path = args.meta_csv
            if not os.path.exists(metadata_csv_path):
                logger.warning(f"Specified metadata CSV not found: {metadata_csv_path}")
                logger.warning("Attempting to auto-download metadata...")
                metadata_csv_path = download_metadata_if_needed(
                    metadata_dir, args.split
                )

        # Load the metadata CSV file containing video information
        df = pd.read_csv(metadata_csv_path, encoding="utf-8")
        logger.info(
            f"Successfully loaded the CSV file with {len(df)} entries from {metadata_csv_path}"
        )

        # If --sample flag is set, reduce to a small subset for testing
        if args.sample:
            df, _ = split_data(
                df,
                train_size=args.n_samples / len(df),
                strata=["source"],
                random_state=42,
            )
            logger.info(
                f"Sample mode enabled: reduced dataset to {args.n_samples} random entries for testing"
            )

        process_videos(
            df,
            args.raw_video_save_dir,
            args.clip_video_save_dir,
            args.remove_raw_videos,
        )

        logger.info("Processing complete! All videos have been downloaded and clipped.")
        logger.info(f"Metadata used: {metadata_csv_path}")
        logger.info(f"Raw videos saved to: {args.raw_video_save_dir}")
        logger.info(f"Clipped videos saved to: {args.clip_video_save_dir}")

        if args.remove_raw_videos:
            os.system(f"bash {Path(__file__).parent}/remove_raw_videos.sh")
            logger.info("Raw videos have been removed after clipping.")

    except KeyboardInterrupt:
        if args.remove_raw_videos:
            os.system(f"bash {Path(__file__).parent}/remove_raw_videos.sh")
            logger.info("Raw videos have been removed after clipping.")
        raise  # re-raise so program still exits with Ctrl+C
    except Exception as e:
        if args.remove_raw_videos:
            os.system(f"bash {Path(__file__).parent}/remove_raw_videos.sh")
            logger.info("Raw videos have been removed after clipping.")
        raise


if __name__ == "__main__":
    main()
