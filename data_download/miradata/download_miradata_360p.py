#!/usr/bin/env python3
"""
Miradata Video Downloader and Clipper

This script downloads and processes video data from the Miradata dataset.
It performs two main operations:
1. Downloads full videos from YouTube or direct URLs at 360p resolution
2. Clips the downloaded videos to specific timestamps as specified in the metadata CSV

The script supports resuming downloads - it will skip files that already exist.
"""

import argparse
import os
import pandas as pd
import requests
import tqdm
from datetime import datetime

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Download and clip Miradata videos")
parser.add_argument("--meta_csv", type=str, 
                   default=os.path.join(os.path.dirname(__file__), "data_download/miradata/train.csv"),
                   help="Path to the metadata CSV file")
parser.add_argument("--raw_video_save_dir", type=str, 
                   default=os.path.join(os.path.dirname(__file__), "data/miradata/raw_video"),
                   help="Directory to save downloaded raw videos")
parser.add_argument("--clip_video_save_dir", type=str, 
                   default=os.path.join(os.path.dirname(__file__), "data/miradata/"),
                   help="Directory to save clipped video segments")
args = parser.parse_args()

# Load the metadata CSV file containing video information
df = pd.read_csv(args.meta_csv, encoding='utf-8')
print(f"Successfully loaded the csv file with {len(df)} entries")

# Process each video entry in the dataset
for i, row in tqdm.tqdm(df.iterrows(), desc="Processing videos"):
    # Only process YouTube videos (skip other sources for now)
    if "youtube" in row["source"]:
        # Extract clip ID and create standardized filename (12-digit zero-padded)
        download_id = int(row["clip_id"])
        raw_video_download_path = os.path.join(args.raw_video_save_dir, str(download_id).zfill(12) + ".mp4")

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
                    video_id = row['video_url'].split('watch?v=')[-1]
                    ret = os.system(f"yt-dlp -S vcodec:h264,res:360,acodec:aac -o '{raw_video_download_path}' -- {video_id}")
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
            end_time = timestamps[1]    # End timestamp for clipping
            
            # Calculate duration by subtracting start time from end time
            duration = str(datetime.strptime(end_time, "%H:%M:%S.%f") - 
                          datetime.strptime(start_time, "%H:%M:%S.%f"))
            
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
