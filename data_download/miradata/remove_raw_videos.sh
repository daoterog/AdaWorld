#!/bin/bash

# Get path to the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Removing all raw videos in $SCRIPT_DIR/../../data/miradata/raw_video/"
rm -rf "$SCRIPT_DIR/../../data/miradata/raw_video/"