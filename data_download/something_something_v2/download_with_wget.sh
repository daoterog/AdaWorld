#!/bin/bash
# Smart wget-based downloader for Something-Something v2 dataset
# Checks for existing files, downloads missing ones, and verifies integrity using SHA256

set -e  # Exit on any error

# Configuration
DOWNLOAD_DIR="data/something_something_v2"
LABELS_URL="https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip"
VIDEO_URLS=(
    "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00"
    "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01"
)
VIDEO_FILES=("20bn-something-something-v2-00" "20bn-something-something-v2-01")
LABELS_FILE="20bn-something-something-download-package-labels.zip"
FINAL_FILE="20bn-something-something-v2.tar.gz"

# Hardcoded SHA256 checksums
declare -A FILE_SHA256=(
    ["$LABELS_FILE"]="8e9859f2862c3626dd1cd013dbc0bccccc12ab48d7c18812e3f302988d6f2c0c"
    ["20bn-something-something-v2-00"]="aab5bf3badcdf58aed2ac4b6c219a910ea97069b346bcb3ab19202b42dc8f0aa"
    ["20bn-something-something-v2-01"]="65b505fd26e7e72b745421d33508a5b5fcb7544c6b5510e39aa754282bad8d24"
    ["$FINAL_FILE"]="0a456ec208cc7bf94a6a6949f2fc16463eef80ff181bbd737c071187850f6393"
)

# Create download directory
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo "=== Something-Something v2 Dataset Downloader ==="
echo "Download directory: $(pwd)"

# Ensure sha256sum is available (install if missing)
if ! command -v sha256sum >/dev/null 2>&1; then
    echo "⚙️  sha256sum not found. Attempting to install coreutils..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y && sudo apt-get install -y coreutils
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y coreutils
    else
        echo "✗ Could not install sha256sum (no supported package manager found)."
        exit 1
    fi
fi

# Function to check if a file matches its expected SHA256
verify_sha256() {
    local file="$1"
    local expected_sha="${FILE_SHA256[$file]}"

    if [[ -z "$expected_sha" ]]; then
        echo "⚠️  No hardcoded checksum for $file. Skipping verification."
        return 0
    fi

    if [[ ! -f "$file" ]]; then
        echo "✗ File not found: $file"
        return 1
    fi

    local actual_sha
    actual_sha=$(sha256sum "$file" | awk '{print $1}')

    if [[ "$actual_sha" == "$expected_sha" ]]; then
        return 0
    else
        echo "✗ Checksum mismatch for $file"
        echo "  Expected: $expected_sha"
        echo "  Actual:   $actual_sha"
        return 1
    fi
}

# Function to download a file with resume capability and verify integrity
download_file() {
    local url="$1"
    local filename="$2"
    local description="$3"

    echo "Checking $description..."

    if [[ -f "$filename" ]]; then
        if verify_sha256 "$filename"; then
            echo "✓ $description already exists and passed checksum"
            return 0
        else
            echo "⚠️  $description failed checksum. Re-downloading..."
            rm -f "$filename"
        fi
    fi

    echo "Downloading $description..."
    if wget -c -O "$filename" "$url"; then
        echo "✓ $description downloaded successfully"
        if verify_sha256 "$filename"; then
            echo "✓ Checksum verified for $filename"
        else
            echo "✗ Checksum failed for $filename after download"
            exit 1
        fi
    else
        echo "✗ Failed to download $description"
        exit 1
    fi
}

# Download labels file
download_file "$LABELS_URL" "$LABELS_FILE" "labels file"

# Download video files
for i in "${!VIDEO_URLS[@]}"; do
    download_file "${VIDEO_URLS[$i]}" "${VIDEO_FILES[$i]}" "video file part $((i+1))"
done

echo ""
echo "=== Processing Downloads ==="

# Extract labels if not already extracted
if [[ -f "$LABELS_FILE" ]]; then
    echo "Extracting labels file..."
    unzip -q -o "$LABELS_FILE" 2>/dev/null || echo "Labels already extracted or extraction failed"
fi

# Check all video parts
for video_file in "${VIDEO_FILES[@]}"; do
    if ! verify_sha256 "$video_file"; then
        echo "✗ Missing or invalid: $video_file"
        echo "Please re-run the script to retry the download."
        exit 1
    fi
done

echo "Concatenating video files..."
cat "${VIDEO_FILES[@]}" > "$FINAL_FILE"

if verify_sha256 "$FINAL_FILE"; then
    echo "✓ Final dataset archive verified successfully."
else
    echo "✗ Final archive checksum mismatch. Recreating..."
    rm -f "$FINAL_FILE"
    cat "${VIDEO_FILES[@]}" > "$FINAL_FILE"
    if ! verify_sha256 "$FINAL_FILE"; then
        echo "✗ Final archive still invalid. Aborting."
        exit 1
    fi
fi

echo "Extracting dataset..."
if tar -xzf "$FINAL_FILE"; then
    echo "✓ Extraction completed successfully."
    echo "✓ Dataset ready in: $(pwd)"
else
    echo "✗ Extraction failed."
    exit 1
fi
