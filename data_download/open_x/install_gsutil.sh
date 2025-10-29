#!/bin/bash

echo "gsutil not found. Attempting standalone installation for unauthenticated use..."

# Define a local installation path
GSUTIL_INSTALL_DIR="$HOME/gsutil_standalone"
mkdir -p "$GSUTIL_INSTALL_DIR"

# Download the standalone gsutil tarball
GSUTIL_TAR="gsutil.tar.gz"
curl -o $GSUTIL_TAR https://storage.googleapis.com/pub/gsutil.tar.gz

# Extract to the installation directory
tar -xf $GSUTIL_TAR -C $GSUTIL_INSTALL_DIR
rm $GSUTIL_TAR

# Add the local gsutil to the PATH for the rest of this script
export PATH="$GSUTIL_INSTALL_DIR/gsutil:$PATH"
echo "gsutil installed to $GSUTIL_INSTALL_DIR and added to PATH for this job."
