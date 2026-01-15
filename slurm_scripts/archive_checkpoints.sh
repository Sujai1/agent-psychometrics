#!/bin/bash
# Archive existing SAD-IRT checkpoints to permanent storage
# Run this on the cluster before starting fresh training

set -e

# Source and destination
SRC_DIR="chris_output/sad_irt"
DEST_DIR="$HOME/orcd/pool/sad_irt_checkpoints/$(date +%Y%m%d_%H%M%S)"

# Check if source has checkpoints
if ! ls "$SRC_DIR"/checkpoint_*.pt 1>/dev/null 2>&1; then
    echo "No checkpoints found in $SRC_DIR"
    exit 0
fi

# Create destination directory
echo "Creating archive directory: $DEST_DIR"
mkdir -p "$DEST_DIR"

# Copy checkpoints
echo "Copying checkpoints..."
cp -v "$SRC_DIR"/checkpoint_*.pt "$DEST_DIR/"

# Copy any results files too
if [ -f "$SRC_DIR/results.json" ]; then
    cp -v "$SRC_DIR/results.json" "$DEST_DIR/"
fi

# List what was archived
echo ""
echo "Archived to $DEST_DIR:"
ls -lh "$DEST_DIR"

# Optionally clear source (uncomment if desired)
# echo ""
# echo "Clearing source checkpoints..."
# rm "$SRC_DIR"/checkpoint_*.pt

echo ""
echo "Done! Checkpoints archived to: $DEST_DIR"
echo ""
echo "To start fresh training, either:"
echo "  1. Delete checkpoints: rm $SRC_DIR/checkpoint_*.pt"
echo "  2. Use a new output_dir in the SLURM script"
