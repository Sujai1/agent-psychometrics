#!/bin/bash
# Migrate HuggingFace cache from home directory to scratch storage
# Run this once on the cluster to move existing cache and set up symlink

set -euo pipefail

# Configuration
OLD_CACHE="$HOME/.cache/huggingface"
NEW_CACHE="$HOME/orcd/scratch/.cache/huggingface"

echo "=== HuggingFace Cache Migration ==="
echo "From: $OLD_CACHE"
echo "To:   $NEW_CACHE"
echo ""

# Check current usage
if [ -d "$OLD_CACHE" ]; then
    echo "Current cache size:"
    du -sh "$OLD_CACHE" 2>/dev/null || echo "  (empty or inaccessible)"
    echo ""
fi

# Create new cache directory
echo "Creating new cache directory..."
mkdir -p "$NEW_CACHE"

# Move existing cache if it exists and is not a symlink
if [ -d "$OLD_CACHE" ] && [ ! -L "$OLD_CACHE" ]; then
    echo "Moving existing cache to scratch..."

    # Use rsync for safe transfer (handles partial files)
    if command -v rsync &> /dev/null; then
        rsync -av --remove-source-files "$OLD_CACHE/" "$NEW_CACHE/"
        # Remove empty directories left behind
        find "$OLD_CACHE" -type d -empty -delete 2>/dev/null || true
        rmdir "$OLD_CACHE" 2>/dev/null || true
    else
        # Fallback to mv
        mv "$OLD_CACHE"/* "$NEW_CACHE/" 2>/dev/null || true
        rmdir "$OLD_CACHE" 2>/dev/null || true
    fi

    echo "Cache moved successfully."
elif [ -L "$OLD_CACHE" ]; then
    echo "Cache is already a symlink, skipping move."
else
    echo "No existing cache found, skipping move."
fi

# Create symlink from old location to new
if [ ! -L "$OLD_CACHE" ]; then
    echo "Creating symlink..."
    mkdir -p "$(dirname "$OLD_CACHE")"
    ln -sf "$NEW_CACHE" "$OLD_CACHE"
    echo "Symlink created: $OLD_CACHE -> $NEW_CACHE"
else
    echo "Symlink already exists."
fi

# Also handle project-local cache if it exists
PROJECT_CACHE="$HOME/model_irt/.cache/huggingface"
if [ -d "$PROJECT_CACHE" ] && [ ! -L "$PROJECT_CACHE" ]; then
    echo ""
    echo "Moving project-local cache..."
    rsync -av --remove-source-files "$PROJECT_CACHE/" "$NEW_CACHE/" 2>/dev/null || true
    find "$PROJECT_CACHE" -type d -empty -delete 2>/dev/null || true
    rmdir "$PROJECT_CACHE" 2>/dev/null || true
    ln -sf "$NEW_CACHE" "$PROJECT_CACHE"
    echo "Project cache linked."
fi

echo ""
echo "=== Migration Complete ==="
echo ""
echo "New cache location: $NEW_CACHE"
echo "New cache size:"
du -sh "$NEW_CACHE" 2>/dev/null || echo "  (empty)"
echo ""
echo "Add this to your .bashrc or SLURM scripts:"
echo "  export HF_HOME=\"$NEW_CACHE\""
echo ""
echo "The symlink ensures existing tools still work with ~/.cache/huggingface"
