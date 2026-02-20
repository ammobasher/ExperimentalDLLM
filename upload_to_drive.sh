#!/bin/bash
# ============================================================
# Upload ExperimentalDLLM to Google Drive for Colab Training
# ============================================================
# This script packages the source code + checkpoint into a
# tarball that you can upload to Google Drive.
#
# Usage:
#   cd /Volumes/Segate/ExperimentalDLLM
#   bash upload_to_drive.sh
#
# Then upload the resulting .tar.gz to Google Drive root.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_NAME="ExperimentalDLLM_colab"
OUTPUT_DIR="/Volumes/Segate/tmp/${OUTPUT_NAME}"
TARBALL="/Volumes/Segate/tmp/${OUTPUT_NAME}.tar.gz"

echo "============================================================"
echo "  Packaging ExperimentalDLLM for Google Colab"
echo "============================================================"

# Clean previous
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/src"
mkdir -p "$OUTPUT_DIR/checkpoints"

# Copy source files
echo ">> Copying source code..."
cp -r src/*.py "$OUTPUT_DIR/src/"
cp train_episodic.py "$OUTPUT_DIR/"
cp cache_data.py "$OUTPUT_DIR/"
cp benchmark_all.py "$OUTPUT_DIR/"
cp requirements.txt "$OUTPUT_DIR/"
cp colab_train.ipynb "$OUTPUT_DIR/" 2>/dev/null || true

# Copy checkpoint (1.5 GB)
CKPT="checkpoints/checkpoint_small_pretrain_step610000.pt"
if [ -f "$CKPT" ]; then
    echo ">> Copying checkpoint ($(du -h "$CKPT" | cut -f1))..."
    cp "$CKPT" "$OUTPUT_DIR/checkpoints/"
else
    echo "!! Checkpoint not found: $CKPT"
    echo "   Will need to train from scratch on Colab."
fi

# Create tarball
echo ">> Creating tarball..."
cd /tmp
tar -czf "$TARBALL" "$OUTPUT_NAME"

SIZE=$(du -h "$TARBALL" | cut -f1)
echo ""
echo "============================================================"
echo "  ✅ Package created: $TARBALL ($SIZE)"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "  1. Upload $TARBALL to Google Drive (root folder)"
echo "  2. Open Google Colab (colab.research.google.com)"
echo "  3. Upload colab_train.ipynb or create a new notebook"
echo "  4. Follow the notebook instructions"
echo ""
echo "  Tip: You can also drag-and-drop the tarball into"
echo "  drive.google.com for the fastest upload."
echo "============================================================"
