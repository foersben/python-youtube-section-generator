#!/bin/bash
# Download Phi-3 GGUF model for CPU inference

echo "======================================================================"
echo "Phi-3 GGUF Model Downloader"
echo "======================================================================"
echo ""

MODEL_DIR="models"
MODEL_FILE="Phi-3-mini-4k-instruct-q4.gguf"
MODEL_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
EXPECTED_SIZE_GB=2.2  # Minimum expected size in GB

# Create models directory
mkdir -p "$MODEL_DIR"

# Check if model already exists and verify size
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    SIZE_BYTES=$(stat -c%s "$MODEL_DIR/$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_DIR/$MODEL_FILE" 2>/dev/null)
    SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc)

    if (( $(echo "$SIZE_GB > $EXPECTED_SIZE_GB" | bc -l) )); then
        echo "✅ Model already downloaded: $MODEL_DIR/$MODEL_FILE"
        echo "   Size: ${SIZE_GB}GB (valid)"
        echo ""
        echo "To re-download, delete the file first:"
        echo "   rm $MODEL_DIR/$MODEL_FILE"
        exit 0
    else
        echo "⚠️  Incomplete download detected (${SIZE_GB}GB < ${EXPECTED_SIZE_GB}GB expected)"
        echo "   Removing incomplete file and re-downloading..."
        rm -f "$MODEL_DIR/$MODEL_FILE"
    fi
fi

echo "Downloading Phi-3-mini-4k-instruct Q4 GGUF model..."
echo "Size: ~2.4GB"
echo "This may take 5-15 minutes depending on your connection."
echo ""

# Download with wget or curl (with resume support)
if command -v wget &> /dev/null; then
    echo "Using wget (with resume support)..."
    wget --continue --show-progress -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
    DOWNLOAD_STATUS=$?
elif command -v curl &> /dev/null; then
    echo "Using curl (with resume support)..."
    curl -L -C - --progress-bar -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
    DOWNLOAD_STATUS=$?
else
    echo "❌ Error: Neither wget nor curl found."
    echo ""
    echo "Please install wget or curl, or download manually:"
    echo "  $MODEL_URL"
    echo ""
    echo "Save to: $MODEL_DIR/$MODEL_FILE"
    exit 1
fi

# Check download status
if [ $DOWNLOAD_STATUS -ne 0 ]; then
    echo ""
    echo "❌ Download failed or was interrupted!"
    echo ""
    echo "To resume download, run this script again:"
    echo "  ./scripts/download_model.sh"
    exit 1
fi

# Verify download
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    SIZE_BYTES=$(stat -c%s "$MODEL_DIR/$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_DIR/$MODEL_FILE" 2>/dev/null)
    SIZE_GB=$(echo "scale=2; $SIZE_BYTES / 1024 / 1024 / 1024" | bc)

    if (( $(echo "$SIZE_GB < $EXPECTED_SIZE_GB" | bc -l) )); then
        echo ""
        echo "⚠️  Warning: Downloaded file seems incomplete"
        echo "   Size: ${SIZE_GB}GB (expected >$EXPECTED_SIZE_GB GB)"
        echo ""
        echo "To retry download:"
        echo "  rm $MODEL_DIR/$MODEL_FILE"
        echo "  ./scripts/download_model.sh"
        exit 1
    fi

    SIZE=$(du -h "$MODEL_DIR/$MODEL_FILE" | cut -f1)
    echo ""
    echo "======================================================================"
    echo "✅ Download complete!"
    echo "======================================================================"
    echo ""
    echo "Model: $MODEL_FILE"
    echo "Size: $SIZE"
    echo "Location: $MODEL_DIR/"
    echo ""
    echo "Next steps:"
    echo "  1. Test the model: poetry run python scripts/test_llm_cpu.py"
    echo "  2. Use in web app: poetry run python src/web_app.py"
    echo ""
else
    echo ""
    echo "❌ Download failed!"
    echo ""
    echo "Please download manually from:"
    echo "  $MODEL_URL"
    echo ""
    echo "Save to: $MODEL_DIR/$MODEL_FILE"
    exit 1
fi
