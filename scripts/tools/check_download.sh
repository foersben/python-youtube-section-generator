#!/bin/bash
# Check Phi-3 model download status

MODEL_FILE="models/Phi-3-mini-4k-instruct-q4.gguf"
EXPECTED_SIZE=2393231072

echo "======================================================================"
echo "MODEL DOWNLOAD STATUS"
echo "======================================================================"
echo ""

if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Model file not found: $MODEL_FILE"
    echo ""
    echo "Start download with:"
    echo "  ./scripts/download_model.sh"
    echo ""
    exit 1
fi

SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null)
SIZE_MB=$(echo "scale=1; $SIZE / 1024 / 1024" | bc)
SIZE_GB=$(echo "scale=2; $SIZE / 1024 / 1024 / 1024" | bc)
PERCENT=$(echo "scale=1; 100 * $SIZE / $EXPECTED_SIZE" | bc)

echo "File: $MODEL_FILE"
echo "Current size: ${SIZE_GB}GB (${SIZE_MB}MB)"
echo "Progress: ${PERCENT}%"
echo ""

if [ "$SIZE" -ge "$EXPECTED_SIZE" ]; then
    echo "✅ Download COMPLETE!"
    echo ""
    echo "Next steps:"
    echo "  1. Test: poetry run python scripts/test_llm_cpu.py"
    echo "  2. Use: poetry run python src/web_app.py"
else
    REMAINING=$(echo "scale=2; ($EXPECTED_SIZE - $SIZE) / 1024 / 1024 / 1024" | bc)
    echo "⏳ Download IN PROGRESS..."
    echo "Remaining: ${REMAINING}GB"
    echo ""
    echo "To resume if interrupted:"
    echo "  ./scripts/download_model.sh"
fi

echo ""
