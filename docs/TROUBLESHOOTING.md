# Troubleshooting Guide

Common issues and solutions for the YouTube Transcript Section Generator.

## 🚨 Quick Diagnosis

### System Health Check
Run this command to check your system status:
```bash
python -c "
from src.core.services.section_generation import SectionGenerationService
from pathlib import Path
import os

print('🔍 System Health Check')
print('=' * 50)

# Check model
model_path = 'models/Phi-3-mini-4k-instruct-q4.gguf'
model_exists = Path(model_path).exists()
print(f'✅ Model file: {model_exists}')
if not model_exists:
    print(f'   Expected at: {model_path}')

# Check DeepL
deepl_key = os.getenv('DEEPL_API_KEY')
print(f'✅ DeepL API key: {bool(deepl_key)}')

# Check Python version
import sys
version_ok = sys.version_info >= (3, 11)
print(f'✅ Python 3.11+: {version_ok} ({sys.version.split()[0]})')

# Check imports
try:
    import torch
    import sentence_transformers
    import chromadb
    import llama_cpp
    print('✅ Core dependencies: OK')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')

print('\\n🏥 Health check complete')
"
```

## 🔧 Common Issues & Solutions

### Translation Issues

#### "Titles not translated back to German"
**Symptoms:**
- Titles appear in English instead of original language
- Logs show: `INFO - 🔄 Back-translating 12 section titles`

**Solutions:**
1. **Check DeepL API key:**
   ```bash
   echo $DEEPL_API_KEY  # Should show your key
   ```

2. **Verify `.env` file:**
   ```bash
   cat .env | grep -E "(DEEPL|TRANSLATE)"
   # Should show: DEEPL_API_KEY=your_key_here
   # And: USE_TRANSLATION=true
   ```

3. **Restart Flask server:**
   ```bash
   # Kill existing server
   pkill -f "python src/web_app.py"
   # Restart
   poetry run python src/web_app.py
   ```

4. **Check API quota:**
   - Visit [DeepL API dashboard](https://www.deepl.com/account)
   - Verify you have remaining characters

#### "Translation too slow (>5 minutes)"
**Symptoms:**
- Processing takes 5-10 minutes for 1-hour videos
- Logs show individual segment translation

**Root Cause:** Old code without batching optimization

**Solution:**
```bash
# Ensure you're using latest code
git pull origin main

# Verify batching is active
poetry run python src/web_app.py
# Look for: "INFO - Batched 1586 segments into 32 chunks"
```

**Expected Performance:**
- Translation: ~30-60 seconds (32 API calls)
- Total processing: ~90-120 seconds for 1-hour video

### Model Loading Issues

#### "Model file not found"
**Error:**
```
ERROR - Model file not found: models/Phi-3-mini-4k-instruct-q4.gguf
```

**Solutions:**

1. **Download manually:**
   ```bash
   mkdir -p models
   cd models
   # Download from HuggingFace
   wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
   ```

2. **Auto-download (requires huggingface_hub):**
   ```bash
   poetry run pip install huggingface-hub
   # Model will download on first use
   ```

3. **Check file permissions:**
   ```bash
   ls -la models/Phi-3-mini-4k-instruct-q4.gguf
   # Should show readable file
   ```

#### "CUDA/GPU errors despite CPU-only setup"
**Error:**
```
RuntimeError: CUDA out of memory
ERROR - torchvision::nms does not exist
```

**Solution:** Clean GPU packages and reinstall CPU-only:
```bash
# Remove GPU versions
poetry run pip uninstall torch torchvision torchaudio nvidia-cuda* nvidia-cublas* -y

# Install CPU-only versions
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
poetry run pip install sentence-transformers chromadb

# Restart application
poetry run python src/web_app.py
```

### Title Quality Issues

#### "Titles contain fragments like 'Look accompanying'"
**Symptoms:**
```
❌ "Look accompanying"
❌ "This event not be"
❌ "Come fear"
```

**Solutions:**

1. **Enable multi-stage LLM refinement:**
   ```bash
   # In .env
   USE_LLM_TITLES=true
   ```

2. **Check logs for LLM activity:**
   ```
   INFO - [llama.cpp] Refined main title at 723.0s: 'Student council' -> 'Student council politics'
   ```

3. **If LLM fails, falls back to heuristics:**
   ```bash
   # Disable LLM for faster processing
   USE_LLM_TITLES=false
   ```

#### "All titles are 'Section'"
**Cause:** Title validation is too aggressive

**Solution:** Check the title validation logic in `web_app.py`:
```python
def _is_valid_title(title: str) -> bool:
    # Should allow proper German titles
    # Check for: minimum length, has letters, not all digits
```

### Performance Issues

#### "Processing takes too long"
**Expected times:**
- 10 min video: ~30 seconds
- 30 min video: ~50 seconds
- 60 min video: ~90 seconds

**If slower:**
1. **Check CPU cores:** `nproc` (more cores = faster)
2. **Disable LLM refinement:** `USE_LLM_TITLES=false`
3. **Use smaller model:** Consider Phi-3-mini-4k-instruct (2.2GB) instead of larger models
4. **Check memory:** `free -h` (need ~4GB free RAM)

#### "High memory usage (>8GB)"
**Solutions:**
- Close other applications
- Use 4-bit quantized model (already default)
- Process shorter videos
- Add swap space: `sudo fallocate -l 4G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`

### Web Interface Issues

#### "Port 5000 already in use"
**Solution:**
```bash
# Find process
lsof -ti:5000

# Kill it
kill -9 $(lsof -ti:5000)

# Or use different port
FLASK_RUN_PORT=5001 poetry run python src/web_app.py
```

#### "Module not found" errors
**Error:**
```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solution:**
```bash
# Install missing packages
poetry run pip install sentence-transformers chromadb
poetry run pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### RAG System Issues

#### "ChromaDB errors"
**Symptoms:**
```
ERROR - ChromaDB collection error
```

**Solutions:**
1. **Clear ChromaDB cache:**
   ```bash
   rm -rf .chromadb/
   ```

2. **Check disk space:**
   ```bash
   df -h  # Need ~500MB free
   ```

3. **Reinstall ChromaDB:**
   ```bash
   poetry run pip uninstall chromadb -y
   poetry run pip install chromadb
   ```

#### "Vector search fails"
**Symptoms:**
```
ERROR - similarity_search failed
```

**Solutions:**
1. **Disable RAG for short videos:**
   ```bash
   USE_RAG=never
   ```

2. **Check embeddings model:**
   ```bash
   python -c "import sentence_transformers; print('OK')"
   ```

### Configuration Issues

#### "Environment variables not working"
**Symptoms:** Settings in `.env` are ignored

**Solutions:**
1. **Check `.env` location:**
   ```bash
   ls -la .env  # Should be in project root
   ```

2. **Verify variable names:**
   ```bash
   cat .env | grep -v "^#" | grep -v "^$"
   # Should show: VARIABLE_NAME=value
   ```

3. **Restart application:**
   ```bash
   # Flask needs restart for .env changes
   pkill -f "python src/web_app.py"
   poetry run python src/web_app.py
   ```

4. **Check variable loading:**
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('USE_TRANSLATION'))"
   ```

### Network Issues

#### "YouTube API blocked"
**Error:**
```
ERROR - Transcript extraction failed
```

**Solutions:**
1. **Check video availability:**
   - Video exists and is public
   - Not region-locked
   - Has transcripts available

2. **Try different video:**
   ```bash
   # Test with known working video
   echo "cZ9PHPta9v0"  # German video with transcripts
   ```

3. **Check network connectivity:**
   ```bash
   curl -I https://www.youtube.com/watch?v=cZ9PHPta9v0
   ```

#### "DeepL API connection failed"
**Symptoms:**
```
ERROR - DeepL API request failed
```

**Solutions:**
1. **Check API key:**
   ```bash
   curl "https://api-free.deepl.com/v2/usage?auth_key=$DEEPL_API_KEY"
   ```

2. **Verify endpoint:**
   ```bash
   # Free tier vs Pro tier
   curl -X POST "https://api-free.deepl.com/v2/translate" \
     -d "auth_key=$DEEPL_API_KEY" \
     -d "text=Hello" \
     -d "target_lang=DE"
   ```

3. **Check rate limits:**
   - Free tier: 500,000 characters/month
   - Pro tier: Higher limits

### File System Issues

#### "Permission denied"
**Error:**
```
ERROR - Permission denied: models/
```

**Solution:**
```bash
# Fix permissions
chmod -R 755 models/
chmod 644 models/*.gguf
```

#### "Disk space full"
**Error:**
```
ERROR - No space left on device
```

**Solutions:**
1. **Check disk usage:**
   ```bash
   df -h
   du -sh models/ .chromadb/
   ```

2. **Clean up:**
   ```bash
   # Remove old ChromaDB caches
   rm -rf .chromadb/

   # Remove temporary files
   find . -name "*.tmp" -delete
   ```

3. **Free space:**
   ```bash
   # Clear package cache
   poetry cache clear --all .
   pip cache purge
   ```

## 🐛 Reporting Issues

When reporting bugs, please include:

1. **Full error message and traceback**
2. **Your `.env` file (without API keys)**
3. **System information:**
   ```bash
   python --version
   pip list | grep -E "(torch|sentence|chroma|llama)"
   uname -a
   free -h
   ```
4. **Video ID that causes the issue**
5. **Full application logs**

## 🚀 Performance Tuning

### For Speed
```bash
# .env settings for maximum speed
USE_LLM_TITLES=false      # Skip LLM refinement
USE_TRANSLATION=false     # Skip translation
USE_RAG=never            # Skip vector search
LLM_TEMPERATURE=0.0      # Maximum determinism
```

### For Quality
```bash
# .env settings for maximum quality
USE_LLM_TITLES=true       # Enable LLM refinement
USE_TRANSLATION=true      # Enable translation
USE_RAG=always           # Always use RAG
LLM_TEMPERATURE=0.05     # Balanced creativity
DEEPL_API_KEY=your_key   # Use DeepL for translation
```

### For Reliability
```bash
# .env settings for maximum reliability
USE_LLM_TITLES=false      # Fallback to heuristics
USE_RAG=auto             # Auto-detect when needed
LOG_LEVEL=DEBUG          # Detailed logging
LOG_TO_FILE=true         # Persistent logs
```

## 📞 Getting Help

1. **Check this guide first** - Most issues are covered here
2. **Run the health check** - Identifies common problems
3. **Check logs** - Look for error patterns
4. **Try different settings** - Isolate the issue
5. **Report with details** - Include all diagnostic information

**Still stuck?** Open an issue on GitHub with the diagnostic information above.
