# Transcript Refinement with AI/LLM

## Overview

The YouTube Transcript Generator now includes **AI-powered transcript refinement** that intelligently cleans and corrects YouTube transcripts before processing. This addresses common issues with auto-generated transcripts:

- **Filler words** (uh, um, hmm, äh, ähm, like, you know, etc.)
- **Phonetic errors** and misspellings based on context
- **Grammar issues** and sentence structure
- **Repeated words** or phrases

Unlike simple regex-based filtering, the LLM refinement uses **context-aware AI** to preserve meaning while cleaning the text.

## Features

### Context-Aware Processing
- Processes transcripts in **batches** with surrounding context
- Preserves technical terms and domain-specific vocabulary
- Maintains original timestamps and duration information
- Uses hierarchical processing for better understanding

### Intelligent Cleaning
- Removes filler words **only when they add no semantic value**
- Corrects phonetic errors based on full context
- Fixes grammar while preserving the speaker's intent
- Maintains natural flow and readability

### Fallback Safety
- Falls back to original text if refinement fails
- Graceful error handling ensures processing continues
- Logs warnings for debugging without blocking execution

## Configuration

### Environment Variable
Add to your `.env` file:

```env
# Enable/disable transcript refinement
REFINE_TRANSCRIPTS=true  # Default: true
```

### Programmatic Control
You can also control refinement per-call:

```python
from src.core.transcript.extractor import extract_transcript

# Enable refinement (uses .env setting if not specified)
transcript = extract_transcript(video_id="abc123", refine_with_llm=True)

# Disable refinement for specific call
transcript = extract_transcript(video_id="abc123", refine_with_llm=False)
```

## How It Works

### 1. Extraction
YouTube transcript is extracted using `youtube-transcript-api`:
```
[{"text": "um so like I think that...", "start": 0.0, "duration": 3.5}, ...]
```

### 2. Batch Processing
Segments are grouped into batches (default: 5 segments) with context:
```
Context Before: [Previous 2 segments]
Batch: [5 segments to refine]
Context After: [Next 2 segments]
```

### 3. LLM Refinement
Each batch is sent to the LLM with a specialized prompt:

```
You are a transcript refinement expert. Clean and correct the following...

Instructions:
1. Remove filler words ONLY when contextually appropriate
2. Correct phonetic/spelling errors using the full context
3. Fix grammar while preserving the speaker's intent
...

Text to refine:
[batch text]

Refined text:
```

### 4. Re-segmentation
The refined text is split back into segments matching original timestamps:
```
[{"text": "I think that...", "start": 0.0, "duration": 3.5}, ...]
```

## Architecture

```
TranscriptRefinementService
├── refine_transcript_batch()      # Main entry point
│   ├── Processes in batches of 5 segments
│   └── Builds context from surrounding segments
│
├── _refine_batch_with_context()   # Refines a batch with LLM
│   ├── Combines batch text
│   ├── Builds full context
│   └── Calls LLM provider
│
└── _split_refined_text()          # Splits refined text back
    ├── Preserves original timestamps
    └── Distributes text proportionally
```

## LLM Provider Integration

The refinement service uses the existing **LLM Factory** pattern:

- **Local LLM** (Phi-3-mini-4k-instruct-q4.gguf) - Default when `USE_LOCAL_LLM=true`
- **Google Gemini API** - When `GOOGLE_API_KEY` is set and `USE_LOCAL_LLM=false`
- **Fallback** - Returns original text on failure

### Temperature Setting
Uses **low temperature (0.3)** for deterministic, consistent corrections:
- Reduces creativity/randomness
- Ensures predictable cleaning
- Preserves original meaning

## Example Usage

### CLI
```bash
# With refinement (default)
python src/main.py --video-id abc123

# Without refinement
REFINE_TRANSCRIPTS=false python src/main.py --video-id abc123
```

### Web App
The Flask web app automatically applies refinement based on `.env` settings:
```bash
# Start web app
python src/web_app.py

# Refinement is applied automatically during transcript extraction
```

### Programmatic
```python
from src.core.transcript.extractor import extract_transcript
from src.core.services.transcript_refinement import TranscriptRefinementService

# Method 1: Automatic refinement during extraction
transcript = extract_transcript(
    video_id="abc123",
    refine_with_llm=True
)

# Method 2: Manual refinement of existing transcript
service = TranscriptRefinementService()
refined = service.refine_transcript_batch(
    segments=transcript,
    batch_size=5
)
```

## Performance Considerations

### Batch Size
- **Default: 5 segments** - Balances context and speed
- Larger batches = more context but slower processing
- Smaller batches = faster but less context

### Processing Time
- **Local LLM**: ~1-3 seconds per batch (CPU), faster on GPU
- **Gemini API**: ~0.5-1 second per batch (network latency)
- For a 30-minute video (~1000 segments): ~3-10 minutes total

### Cost (Gemini API)
- **Tokens per batch**: ~200-500 input, ~150-300 output
- **Cost per 1M tokens**: $0.07 input, $0.30 output (Gemini 1.5 Flash)
- **Estimated cost for 30-min video**: ~$0.02-$0.05

### Local LLM (Free)
- **No API costs** - Runs entirely locally
- Requires: 5-8GB RAM with 4-bit quantization
- Recommended: 16GB RAM for best performance

## Testing

Run the test suite:
```bash
# All refinement tests
poetry run pytest tests/test_transcript_refinement.py -v

# Specific test
poetry run pytest tests/test_transcript_refinement.py::test_refine_single_segment -v
```

### Test Coverage
- ✅ Service initialization
- ✅ Single segment refinement
- ✅ Batch refinement with context
- ✅ Timestamp preservation
- ✅ Text cleaning verification
- ✅ Failure handling and fallback
- ✅ Empty segment handling
- ✅ Context building verification

## Troubleshooting

### Issue: Refinement is slow
**Solution**: 
- Reduce batch size: modify `refine_transcript_batch(segments, batch_size=3)`
- Use Gemini API instead of local LLM
- Use GPU for local LLM processing

### Issue: Refinement removes important words
**Solution**:
- Check LLM temperature (should be low: 0.2-0.4)
- Review LLM provider logs for prompt/response
- Consider disabling for technical/specialized content

### Issue: Out of memory errors
**Solution**:
- Use 4-bit quantized model (Phi-3-mini-4k-instruct-q4.gguf)
- Reduce context window size
- Use Gemini API instead of local LLM

### Issue: Refinement not applied
**Solution**:
- Check `.env` setting: `REFINE_TRANSCRIPTS=true`
- Verify LLM provider is configured (local model or API key)
- Check logs for error messages

## Best Practices

1. **Enable for user-facing applications** - Improves readability significantly
2. **Disable for archival/research** - When original text must be preserved
3. **Use local LLM for privacy** - No data sent to external APIs
4. **Monitor LLM outputs** - Check logs to ensure quality
5. **Adjust batch size** - Balance context vs. speed for your use case

## Future Enhancements

Potential improvements:
- [ ] Language-specific filler word lists
- [ ] Confidence scoring for corrections
- [ ] User-configurable correction rules
- [ ] Integration with speaker diarization
- [ ] Advanced context windows (sliding/overlapping)
- [ ] Fine-tuned models for transcript refinement

## Related Documentation

- [Architecture Overview](ARCHITECTURE.md)
- [LLM Configuration](LLM_CONFIGURATION.md)
- [API Documentation](api/index.md)
- [Contributing Guidelines](CONTRIBUTING.md)
