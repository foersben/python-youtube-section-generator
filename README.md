# YouTube Transcript to Section Timestamps

Automatically generate YouTube-style section timestamps from video transcripts using AI.

![Example Output](https://via.placeholder.com/800x400.png?text=YouTube+Section+Timestamps+Example)

## Features

- 🎬 Extract YouTube transcripts (including auto-generated ones)
- 🤖 Generate section timestamps using Google Gemini AI
- ⏱️ Convert timestamps to YouTube-ready format
- 🌐 Support for transcript translation
- 📁 Save outputs as JSON and text files

## Requirements

- Python 3.9+
- Poetry (for dependency management)
- Google API Key (for Gemini AI)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/foersben/pythonyoutubetranscript.git
cd pythonyoutubetranscript
```

2. **Install dependencies with Poetry:**
```bash
poetry install
```

3. **Set up environment variables:**
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

## Configuration

Edit `src/main.py` to configure:
```python
# Configuration
VIDEO_ID = "your_video_id_here"  # YouTube video ID (11 characters)
TRANSCRIPT_FILE = "transcript.json"  # Transcript output file
SECTIONS_FILE = "sections.json"    # Raw sections output file
```

## Usage

Run the script with Poetry:
```bash
poetry run python src/main.py
```

### Output Files

| File | Description |
|------|-------------|
| `transcript.json` | Full transcript in JSON format |
| `sections.json` | AI-generated sections with timestamps |
| `youtube_sections.txt` | YouTube-ready timestamp format |

### Command Line Options
Modify the main function in `src/main.py` to:
- Enable translation: `translate_to="en"`
- Adjust section count: `section_count_range=(8, 12)`
- Customize title length: `title_length_range=(4, 6)`

## Example Workflow

1. Get a YouTube video ID (e.g., from URL: `https://youtu.be/kXhCEyix180`)
2. Update `VIDEO_ID` in `src/main.py`
3. Run the script:
```bash
Fetching transcript...

Transcript Details:
- Video ID: kXhCEyix180
- Language: English (en)
- Generated: Yes
Transcript saved to transcript.json

Generating section timestamps...
Section timestamps saved to sections.json

YouTube-ready Section Timestamps:

1. 00:00 Introduction to AI
2. 01:25 Machine Learning Basics
3. 04:10 Deep Learning Explained
4. 07:45 Real-world Applications
5. 12:30 Future Predictions

YouTube-formatted sections saved to youtube_sections.txt
```

## Advanced Usage

### Using Different AI Models
Modify the model in `create_section_timestamps()`:
```python
response = client.models.generate_content(
    model="gemini-1.5-pro",  # Alternative model
    # ...
)
```

### Handling Long Videos
For videos >30 minutes:
```python
sections = create_section_timestamps(
    transcript=transcript,
    section_count_range=(20, 30),  # More sections for long videos
    title_length_range=(5, 8)      # Longer titles
)
```

## Troubleshooting

**Error: "GOOGLE_API_KEY not found"**
- Verify `.env` file exists with valid API key
- Ensure file is in project root directory

**Error: "Failed to extract JSON from response"**
- AI returned malformed response
- Try reducing section count or title length
- Check Google AI quota limits

**Error: Transcript not available**
- Some videos don't have transcripts
- Try a different video ID

## Project Structure

```
.
├── .env                   # Environment variables (not versioned)
├── .gitignore             # Ignores virtualenv and output files
├── poetry.lock            # Dependency lockfile
├── pyproject.toml         # Poetry configuration
├── README.md              # This file
└── src/
    ├── main.py            # Main application script
    └── ...                # Other source files
```

## Dependencies

- [Poetry](https://python-poetry.org/) - Dependency management
- [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/) - Transcript extraction
- [google-generativeai](https://pypi.org/project/google-generativeai/) - Gemini AI integration
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment management

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

MIT License - see [LICENSE](LICENSE) for details
```

## Important Setup Notes

1. Create a `.gitignore` file with:
```gitignore
# .gitignore
.venv/
.env
*.json
*.txt
__pycache__/
```

2. Add a LICENSE file (e.g., MIT License)

3. Recommended project structure:
```
pythonyoutubetranscript/
├── .env
├── .gitignore
├── LICENSE
├── poetry.lock
├── pyproject.toml
├── README.md
└── src/
    ├── __init__.py
    └── main.py
```

4. For first-time Poetry users:
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Initialize virtual environment
poetry shell

# Install dependencies
poetry install
```

This README provides:
- Clear installation instructions
- Usage examples
- Configuration guidance
- Troubleshooting tips
- Project structure overview
- Contribution guidelines
- License information

The setup ensures:
- API keys stay secure (not committed to Git)
- Virtual environments are self-contained
- Output files are automatically ignored
- Dependencies are explicitly managed
