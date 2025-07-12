# YouTube Transcript to Section Timestamps

Automatically generate YouTube-style section timestamps from video transcripts using AI. Available as both a command-line tool and web application.

## Features

- 🎬 Extract YouTube transcripts (including auto-generated ones)
- 🤖 Generate section timestamps using Google Gemini AI
- ⏱️ Convert timestamps to YouTube-ready format
- 🌐 Support for transcript translation
- 📁 Save outputs as JSON and text files
- 🌍 Web interface for easy browser access
- 🧩 Modular architecture for easy maintenance

## Requirements

- Python 3.9+
- Poetry (for dependency management)
- Google API Key (for Gemini AI)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/foersben/python-youtube-transcript.git
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

## Usage

### Command Line Interface (CLI)
Edit `src/main.py` to configure:
```python
# Configuration
VIDEO_ID = "your_video_id_here"  # YouTube video ID (11 characters)
TRANSCRIPT_FILE = "transcript.json"  # Transcript output file
SECTIONS_FILE = "sections.json"    # Raw sections output file
```

Run the script:
```bash
poetry run python src/main.py
```

### Web Interface

<img width="1622" height="1418" alt="image" src="https://github.com/user-attachments/assets/9044ffc6-f3f3-431f-bd22-d599b30022e3" />

Start the web server:
```bash
poetry run python src/web_app.py
```

Then open in your browser:
```
http://localhost:5000
```

### Output Files (CLI)

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

### CLI Example
1. Get a YouTube video ID (e.g., from URL: `https://youtu.be/kXhCEyix180`)
2. Update `VIDEO_ID` in `src/main.py`
3. Run the script:
```bash
Fetching transcript...

Transcript Details:
- Video ID: kXhCEyix180
- Language: German (auto-generated) (de)
- Generated: Yes
Transcript saved to transcript.json

Generating section timestamps...
Section timestamps saved to sections.json

YouTube-ready Section Timestamps:

1. 00:33 Willkommen: Das Thema Freiheit
2. 01:47 Projekt: Grundrechte und Justiz
3. 03:20 Aktualität, Aufzeichnung, Feedback
4. 03:48 Große Brocken: Freiheit, Gleichheit, Eigentum
5. 07:35 Kritik an Freiheit als Höchstwert
6. 08:45 Anekdote: Freiheit oder Sklaverei?
7. 13:20 Artikel 2 GG: Allgemeine Freiheit
8. 19:09 Analyse: Freie Entfaltung der Persönlichkeit
9. 22:12 Freiheit, Grenzen und Konflikte
10. 35:19 Elfes-Urteil: Staatliche Kontrolle der Freiheit
11. 45:52 Abwehrrecht: Illusorisch gegen Staat
12. 55:55 Reiten im Walde: Freiheitskollisionen
13. 62:30 Freiheit: Rücksichtslosigkeit und Konflikte
14. 84:15 Recht auf Leben und Körper
15. 98:55 Fluglärm-Urteil: Zumutbarer Körperschaden
16. 118:26 Fazit: Freiheit als Staatsinstrument
17. 121:55 Ideologische Darstellung von Freiheit
18. 140:00 Freiheit und Pandemie-Maßnahmen
19. 154:41 Nächste Themen & Support

YouTube-formatted sections saved to youtube_sections.txt
```

### Web Interface Example
1. Enter YouTube URL or video ID
2. Adjust settings (optional)
3. Click "Generate Sections"
4. Copy or download results

## Advanced Usage

### Using Different AI Models
Modify the model in `create_section_timestamps()` (in `src/core/sections.py`):
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

## Project Structure (Updated)
```
.
├── .env                   # Environment variables (not versioned)
├── .gitignore             # Ignores virtualenv and output files
├── LICENSE                # Project license
├── poetry.lock            # Dependency lockfile
├── pyproject.toml         # Poetry configuration
├── README.md              # This file
├── static/                # Web assets (CSS, JS)
│   └── style.css
├── src/
│   ├── __init__.py        # Package initialization
│   ├── main.py            # CLI application
│   ├── web_app.py         # Web application
│   ├── core/              # Core functionality
│   │   ├── __init__.py
│   │   ├── transcript.py  # Transcript handling
│   │   ├── sections.py    # Section generation
│   │   └── formatting.py  # Output formatting
│   ├── templates/         # HTML templates
│   │   └── index.html
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── file_io.py     # File operations
│       └── json_utils.py  # JSON handling
└── tests/                 # Unit tests (optional)
```

## Web Interface Features
- 🖥️ Simple browser-based interface
- 📱 Responsive design works on mobile devices
- ⚙️ Adjustable settings for section generation
- 📋 One-click copy to clipboard
- 💾 Download section timestamps as text file
- 🔄 Process multiple videos in one session

## Troubleshooting

**Error: "GOOGLE_API_KEY not found"**
- Verify `.env` file exists with valid API key
- Ensure file is in project root directory
- Restart application after changing .env

**Error: "Failed to extract JSON from response"**
- AI returned malformed response
- Try reducing section count or title length
- Check Google AI quota limits

**Error: Transcript not available**
- Some videos don't have transcripts
- Try a different video ID
- Ensure video has captions enabled

**Web interface not loading**
- Ensure port 5000 is available
- Check firewall settings
- Verify all dependencies are installed (`poetry install`)

## Deployment

For production deployment of the web interface:

*(only consider this, if you are completely nuts)*

1. **Use a production WSGI server:**
```bash
poetry add gunicorn
gunicorn -w 4 "src.web_app:app"
```

2. **Set environment variables:**
```bash
export FLASK_ENV=production
export GOOGLE_API_KEY=your_key_here
```

3. **Use a reverse proxy (Nginx example):**
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Dependencies

- [Poetry](https://python-poetry.org/) - Dependency management
- [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/) - Transcript extraction
- [google-generativeai](https://pypi.org/project/google-generativeai/) - Gemini AI integration
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment management
- [Flask](https://flask.palletsprojects.com/) - Web application framework

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

MIT License - see [LICENSE](LICENSE) for details
