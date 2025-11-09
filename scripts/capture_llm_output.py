"""Simple test to capture LLM raw output."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

print("Loading model...")
from src.core.adapters.local_llm_client import LocalLLMClient
client = LocalLLMClient()

# Build a simple prompt
transcript = [
    {"start": 0.0, "text": "Welcome to Python tutorial", "duration": 2.0},
    {"start": 2.0, "text": "Today we learn functions", "duration": 2.0}
]

prompt = client._build_section_prompt(transcript, num_sections=2)

print("\n" + "="*70)
print("PROMPT:")
print("="*70)
print(prompt)
print()

print("="*70)
print("GENERATING...")
print("="*70)

response = client._generate(prompt, max_new_tokens=512)

print("\n" + "="*70)
print("RAW RESPONSE:")
print("="*70)
print(response)
print()

# Save to file
output_file = Path(__file__).parent.parent / "docs" / "llm_debug_output.txt"
with open(output_file, "w") as f:
    f.write("PROMPT:\n")
    f.write("="*70 + "\n")
    f.write(prompt)
    f.write("\n\n")
    f.write("RESPONSE:\n")
    f.write("="*70 + "\n")
    f.write(response)

print(f"Output saved to: {output_file}")
