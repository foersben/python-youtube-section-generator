"""Diagnostic: checks config and attempts to resolve the local model path.
Prints the model candidates tried by LocalLLMProvider if not found.
"""
import os
from pathlib import Path
from src.core.config import config

print('PROJECT_ROOT:', config.project_root)
print('CONFIG LOCAL_MODEL_PATH:', config.local_model_path)
print('CONFIG local_model_exists:', config.local_model_path.exists())
print('ENV LOCAL_MODEL_PATH:', os.getenv('LOCAL_MODEL_PATH'))
print('CWD:', Path.cwd())
print('models/ listing:', list(Path('models').glob('*')) if Path('models').exists() else 'models dir not found')

try:
    from src.core.llm.local_provider import LocalLLMProvider
    p = LocalLLMProvider(config.local_model_path)
    print('LocalLLMProvider resolved model to:', p.model_path)
except Exception as e:
    print('ERROR:', type(e).__name__, e)

