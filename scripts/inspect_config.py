from src.core.config import config
import os
print('project_root=', config.project_root)
print('local_model_path=', config.local_model_path)
print('local_model_exists=', config.local_model_path.exists())
print('cwd=', os.getcwd())
print('env LOCAL_MODEL_PATH=', os.getenv('LOCAL_MODEL_PATH'))

