import os
import torch
from dotenv import load_dotenv
from pathlib import Path
from fastai.vision.all import PILImage

repo_root = Path(__file__).parent.parent

def check_cuda():
    if torch.cuda.is_available():
        print('CUDA is available via:')
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))
    else:
        print('CUDA is not available.')
        print(f'Training model will be much slower.')

def load_env():
    here = Path(__file__)
    repo_root = here.parent.parent
    env_path = repo_root / '.env'
    load_dotenv(dotenv_path=env_path)
    print(f'Loaded environment variables.')

def get_data_path():
    return Path(repo_root / '.data')
