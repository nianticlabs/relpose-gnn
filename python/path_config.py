import os
from pathlib import Path

PATH_PROJECT = Path(__file__).parent.parent
PATH_PYTHON = Path(__file__).parent

if 'TORCH_HOME' not in os.environ:
    os.environ['TORCH_HOME'] = str(PATH_PROJECT / 'models')
