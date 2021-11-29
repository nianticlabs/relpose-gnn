import sys
from pathlib import Path

_p_sanet_relocal_demo = str(Path(__file__).parent)
if _p_sanet_relocal_demo not in sys.path:
    sys.path.insert(0, _p_sanet_relocal_demo)
