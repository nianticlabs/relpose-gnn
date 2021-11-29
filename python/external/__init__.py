import sys
from pathlib import Path

p_external = Path(__file__).parent
if str(p_external) not in sys.path:
    sys.path.insert(0, str(p_external))
