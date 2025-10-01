import logging, os
from pathlib import Path

def SYS_PATH():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return script_dir


def PARENT(levels: int = 1) -> Path:
    current_dir = Path.cwd()
    try:
        return current_dir.parents[levels - 1]
    except IndexError:
        raise ValueError(f"Cannot go up {levels} levels from {current_dir}")


