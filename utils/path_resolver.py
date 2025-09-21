import logging, os
from pathlib import Path

def SYS_PATH():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return script_dir

def PARENT():
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    return parent_dir


