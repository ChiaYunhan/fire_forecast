"""pytest configuration - makes src/ discoverable"""
import sys
from pathlib import Path

# Add the project root to Python path so 'import src' works
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
