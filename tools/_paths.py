"""Shared path bootstrap for tool scripts.

Importing this first makes a tool runnable from any CWD: it puts the repo root on
sys.path (so `import cg` and `from agents import ...` resolve) and exposes DATA.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DECKS_DIR = DATA / "decks"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
