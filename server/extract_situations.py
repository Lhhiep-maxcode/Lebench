"""Convenience entry point: python src/server/extract_situations.py --help."""

import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.situation_extraction.__main__ import main

if __name__ == '__main__':
    raise SystemExit(main())
