"""Enables `python -m scripts.dedup_calibration`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
