"""Entry point for `python -m scripts.voice_data`.

The call is guarded so that importing the module (the repository's import-walk tests import
every module under `scripts/`) parses no argv and exits nothing.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
