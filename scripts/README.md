# Scripts Directory

## check_ai_slop.py

AI Slop Detection and Removal Tool - Detects and removes emojis, unicode symbols, and other AI-generated patterns.

**Quick Start:**
```bash
# Check for emojis (fast)
python scripts/check_ai_slop.py --critical-only

# Auto-fix emojis and unicode symbols
python scripts/check_ai_slop.py --fix

# LLM-friendly markdown report
python scripts/check_ai_slop.py --format markdown --output slop-report.md
```

**Full Documentation:** [docs/AI_SLOP_CHECKER.md](../docs/AI_SLOP_CHECKER.md)

---

## install-git-hooks.sh / install-git-hooks.bat

Git hook installation scripts for setting up pre-commit and commit-msg hooks.

**Quick Start:**
```bash
# Linux/macOS
bash scripts/install-git-hooks.sh

# Windows
scripts\install-git-hooks.bat

# Or use pre-commit framework directly
pip install pre-commit
pre-commit install
```

**Full Documentation:** [docs/GIT_WORKFLOWS.md](../docs/GIT_WORKFLOWS.md)

---

## Future Scripts

Additional utility scripts will be documented here as they are added.
