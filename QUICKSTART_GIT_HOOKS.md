# Quick Start: Git Hooks Setup

Get pre-commit hooks running in under 2 minutes.

## Step 1: Install pre-commit

```bash
pip install pre-commit
```

## Step 2: Install hooks

```bash
pre-commit install
```

## Step 3: Test installation

```bash
pre-commit run --all-files
```

## Done!

Hooks will now run automatically on every commit.

---

## What gets checked?

All **16 hooks** run automatically on every commit:

**Python Quality (5 hooks):**
- Black - Code formatting (auto-fix)
- Ruff - Linting + auto-fix
- Mypy - Type checking (non-blocking)
- Bandit - Security scanning
- Codespell - Spell checking

**Dart/Flutter (2 hooks):**
- Dart format - Code formatting (auto-fix)
- Flutter analyze - Linting

**General Quality (9 hooks):**
- **AI Slop Checker** - Emoji/unicode detection (auto-fix, runs on changed files only)
- Trailing whitespace removal
- End-of-file fixer
- YAML/TOML/JSON validation
- Large file detection (>1MB)
- Merge conflict detection
- Private key detection
- Line ending normalization

---

## Common commands

```bash
# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run check-ai-slop --all-files

# Fix AI slop manually (all files)
python scripts/check_ai_slop.py --fix --critical-only

# Fix AI slop (changed files only - faster)
python scripts/check_ai_slop.py --incremental --fix

# Fix Python formatting
black backend/

# Fix Flutter formatting
cd mist_desktop && dart format .

# Skip hooks (emergency only - use sparingly!)
git commit --no-verify
```

---

## Need more details?

See [Git Workflows Documentation](docs/GIT_WORKFLOWS.md) for:
- CI/CD setup
- Troubleshooting
- Configuration options
- Best practices

---

## Installation script alternative

Instead of manual installation, use the installation script:

```bash
# Linux/macOS
bash scripts/install-git-hooks.sh

# Windows
scripts\install-git-hooks.bat
```

This installs hooks automatically and provides detailed status messages.
