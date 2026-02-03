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

- [CRITICAL] Emojis and unicode symbols (auto-fixed)
- [CHECK] Python formatting (Black)
- [CHECK] Python linting (Ruff)
- [CHECK] Flutter formatting
- [CHECK] Flutter analysis
- [CHECK] YAML/JSON validation
- [CHECK] Trailing whitespace
- [CHECK] Secret detection

---

## Common commands

```bash
# Manual run
pre-commit run --all-files

# Fix AI slop (emojis)
python scripts/check_ai_slop.py --fix

# Fix Python formatting
black backend/

# Fix Flutter formatting
cd mist_desktop && flutter format lib/

# Skip hooks (emergency only)
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
