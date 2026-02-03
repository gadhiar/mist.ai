# Git Workflows and Code Quality Setup

This document explains how to set up and use git hooks and CI/CD workflows for automated code quality checks.

---

## Overview

MIST.AI uses two levels of code quality enforcement:

1. **Local Git Hooks** - Run on your machine before commits
2. **GitHub Actions CI/CD** - Run on GitHub when you push

Both levels check for:
- Emojis and unicode symbols (critical)
- Code formatting (Black for Python, flutter format for Dart)
- Linting issues (Ruff for Python, flutter analyze for Dart)

---

## Part 1: Local Git Hooks

### Installation

**Option A: Using pre-commit framework (Recommended)**

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Test installation
pre-commit run --all-files
```

**Option B: Using installation script**

```bash
# Linux/macOS
bash scripts/install-git-hooks.sh

# Windows
scripts\install-git-hooks.bat
```

**Option C: Manual installation**

Copy hook scripts from the installation script into `.git/hooks/` and make them executable.

---

### What Gets Checked

**Pre-commit hook (before commit):**
1. AI slop check (critical issues only - emojis/unicode)
2. Black formatting (Python)
3. Ruff linting (Python)
4. Flutter formatting (Dart)
5. Flutter analysis (Dart)
6. YAML/JSON validation
7. Trailing whitespace removal
8. Secret detection

**Commit-msg hook (after commit message):**
1. Check for emojis in commit message

---

### Using Git Hooks

**Normal workflow (hooks run automatically):**

```bash
# Stage changes
git add .

# Commit (hooks run automatically)
git commit -m "fix: remove emoji from docs"
```

**If hooks fail:**

```bash
# Hooks will show what failed and how to fix

# Fix the issues
python scripts/check_ai_slop.py --fix
black backend/
flutter format mist_desktop/lib/

# Try commit again
git commit -m "fix: remove emoji from docs"
```

**Skip hooks (emergency only):**

```bash
# Skip all hooks (not recommended!)
git commit --no-verify -m "emergency fix"
```

---

### Testing Hooks

**Test all hooks manually:**

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific hook
pre-commit run check-ai-slop

# Run on staged files only
pre-commit run
```

**Test AI slop checker directly:**

```bash
# Check for emojis
python scripts/check_ai_slop.py --critical-only

# Check changed files only
python scripts/check_ai_slop.py --incremental --critical-only
```

---

### Updating Hooks

**Update pre-commit framework:**

```bash
# Update to latest hook versions
pre-commit autoupdate

# Reinstall hooks
pre-commit install --install-hooks
```

**Modify hook behavior:**

Edit `.pre-commit-config.yaml` and change the `args` for any hook:

```yaml
- repo: local
  hooks:
    - id: check-ai-slop
      args: [--critical-only]  # Change behavior here
```

---

## Part 2: GitHub Actions CI/CD

### What Gets Checked

**Python Quality Workflow** (`.github/workflows/python-quality.yml`)

Runs on:
- Push to `main`, `feat/**`, `dev/**` branches
- Pull requests to `main`
- When Python files change

Checks:
1. Black formatting
2. Ruff linting
3. mypy type checking (non-blocking)
4. AI slop detection (critical only)

**Flutter Quality Workflow** (`.github/workflows/flutter-quality.yml`)

Runs on:
- Push to `main`, `feat/**`, `dev/**` branches
- Pull requests to `main`
- When Flutter files change

Checks:
1. Flutter formatting
2. Flutter analysis

---

### Viewing CI/CD Results

**On GitHub:**

1. Go to repository on GitHub
2. Click "Actions" tab
3. View workflow runs and logs

**On Pull Requests:**

- Checks appear at bottom of PR
- Green checkmark = passed
- Red X = failed (click for details)
- PR cannot merge until checks pass

---

### CI/CD Status Badges

Add to README.md (replace `YOUR_USERNAME` and `YOUR_REPO`):

```markdown
[![Python Quality](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Python%20Code%20Quality/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions)
[![Flutter Quality](https://github.com/YOUR_USERNAME/YOUR_REPO/workflows/Flutter%20Code%20Quality/badge.svg)](https://github.com/YOUR_USERNAME/YOUR_REPO/actions)
```

---

### Fixing CI/CD Failures

**Python formatting failed:**

```bash
# Format code
black backend/ scripts/

# Commit and push
git add -A
git commit -m "style: format Python code with black"
git push
```

**Ruff linting failed:**

```bash
# Fix auto-fixable issues
ruff check backend/ scripts/ --fix

# Review remaining issues
ruff check backend/ scripts/

# Commit fixes
git add -A
git commit -m "fix: resolve linting issues"
git push
```

**AI slop detected:**

```bash
# Auto-fix emojis
python scripts/check_ai_slop.py --fix

# Commit
git add -A
git commit -m "fix: remove emojis and unicode symbols"
git push
```

**Flutter analysis failed:**

```bash
# Format code
cd mist_desktop
flutter format lib/

# Check analysis
flutter analyze

# Fix issues reported
# ... make fixes ...

# Commit
git add -A
git commit -m "fix: resolve Flutter analysis issues"
git push
```

---

## Part 3: Workflow Examples

### Example 1: Daily Development

```bash
# Make changes
vim backend/server.py

# Stage changes
git add backend/server.py

# Commit (hooks run automatically)
git commit -m "feat: add new endpoint"

# If hooks fail, fix and retry
python scripts/check_ai_slop.py --fix
git add backend/server.py
git commit -m "feat: add new endpoint"

# Push (triggers CI/CD)
git push origin feat/my-feature
```

### Example 2: Creating a Pull Request

```bash
# Create feature branch
git checkout -b feat/my-feature

# Make changes
# ... edit files ...

# Run quality checks locally
pre-commit run --all-files

# Fix any issues
python scripts/check_ai_slop.py --fix
black backend/
flutter format mist_desktop/lib/

# Commit all changes
git add -A
git commit -m "feat: implement new feature"

# Push to GitHub (triggers CI/CD)
git push origin feat/my-feature

# Create PR on GitHub
# - CI/CD checks will run automatically
# - Wait for green checkmarks
# - Request review
```

### Example 3: Fixing Failed CI/CD

```bash
# CI/CD failed on GitHub

# Pull latest
git pull

# Run checks locally to reproduce
pre-commit run --all-files

# Fix issues
python scripts/check_ai_slop.py --fix
black backend/
ruff check backend/ --fix

# Commit fixes
git add -A
git commit -m "fix: resolve code quality issues"

# Push (CI/CD runs again)
git push

# Check GitHub Actions for green checkmarks
```

---

## Part 4: Configuration

### Disable Specific Checks

**Disable AI slop check in pre-commit:**

Edit `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    # Comment out to disable
    # - id: check-ai-slop
    #   ...
```

**Disable specific CI/CD workflow:**

Rename workflow file to disable:

```bash
mv .github/workflows/python-quality.yml .github/workflows/python-quality.yml.disabled
```

Or add condition to workflow:

```yaml
jobs:
  lint-and-format:
    if: false  # Disables this job
    runs-on: ubuntu-latest
```

---

### Adjust Check Strictness

**Make AI slop check less strict (local only):**

Edit `.pre-commit-config.yaml`:

```yaml
- id: check-ai-slop
  args: [--critical-only]  # Already set - only emojis
```

**Make Black less strict:**

Edit `pyproject.toml`:

```toml
[tool.black]
line-length = 120  # Change from 100 to 120
```

**Skip type checking in CI:**

Already configured as non-blocking (continue-on-error: true)

---

## Part 5: Troubleshooting

### Hooks don't run

**Check installation:**

```bash
# Check if hooks are installed
ls -la .git/hooks/

# Should see: pre-commit, commit-msg (without .sample extension)

# Reinstall
pre-commit install
```

**Check Python path:**

```bash
# Hooks need Python in PATH
which python  # Linux/macOS
where python  # Windows

# If not found, add Python to PATH
```

### Hooks run but fail

**Get detailed output:**

```bash
# Run manually for details
pre-commit run --all-files --verbose

# Run specific check
python scripts/check_ai_slop.py --critical-only
```

**Common issues:**

1. Python not in PATH
2. Dependencies not installed (pip install -r requirements.txt)
3. Files not staged (git add .)

### CI/CD fails but local passes

**Common causes:**

1. Pushed before running hooks locally
2. Different Python/Flutter versions
3. Didn't push all files

**Fix:**

```bash
# Run exact same checks as CI
pre-commit run --all-files

# If passes locally, push again
git push
```

### Want to skip hooks temporarily

**For one commit:**

```bash
git commit --no-verify -m "WIP: work in progress"
```

**Disable entirely (not recommended):**

```bash
# Uninstall hooks
pre-commit uninstall

# Or delete hook files
rm .git/hooks/pre-commit .git/hooks/commit-msg
```

**Re-enable:**

```bash
pre-commit install
```

---

## Part 6: Best Practices

1. **Install hooks immediately** after cloning repository
2. **Run `pre-commit run --all-files`** before creating PR
3. **Fix issues locally** before pushing (faster than CI/CD)
4. **Don't use `--no-verify`** unless emergency
5. **Keep hooks updated** with `pre-commit autoupdate`
6. **Test hooks after modifying** `.pre-commit-config.yaml`

---

## Quick Reference

```bash
# Installation
pip install pre-commit
pre-commit install

# Running checks
pre-commit run --all-files          # All hooks
pre-commit run check-ai-slop        # Specific hook
python scripts/check_ai_slop.py     # Direct script

# Fixing issues
python scripts/check_ai_slop.py --fix
black backend/
ruff check backend/ --fix
flutter format mist_desktop/lib/

# Skip hooks (emergency)
git commit --no-verify

# Update hooks
pre-commit autoupdate
```

---

## Related Documentation

- **AI_SLOP_CHECKER.md** - Full AI slop checker documentation
- **CONTRIBUTING.md** - Code quality standards
- **.pre-commit-config.yaml** - Hook configuration
- **pyproject.toml** - Python tool configuration

---

**Last Updated:** 2025-02-03
