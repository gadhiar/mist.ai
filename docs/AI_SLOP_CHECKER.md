# AI Slop Checker Documentation

## Overview

The AI Slop Checker is a Python script that detects and removes "AI slop" patterns from the codebase. AI slop refers to common patterns in AI-generated content that indicate low-quality output or violate project style guidelines.

**Location:** `scripts/check_ai_slop.py`

**Purpose:** Enforce the NO EMOJIS rule and detect other AI-generated content patterns that don't match project standards.

---

## What It Detects

### Critical Issues (Must Fix)

These patterns will cause the script to exit with error code 1:

1. **emoji** - Unicode emojis (U+1F300-U+1F9FF range)
   - Example: 🎯, 🚀, 💡, 🔥
   - Auto-fixable: Yes (removes them)

2. **emoji_symbols** - Common emoji-like unicode symbols
   - Example: ✓, ✅, ❌, ⚠️, 📝
   - Auto-fixable: Yes (removes them)

3. **arrow_symbols** - Unicode arrow characters
   - Example: →, ←, ⇒, ➜
   - Auto-fixable: Yes (replaces with `->`)

### Warning Issues (Consider Fixing)

These are flagged but don't cause script failure:

4. **superlatives** - Hyperbolic adjectives typical of AI
   - Example: amazing, awesome, fantastic, incredible, wonderful
   - Auto-fixable: No (requires manual review)

5. **hype_words** - Over-hyped technical marketing terms
   - Example: seamless, cutting-edge, world-class, game-changing
   - Auto-fixable: No (sometimes legitimate use)
   - Note: "excellent", "innovative", "revolutionary" removed - sometimes valid

### Info Issues (Optional)

6. **filler_phrases** - Common AI filler language
   - Example: "let's dive in", "first and foremost", "moving forward"
   - Auto-fixable: No

7. **exclamation_spam** - Excessive exclamation marks
   - Example: `!!!`, `!!!!` (but `!!` is allowed)
   - Auto-fixable: Yes (reduces to single `!`)

---

## Installation

The script is already included in the repository. Ensure you have Python 3.11+ installed:

```bash
# No installation needed - uses only standard library
python scripts/check_ai_slop.py --help
```

---

## Usage

### Basic Commands

**Check entire codebase:**
```bash
python scripts/check_ai_slop.py
```

**Auto-fix fixable issues:**
```bash
python scripts/check_ai_slop.py --fix
```

**Check specific directory:**
```bash
python scripts/check_ai_slop.py --path backend/
```

**Only check for emojis (fast):**
```bash
python scripts/check_ai_slop.py --critical-only
```

### Command-Line Options

```
--fix
    Auto-fix all fixable patterns (emojis, arrows, exclamation spam)
    Modifies files in-place

--path PATH
    Path to check (default: current directory)
    Can be a file or directory

--format {text,markdown,json}
    Output format (default: text)
    - text: Human-readable colored output
    - markdown: Ultra-compact LLM-friendly format
    - json: Machine-readable structured data

--output FILE
    Write report to file instead of stdout
    Useful for saving reports or feeding to other tools

--critical-only
    Only check critical issues (emojis and unicode symbols)
    Faster, focuses on must-fix items
    Disables warning and info patterns

--incremental
    Only check files changed in git (git diff HEAD)
    Fast pre-commit workflow
    Exits cleanly if no changed files

--parallel
    Process files in parallel (4 workers)
    Faster for large codebases (>10 files)

--no-color
    Disable ANSI color codes in text output
    Useful for CI/CD or piping to files
```

---

## Output Formats

### Text Format (Default)

Human-readable output with colored severity indicators:

```
Checking 50 files for AI slop patterns...

backend/server.py:
  [CRITICAL] emoji: 2 occurrence(s)
    Line 45: ...🎯...
    Line 120: ...✅...

docs/README.md:
  [WARNING] superlatives: 1 occurrence(s)
    Line 34: ...amazing...

======================================================================
SUMMARY
======================================================================
Total files checked: 50
Files with issues: 2
Total issues found: 3

Issues by pattern:
  [CRITICAL] emoji: 2 (fixable)
  [WARNING] superlatives: 1 (manual fix required)

Run with --fix to automatically fix fixable patterns
```

### Markdown Format (LLM-Optimized)

Ultra-compact format designed for LLM consumption with minimal tokens:

```markdown
# AI Slop Report
Scanned 50 files: 3 issues (2c,1w)

## Issues
[C+] emoji: 2 @backend/server.py:45,120
[W] superlatives: 1 @docs/README.md:34
```

**Format Legend:**
- `[C+]` = Critical severity, auto-fixable (`+` means fixable)
- `[W]` = Warning severity, not auto-fixable
- `[I]` = Info severity
- `2c,1w,3i` = Issue count by severity (critical, warning, info)
- `@file:12,34+5` = Issues at lines 12, 34, and 5 more lines

**Token Efficiency:** 80-85% fewer tokens than verbose format

### JSON Format (Programmatic)

Structured data for scripts and automation:

```json
{
  "summary": {
    "files_checked": 50,
    "files_with_issues": 2,
    "total_issues": 3,
    "total_fixes": 0
  },
  "pattern_counts": {
    "emoji": 2,
    "superlatives": 1
  },
  "files": {
    "backend/server.py": {
      "emoji": 2
    },
    "docs/README.md": {
      "superlatives": 1
    }
  }
}
```

---

## Common Workflows

### Pre-Commit Check (Fast)

Check only changed files for critical issues:

```bash
python scripts/check_ai_slop.py --incremental --critical-only
```

Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
python scripts/check_ai_slop.py --incremental --critical-only
if [ $? -ne 0 ]; then
    echo "Critical AI slop detected. Run: python scripts/check_ai_slop.py --fix"
    exit 1
fi
```

### Auto-Fix Workflow

Automatically fix all fixable issues:

```bash
# Fix and show what was changed
python scripts/check_ai_slop.py --fix

# Review changes
git diff

# Commit if satisfied
git add -A
git commit -m "fix: remove emojis and unicode symbols"
```

### LLM Code Review

Generate compact report for AI assistant:

```bash
# Generate report
python scripts/check_ai_slop.py --critical-only --format markdown --output ai-slop.md

# Feed to LLM
cat ai-slop.md
# "Review this AI slop report and suggest fixes..."
```

### CI/CD Integration

In GitHub Actions or similar:

```yaml
- name: Check for AI slop
  run: |
    python scripts/check_ai_slop.py --critical-only --no-color
  continue-on-error: false
```

### Large Codebase Scan

For repositories with hundreds of files:

```bash
# Parallel processing for speed
python scripts/check_ai_slop.py --parallel --format json --output slop-report.json

# Review critical issues only
python scripts/check_ai_slop.py --critical-only --format markdown --output critical.md
```

### Gradual Cleanup

Fix critical issues first, then address warnings:

```bash
# Step 1: Fix critical (emojis)
python scripts/check_ai_slop.py --critical-only --fix

# Step 2: Review warnings
python scripts/check_ai_slop.py --format markdown --output warnings.md

# Step 3: Manual review of superlatives and hype words
```

---

## Exit Codes

The script uses exit codes to indicate results:

- **0** - No critical issues found (success)
- **1** - Critical issues detected (emojis or unicode symbols)

Warning and info issues do NOT cause non-zero exit codes.

Use in scripts:
```bash
python scripts/check_ai_slop.py --critical-only
if [ $? -eq 0 ]; then
    echo "Clean!"
else
    echo "Critical issues found"
    exit 1
fi
```

---

## File Skipping

The script automatically skips:

- `.git/` directory
- `venv/`, `node_modules/` directories
- `__pycache__/`, build artifacts
- Binary files (`.jpg`, `.png`, `.wav`, etc.)
- `dependencies/` directory
- Lock files (`package-lock.json`, etc.)

---

## Performance

**Typical performance:**
- Small codebase (<50 files): <1 second
- Medium codebase (100-500 files): 1-3 seconds
- Large codebase (1000+ files): 5-10 seconds with `--parallel`

**Optimization tips:**
- Use `--incremental` for pre-commit checks
- Use `--critical-only` when speed is important
- Use `--parallel` for large repositories
- Use `--path backend/` to check specific directories

---

## Integration with Pre-Commit Framework

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: check-ai-slop
      name: Check for AI slop patterns
      entry: python scripts/check_ai_slop.py
      language: system
      pass_filenames: false
      args: [--critical-only]
      stages: [commit]
```

This runs automatically on every commit.

---

## Customization

### Adjust Pattern Strictness

Edit `scripts/check_ai_slop.py` and modify the `PATTERNS` list:

**Remove a pattern entirely:**
```python
# Comment out or remove unwanted patterns
PATTERNS = [
    # ... other patterns ...
    # SlopPattern(name="hype_words", ...), # Disabled - too strict
]
```

**Change severity:**
```python
SlopPattern(
    name="superlatives",
    severity="info",  # Changed from "warning" to "info"
    # ...
)
```

**Add custom patterns:**
```python
PATTERNS.append(
    SlopPattern(
        name="custom_pattern",
        pattern=re.compile(r'\bmy_forbidden_word\b', re.IGNORECASE),
        severity="warning",
        fixable=False
    )
)
```

### Add File Skip Patterns

Edit the `SKIP_PATTERNS` list:

```python
SKIP_PATTERNS = [
    # ... existing patterns ...
    r'my_special_dir/',
    r'\.generated\.py$',
]
```

---

## Troubleshooting

### "No files to check"

**Cause:** All files in the path are being skipped

**Solutions:**
- Check if you're in the right directory
- Verify file extensions are supported (`.py`, `.md`, `.dart`, etc.)
- Check if files are in excluded directories (venv, node_modules, etc.)

### "git diff" error in incremental mode

**Cause:** Not in a git repository or git not installed

**Solution:** Don't use `--incremental` flag, or ensure you're in a git repo

### Script is slow

**Solutions:**
- Use `--parallel` flag
- Use `--critical-only` to skip warnings
- Use `--incremental` to check only changed files
- Use `--path` to check specific directories

### False positives

**For superlatives/hype words:**
These are warnings, not errors. Review manually and decide if they're appropriate in context.

**To disable entirely:**
```bash
# Only check critical issues
python scripts/check_ai_slop.py --critical-only
```

---

## Examples

### Example 1: Daily Development Workflow

```bash
# Morning: Check what changed
python scripts/check_ai_slop.py --incremental

# Fix issues before committing
python scripts/check_ai_slop.py --fix

# Verify fix worked
python scripts/check_ai_slop.py --critical-only
```

### Example 2: Code Review Preparation

```bash
# Generate report for reviewer
python scripts/check_ai_slop.py --format markdown --output slop-report.md

# Attach slop-report.md to PR description
```

### Example 3: Cleaning Up Old Code

```bash
# Scan backend directory
python scripts/check_ai_slop.py --path backend/ --format markdown

# Fix all auto-fixable issues
python scripts/check_ai_slop.py --path backend/ --fix

# Review remaining warnings
python scripts/check_ai_slop.py --path backend/ --format text | grep WARNING
```

### Example 4: CI/CD Pipeline

```bash
# In CI script
python scripts/check_ai_slop.py --critical-only --no-color
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "AI slop detected. Please remove emojis and unicode symbols."
    python scripts/check_ai_slop.py --format json --output slop-report.json
    # Upload slop-report.json as artifact
    exit 1
fi
```

---

## Best Practices

1. **Run on every commit** - Use pre-commit hooks or CI/CD
2. **Fix critical issues immediately** - Emojis are never acceptable
3. **Review warnings manually** - Context matters for superlatives
4. **Use incremental mode** - Faster for daily development
5. **Generate reports for review** - Markdown format for PRs
6. **Don't over-fix** - Some "hype words" are legitimate technical terms

---

## FAQs

**Q: Why can't I use emojis?**
A: This is a project rule documented in CLAUDE.md. Emojis are decorative and don't render consistently across terminals. Use text alternatives like `[SUCCESS]` instead.

**Q: Is "excellent" really AI slop?**
A: No - it was removed from the hype words list. The script focuses on clearly problematic patterns.

**Q: Can I use `!!` for emphasis?**
A: Yes, only `!!!` or more triggers the warning. Double exclamation is acceptable.

**Q: What if I legitimately need to use an arrow symbol?**
A: Use `->` instead of `→`. The script auto-fixes this.

**Q: How do I disable warnings for a specific file?**
A: Add the file pattern to `SKIP_PATTERNS` in the script, or use `--critical-only` to skip all warnings.

**Q: Does this work on Windows?**
A: Yes, fully compatible with Windows, macOS, and Linux.

---

## Related Documentation

- **CLAUDE.md** - Full AI integration guidelines (includes NO EMOJIS rule)
- **CONTRIBUTING.md** - Code quality standards
- **.pre-commit-config.yaml** - Pre-commit hook configuration
- **pyproject.toml** - Python linting and formatting rules

---

**Last Updated:** 2025-02-03
