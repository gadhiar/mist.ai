#!/bin/bash
# Install git hooks for MIST.AI project
# This script sets up pre-commit hooks for code quality enforcement

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "Installing git hooks for MIST.AI..."
echo ""

# Check if we're in a git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Method 1: Install pre-commit framework (recommended)
echo "[1/3] Installing pre-commit framework hooks..."
if command -v pre-commit &> /dev/null; then
    cd "$PROJECT_ROOT"
    pre-commit install
    pre-commit install --hook-type commit-msg
    echo "  [SUCCESS] pre-commit hooks installed"
else
    echo "  [WARNING] pre-commit not found. Install with: pip install pre-commit"
    echo "  Falling back to manual hook installation..."

    # Method 2: Manual hook installation (fallback)
    echo ""
    echo "[2/3] Installing manual git hooks..."

    # Pre-commit hook
    cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook: Check for AI slop and code quality issues

echo "Running pre-commit checks..."

# Check for critical AI slop (emojis)
python scripts/check_ai_slop.py --incremental --critical-only --no-color
if [ $? -ne 0 ]; then
    echo ""
    echo "[BLOCKED] Critical AI slop detected (emojis or unicode symbols)"
    echo "Fix with: python scripts/check_ai_slop.py --fix"
    echo ""
    echo "To skip this check (not recommended): git commit --no-verify"
    exit 1
fi

echo "[SUCCESS] Pre-commit checks passed"
exit 0
EOF
    chmod +x "$HOOKS_DIR/pre-commit"
    echo "  [SUCCESS] pre-commit hook installed"

    # Commit-msg hook
    cat > "$HOOKS_DIR/commit-msg" << 'EOF'
#!/bin/bash
# Commit-msg hook: Check commit message for emojis

COMMIT_MSG_FILE=$1

# Check for emojis in commit message
if grep -qP '[\x{1F300}-\x{1F9FF}]' "$COMMIT_MSG_FILE"; then
    echo ""
    echo "[BLOCKED] Commit message contains emojis"
    echo "Remove emojis from your commit message and try again"
    echo ""
    echo "To skip this check (not recommended): git commit --no-verify"
    exit 1
fi

exit 0
EOF
    chmod +x "$HOOKS_DIR/commit-msg"
    echo "  [SUCCESS] commit-msg hook installed"
fi

echo ""
echo "[3/3] Testing hooks..."

# Test that Python is available
if ! command -v python &> /dev/null; then
    echo "  [WARNING] Python not found in PATH"
    echo "  Hooks may not work correctly"
else
    echo "  [SUCCESS] Python found"
fi

# Test that scripts are executable
if [ -x "$PROJECT_ROOT/scripts/check_ai_slop.py" ]; then
    echo "  [SUCCESS] check_ai_slop.py is executable"
else
    echo "  [WARNING] check_ai_slop.py is not executable"
    chmod +x "$PROJECT_ROOT/scripts/check_ai_slop.py"
    echo "  [FIXED] Made check_ai_slop.py executable"
fi

echo ""
echo "========================================================================"
echo "Git hooks installed successfully!"
echo "========================================================================"
echo ""
echo "What happens now:"
echo "  - Pre-commit hook checks for emojis before each commit"
echo "  - Commit-msg hook blocks emojis in commit messages"
echo "  - Hooks run automatically (use --no-verify to skip)"
echo ""
echo "Manual testing:"
echo "  1. Run pre-commit manually: pre-commit run --all-files"
echo "  2. Test AI slop check: python scripts/check_ai_slop.py --critical-only"
echo ""
echo "To uninstall hooks:"
echo "  - pre-commit framework: pre-commit uninstall"
echo "  - Manual hooks: rm .git/hooks/pre-commit .git/hooks/commit-msg"
echo ""
