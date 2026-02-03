@echo off
REM Install git hooks for MIST.AI project (Windows version)
REM This script sets up pre-commit hooks for code quality enforcement

setlocal enabledelayedexpansion

echo Installing git hooks for MIST.AI...
echo.

REM Check if we're in a git repository
if not exist ".git" (
    echo Error: Not in a git repository
    exit /b 1
)

REM Method 1: Install pre-commit framework (recommended)
echo [1/3] Installing pre-commit framework hooks...
where pre-commit >nul 2>&1
if %errorlevel% == 0 (
    pre-commit install
    pre-commit install --hook-type commit-msg
    echo   [SUCCESS] pre-commit hooks installed
) else (
    echo   [WARNING] pre-commit not found. Install with: pip install pre-commit
    echo   Falling back to manual hook installation...
    echo.

    REM Method 2: Manual hook installation (fallback)
    echo [2/3] Installing manual git hooks...

    REM Pre-commit hook
    (
        echo #!/bin/bash
        echo # Pre-commit hook: Check for AI slop and code quality issues
        echo.
        echo echo "Running pre-commit checks..."
        echo.
        echo # Check for critical AI slop ^(emojis^)
        echo python scripts/check_ai_slop.py --incremental --critical-only --no-color
        echo if [ $? -ne 0 ]; then
        echo     echo ""
        echo     echo "[BLOCKED] Critical AI slop detected ^(emojis or unicode symbols^)"
        echo     echo "Fix with: python scripts/check_ai_slop.py --fix"
        echo     echo ""
        echo     echo "To skip this check ^(not recommended^): git commit --no-verify"
        echo     exit 1
        echo fi
        echo.
        echo echo "[SUCCESS] Pre-commit checks passed"
        echo exit 0
    ) > .git\hooks\pre-commit
    echo   [SUCCESS] pre-commit hook installed

    REM Commit-msg hook
    (
        echo #!/bin/bash
        echo # Commit-msg hook: Check commit message for emojis
        echo.
        echo COMMIT_MSG_FILE=$1
        echo.
        echo # Check for emojis in commit message
        echo if grep -qP '[\x{1F300}-\x{1F9FF}]' "$COMMIT_MSG_FILE"; then
        echo     echo ""
        echo     echo "[BLOCKED] Commit message contains emojis"
        echo     echo "Remove emojis from your commit message and try again"
        echo     echo ""
        echo     echo "To skip this check ^(not recommended^): git commit --no-verify"
        echo     exit 1
        echo fi
        echo.
        echo exit 0
    ) > .git\hooks\commit-msg
    echo   [SUCCESS] commit-msg hook installed
)

echo.
echo [3/3] Testing hooks...

REM Test that Python is available
where python >nul 2>&1
if %errorlevel% == 0 (
    echo   [SUCCESS] Python found
) else (
    echo   [WARNING] Python not found in PATH
    echo   Hooks may not work correctly
)

REM Test that scripts exist
if exist "scripts\check_ai_slop.py" (
    echo   [SUCCESS] check_ai_slop.py found
) else (
    echo   [WARNING] check_ai_slop.py not found
)

echo.
echo ========================================================================
echo Git hooks installed successfully!
echo ========================================================================
echo.
echo What happens now:
echo   - Pre-commit hook checks for emojis before each commit
echo   - Commit-msg hook blocks emojis in commit messages
echo   - Hooks run automatically (use --no-verify to skip)
echo.
echo Manual testing:
echo   1. Run pre-commit manually: pre-commit run --all-files
echo   2. Test AI slop check: python scripts\check_ai_slop.py --critical-only
echo.
echo To uninstall hooks:
echo   - pre-commit framework: pre-commit uninstall
echo   - Manual hooks: del .git\hooks\pre-commit .git\hooks\commit-msg
echo.

pause
