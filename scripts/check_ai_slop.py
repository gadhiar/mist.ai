#!/usr/bin/env python3
"""AI Slop Detection and Removal Tool.

Detects and removes "AI slop" patterns that violate project style guidelines.
See docs/AI_SLOP_CHECKER.md for complete documentation.

What it detects:
- Emojis and unicode decorative symbols (CRITICAL - auto-fixable)
- Superlative language like "amazing" (WARNING - manual review)
- AI filler phrases like "let's dive in" (INFO - optional)

Quick examples:
    python scripts/check_ai_slop.py                    # Check all
    python scripts/check_ai_slop.py --fix              # Auto-fix emojis/symbols
    python scripts/check_ai_slop.py --critical-only    # Only check emojis (fast)
    python scripts/check_ai_slop.py --incremental      # Only check git diff
    python scripts/check_ai_slop.py --format markdown  # LLM-friendly output
    python scripts/check_ai_slop.py --output report.md # Save to file

Exit codes:
    0 - No critical issues found
    1 - Critical issues detected (emojis or unicode symbols)

For full documentation, see: docs/AI_SLOP_CHECKER.md
"""

import argparse
import json
import re
import subprocess  # nosec B404 - used only for safe git commands
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure the project root (parent of scripts/) is on sys.path so that
# `backend` is importable whether the script is run as:
#   python scripts/check_ai_slop.py   (from project root)
#   python check_ai_slop.py           (from scripts/ dir)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Canonical pattern catalogue lives in backend.chat.slop_detector so backend
# code can import it. This script is the CLI wrapper.
from backend.chat.slop_detector import (  # noqa: E402
    PATTERNS,
    SlopPattern,
)

# Files to always skip
SKIP_PATTERNS = [
    r"\.git[/\\]",
    r"[/\\]?venv[/\\]",
    r"[/\\]?\.venv[/\\]",
    r"node_modules[/\\]",
    r"__pycache__[/\\]",
    r"\.pyc$",
    r"\.jpg$",
    r"\.png$",
    r"\.ico$",
    r"\.wav$",
    r"\.mp3$",
    r"dependencies[/\\]",
    r"\.lock$",
    r"package-lock\.json$",
    r"check_ai_slop\.py$",  # Skip self - contains patterns by design
    r"slop_detector\.py$",  # Skip library source - contains pattern literals by design
    r"test_slop_detector\.py$",  # Skip slop detector tests - contain literal test inputs
]


def should_skip_file(file_path: Path) -> bool:
    """Check if file should be skipped."""
    path_str = str(file_path)
    return any(re.search(pattern, path_str) for pattern in SKIP_PATTERNS)


def get_git_diff_files() -> set[Path]:
    """Get list of files changed in git (unstaged + staged)."""
    try:
        result = subprocess.run(  # nosec B603, B607 - safe git command with fixed arguments
            ["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True, check=True
        )
        files = result.stdout.strip().split("\n")
        return {Path(f) for f in files if f}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()


def filter_by_severity(patterns: list[SlopPattern], severity_threshold: str) -> list[SlopPattern]:
    """Filter patterns by severity threshold."""
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    threshold_level = severity_order.get(severity_threshold, 0)
    return [p for p in patterns if severity_order[p.severity] <= threshold_level]


def check_file(file_path: Path) -> dict[str, list[tuple[int, str, str]]]:
    """Check a single file for AI slop patterns.

    Returns:
        Dict mapping pattern names to list of (line_num, line_content, match)
    """
    if should_skip_file(file_path):
        return {}

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
        return {}

    findings = {}

    for pattern in PATTERNS:
        matches = []
        for line_num, line in enumerate(lines, 1):
            for match in pattern.pattern.finditer(line):
                matches.append((line_num, line.rstrip(), match.group()))

        if matches:
            findings[pattern.name] = matches

    return findings


def fix_file(file_path: Path, dry_run: bool = False) -> int:
    """Fix AI slop in a file.

    Returns:
        Number of fixes applied
    """
    if should_skip_file(file_path):
        return 0

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return 0

    original_content = content
    fixes_applied = 0

    for pattern in PATTERNS:
        if pattern.fixable:
            matches = list(pattern.pattern.finditer(content))
            if matches:
                content = pattern.pattern.sub(pattern.replacement, content)
                fixes_applied += len(matches)

    if content != original_content and not dry_run:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"[ERROR] Failed to write {file_path}: {e}")
            return 0

    return fixes_applied


def get_severity_color(severity: str) -> str:
    """Get ANSI color code for severity."""
    colors = {
        "critical": "\033[91m",  # Red
        "warning": "\033[93m",  # Yellow
        "info": "\033[94m",  # Blue
    }
    return colors.get(severity, "")


def reset_color() -> str:
    """Get ANSI reset code."""
    return "\033[0m"


def format_markdown_report(
    total_files: int,
    files_with_issues: int,
    total_issues: int,
    total_fixes: int,
    pattern_counts: dict[str, int],
    file_issues: dict[Path, dict[str, list]],
) -> str:
    """Generate ultra-compact markdown for LLM consumption."""
    report = ["# AI Slop Report"]

    # Calculate severity counts
    critical = sum(
        v
        for k, v in pattern_counts.items()
        if any(p.name == k and p.severity == "critical" for p in PATTERNS)
    )
    warning = sum(
        v
        for k, v in pattern_counts.items()
        if any(p.name == k and p.severity == "warning" for p in PATTERNS)
    )
    info = sum(
        v
        for k, v in pattern_counts.items()
        if any(p.name == k and p.severity == "info" for p in PATTERNS)
    )

    # Ultra-compact summary (one line)
    severity_str = f"{critical}c" if critical else ""
    severity_str += f",{warning}w" if warning else ""
    severity_str += f",{info}i" if info else ""
    fix_str = f", {total_fixes} fixed" if total_fixes > 0 else ""
    report.append(f"Scanned {total_files} files: {total_issues} issues ({severity_str}){fix_str}\n")

    # Issue locations (ultra-compact: pattern: count @file:lines)
    if total_issues > 0:
        report.append("## Issues")
        for pattern in PATTERNS:
            count = pattern_counts.get(pattern.name, 0)
            if count == 0:
                continue

            # Severity indicator: [C], [W], [I]
            sev = pattern.severity[0].upper()
            fixable = "+" if pattern.fixable else ""

            # Collect all locations for this pattern
            locations = []
            for file_path, issues in file_issues.items():
                if pattern.name in issues:
                    lines = [str(m[0]) for m in issues[pattern.name][:5]]  # Max 5 lines per file
                    more = (
                        f"+{len(issues[pattern.name]) - 5}" if len(issues[pattern.name]) > 5 else ""
                    )
                    loc = f"@{file_path}:{','.join(lines)}{more}"
                    locations.append(loc)

            # Format: [C+] emoji: 5 @file1:12,34 @file2:56,78
            report.append(f"[{sev}{fixable}] {pattern.name}: {count} {' '.join(locations)}")

    return "\n".join(report)


def main():
    global PATTERNS

    # Fix Windows encoding issues with Unicode output
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Detect and remove AI slop patterns from codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check everything, text output
  python scripts/check_ai_slop.py

  # Only critical issues (emojis), write to file
  python scripts/check_ai_slop.py --critical-only --output report.md

  # Check only changed files in git
  python scripts/check_ai_slop.py --incremental

  # Auto-fix and output markdown
  python scripts/check_ai_slop.py --fix --format markdown
        """,
    )
    parser.add_argument("--fix", action="store_true", help="Auto-fix fixable patterns")
    parser.add_argument(
        "--path", type=str, default=".", help="Path to check (default: current directory)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument("--output", type=str, help="Write report to file instead of stdout")
    parser.add_argument(
        "--critical-only", action="store_true", help="Only check critical issues (emojis/symbols)"
    )
    parser.add_argument(
        "--incremental", action="store_true", help="Only check files changed in git"
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output (for text format)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process files in parallel (faster for large codebases)",
    )

    args = parser.parse_args()

    # Filter patterns by severity if requested
    patterns_to_check = PATTERNS
    if args.critical_only:
        patterns_to_check = [p for p in PATTERNS if p.severity == "critical"]

    # Update global PATTERNS for check_file to use
    original_patterns = PATTERNS
    PATTERNS = patterns_to_check

    root_path = Path(args.path)
    if not root_path.exists():
        print(f"[ERROR] Path does not exist: {root_path}")
        sys.exit(1)

    # Get all text files to check
    file_extensions = [".py", ".md", ".txt", ".yaml", ".yml", ".json", ".dart", ".sh"]
    files_to_check = []

    if root_path.is_file():
        files_to_check = [root_path]
    else:
        # If incremental mode, only check git diff files
        if args.incremental:
            git_files = get_git_diff_files()
            for ext in file_extensions:
                files_to_check.extend(
                    [
                        f
                        for f in root_path.rglob(f"*{ext}")
                        if f in git_files or any(p in git_files for p in f.parents)
                    ]
                )
            if not files_to_check:
                print("No changed files to check (git diff is empty)")
                sys.exit(0)
        else:
            for ext in file_extensions:
                files_to_check.extend(root_path.rglob(f"*{ext}"))

    # Filter out skipped files
    files_to_check = [f for f in files_to_check if not should_skip_file(f)]

    if not files_to_check:
        print("No files to check")
        sys.exit(0)

    # Processing message
    mode_msg = "changed files" if args.incremental else "files"
    severity_msg = "critical issues only" if args.critical_only else "AI slop patterns"
    print(f"Checking {len(files_to_check)} {mode_msg} for {severity_msg}...")

    total_issues = 0
    total_files_with_issues = 0
    total_fixes = 0
    pattern_counts = {p.name: 0 for p in patterns_to_check}
    file_issues = {}  # Store for markdown output

    # Process files (optionally in parallel)
    def process_file(file_path):
        findings = check_file(file_path)
        fixes = 0
        if args.fix and findings:
            fixes = fix_file(file_path)
        return file_path, findings, fixes

    if args.parallel and len(files_to_check) > 10:
        # Parallel processing for large codebases
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_file, f): f for f in files_to_check}
            results = []
            for future in as_completed(futures):
                results.append(future.result())
    else:
        # Sequential processing
        results = [process_file(f) for f in sorted(files_to_check)]

    # Collect results
    for file_path, findings, fixes in results:
        if findings:
            total_files_with_issues += 1
            file_issues[file_path] = findings

            for pattern_name, matches in findings.items():
                pattern_counts[pattern_name] += len(matches)
                total_issues += len(matches)

            if fixes > 0:
                total_fixes += fixes

    # Output based on format
    if args.format == "markdown":
        output = format_markdown_report(
            len(files_to_check),
            total_files_with_issues,
            total_issues,
            total_fixes,
            pattern_counts,
            file_issues,
        )
    elif args.format == "json":
        output = json.dumps(
            {
                "summary": {
                    "files_checked": len(files_to_check),
                    "files_with_issues": total_files_with_issues,
                    "total_issues": total_issues,
                    "total_fixes": total_fixes,
                },
                "pattern_counts": pattern_counts,
                "files": {
                    str(path): {name: len(matches) for name, matches in issues.items()}
                    for path, issues in file_issues.items()
                },
            },
            indent=2,
        )
    else:  # text format
        output_lines = []
        if args.format == "text":
            for file_path, findings in sorted(file_issues.items()):
                output_lines.append(f"\n{file_path}:")
                for pattern_name, matches in findings.items():
                    pattern = next(p for p in patterns_to_check if p.name == pattern_name)

                    if not args.no_color:
                        color = get_severity_color(pattern.severity)
                        reset = reset_color()
                    else:
                        color = reset = ""

                    output_lines.append(
                        f"  {color}[{pattern.severity.upper()}] {pattern.name}: "
                        f"{len(matches)} occurrence(s){reset}"
                    )

                    for line_num, _, match_text in matches[:5]:
                        output_lines.append(f"    Line {line_num}: ...{match_text}...")

                    if len(matches) > 5:
                        output_lines.append(f"    ... and {len(matches) - 5} more")

                if args.fix and total_fixes > 0:
                    output_lines.append(f"  [FIXED] Applied {total_fixes} fix(es)")

        # Summary
        output_lines.append("\n" + "=" * 70)
        output_lines.append("SUMMARY")
        output_lines.append("=" * 70)
        output_lines.append(f"Total files checked: {len(files_to_check)}")
        output_lines.append(f"Files with issues: {total_files_with_issues}")
        output_lines.append(f"Total issues found: {total_issues}")

        if args.fix:
            output_lines.append(f"Total fixes applied: {total_fixes}")

        output_lines.append("\nIssues by pattern:")
        for pattern in patterns_to_check:
            count = pattern_counts[pattern.name]
            if count > 0:
                fixable = "(fixable)" if pattern.fixable else "(manual fix required)"
                output_lines.append(
                    f"  [{pattern.severity.upper()}] {pattern.name}: {count} {fixable}"
                )

        if total_issues > 0 and not args.fix:
            output_lines.append("\nRun with --fix to automatically fix fixable patterns")

        output = "\n".join(output_lines)

    # Write output
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(output)

    # Restore original patterns
    PATTERNS = original_patterns

    # Exit code
    critical_count = sum(
        pattern_counts.get(p.name, 0) for p in patterns_to_check if p.severity == "critical"
    )

    if critical_count > 0:
        sys.exit(1)  # Critical issues found
    else:
        sys.exit(0)  # No critical issues


if __name__ == "__main__":
    main()
