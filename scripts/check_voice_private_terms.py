"""Refuse private voice-data terms in files or in unpushed history.

This repository is public. The words that identify Raj's voice sources (show,
character and actor names) live in `<private root>/guard/terms.txt`, outside
the repository, and this script fails when one of them appears in a passed
file's path or text, or anywhere in a commit range. It never prints a term,
and it never prints a path that holds one, because its output lands in logs.

File mode (the `no-voice-private-terms` pre-commit hook):
    python scripts/check_voice_private_terms.py <file>...
With no list it prints `voice private terms: no list, skipped` and exits 0,
which is the CI and container path.

History mode (the binding check before a push of the integration branch):
    python scripts/check_voice_private_terms.py --range <A>..<B>
It reads one `git log -p --cc` stream and checks every commit message, every
patch path and every added line, merges included. With no list it exits 2,
because a history check that passes without a list proves nothing.

The private root is `MIST_VOICE_PRIVATE_ROOT`, else `mist-voice` beside the
main checkout (the parent of `git rev-parse --git-common-dir`, which every
worktree shares). A term is a line of `terms.txt` of 3 or more characters that
does not start with `#`, matched case-insensitively; `#<n>` in the output is
its line number in `terms.txt`. Standard library only: the hook runs on a CI
runner that has nothing but pre-commit installed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

NO_LIST = "voice private terms: no list"
MARKER = "\x1evoice-private-terms\x1e"
GIT_LOG = [
    "git",
    "-c",
    "core.quotePath=false",
    "log",
    "--format=%x1evoice-private-terms%x1e%H%n%B",
    "-p",
    "--cc",
    "--no-renames",
    "--no-color",
    "--unified=0",
    "--no-ext-diff",
]

Terms = list[tuple[int, str]]


def private_root() -> Path | None:
    explicit = os.environ.get("MIST_VOICE_PRIVATE_ROOT")
    if explicit:
        return Path(explicit)
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    main_checkout = Path(result.stdout.strip()).parent
    return main_checkout.parent / "mist-voice"


def load_terms(root: Path | None) -> Terms | None:
    if root is None:
        return None
    path = root / "guard" / "terms.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    terms = []
    for number, line in enumerate(text.splitlines(), 1):
        term = line.strip()
        if len(term) >= 3 and not term.startswith("#"):
            terms.append((number, term.casefold()))
    return terms


def find(terms: Terms, text: str) -> int | None:
    folded = text.casefold()
    for number, term in terms:
        if term in folded:
            return number
    return None


def check_files(terms: Terms, paths: list[str]) -> int:
    failed = False
    for k, path in enumerate(paths, 1):
        label = path
        number = find(terms, path)
        if number is not None:
            print(f"file #{k} of the batch: path matches private term #{number}")
            label = f"file #{k}"
            failed = True
        try:
            text = Path(path).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.split("\n"), 1):
            number = find(terms, line)
            if number is not None:
                print(f"{label}:{line_no}: matches private term #{number}")
                failed = True
    return 1 if failed else 0


def check_range(terms: Terms, commit_range: str) -> int:
    process = subprocess.Popen([*GIT_LOG, commit_range], stdout=subprocess.PIPE)
    assert process.stdout is not None
    reported: set[str] = set()
    sha = ""
    in_message = in_header = False
    parents = 1
    path = ""
    path_hit = False

    def report(line: str) -> None:
        if line not in reported:
            reported.add(line)
            print(line)

    for raw in process.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        if line.startswith(MARKER):
            sha = line[len(MARKER) : len(MARKER) + 12]
            in_message, in_header = True, False
            continue
        if line.startswith(("diff --git ", "diff --cc ", "diff --combined ")):
            in_message, in_header = False, True
            parents = 1 if line.startswith("diff --git ") else 2
            path, path_hit = "", False
        if in_message:
            number = find(terms, line)
            if number is not None:
                report(f"{sha}: message matches private term #{number}")
            continue
        if line.startswith("@@"):
            in_header = False
            parents = len(line) - len(line.lstrip("@")) - 1
            continue
        if in_header:
            if line.startswith(("diff ", "--- ", "+++ ", "Binary files ")):
                if line.startswith("+++ "):
                    path = line[4:].removeprefix("b/")
                number = find(terms, line)
                if number is not None:
                    report(f"{sha}: a path matches private term #{number}")
                    path_hit = True
            continue
        prefix = line[:parents]
        if "+" in prefix and "-" not in prefix:
            number = find(terms, line[parents:])
            if number is not None:
                where = "path withheld" if path_hit else path
                report(f"{sha} {where}: added line matches private term #{number}")
    if process.wait() != 0:
        print("voice private terms: git log failed", file=sys.stderr)
        return 2
    return 1 if reported else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--range", dest="commit_range", metavar="A..B")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="backslashreplace")

    terms = load_terms(private_root())
    if args.commit_range:
        if terms is None:
            print(NO_LIST)
            return 2
        return check_range(terms, args.commit_range)
    if terms is None:
        print(f"{NO_LIST}, skipped")
        return 0
    return check_files(terms, args.paths)


if __name__ == "__main__":
    sys.exit(main())
