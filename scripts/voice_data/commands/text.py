"""`text build --version V --sources DIR [--assistant-lines FILE]`: the teacher's script.

Merges the public transcripts under `--sources` (a rehearsal root's `public/`, holding
`librittsr/` and `expresso/`) with the committed assistant-register lines, holds out the ear-test
lines, drops every line an earlier version under the root's `text/` holds and every line shorter
than the `text.line_chars_min` gate, and writes `text/<V>/lines.jsonl` and `text/<V>/summary.json`
in the root. Output names counts, version names, source names and root-relative paths only: never
a line's text or a source path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import gates, text_corpus
from ..errors import InputError
from ..records import write_json, write_jsonl
from ..root import open_root


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `text` command and its `build` action."""
    parser = subparsers.add_parser(
        "text",
        help="the training text the teacher speaks, with the ear-test lines held out",
        description="Build text/<V>/lines.jsonl and its summary from the text sources.",
    )
    actions = parser.add_subparsers(dest="text_action", metavar="ACTION", required=True)
    build = actions.add_parser(
        "build",
        help="merge the public transcripts and the assistant lines into one neutral script",
        description=(
            "Merge LibriTTS-R test-clean and Expresso's read transcriptions with the assistant "
            "lines; drop, in this order, lines that overlap an ear-test line, lines whose id or "
            "text an earlier version under text/ holds, lines shorter than the "
            f"{text_corpus.LINE_CHARS_GATE} gate or longer than {text_corpus.MAX_CHARS} chars, "
            "and duplicates; write text/<V>/lines.jsonl and text/<V>/summary.json."
        ),
    )
    build.add_argument("--version", required=True, help="the script version, a folder under text")
    build.add_argument(
        "--sources",
        required=True,
        help="the folder holding librittsr/ and expresso/ (a rehearsal root's public/)",
    )
    build.add_argument(
        "--assistant-lines",
        default=str(text_corpus.ASSISTANT_LINES_PATH),
        help="the assistant-register lines, JSONL (default: the committed text_lines.jsonl)",
    )
    build.set_defaults(func=run_build)


def run_build(args: argparse.Namespace) -> int:
    """Read the sources, apply the rules, write the script and print its counts."""
    root = open_root(args.root)
    if not text_corpus.VERSION.fullmatch(args.version) or ".." in args.version:
        raise InputError("refused: a text version is one folder name")
    version_dir = f"{text_corpus.TEXT_DIR}/{args.version}"
    if (root.path / version_dir).exists():
        raise InputError(f"refused: {version_dir} already exists; choose a new version")
    line_chars_min = text_corpus.line_chars_gate(gates.load(root.path))
    sources = Path(args.sources)
    if not sources.is_dir():
        raise InputError("refused: --sources is not a folder")
    earlier = text_corpus.read_earlier(root.path)
    corpus = text_corpus.build(
        text_corpus.read_sources(sources, Path(args.assistant_lines)),
        text_corpus.HoldOut(text_corpus.load_ear_test()),
        earlier,
        line_chars_min=line_chars_min,
    )
    lines = write_jsonl(root, f"{version_dir}/{text_corpus.LINES_NAME}", corpus.rows())
    summary = write_json(root, f"{version_dir}/{text_corpus.SUMMARY_NAME}", corpus.summary())
    for line in corpus.report_lines():
        print(line)
    print(f"wrote {root.rel(lines)} and {root.rel(summary)}")
    return 0
