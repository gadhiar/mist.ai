"""Measure the need bank's topic coverage by word, so authors can find label-signal words.

A classifier fit on the bank can learn a topic word as a label signal when that word occurs in
rows of one label only. For each word, with `n` need rows and `m` no-need rows holding it, the
coverage invariant is `min(n, m) >= min(max(n, m), CAP)`; `gaps` lists each word breaking it
and the rows the lighter label still needs. Words that name no topic (recall markers, function
words, numbers, interjections, fragments) are exempt, since covering them would cost rows
without removing a topic signal; and a marker such as "usual" is the cue the gate should learn.

Words are folded plural tokens, not stems: the fixture's `_stem` merges unrelated words on
purpose, which here would hide a topic word behind another.

Bank files are read as plain JSON lines, and nothing here prints a row's text: the output holds
words, labels and counts only.

Run: `python -m tests.unit.memory.need_bank_topics --side both <path> ...`
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from tests.unit.eval_harness.test_memory_recall_fixture import _content_tokens

REPO_ROOT = Path(__file__).resolve().parents[3]
BANK_DIR = REPO_ROOT / "backend" / "memory" / "need_bank"
DEFAULT_BANK_PATHS = (BANK_DIR / "writer_a.jsonl", BANK_DIR / "writer_b.jsonl")

CAP = 5
LABELS = ("need", "no_need")
SIDES = ("need_heavy", "no_need_heavy", "both")
Side = Literal["need_heavy", "no_need_heavy", "both"]

# Token to the word it counts as, applied before the plural fold. Each entry is a token, in the
# bank or listed ahead of the rows that will hold it, whose fold is not a readable word or merges
# it with an unrelated word. An entry maps the token to the readable word it stands for, to its
# plain singular, or to itself, kept whole. Reviewers add to this list.
FOLD_EXCEPTIONS: dict[str, str] = {
    # Kept whole: the fold would merge them with an unrelated word.
    "clothes": "clothes",  # garments, not the verb clothe
    "eighties": "eighties",  # the decade, not the number eighty
    "hits": "hits",  # popular songs, not hit the gym
    "lands": "lands",  # a flight landing, not where a decision landed
    "news": "news",
    "ones": "ones",  # a pronoun, not the number one
    "points": "points",  # bullet points, not a boiling point
    "seconds": "seconds",  # the unit, not the ordinal
    "times": "times",  # multiplication, not the clock
    "turns": "turns",  # reaching an age, not turn on
    # Kept whole: the fold is not a readable word.
    "always": "always",
    "cheers": "cheers",
    "christmas": "christmas",
    "downstairs": "downstairs",
    "lens": "lens",
    "upstairs": "upstairs",
    # Kept whole: the fold is not a readable word. Series occurs in no bank row yet; it is
    # listed ahead of the rows authors will write.
    "gas": "gas",
    "series": "series",
    # Mapped to the readable word the token stands for: the token left of "café" once the
    # tokenizer drops the accent. The escape keeps codespell (v2.2.6, without inline ignores)
    # from reading the token as a misspelling.
    "ca\x66": "cafe",
    # Mapped to the plain singular the fold misses.
    "calories": "calorie",
    "causes": "cause",
    "goes": "go",
    "movies": "movie",
    "phrases": "phrase",
    "volcanoes": "volcano",
}

EXEMPT_CLASSES = ("marker", "function", "number", "interjection", "fragment")

# Word to class, for words that name nothing a turn could act on or ask about. A word that
# names a device, item, food, place, person, activity, media, service, weather condition or a
# device's action stays out of this list. One entry a line; reviewers read every entry.
EXEMPT_WORDS: dict[str, str] = {
    # Recall, preference or ownership: the cue the gate should learn.
    "always": "marker",
    "discussed": "marker",
    "enjoy": "marker",
    "favourite": "marker",
    "hate": "marker",
    "hated": "marker",
    "leaning": "marker",
    "like": "marker",
    "liked": "marker",
    "love": "marker",
    "mention": "marker",
    "mentioned": "marker",
    "mine": "marker",
    "normal": "marker",
    "normally": "marker",
    "prefer": "marker",
    "regular": "marker",
    "remember": "marker",
    "said": "marker",
    "told": "marker",
    "usual": "marker",
    "usually": "marker",
    # Verbs, adverbs and adjectives that name no topic.
    "actually": "function",
    "ago": "function",
    "aiming": "function",
    "along": "function",
    "already": "function",
    "appropriate": "function",
    "ask": "function",
    "asked": "function",
    "asking": "function",
    "away": "function",
    "best": "function",
    "bit": "function",
    "called": "function",
    "cheapest": "function",
    "clever": "function",
    "closest": "function",
    "come": "function",
    "coming": "function",
    "conclude": "function",
    "current": "function",
    "dangerous": "function",
    "decide": "function",
    "dumb": "function",
    "earlier": "function",
    "early": "function",
    "else": "function",
    "end": "function",
    "ended": "function",
    "enough": "function",
    "even": "function",
    "ever": "function",
    "every": "function",
    "everything": "function",
    "exactly": "function",
    "explain": "function",
    "finally": "function",
    "find": "function",
    "fit": "function",
    "forget": "function",
    "form": "function",
    "fresh": "function",
    "funny": "function",
    "gave": "function",
    "getting": "function",
    "given": "function",
    "glad": "function",
    "going": "function",
    "good": "function",
    "grab": "function",
    "great": "function",
    "halfway": "function",
    "hang": "function",
    "happen": "function",
    "happening": "function",
    "happy": "function",
    "heard": "function",
    "helped": "function",
    "holding": "function",
    "instead": "function",
    "interrupted": "function",
    "keep": "function",
    "kind": "function",
    "last": "function",
    "late": "function",
    "lately": "function",
    "least": "function",
    "left": "function",
    "little": "function",
    "look": "function",
    "looked": "function",
    "made": "function",
    "main": "function",
    "make": "function",
    "mean": "function",
    "middle": "function",
    "might": "function",
    "missed": "function",
    "much": "function",
    "near": "function",
    "necessary": "function",
    "never": "function",
    "new": "function",
    "nice": "function",
    "nothing": "function",
    "older": "function",
    "ones": "function",
    "option": "function",
    "originally": "function",
    "part": "function",
    "perfect": "function",
    "pick": "function",
    "properly": "function",
    "put": "function",
    "quarterly": "function",
    "quick": "function",
    "reach": "function",
    "ready": "function",
    "recommend": "function",
    "recommended": "function",
    "remain": "function",
    "right": "function",
    "roughly": "function",
    "safe": "function",
    "see": "function",
    "simply": "function",
    "since": "function",
    "someone": "function",
    "sort": "function",
    "sorted": "function",
    "spoke": "function",
    "still": "function",
    "stuff": "function",
    "suggest": "function",
    "suggested": "function",
    "take": "function",
    "taking": "function",
    "talk": "function",
    "talking": "function",
    "tallest": "function",
    "think": "function",
    "thinking": "function",
    "touch": "function",
    "tricky": "function",
    "try": "function",
    "use": "function",
    "used": "function",
    "wait": "function",
    "waiting": "function",
    "want": "function",
    "wanted": "function",
    "well": "function",
    "went": "function",
    "whether": "function",
    "willing": "function",
    "wrong": "function",
    "yearly": "function",
    "yet": "function",
    "younger": "function",
    # Numerals and number words, cardinal and ordinal.
    "12": "number",
    "180": "number",
    "196": "number",
    "2": "number",
    "22": "number",
    "350": "number",
    "38": "number",
    "48": "number",
    "5": "number",
    "8": "number",
    "900": "number",
    "eight": "number",
    "eighth": "number",
    "eighty": "number",
    "eleven": "number",
    "fifteen": "number",
    "fifty": "number",
    "first": "number",
    "five": "number",
    "forty": "number",
    "four": "number",
    "half": "number",
    "hundred": "number",
    "nine": "number",
    "ninety": "number",
    "second": "number",
    "seven": "number",
    "seventy": "number",
    "six": "number",
    "sixty": "number",
    "ten": "number",
    "third": "number",
    "thirteen": "number",
    "thirty": "number",
    "three": "number",
    "twelfth": "number",
    "twelve": "number",
    "twenty": "number",
    "twice": "number",
    "two": "number",
    # Interjections and greetings.
    "alright": "interjection",
    "bye": "interjection",
    "cheers": "interjection",
    "goodnight": "interjection",
    "ha": "interjection",
    "haha": "interjection",
    "hello": "interjection",
    "hmm": "interjection",
    "phew": "interjection",
    "righto": "interjection",
    "sorry": "interjection",
    "ta": "interjection",
    "thank": "interjection",
    "yep": "interjection",
    # Pieces of a split word or contraction, naming nothing on their own.
    "didn": "fragment",
    "doesn": "fragment",
    "en": "fragment",
    "haven": "fragment",
    "o": "fragment",
    "sec": "fragment",
    "self": "fragment",
    "wide": "fragment",
}


def _fold(token: str) -> str:
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("es") and token[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return token[:-2]
    if token.endswith("s") and not token[:-1].endswith(("s", "u", "i")):
        return token[:-1]
    return token


def word(token: str) -> str:
    """The word a content token counts as: its exception if listed, else its plural fold."""
    if token in FOLD_EXCEPTIONS:
        return FOLD_EXCEPTIONS[token]
    return _fold(token)


def row_words(text: str) -> frozenset[str]:
    """The distinct words of one row's text."""
    return frozenset(word(token) for token in _content_tokens(text))


def _rows(path: Path) -> Iterable[tuple[str, str, str]]:
    """Yield (id, label, text) for each non-empty JSON line of `path`."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            yield row["id"], row["label"], row["text"]


def word_counts(paths: Iterable[Path | str]) -> dict[str, tuple[int, int]]:
    """Per word, the number of need rows and of no-need rows holding it.

    Raises:
        ValueError: a row's label is neither `need` nor `no_need`; the message names the row's id.
    """
    counts: dict[str, list[int]] = {}
    for path in paths:
        for row_id, label, text in _rows(Path(path)):
            if label not in LABELS:
                raise ValueError(f"row {row_id} has label {label!r}, not one of {LABELS}")
            column = LABELS.index(label)
            for found in row_words(text):
                counts.setdefault(found, [0, 0])[column] += 1
    return {found: (pair[0], pair[1]) for found, pair in counts.items()}


def gaps(
    paths: Iterable[Path | str],
    *,
    side: Side,
    exempt: Mapping[str, str] = EXEMPT_WORDS,
) -> list[tuple[str, str, int]]:
    """Each non-exempt word breaking the invariant on `side`, sorted by word.

    A gap is (word, the label that needs rows, how many): a need-heavy word needs `no_need`
    rows, a no-need-heavy word needs `need` rows, and the count is
    `min(max(n, m), CAP) - min(n, m)`.

    Raises:
        ValueError: `side` is not one of `SIDES`, or an exempt word's class is not one of
            `EXEMPT_CLASSES`; the message names the side or the word.
    """
    if side not in SIDES:
        raise ValueError(f"side {side!r} is not one of {SIDES}")
    unknown = sorted(found for found, cls in exempt.items() if cls not in EXEMPT_CLASSES)
    if unknown:
        raise ValueError(f"exempt words with a class outside {EXEMPT_CLASSES}: {unknown}")
    found_gaps = []
    for found, (need, no_need) in word_counts(paths).items():
        if found in exempt:
            continue
        rows_needed = min(max(need, no_need), CAP) - min(need, no_need)
        if rows_needed <= 0:
            continue
        if need > no_need and side in ("need_heavy", "both"):
            found_gaps.append((found, "no_need", rows_needed))
        elif no_need > need and side in ("no_need_heavy", "both"):
            found_gaps.append((found, "need", rows_needed))
    return sorted(found_gaps)


def main(argv: Sequence[str] | None = None) -> int:
    """Print one `word<TAB>label<TAB>rows` line per gap."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", choices=SIDES, required=True)
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_BANK_PATHS))
    args = parser.parse_args(argv)
    for found, label, rows_needed in gaps(args.paths, side=args.side):
        sys.stdout.write(f"{found}\t{label}\t{rows_needed}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
