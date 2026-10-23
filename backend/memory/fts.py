"""Build the OR-mode FTS5 query the memory index's lexical pools search with.

A recall query is a sentence ("where did I work before Acme?"), not a keyword list. The vault
sidecar ANDs its tokens, so one word absent from a fact ("before") empties the pool. Here the
content words are ORed and bm25 ranks the rows that share the most of them first, which is
what a candidate pool feeding a reranker needs.

A token is a run of letters, digits and marks. Everything else separates tokens, including
every FTS5 operator character and `_` and `'`, which are Unicode punctuation and so also
separators to the index's `unicode61` tokenizer. So "Kestrel's" yields "kestrel" (the `s` is dropped as too short), as the
index stored it, rather than a phrase "kestrel s" that no stored text contains. Queries and the
index's FTS mirror both pass through `fold`, so a fullwidth or dotted-capital letter on either
side still matches.
"""

from __future__ import annotations

import re
import unicodedata

# NLTK's English stopword list without its apostrophe forms ("don't", "you're"), which the
# tokenizer below never yields; the fragments those forms leave when split at the apostrophe
# ("don", "ll", "re", ...) are on it. Question words are on it too, so "where", "when" and
# "who" never decide what a recall query matches.
_NLTK_STOPWORDS = """
    a about above after again against ain all am an and any are as at
    be because been before being below between both but by
    can d did do does doing don down during
    each few for from further
    had has have having he her here hers herself him himself his how
    i if in into is it its itself just
    ll m ma me more most my myself
    no nor not now o of off on once only or other our ours ourselves out over own
    re s same shan she should so some such
    t than that the their theirs them themselves then there these they this those through
    to too under until up ve very
    was we were what when where which while who whom why will with won
    y you your yours yourself yourselves
"""

# Verbs whose regular "n't" contraction NLTK lists by its fragment ("isn't" -> "isn").
# The fragments are built from the verbs because codespell flags some of them as misspellings.
_NT_CONTRACTION_VERBS = (
    "are",
    "could",
    "did",
    "does",
    "had",
    "has",
    "have",
    "is",
    "might",
    "must",
    "need",
    "should",
    "was",
    "were",
    "would",
)

STOPWORDS: frozenset[str] = frozenset(_NLTK_STOPWORDS.split()) | frozenset(
    f"{verb}n" for verb in _NT_CONTRACTION_VERBS
)

_MIN_TOKEN_CHARS = 2

# The Combining Diacritical Marks block: the accents Latin, Greek and Cyrillic letters
# decompose into, and the dot above that lowercasing "İ" leaves behind.
_COMBINING_DIACRITICS = re.compile("[̀-ͯ]")


def fold(text: str) -> str:
    """`text` in the one form the index stores and every query is read in.

    NFKC turns a fullwidth or compatibility letter into its plain form; lowercasing follows;
    then the combining diacritics are removed, so "Café", "Cafe" plus a combining accent and
    "café" fold alike, and "İstanbul", whose lowercase is "i" plus a combining dot, folds to
    "istanbul". FactIndex writes its FTS mirror through this function and every query passes
    through it, so stored text and queries can only differ in ways the tokenizer treats alike.
    """
    lowered = unicodedata.normalize("NFKC", text).lower()
    stripped = _COMBINING_DIACRITICS.sub("", unicodedata.normalize("NFD", lowered))
    return unicodedata.normalize("NFC", stripped)


def words(text: str) -> list[str]:
    """Every token of `fold(text)`, in order, with nothing dropped.

    Name matching uses this unfiltered form, because a name may contain a stopword
    ("The Hague").
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in fold(text):
        category = unicodedata.category(char)
        # Letters, numbers, marks (a Devanagari vowel sign belongs to its word) and private-use
        # characters; everything else, `_` and `'` included, separates tokens.
        if category[0] in "LNM" or category == "Co":
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def query_tokens(text: str) -> list[str]:
    """The content words of `text`, lowercased, in first-seen order and without repeats.

    Stopwords and tokens shorter than two characters are dropped: they match nearly every
    stored fact and would only add noise to bm25.

    Args:
        text: Free text, such as a user utterance.

    Returns:
        The surviving tokens; empty when none survive.
    """
    seen: dict[str, None] = {}
    for token in words(text):
        if len(token) >= _MIN_TOKEN_CHARS and token not in STOPWORDS:
            seen.setdefault(token, None)
    return list(seen)


def quote_token(token: str) -> str:
    """`token` as an FTS5 string literal, with each embedded `"` doubled.

    A quoted token is always a literal: an operator word ("NEAR", "NOT") or a character such as
    `*` or `:` inside it is matched as text, never parsed. `query_tokens` never yields a `"`, but
    the escape keeps any caller that quotes other text safe.
    """
    return '"' + token.replace('"', '""') + '"'


def build_or_query(text: str) -> str | None:
    """An FTS5 MATCH expression that matches a row holding any content word of `text`.

    Args:
        text: Free text, such as the user's query or utterance.

    Returns:
        The quoted tokens joined with `OR`, for example `"capital" OR "france"`, or None when
        no token survives. The caller skips the search on None, because an empty MATCH
        expression is an FTS5 syntax error.
    """
    tokens = query_tokens(text)
    if not tokens:
        return None
    return " OR ".join(quote_token(token) for token in tokens)
