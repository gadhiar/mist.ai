"""Tests for backend.memory.fts: the OR-mode FTS5 query the fact and episode pools search with."""

import sqlite3

import pytest

from backend.memory.fts import STOPWORDS, build_or_query, fold, quote_token


@pytest.fixture
def fts_table() -> sqlite3.Connection:
    """An in-memory FTS5 table with the memory index's tokenizer and two rows."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(text, tokenize='porter unicode61')")
    conn.executemany(
        "INSERT INTO t(rowid, text) VALUES (?, ?)",
        [(1, "Sam works at Kestrel Labs near the harbour"), (2, "Sam visited the café in Zürich")],
    )
    return conn


def _match(conn: sqlite3.Connection, expr: str) -> list[int]:
    return [row[0] for row in conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (expr,))]


def test_lowercases_tokens() -> None:
    assert build_or_query("Python RUST") == '"python" OR "rust"'


def test_drops_stopwords() -> None:
    assert build_or_query("What is the capital of France") == '"capital" OR "france"'


def test_drops_tokens_shorter_than_two_characters() -> None:
    assert build_or_query("x y z go") == '"go"'


def test_repeated_tokens_appear_once() -> None:
    assert build_or_query("rust Rust RUST tooling") == '"rust" OR "tooling"'


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("   \n\t", id="whitespace"),
        pytest.param("?!... -- ***", id="punctuation-only"),
        pytest.param("what is it that you are", id="stopwords-only"),
        pytest.param("a b c 1 2", id="single-characters-only"),
        pytest.param("I couldn't, it wasn't there", id="negated-contractions"),
    ],
)
def test_returns_none_when_no_token_remains(text: str) -> None:
    assert build_or_query(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("rock AND roll", '"rock" OR "roll"', id="and"),
        pytest.param("rock OR roll", '"rock" OR "roll"', id="or"),
        pytest.param("NOT sure", '"sure"', id="not"),
        pytest.param("NEAR harbour", '"near" OR "harbour"', id="near-kept-as-a-word"),
        pytest.param("NEAR(sam kestrel, 2)", '"near" OR "sam" OR "kestrel"', id="near-group"),
    ],
)
def test_fts5_operator_words_become_quoted_words_or_stopwords(text: str, expected: str) -> None:
    assert build_or_query(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param('he said "kestrel labs"', '"said" OR "kestrel" OR "labs"', id="phrase"),
        pytest.param('"unbalanced kestrel', '"unbalanced" OR "kestrel"', id="unbalanced"),
        pytest.param("Kestrel's harbour", '"kestrel" OR "harbour"', id="apostrophe"),
        pytest.param("Kestrel’s harbour", '"kestrel" OR "harbour"', id="curly-apostrophe"),
        pytest.param("don't forget", '"forget"', id="contraction"),
    ],
)
def test_quotes_in_the_text_never_reach_the_expression_unescaped(text: str, expected: str) -> None:
    assert build_or_query(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Where's Kestrel?", '"kestrel"', id="question"),
        pytest.param("type:session", '"type" OR "session"', id="column-filter"),
        pytest.param("prefix* ^start", '"prefix" OR "start"', id="prefix-and-initial"),
        pytest.param("sqlite-vec + fts5", '"sqlite" OR "vec" OR "fts5"', id="hyphen-plus"),
        pytest.param("(kestrel) {labs}", '"kestrel" OR "labs"', id="brackets"),
        pytest.param("snake_case_name", '"snake" OR "case" OR "name"', id="underscore"),
    ],
)
def test_punctuation_splits_tokens(text: str, expected: str) -> None:
    assert build_or_query(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Café Zürich", '"cafe" OR "zurich"', id="accented-latin"),
        pytest.param("ÉCOLE", '"ecole"', id="accented-capital"),
        pytest.param("Café", '"cafe"', id="decomposed-accent"),
        pytest.param("İstanbul", '"istanbul"', id="dotted-capital-i"),
        pytest.param("İSTANBUL", '"istanbul"', id="dotted-capital-i-upper"),
        pytest.param("東京 タワー", '"東京" OR "タワー"', id="cjk"),
        pytest.param("Ｋｅｓｔｒｅｌ", '"kestrel"', id="fullwidth"),
        pytest.param("Москва", '"москва"', id="cyrillic"),
        pytest.param("हिंदी", '"हिंदी"', id="marks"),
    ],
)
def test_unicode_letters_are_tokens(text: str, expected: str) -> None:
    assert build_or_query(text) == expected


def test_fold_gives_stored_text_and_queries_one_form() -> None:
    stored = fold("Ｋｅｓｔｒｅｌ visited İstanbul's Café")

    assert stored == "kestrel visited istanbul's cafe"


def test_quote_token_doubles_embedded_quotes() -> None:
    assert quote_token('x"y') == '"x""y"'


def test_stopwords_are_lowercase() -> None:
    assert all(word == word.lower() for word in STOPWORDS)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('NEAR(" AND OR NOT *', id="operators-and-quote"),
        pytest.param("harbour?", id="trailing-question-mark"),
        pytest.param('kestrel "labs', id="unbalanced-quote"),
        pytest.param("col:harbour AND -sam +kestrel ^labs", id="syntax-characters"),
        pytest.param("Zürich café 東京", id="unicode"),
    ],
)
def test_expression_is_valid_fts5_syntax(fts_table: sqlite3.Connection, text: str) -> None:
    expr = build_or_query(text)

    assert expr is not None
    _match(fts_table, expr)


def test_or_mode_matches_a_row_containing_any_token(fts_table: sqlite3.Connection) -> None:
    expr = build_or_query("which harbour did the unknownword visit")

    assert expr is not None
    assert _match(fts_table, expr) == [1, 2]


def test_quoted_operator_word_matches_the_literal_word(fts_table: sqlite3.Connection) -> None:
    expr = build_or_query("NEAR")

    assert expr is not None
    assert _match(fts_table, expr) == [1]


def test_escaped_quote_token_is_valid_fts5(fts_table: sqlite3.Connection) -> None:
    assert _match(fts_table, quote_token('kestrel"labs')) == [1]
