"""Tests for the Tier-3 numeric/date name veto (`backend.knowledge.curation.name_veto`)."""

import pytest

from backend.knowledge.curation.name_veto import NameTokens, name_tokens, names_conflict


class TestNameTokens:
    def test_digit_runs_become_integer_tokens(self):
        assert name_tokens("P95") == NameTokens(numbers=frozenset({"95"}), months=frozenset())
        assert name_tokens("Q1").numbers == frozenset({"1"})

    def test_date_has_numeric_and_month_tokens(self):
        tokens = name_tokens("March 13, 2026")
        assert tokens.numbers == frozenset({"13", "2026"})
        assert tokens.months == frozenset({3})

    def test_leading_zeros_do_not_change_the_value(self):
        assert name_tokens("03") == name_tokens("3")
        assert name_tokens("000").numbers == frozenset({"0"})

    def test_non_ascii_decimal_digits_map_to_their_value(self):
        # U+0663 ARABIC-INDIC DIGIT THREE is category Nd with value 3.
        assert name_tokens("٣").numbers == frozenset({"3"})

    def test_very_long_digit_run_does_not_raise(self):
        run = "9" * 5000  # beyond int()'s default 4300-digit string limit
        assert name_tokens(run).numbers == frozenset({run})

    @pytest.mark.parametrize(
        "text,month",
        [
            ("January", 1),
            ("jan", 1),
            ("FEB", 2),
            ("Mar", 3),
            ("April", 4),
            ("may", 5),
            ("Jun", 6),
            ("July", 7),
            ("Aug", 8),
            ("Sep", 9),
            ("Sept", 9),
            ("september", 9),
            ("Oct", 10),
            ("November", 11),
            ("DEC", 12),
        ],
    )
    def test_month_names_and_abbreviations_normalise_to_month_number(self, text, month):
        assert name_tokens(text).months == frozenset({month})

    def test_month_must_be_a_whole_letter_run(self):
        # 'Mars', 'Octopus', 'Maybe' contain month prefixes but are not months.
        assert name_tokens("Mars Octopus Maybe").months == frozenset()

    def test_month_adjacent_to_digits_is_still_a_month(self):
        tokens = name_tokens("March13")
        assert tokens.months == frozenset({3})
        assert tokens.numbers == frozenset({"13"})

    def test_name_without_numbers_or_months_has_empty_tokens(self):
        assert name_tokens("Neo4j graph database").months == frozenset()
        assert name_tokens("Python") == NameTokens(numbers=frozenset(), months=frozenset())


class TestNamesConflict:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("P95", "P99 latency"),
            ("March 3", "March 13, 2026"),
            ("Q1", "Q2 2026"),
            ("March 3", "April 3"),
            ("Python 3", "Python"),
            ("March", "3"),
        ],
    )
    def test_vetoes_names_with_different_numeric_or_month_tokens(self, a, b):
        assert names_conflict(a, b) is True
        assert names_conflict(b, a) is True

    @pytest.mark.parametrize(
        "a,b",
        [
            ("P95 latency", "P95 latency"),
            ("03", "3"),
            ("March 03", "march 3"),
            ("MARCH 3", "March 3"),
            ("Sept 9", "September 9"),
            ("Python programming language", "Python"),
            ("Tabs over spaces", "Spaces over tabs"),
            ("", ""),
        ],
    )
    def test_does_not_veto_names_with_equal_tokens(self, a, b):
        assert names_conflict(a, b) is False
        assert names_conflict(b, a) is False
