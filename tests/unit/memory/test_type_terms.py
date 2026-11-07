"""Tests for backend.memory.type_terms: the words a type-scoped question uses for each type."""

from __future__ import annotations

import re

import pytest

from backend.knowledge.ontologies import EXTRACTABLE_NODE_TYPES
from backend.memory.fts import STOPWORDS, fold, words
from backend.memory.type_terms import TYPE_TERMS, type_terms_text


def test_covers_exactly_the_extractable_node_types():
    assert set(TYPE_TERMS) == set(EXTRACTABLE_NODE_TYPES)


@pytest.mark.parametrize("type_name", sorted(TYPE_TERMS))
def test_every_type_has_at_least_one_term(type_name):
    assert TYPE_TERMS[type_name]
    assert isinstance(TYPE_TERMS[type_name], tuple)


@pytest.mark.parametrize("type_name", sorted(TYPE_TERMS))
def test_each_term_is_one_searchable_token_in_folded_form(type_name):
    for term in TYPE_TERMS[type_name]:
        # One token, already in the form the index stores, and one a query keeps: a stopword
        # or a one-letter token is dropped from every query, so it could never match.
        assert words(term) == [term], term
        assert fold(term) == term, term
        assert len(term) >= 2, term
        assert term not in STOPWORDS, term


@pytest.mark.parametrize("type_name", sorted(TYPE_TERMS))
def test_terms_include_the_words_of_the_type_name(type_name):
    name_words = [part.lower() for part in re.findall(r"[A-Z][a-z]*", type_name)]

    assert set(name_words) <= set(TYPE_TERMS[type_name])


@pytest.mark.parametrize("type_name", sorted(TYPE_TERMS))
def test_terms_do_not_repeat(type_name):
    terms = TYPE_TERMS[type_name]

    assert len(terms) == len(set(terms))


def test_person_covers_the_irregular_plural():
    # Porter stems "technologies" to the stem of "technology", but not "people" to "person".
    assert "people" in TYPE_TERMS["Person"]


class TestTypeTermsText:
    def test_joins_the_terms_of_both_types(self):
        text = type_terms_text("User", "Technology")

        assert words(text) == list(TYPE_TERMS["User"]) + list(TYPE_TERMS["Technology"])

    def test_a_term_shared_by_both_types_appears_once(self):
        text = type_terms_text("Person", "Person")

        assert words(text) == list(TYPE_TERMS["Person"])

    @pytest.mark.parametrize(
        ("subject_type", "object_type"),
        [(None, None), ("MistTrait", None), (None, "NotAType"), ("", "")],
    )
    def test_an_absent_or_unknown_type_adds_nothing(self, subject_type, object_type):
        assert type_terms_text(subject_type, object_type) == ""

    def test_one_known_type_is_enough(self):
        assert words(type_terms_text(None, "Location")) == list(TYPE_TERMS["Location"])
