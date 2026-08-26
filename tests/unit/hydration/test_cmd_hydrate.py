"""`mist_admin hydrate`: the composition, and the order its checks run in.

The command deliberately writes no driver of its own. `run_replay` already
reads a per-line `session_id` from JSONL and already builds a real
`ConversationHandler` through the factories, which is where the hydration clock
is wired -- MIS-129 names writing a second driver as the specific wrong turn.
What this command adds is preflight -> replay -> postcondition.

These tests cover the parts that are cheap to exercise without an LLM: the
refusal ordering and the corpus-shape guard. The full path was verified against
the live container by running the command there and confirming exit 2 with no
handler constructed.
"""

from __future__ import annotations

import argparse
import json

import pytest

from scripts import mist_admin

_ISOLATION = "MIST_HYDRATION_ISOLATION"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_ISOLATION, raising=False)


def _args(input_path, **kw):
    return argparse.Namespace(input=str(input_path), session_id=None, user_id="User", **kw)


def _corpus(tmp_path, rows):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


class TestRefusesBeforeTouchingAnything:
    def test_no_isolation_returns_two_without_loading_the_backend(self, tmp_path, monkeypatch):
        """Ordering, and the reason for it.

        Building a `ConversationHandler` on the live container attaches the
        live event store. Harmless in itself, but a command whose answer is
        "no" should reach that answer before touching what it is refusing to
        touch. Isolation needs neither the corpus nor a handler, so it runs
        first.

        Asserting `_load_backend` was never called is what makes this a test of
        ORDER rather than of outcome -- the exit code alone would pass even if
        the check ran last.
        """
        called = []
        monkeypatch.setattr(mist_admin, "_load_backend", lambda: called.append(1))

        corpus = _corpus(tmp_path, [{"session_id": "s", "turn_index": 0, "utterance": "hi"}])
        assert mist_admin.cmd_hydrate(_args(corpus)) == 2
        assert called == [], "backend must not be loaded when the command refuses"

    def test_a_missing_corpus_is_not_reached_before_the_isolation_refusal(
        self, tmp_path, monkeypatch
    ):
        """A nonexistent path would raise FileNotFoundError if reached.

        Getting 2 instead proves isolation was checked first, which is the
        same ordering property from the other side.
        """
        monkeypatch.setattr(mist_admin, "_load_backend", lambda: None)
        assert mist_admin.cmd_hydrate(_args(tmp_path / "nope.jsonl")) == 2


class TestCorpusShapeGuard:
    def test_rows_without_the_clock_keys_are_refused(self, tmp_path, monkeypatch):
        """The clock keys on session_id and turn_index; a corpus lacking them
        cannot carry an authored timeline at all.
        """
        monkeypatch.setenv(_ISOLATION, "1")
        monkeypatch.setattr(mist_admin, "_load_backend", lambda: None)

        corpus = _corpus(tmp_path, [{"utterance": "hi"}])
        assert mist_admin.cmd_hydrate(_args(corpus)) == 1

    def test_it_names_the_first_offending_line(self, tmp_path, monkeypatch, capsys):
        """A corpus author needs a line number, not a count alone."""
        monkeypatch.setenv(_ISOLATION, "1")
        monkeypatch.setattr(mist_admin, "_load_backend", lambda: None)

        rows = [
            {"session_id": "s", "turn_index": 0, "utterance": "ok"},
            {"utterance": "missing keys"},
        ]
        mist_admin.cmd_hydrate(_args(_corpus(tmp_path, rows)))
        assert "line 2" in capsys.readouterr().err

    def test_an_empty_corpus_is_a_noop_not_a_failure(self, tmp_path, monkeypatch):
        """Nothing to drive is not the same as something went wrong."""
        monkeypatch.setenv(_ISOLATION, "1")
        monkeypatch.setattr(mist_admin, "_load_backend", lambda: None)

        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert mist_admin.cmd_hydrate(_args(path)) == 0


class TestRegistered:
    def test_hydrate_is_a_real_subcommand(self):
        """A command nothing can invoke is not a command."""
        parser = mist_admin.build_parser()
        args = parser.parse_args(["hydrate", "some/corpus.jsonl"])
        assert args.func is mist_admin.cmd_hydrate
        assert args.input == "some/corpus.jsonl"
