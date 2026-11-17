"""Tests for the music rows of the fast-path grammar (`music/intents.py`).

Every utterance goes through `normalize_utterance` first, as the host's fast path does, so the
fixtures are written as a user would say or type them.
"""

from __future__ import annotations

import pytest

from backend.capabilities.contracts import FastIntent
from backend.capabilities.intents import normalize_utterance
from backend.capabilities.music.intents import (
    DAY_WORDS,
    MAX_QUERY_WORDS,
    SLOT_STOP_WORDS,
    match_music_intent,
)

WAKE_WORDS = ("mist", "hey mist", "ok mist", "okay mist")


def _match(utterance: str) -> FastIntent | None:
    return match_music_intent(normalize_utterance(utterance, WAKE_WORDS))


def _control(action: str) -> FastIntent:
    return FastIntent(tool="music.control", args={"action": action})


def _change(change: str) -> FastIntent:
    return FastIntent(tool="music.volume", args={"change": change})


def _level(percent: int) -> FastIntent:
    return FastIntent(tool="music.volume", args={"level_pct": percent})


def _play(query: str | None) -> FastIntent:
    return FastIntent(tool="music.play", args={} if query is None else {"query": query})


NOW_PLAYING = FastIntent(tool="music.now_playing", args={})

MATCHING = [
    ("pause", _control("pause")),
    ("Pause the music.", _control("pause")),
    ("pause music", _control("pause")),
    ("stop the music", _control("pause")),
    ("stop music", _control("pause")),
    ("turn off the music", _control("pause")),
    ("switch the music off", _control("pause")),
    ("resume", _control("resume")),
    ("resume the music please", _control("resume")),
    ("play", _control("resume")),
    ("Hey MIST, play", _control("resume")),
    ("skip", _control("next")),
    ("skip this song", _control("next")),
    ("next", _control("next")),
    ("next song", _control("next")),
    ("skip track", _control("next")),
    ("go back", _control("previous")),
    ("previous", _control("previous")),
    ("previous song", _control("previous")),
    ("go back track", _control("previous")),
    ("volume up", _change("up")),
    ("volume down", _change("down")),
    ("louder", _change("up")),
    ("quieter, please", _change("down")),
    ("turn it up", _change("up")),
    ("turn it down", _change("down")),
    ("set the volume to 40 percent", _level(40)),
    ("volume 70", _level(70)),
    ("set volume to forty per cent", _level(40)),
    ("the volume to 25%", _level(25)),
    ("what's playing", NOW_PLAYING),
    ("What is playing?", NOW_PLAYING),
    ("what song is this", NOW_PLAYING),
    ("play some jazz", _play("jazz")),
    ("play me Bohemian Rhapsody", _play("bohemian rhapsody")),
    ("play 7 rings", _play("7 rings")),
    ("play seven rings", _play("7 rings")),
    ("Can you play me some Miles Davis", _play("miles davis")),
    ("play kind of blue", _play("kind of blue")),
    ("play one two three four five six seven eight", _play("1 2 3 4 5 6 7 8")),
    # "music" alone names no title, so it plays on from where Spotify stopped.
    ("play some music", _play(None)),
    ("play music", _play(None)),
]

NOT_MATCHING = [
    "play tennis with me tomorrow",
    "play the piano at 5",
    "play something tonight",
    "play some jazz and tell me about Coltrane",
    "I love this song",
    "stop",
    "turn off",
    "playing music helps me focus",
    "play rock and roll",
    "play jazz then pause",
    "play jazz also",
    "play jazz at 7pm",
    "play jazz on monday",
    "play the news today",
    "play a song at 7:30",
    "play one two three four five six seven eight nine",
    "play me",
    "play some",
    "skip breakfast",
    "go back home",
    "volume",
    "turn off the lights",
    "what time is it",
    "set the volume to the max",
    "pause the alarm",
    "",
]


@pytest.mark.parametrize(("utterance", "expected"), MATCHING)
def test_matching_utterances_give_the_expected_call(utterance: str, expected: FastIntent) -> None:
    assert _match(utterance) == expected


@pytest.mark.parametrize("utterance", NOT_MATCHING)
def test_non_matching_utterances_give_none(utterance: str) -> None:
    assert _match(utterance) is None


def test_the_fixtures_meet_the_acceptance_minimums() -> None:
    assert len(MATCHING) >= 25
    assert len(NOT_MATCHING) >= 15


def test_stop_words_are_the_fast_path_tables() -> None:
    assert frozenset({"and", "then", "also", "with"}) == SLOT_STOP_WORDS


@pytest.mark.parametrize("word", sorted(SLOT_STOP_WORDS))
def test_a_query_holding_any_stop_word_goes_to_the_decode(word: str) -> None:
    assert _match(f"play miles {word} davis") is None


@pytest.mark.parametrize("word", sorted(DAY_WORDS))
def test_a_query_holding_any_day_word_goes_to_the_decode(word: str) -> None:
    assert _match(f"play jazz {word}") is None


@pytest.mark.parametrize(
    "phrase", ["at 5", "at 11", "7:30", "5pm", "6 am", "at noon", "until 6", "by midnight"]
)
def test_a_query_holding_a_time_goes_to_the_decode(phrase: str) -> None:
    assert _match(f"play jazz {phrase}") is None


def test_the_query_is_one_to_eight_words() -> None:
    assert MAX_QUERY_WORDS == 8
    assert _match("play a") == _play("a")
    assert _match("play " + " ".join(["la"] * 8)) == _play(" ".join(["la"] * 8))
    assert _match("play " + " ".join(["la"] * 9)) is None


def test_numbers_without_a_time_word_stay_in_the_query() -> None:
    # "7 rings" and "1999" are titles; only a number after "at", "until" or "till", or before
    # am or pm, reads as a time.
    assert _match("play 1999") == _play("1999")
    assert _match("play 5 years") == _play("5 years")


def test_am_and_by_alone_stay_in_the_query() -> None:
    assert _match("play I am the walrus") == _play("i am the walrus")
    assert _match("play hits by 50 cent") == _play("hits by 50 cent")
