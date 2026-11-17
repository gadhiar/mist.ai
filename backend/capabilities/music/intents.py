"""The music rows of the fast-path grammar, over `normalize_utterance` output.

A common playback phrase ("pause", "skip this song", "volume 40") is an explicit command, so it
runs without the model. The rows (build plan, Fast-path intents; `[]` optional, `|`
alternatives):

- pause [the] [music]; stop [the] music; (turn|switch) off [the] music;
  (turn|switch) [the] music off -> `music.control` pause
- resume [the] [music]; play -> `music.control` resume
- (skip|next) [this] [song|track] -> `music.control` next
- (previous|go back) [song|track] -> `music.control` previous
- volume (up|down); (louder|quieter); turn it (up|down) -> `music.volume` `change`
- [set] [the] volume [to] <N> [percent] -> `music.volume` `level_pct`
- whats playing; what is playing; what song is this -> `music.now_playing`
- play [me] [some] <query> -> `music.play` `query`

The play row is the one that can catch speech not meant as a command ("play tennis with me
tomorrow"), so its `<query>` is held to these rules, and an utterance that breaks one goes to
the decode instead:

- it is 1 to `MAX_QUERY_WORDS` words;
- it holds no slot stop word (`SLOT_STOP_WORDS`), so a compound request ("play some jazz and
  tell me about Coltrane") is decoded as a whole;
- it holds no day word (`DAY_WORDS`) and no time: a clock time ("7:30", "5pm"), "am" or "pm"
  after a number ("6 am"), "noon" or "midnight", or a number after "at", "until" or "till". A
  scheduled request ("play something tonight", "play the piano at 5") is not a command to play
  now. A bare number stays in the query, because titles hold numbers ("7 rings", "1999"), and
  so do "am" and "by" on their own ("play i am the walrus", "play hits by 50 cent").

A query of "music" alone names no title, so "play some music" gives `music.play` with no query,
which resumes.
"""

from __future__ import annotations

import re

from ..contracts import FastIntent

SLOT_STOP_WORDS: frozenset[str] = frozenset({"and", "then", "also", "with"})
"""Words a play query never holds; a request containing one is a compound for the decode."""

MAX_QUERY_WORDS = 8
"""The longest play query the fast path takes; a longer one is more likely a sentence."""

DAY_WORDS: frozenset[str] = frozenset(
    {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "today",
        "tomorrow",
        "tonight",
        "yesterday",
        "weekend",
        "weekends",
        "weekday",
        "weekdays",
        "morning",
        "afternoon",
        "evening",
    }
)
"""Words that place a play query in time, so the request is not one to play now."""

_TIME_WORDS = frozenset({"noon", "midnight"})
# "am" and "pm" are a time only after a number, because "am" is also a common word.
_MERIDIEMS = frozenset({"am", "pm"})
_CLOCK_TIME = re.compile(r"^\d{1,2}(?::\d{2})?(?:am|pm)$|^\d{1,2}:\d{2}$")
_NUMBER = re.compile(r"^\d+$")
# Words that, before a number, make it a time ("at 5", "until 6"). "by" is left out: "hits by
# 50 cent" names an artist.
_TIME_PREPOSITIONS = frozenset({"at", "until", "till"})
# The query words that name no title, and so mean "play on".
_GENERIC_QUERIES = frozenset({"music"})
# A query of only the play row's own optional words is no query.
_FILLER = frozenset({"me", "some"})

_CONTROL = "music.control"
_VOLUME = "music.volume"
_PLAY = "music.play"
_NOW_PLAYING = "music.now_playing"

_PAUSE = re.compile(
    r"^(?:pause(?: the)?(?: music)?"
    r"|stop(?: the)? music"
    r"|(?:turn|switch) off(?: the)? music"
    r"|(?:turn|switch)(?: the)? music off)$"
)
_RESUME = re.compile(r"^(?:resume(?: the)?(?: music)?|play)$")
_NEXT = re.compile(r"^(?:(?:skip|next)(?: this)?(?: (?:song|track))?|next song)$")
_PREVIOUS = re.compile(r"^(?:(?:previous|go back)(?: (?:song|track))?|previous song)$")
_CHANGE = re.compile(r"^(?:volume (?P<a>up|down)|(?P<b>louder|quieter)|turn it (?P<c>up|down))$")
_LEVEL = re.compile(r"^(?:set )?(?:the )?volume(?: to)? (?P<percent>\d{1,3})(?: percent)?$")
_NOW = re.compile(r"^(?:whats playing|what is playing|what song is this)$")
_PLAY_QUERY = re.compile(r"^play(?: me)?(?: some)? (?P<query>.+)$")

_LOUDER = {"louder": "up", "quieter": "down"}


def match_music_intent(text: str) -> FastIntent | None:
    """Match one normalised utterance against the music rows.

    Args:
        text: `normalize_utterance` output.

    Returns:
        The matched call with unvalidated arguments, as `FastIntent` requires: `music.control`
        with `action`, `music.volume` with `change` or `level_pct`, `music.now_playing` with
        none, or `music.play` with `query` (none for a generic query). None when no row
        matches.
    """
    if _PAUSE.match(text):
        return _control("pause")
    if _RESUME.match(text):
        return _control("resume")
    if _NEXT.match(text):
        return _control("next")
    if _PREVIOUS.match(text):
        return _control("previous")
    match = _CHANGE.match(text)
    if match is not None:
        word = match["a"] or match["c"] or _LOUDER[match["b"]]
        return FastIntent(tool=_VOLUME, args={"change": word})
    match = _LEVEL.match(text)
    if match is not None:
        return FastIntent(tool=_VOLUME, args={"level_pct": int(match["percent"])})
    if _NOW.match(text):
        return FastIntent(tool=_NOW_PLAYING, args={})
    match = _PLAY_QUERY.match(text)
    if match is not None:
        return _play(match["query"].split())
    return None


def _control(action: str) -> FastIntent:
    return FastIntent(tool=_CONTROL, args={"action": action})


def _play(words: list[str]) -> FastIntent | None:
    if all(word in _FILLER for word in words) or len(words) > MAX_QUERY_WORDS:
        return None
    if any(word in SLOT_STOP_WORDS or word in DAY_WORDS for word in words):
        return None
    if _holds_time(words):
        return None
    if all(word in _GENERIC_QUERIES for word in words):
        return FastIntent(tool=_PLAY, args={})
    return FastIntent(tool=_PLAY, args={"query": " ".join(words)})


def _holds_time(words: list[str]) -> bool:
    for index, word in enumerate(words):
        if word in _TIME_WORDS or _CLOCK_TIME.match(word):
            return True
        if word in _MERIDIEMS and index > 0 and _NUMBER.match(words[index - 1]):
            return True
        if (
            word in _TIME_PREPOSITIONS
            and index + 1 < len(words)
            and _NUMBER.match(words[index + 1])
        ):
            return True
    return False
