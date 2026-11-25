"""The speaking states the voice data is labelled and generated in.

Clips are not labelled with these names directly: `labels.py` records each clip's energy, mood and
tags, and `delivery.json` gives each state a region over them, onto which `labels.project` maps a
clip (or onto no state, when it lies outside every region).

Goal mist-realtime-voice's realtime contract owns this vocabulary (its v6 plan, line 583): the
live speech path selects a state per turn, so the data this pipeline produces must use exactly its
names, in its order. These are a copy kept equal to that contract's, not a second source of
truth: a change of vocabulary starts there.
"""

NEUTRAL_STATE = "neutral"

SPEAKING_STATES = ("neutral", "calm", "awake", "sleepy", "excited")
