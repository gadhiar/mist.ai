"""Deterministic correctness/regression probe against a running llama-server.

20 fixed prompts (below), sent non-streamed with greedy, zero-temperature
sampling so the same arm on the same weights should reproduce the same
token ids run to run. Comparing `tokens_sha256` across arms/runs is how the
lead spots a regression without diffing raw text.

UNVERIFIED: `return_tokens: true` is assumed to add a `"tokens": [ids]`
field to llama-server's non-streamed /completion response (current
llama.cpp server README documents `return_tokens` as a request option but
this driver's fixtures are hand-built, not recorded against a live b11151
server -- see README.md). If the deployed server instead omits the field,
uses a different key, or nests it under `tokens_predicted` semantics,
`parse_correctness_response` raises `CorrectnessProbeError` naming the
response keys it actually saw rather than guessing at a field.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Any

N_PREDICT = 256
SEED = 3407

# 20 fixed, deterministic prompts covering a spread of task shapes (factual
# recall, arithmetic, instruction-following, short code, refusal-adjacent).
# Content is arbitrary; determinism (identical text every run) is what a
# regression comparison needs.
PROMPTS: tuple[dict[str, str], ...] = (
    {"id": "p01", "prompt": "What is the capital of France? Answer in one word."},
    {"id": "p02", "prompt": "Compute 17 * 23 and give only the number."},
    {"id": "p03", "prompt": "List the first five prime numbers, comma separated."},
    {"id": "p04", "prompt": "Write a one-sentence summary of what a knowledge graph is."},
    {"id": "p05", "prompt": "Translate 'good morning' into Spanish."},
    {"id": "p06", "prompt": "Reverse the string 'benchmark' and output only the result."},
    {"id": "p07", "prompt": "Name the largest planet in our solar system."},
    {"id": "p08", "prompt": "Write a Python function signature for adding two integers."},
    {"id": "p09", "prompt": "What year did the first moon landing occur?"},
    {"id": "p10", "prompt": "Explain what a hash function is, in exactly two sentences."},
    {"id": "p11", "prompt": "Convert 100 Fahrenheit to Celsius, rounded to one decimal place."},
    {"id": "p12", "prompt": "Give a synonym for 'happy' that is not 'glad'."},
    {"id": "p13", "prompt": "Sort these numbers ascending: 9, 2, 7, 1, 5."},
    {"id": "p14", "prompt": "What does the acronym 'HTTP' stand for?"},
    {"id": "p15", "prompt": "Write a haiku about a river."},
    {"id": "p16", "prompt": "Is the number 91 prime? Answer yes or no and why, briefly."},
    {"id": "p17", "prompt": "Name three colors that are not primary colors."},
    {"id": "p18", "prompt": "What is the chemical symbol for gold?"},
    {"id": "p19", "prompt": "Write a single-line regex that matches a US zip code."},
    {"id": "p20", "prompt": "Describe the difference between a list and a tuple in Python."},
)


class CorrectnessProbeError(RuntimeError):
    """The server response did not carry the token id field this probe needs."""


def build_request_payload(prompt: str) -> dict[str, Any]:
    """Build the non-streamed /completion request body for one prompt."""
    return {
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": 0,
        "top_k": 1,
        "cache_prompt": False,
        "seed": SEED,
        "return_tokens": True,
        "stream": False,
    }


def parse_correctness_response(prompt_id: str, response: dict[str, Any]) -> dict[str, Any]:
    """Reduce a decoded /completion JSON response to one correctness.jsonl row.

    Raises CorrectnessProbeError if the response has no usable token id
    list under the assumed `"tokens"` key -- see the module docstring's
    UNVERIFIED note. Callers should let this propagate as a loud, per-prompt
    `error` field rather than silently recording an empty token list.
    """
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        raise CorrectnessProbeError(
            f"response for {prompt_id!r} has no int 'tokens' list "
            f"(return_tokens field name is UNVERIFIED; response keys were {sorted(response)!r})"
        )
    digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode("utf-8")).hexdigest()
    n_predict = response.get("tokens_predicted", len(tokens))
    return {
        "prompt_id": prompt_id,
        "n_predict": n_predict,
        "tokens": tokens,
        "tokens_sha256": digest,
        "error": None,
    }


def send_request(base_url: str, prompt: str, *, timeout: float = 300.0) -> dict[str, Any]:
    """POST one non-streamed /completion request and return the decoded body.

    Not unit tested (needs a live server); `parse_correctness_response`,
    which consumes its output, is.
    """
    data = json.dumps(build_request_payload(prompt)).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_correctness_probe(base_url: str, *, timeout: float = 300.0) -> list[dict[str, Any]]:
    """Run all 20 fixed prompts and return correctness.jsonl rows in order."""
    rows: list[dict[str, Any]] = []
    for entry in PROMPTS:
        try:
            response = send_request(base_url, entry["prompt"], timeout=timeout)
            rows.append(parse_correctness_response(entry["id"], response))
        except Exception as exc:  # noqa: BLE001 -- one bad prompt must not abort the other 19
            rows.append(
                {
                    "prompt_id": entry["id"],
                    "n_predict": None,
                    "tokens": None,
                    "tokens_sha256": None,
                    "error": str(exc),
                }
            )
    return rows
