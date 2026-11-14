"""The gpt-oss chat template llama-server loads from the repository (T62, A71).

`docker/extraction/chat_templates/gpt-oss.jinja` is the template embedded in
gpt-oss-20b's GGUF file (`tokenizer.chat_template`), byte for byte, with two
edits:

1. The system message's date comes from the request's `current_date` template
   variable, not from `strftime_now`, which reads llama-server's clock at every
   render. A run on another day then sends the same bytes for the same turn.
2. The assistant-content check that llama.cpp b11151 rewrites at load
   (`common/chat.cpp:788-797`) is rewritten in the file itself, so llama.cpp
   finds nothing to rewrite and `/props` serves the file's bytes.

Reversing both edits must give back the model's own template, whose sha256 is
the one H30 printed for the 16,738 bytes of the GGUF key's value; that is also
the sha256 of llama.cpp's `models/templates/openai-gpt-oss-120b.jinja` at
b11151. Rendering is not tested: the test image has no llama.cpp.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "docker" / "extraction" / "chat_templates" / "gpt-oss.jinja"

# H30: `sha256sum` of the model's embedded template.
MODEL_TEMPLATE_SHA256 = "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146"

# Edit 1, line 202: the model's line, and the shipped one.
MODEL_DATE_LINE = rb'    {{- "Current date: " + strftime_now("%Y-%m-%d") + "\n\n" }}'
SHIPPED_DATE_LINE = (
    rb"    {%- if current_date is defined %}"
    rb'{{- "Current date: " + current_date + "\n\n" }}'
    rb"{%- endif %}"
)

# Edit 2, line 264: the model's clause, and the shipped one.
MODEL_CONTENT_CHECK = (
    rb'{%- if "<|channel|>analysis<|message|>" in message.content'
    rb' or "<|channel|>final<|message|>" in message.content %}'
)
SHIPPED_CONTENT_CHECK = b"{%- if false %}"


def _shipped() -> bytes:
    return TEMPLATE_PATH.read_bytes()


class TestTheShippedTemplateIsTheModelsOwn:
    def test_each_edit_is_found_exactly_once(self):
        shipped = _shipped()

        assert shipped.count(SHIPPED_DATE_LINE) == 1
        assert shipped.count(SHIPPED_CONTENT_CHECK) == 1

    def test_reversing_both_edits_gives_the_models_template(self):
        reversed_bytes = (
            _shipped()
            .replace(SHIPPED_DATE_LINE, MODEL_DATE_LINE)
            .replace(SHIPPED_CONTENT_CHECK, MODEL_CONTENT_CHECK)
        )

        assert len(reversed_bytes) == 16738
        assert hashlib.sha256(reversed_bytes).hexdigest() == MODEL_TEMPLATE_SHA256

    def test_it_ends_without_a_newline_as_the_models_does(self):
        assert _shipped().endswith(b"{%- endif -%}")


class TestTheShippedTemplateReadsNoClock:
    def test_it_never_reads_the_servers_clock(self):
        assert b"strftime_now" not in _shipped()

    def test_llama_cpp_finds_nothing_to_rewrite_at_load(self):
        # b11151 rewrites the template when it holds this clause
        # (`common/chat.cpp:790-792`); `/props` would then serve other bytes.
        assert b"in message.content or" not in _shipped()

    def test_current_date_is_read_on_the_date_line_alone(self):
        lines = [line for line in _shipped().split(b"\n") if b"current_date" in line]

        assert lines == [SHIPPED_DATE_LINE]
