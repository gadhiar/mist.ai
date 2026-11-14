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

The service admits this file alone: `CHAT_TEMPLATE_SHA256` pins its sha256
for the gpt-oss adapter, and `chat_template_problem` names any other template
`/props` serves. Edit the file, the pin and this test together.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from backend.extraction_service.adapters import CHAT_TEMPLATE_SHA256, chat_template_problem

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


class TestTheServiceAdmitsTheShippedTemplateAlone:
    """The pin the service checks `/props` against, and the check itself (Acceptance 4)."""

    def test_the_gpt_oss_pin_is_the_shipped_files_sha256(self):
        expected = {"gptoss": hashlib.sha256(_shipped()).hexdigest()}

        assert expected == CHAT_TEMPLATE_SHA256

    def test_the_shipped_template_is_admitted_under_gpt_oss(self):
        assert chat_template_problem("gptoss", hashlib.sha256(_shipped()).hexdigest()) is None

    def test_another_template_or_none_is_refused_under_gpt_oss(self):
        shipped = hashlib.sha256(_shipped()).hexdigest()

        assert chat_template_problem("gptoss", MODEL_TEMPLATE_SHA256) == (
            f"served chat template sha256 {MODEL_TEMPLATE_SHA256} != the gptoss adapter's "
            f"{shipped}"
        )
        assert chat_template_problem("gptoss", None) == (
            f"served chat template sha256 none != the gptoss adapter's {shipped}"
        )

    def test_qwen_and_gemma_are_refused_the_gpt_oss_template(self):
        shipped = hashlib.sha256(_shipped()).hexdigest()

        for adapter in ("qwen", "gemma"):
            assert chat_template_problem(adapter, shipped) == (
                f"served chat template sha256 {shipped} is the gptoss adapter's, "
                f"not the {adapter} adapter's"
            )

    def test_qwen_and_gemma_are_admitted_any_other_template(self):
        for adapter in ("qwen", "gemma"):
            assert chat_template_problem(adapter, MODEL_TEMPLATE_SHA256) is None
            assert chat_template_problem(adapter, None) is None

    def test_an_unregistered_adapter_is_not_checked(self):
        assert chat_template_problem("not-a-registered-adapter", None) is None
        assert chat_template_problem("fake", MODEL_TEMPLATE_SHA256) is None
