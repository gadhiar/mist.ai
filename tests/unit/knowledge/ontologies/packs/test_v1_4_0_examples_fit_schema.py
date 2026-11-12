"""Every example output the prompt teaches fits the Stage 2 schema the service sends.

The schema constrains the model's output, so an example it refuses teaches a shape the
model cannot write. The grammar also emits required keys in the schema's `properties`
order, so each example keeps that order: `id`, `name`, `type` on an entity, `source`,
`target`, `type` on a relationship, and `value`, `unit`, `quantity` in a Metric's
properties.
"""

from __future__ import annotations

import json

import jsonschema
import pytest

from backend.knowledge.ontologies.packs.v1_4_0.conventions import PROMPT_EXAMPLES
from backend.knowledge.ontologies.registry import active_pack


def _output(example: str) -> dict:
    return json.loads(example.split("\nOutput:\n", 1)[1].strip())


def _number(example: str) -> str:
    return example.split("### Example ", 1)[1].split(":", 1)[0]


EXAMPLES = [pytest.param(example, id=f"example-{_number(example)}") for example in PROMPT_EXAMPLES]


def test_every_example_has_an_output():
    assert len(EXAMPLES) > 0
    assert all("\nOutput:\n" in example for example in PROMPT_EXAMPLES)


@pytest.mark.parametrize("example", EXAMPLES)
def test_the_example_output_validates_against_the_generated_schema(example):
    jsonschema.validate(_output(example), active_pack().extraction_schema)


@pytest.mark.parametrize("example", EXAMPLES)
def test_the_example_writes_its_keys_in_the_schema_s_order(example):
    output = _output(example)
    for entity in output["entities"]:
        assert list(entity)[:3] == ["id", "name", "type"]
        if entity["type"] == "Metric" and "properties" in entity:
            assert list(entity["properties"])[:3] == ["value", "unit", "quantity"]
    for relationship in output["relationships"]:
        assert list(relationship)[:3] == ["source", "target", "type"]


def test_a_metric_missing_a_written_key_is_refused():
    example = next(e for e in PROMPT_EXAMPLES if '"type": "Metric", "properties"' in e)
    output = _output(example)
    metric = next(e for e in output["entities"] if e["type"] == "Metric")
    del metric["properties"]["quantity"]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(output, active_pack().extraction_schema)
