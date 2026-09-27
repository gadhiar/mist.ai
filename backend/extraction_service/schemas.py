"""Permissive JSON schemas for constrained decoding of the three stage outputs.

These are deliberately loose: each schema must accept every shape the
corresponding parser in `backend.knowledge.extraction` already accepts
(`parse_scope_output`, `parse_extraction_output`, `parse_derivation_output`).
`additionalProperties: True` throughout so a model that emits an extra
field (e.g. a "reasoning" aside, or a future ontology property) is not
rejected by the constrained-decoding grammar. Used when
`ServiceSettings.constrained_mode` (or the active adapter's
`default_constrained_mode`) resolves to `"schema"`.
"""

SCOPE_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "scope": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["scope"],
    "additionalProperties": True,
}

EXTRACTION_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        },
        "relationships": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        },
    },
    "required": ["entities", "relationships"],
    "additionalProperties": True,
}

DERIVATION_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        },
    },
    "required": ["operations"],
    "additionalProperties": True,
}
