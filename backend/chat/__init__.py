"""Conversation handling with knowledge graph integration.

MCP-like tool access for autonomous knowledge retrieval and extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.chat.conversation_handler import ConversationHandler

__all__ = ["ConversationHandler"]


def __getattr__(name: str):
    """Lazily import ConversationHandler on first access (PEP 562).

    conversation_handler.py pulls in context_budget -> backend.knowledge.config
    -> dotenv, a dependency the CI container (pip + pre-commit only) does not
    have. Eagerly importing ConversationHandler at package-import time meant
    `import backend.chat` -- and therefore `import backend.chat.slop_detector`,
    which sits in this same package -- always paid that cost, even though
    slop_detector.py itself imports only re, dataclasses and typing. Deferring
    the import until ConversationHandler is actually used lets slop_detector
    (and anything else in this package) be imported without dotenv installed.
    """
    if name == "ConversationHandler":
        from backend.chat.conversation_handler import ConversationHandler

        return ConversationHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
