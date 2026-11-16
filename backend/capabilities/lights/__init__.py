"""MIST's lights capability, over Home Assistant's MCP server.

`capability.py` holds `LightsCapability` and the `LightsProvider` protocol it runs behind;
`home_assistant.py` the provider over an `McpConnection`; `catalog.py` the target catalog parsed
from Home Assistant's `GetLiveContext` text; `intents.py` the lights rows of the fast-path
grammar.

This package file holds no imports, as the parent package's does not.
"""
