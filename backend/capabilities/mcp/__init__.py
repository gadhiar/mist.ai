"""MIST's Model Context Protocol client: a transport protocol and a supervised connection.

`transport.py` holds the protocol a transport implements and the plain types it returns;
`connection.py` holds `McpConnection`, which owns one transport's lifetime on the main loop.
Neither module imports the `mcp` SDK.

This package file holds no imports, as the parent package's does not.
"""
