"""MIST's music capability over the Spotify Web API.

`pkce.py` holds the PKCE verifier, challenge, `state` and authorize URL; `token_store.py` the
token file; `spotify_api.py` the accounts token endpoint, the token manager and the Web API
client. Every HTTP client is injected, so tests run against `httpx.MockTransport`.

This package file holds no imports, as the parent package's does not.
"""
