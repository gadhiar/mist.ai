"""Utility for cleaning up Ollama service on application exit."""

import atexit
import logging
import subprocess  # nosec B404 -- fixed-argv call only; see stop_ollama below

logger = logging.getLogger(__name__)


def stop_ollama():
    """Stop Ollama service to free VRAM."""
    try:
        # Suppressions on the call line are deliberate, not blanket: the argv is a
        # hardcoded literal with no interpolated or user-supplied component, and
        # `shell=False` (the default) means no shell parsing occurs. B607 (partial
        # path) is accepted knowingly -- `ollama` resolves from PATH because its
        # install location varies per platform, and a wrong resolution can only
        # make this best-effort cleanup no-op.
        #
        # The IDs below are SPACE-separated on purpose. Under bandit 1.7.6 (the pinned
        # rev) `# nosec B603,B607` suppresses B607 and silently FAILS to suppress B603,
        # so the comma form reads as a working suppression while the hook still fails.
        # Verified by isolating the two edits: comma -> bandit Failed, space -> Passed.
        subprocess.run(["ollama", "stop"], check=False, capture_output=True)  # nosec B603 B607
        logger.info("Ollama service stopped successfully")
    except Exception as e:
        logger.error(f"Failed to stop Ollama: {e}")


def register_cleanup():
    """Register cleanup handler to run on exit."""
    atexit.register(stop_ollama)
    logger.info("Cleanup handler registered")
