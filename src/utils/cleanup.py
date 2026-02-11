"""
Utility for cleaning up Ollama service on application exit.
"""

import atexit
import logging
import subprocess

logger = logging.getLogger(__name__)


def stop_ollama():
    """Stop Ollama service to free VRAM."""
    try:
        subprocess.run(["ollama", "stop"], check=False, capture_output=True)
        logger.info("Ollama service stopped successfully")
    except Exception as e:
        logger.error(f"Failed to stop Ollama: {e}")


def register_cleanup():
    """Register cleanup handler to run on exit."""
    atexit.register(stop_ollama)
    logger.info("Cleanup handler registered")
