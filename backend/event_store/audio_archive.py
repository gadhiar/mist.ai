"""Content-addressable audio file storage.

Stores audio files using their SHA-256 hash as the filename, organized
in a two-level directory hierarchy for filesystem scalability:

    {base_dir}/{hash[:2]}/{hash[2:4]}/{hash}.{format}

Writes are idempotent -- storing the same audio twice is a no-op.
"""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default path under ~/.mist/
_DEFAULT_AUDIO_DIR = Path.home() / ".mist" / "audio"


class AudioArchive:
    """Content-addressable audio file storage."""

    def __init__(self, base_dir: str | None = None) -> None:
        """Initialize the archive.

        Args:
            base_dir: Root directory for audio storage.
                Defaults to ~/.mist/audio/.
        """
        self.base_dir = Path(base_dir) if base_dir else _DEFAULT_AUDIO_DIR

    def store(self, audio_data: bytes, format: str = "wav") -> str:
        """Store audio data by content hash. Idempotent.

        If a file with the same hash already exists, the write is skipped.

        Args:
            audio_data: Raw audio bytes.
            format: File extension without dot (e.g., "wav", "opus").

        Returns:
            SHA-256 hex digest string identifying the stored audio.

        Raises:
            ValueError: If audio_data is empty.
            OSError: If the file cannot be written.
        """
        if not audio_data:
            raise ValueError("audio_data must not be empty")

        content_hash = self._compute_hash(audio_data)
        file_path = self.get_path(content_hash, format)

        if file_path.exists():
            logger.debug("Audio %s already archived, skipping write", content_hash[:12])
            return content_hash

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(audio_data)

        logger.info(
            "Archived audio %s (%d bytes, %s)",
            content_hash[:12],
            len(audio_data),
            format,
        )
        return content_hash

    def retrieve(self, audio_hash: str, format: str = "wav") -> bytes | None:
        """Retrieve audio by content hash.

        Args:
            audio_hash: SHA-256 hex digest.
            format: File extension without dot.

        Returns:
            Raw audio bytes, or None if not found.
        """
        file_path = self.get_path(audio_hash, format)

        if not file_path.exists():
            logger.debug("Audio %s not found in archive", audio_hash[:12])
            return None

        return file_path.read_bytes()

    def exists(self, audio_hash: str, format: str = "wav") -> bool:
        """Check if audio exists in the archive.

        Args:
            audio_hash: SHA-256 hex digest.
            format: File extension without dot.

        Returns:
            True if the file exists.
        """
        return self.get_path(audio_hash, format).exists()

    def get_path(self, audio_hash: str, format: str = "wav") -> Path:
        """Compute the filesystem path for a given content hash.

        Layout: {base_dir}/{hash[:2]}/{hash[2:4]}/{hash}.{format}

        Args:
            audio_hash: SHA-256 hex digest.
            format: File extension without dot.

        Returns:
            pathlib.Path to the (possibly non-existent) file.
        """
        return self.base_dir / audio_hash[:2] / audio_hash[2:4] / f"{audio_hash}.{format}"

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        """Compute SHA-256 hex digest of data.

        Args:
            data: Bytes to hash.

        Returns:
            Lowercase hex digest string (64 characters).
        """
        return hashlib.sha256(data).hexdigest()
