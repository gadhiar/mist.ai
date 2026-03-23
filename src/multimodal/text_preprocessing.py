"""Text preprocessing for TTS pipeline.

Provides prosody-preserving text cleaning for CSM-1B speech synthesis.
Instead of stripping punctuation (which destroys pacing cues the model
learned during training), non-speech characters are replaced with
prosody-equivalent substitutes.
"""

import re


def preprocess_text_for_tts(text: str) -> str:
    """Preprocess text for TTS while preserving prosody-critical punctuation.

    CSM-1B uses punctuation for pacing. Instead of stripping punctuation,
    replace non-speech characters with prosody-equivalent substitutes.

    Substitution rules:
        - Ellipses (...) preserved as-is (trailing off / hesitation)
        - Em/en dashes and ASCII double-dashes -> comma (pause equivalent)
        - Hyphens between words -> space (natural word boundary)
        - Colons -> periods (similar pause weight)
        - Semicolons -> commas (lighter pause)
        - Parentheses removed, content kept
        - Remaining non-speech symbols stripped

    Args:
        text: Raw text to clean for TTS input.

    Returns:
        Cleaned, lowercased text with prosody cues intact.
    """
    # Normalize ellipses (2+ dots -> placeholder) before other rules touch periods.
    # Placeholder prevents the space-after-punctuation rule from splitting "..." into ". .."
    _ELLIPSIS = "\x00ELLIPSIS\x00"
    text = re.sub(r"\.{2,}", _ELLIPSIS, text)

    # Replace dashes/hyphens with pauses (natural word boundary)
    text = re.sub(r"[\u2013\u2014]", ", ", text)  # em/en dash -> comma (pause)
    text = re.sub(r"--+", ", ", text)  # ASCII double-dash -> comma (pause)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # hyphen between words -> space

    # Replace colons with periods (similar pause weight)
    text = re.sub(r":", ".", text)

    # Replace semicolons with commas (lighter pause)
    text = re.sub(r";", ",", text)

    # Remove parentheses but keep content
    text = re.sub(r"[()]", "", text)

    # Remove remaining non-speech characters (brackets, special symbols)
    # Keep: word chars, whitespace, periods, commas, exclamation, question, apostrophe,
    # and the ellipsis placeholder (contains \x00 which is not \w)
    text = re.sub(r"[^\w\s.,!?'\"\x00]", "", text)

    # Remove quotation marks (double quotes and smart quotes)
    text = re.sub(r'["\u201c\u201d\u2018\u2019]', "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Ensure space after punctuation (but not inside ellipsis placeholder)
    text = re.sub(r"([.,!?])(\S)", r"\1 \2", text)

    # Restore ellipses from placeholder
    text = text.replace(_ELLIPSIS, "...")

    # Lowercase for CSM compatibility (trained on lowercase)
    # Trade-off: proper nouns lose emphasis, but CSM expects lowercase input.
    # Changing this would require fine-tuning or adapter work.
    return text.strip().lower()
