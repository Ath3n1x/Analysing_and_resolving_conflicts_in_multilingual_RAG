"""
src/chunker.py
Language-aware passage chunker.

Splits raw passages into overlapping sentence-window chunks using
per-language character-to-token ratios and NLTK sentence boundaries.
Target: 80 tokens per chunk, 2-sentence overlap.
"""

import re
import unicodedata
from typing import List, Dict

try:
    import nltk
    nltk.download("punkt",     quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
    _HAVE_NLTK = True
except Exception:
    _HAVE_NLTK = False

# Default chars-per-token ratios (overridden by config values when passed in)
DEFAULT_CHARS_PER_TOKEN: Dict[str, float] = {
    "zh": 1.5, "hi": 1.8, "ar": 2.5, "th": 1.5,
    "ru": 3.5, "el": 3.5, "tr": 4.0, "vi": 3.0,
    "es": 4.0, "de": 4.5, "en": 4.5,
}

LANG_NLTK_NAME: Dict[str, str] = {
    "en": "english", "de": "german", "es": "spanish",
    "ru": "russian", "ar": "arabic",
}


def _tok_count(text: str, lang: str, chars_per_token: Dict[str, float]) -> int:
    cpt = chars_per_token.get(lang, 4.5)
    return max(1, int(len(text) / cpt))


def _split_sentences(text: str, lang: str) -> List[str]:
    if _HAVE_NLTK:
        nltk_lang = LANG_NLTK_NAME.get(lang, "english")
        try:
            return sent_tokenize(text.strip(), language=nltk_lang)
        except Exception:
            pass
    # Fallback: split on sentence-ending punctuation across scripts
    parts = re.split(
        r'(?<=[.!?।॥。！？؟\u0E2F\u0E5A\u0E5B])\s*', text.strip()
    )
    return [p for p in parts if p.strip()] or [text]


def chunk_passage(
    text: str,
    lang: str,
    target_tokens: int = 80,
    overlap_sentences: int = 2,
    chars_per_token: Dict[str, float] | None = None,
) -> List[str]:
    """
    Split a passage into overlapping chunks of ~target_tokens tokens.

    Args:
        text: Raw passage text.
        lang: BCP-47 language code.
        target_tokens: Soft token budget per chunk.
        overlap_sentences: Number of trailing sentences carried into next chunk.
        chars_per_token: Optional override map; falls back to DEFAULT_CHARS_PER_TOKEN.

    Returns:
        List of chunk strings.
    """
    cpt = chars_per_token if chars_per_token is not None else DEFAULT_CHARS_PER_TOKEN
    soft_max = int(target_tokens * cpt.get(lang, 4.5))

    sentences = _split_sentences(text, lang)
    chunks: List[str] = []
    current: List[str] = []
    cur_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if cur_len + sent_len > soft_max and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            cur_len = sum(len(s) for s in current)
        current.append(sent)
        cur_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks or [text]


def build_chunk_store(
    corpus: Dict[str, List[Dict]],
    cfg: Dict,
) -> Dict[str, List[Dict]]:
    """
    Chunk all passages in the corpus.

    Args:
        corpus: {lang: [{"context": str, "lang": str, ...}]}
        cfg:    Chunking config dict with keys target_tokens, overlap_sentences,
                chars_per_token.

    Returns:
        {lang: [{"text": str, "lang": str, "chunk_id": str, "passage_id": str}]}
    """
    target_tokens    = cfg.get("target_tokens", 80)
    overlap_sents    = cfg.get("overlap_sentences", 2)
    chars_per_token  = cfg.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN)

    store: Dict[str, List[Dict]] = {}
    for lang, rows in corpus.items():
        chunks: List[Dict] = []
        for row in rows:
            pid = row.get("id", row.get("passage_id", ""))
            for i, txt in enumerate(
                chunk_passage(
                    row["context"], lang,
                    target_tokens=target_tokens,
                    overlap_sentences=overlap_sents,
                    chars_per_token=chars_per_token,
                )
            ):
                chunks.append({
                    "chunk_id":   f"{pid}::{i}",
                    "text":       txt,
                    "lang":       lang,
                    "passage_id": pid,
                })
        store[lang] = chunks
    return store
