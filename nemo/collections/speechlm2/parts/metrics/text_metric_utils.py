# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared text preprocessing for BLEU / WER / ASR-BLEU and validation JSON (single code path)."""

import re
import unicodedata
from typing import Callable, Optional

# RNNT / streaming decoders often append BCP-47 style tags, e.g. <fr-FR>, <es-ES>, <de-DE>.
# Strip these before text normalization for scoring; keep raw strings in JSON logs.
_STRIP_LANG_TAG_RE = re.compile(r"<\s*[a-zA-Z]{2,3}(?:\s*[-_]\s*[a-zA-Z]{2,3})?\s*>")
_MULTI_SPACE_RE = re.compile(r"\s+")


def strip_language_tags_for_scoring(text: str) -> str:
    """Remove language tags from strings used only for metric aggregation."""
    if not text:
        return text
    s = _STRIP_LANG_TAG_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", s).strip()


def simple_multilingual_normalize(text: str) -> str:
    """
    Lightweight normalization for multilingual text (source WER/BLEU, target AST BLEU, ASR-BLEU).

    Uses NFC (not NFKC) to avoid compatibility mappings that alter letters (e.g. ß).
    Uses :meth:`str.lower` instead of :meth:`str.casefold` so German ß is preserved (casefold maps ß→ss).
    Maps non-letters (except spaces) to space and collapses whitespace.
    Use with ``normalize_for_metric`` (strip RNNT tags first).
    """
    if not text:
        return text
    text = unicodedata.normalize("NFC", text).lower()
    out: list[str] = []
    for ch in text:
        if ch.isspace():
            out.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N", "M"):
            out.append(ch)
        else:
            out.append(" ")
    return _MULTI_SPACE_RE.sub(" ", "".join(out)).strip()


def normalize_for_metric(text: str, normalizer: Callable[[str], str]) -> str:
    """
    Same pipeline as BLEU.update / WER.update: strip RNNT language tags, then model normalizer.
    ``normalizer`` is e.g. EnglishTextNormalizer, BasicTextNormalizer, or ``simple_multilingual_normalize``.
    """
    return normalizer(strip_language_tags_for_scoring(str(text or "")))


def sentence_bleu_on_normalized(normalized_ref: str, normalized_hyp: str) -> Optional[float]:
    """sacrebleu sentence BLEU on strings already normalized (matches BLEU verbose logging)."""
    import sacrebleu

    if not normalized_hyp.strip() and not normalized_ref.strip():
        return None
    try:
        return float(sacrebleu.sentence_bleu(normalized_hyp, [normalized_ref]).score)
    except Exception:
        return None


def wer_on_normalized(normalized_ref: str, normalized_hyp: str) -> Optional[float]:
    """jiwer WER with normalized ref as ground truth; None if ref is empty (matches WER aggregation)."""
    import jiwer

    if not normalized_ref.strip():
        return None
    return float(jiwer.wer(normalized_ref, normalized_hyp))
