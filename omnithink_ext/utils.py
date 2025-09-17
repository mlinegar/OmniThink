"""Utility helpers for the OmniThink extension modules."""

from collections import Counter
from typing import Iterable, Mapping, Sequence, Tuple


def _as_counter(vec: Iterable[float] | Mapping[str, float] | Sequence[float]) -> Counter:
    """Convert a variety of vector-like inputs into a Counter for cosine math."""

    if isinstance(vec, Mapping):
        return Counter(vec)
    if isinstance(vec, Sequence):
        return Counter({str(i): float(v) for i, v in enumerate(vec)})
    return Counter(vec)


def cosine(a, b) -> float:
    """Cosine similarity that accepts mappings, sequences, or Counters."""

    ca = _as_counter(a)
    cb = _as_counter(b)
    if not ca or not cb:
        return 0.0

    dot = sum(ca[k] * cb.get(k, 0.0) for k in ca)
    if dot == 0.0:
        return 0.0

    import math

    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _ngrams_from_claims(claims: Iterable[str], n: int) -> set[Tuple[str, ...]]:
    tokens = []
    for claim in claims:
        tokens.extend(word.lower() for word in claim.split())

    if n <= 0 or len(tokens) < n:
        return set()

    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_ngrams(claims_a: Iterable[str], claims_b: Iterable[str], n: int = 2) -> float:
    """Jaccard similarity between n-gram sets extracted from claim collections."""

    set_a = _ngrams_from_claims(claims_a, n)
    set_b = _ngrams_from_claims(claims_b, n)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)

