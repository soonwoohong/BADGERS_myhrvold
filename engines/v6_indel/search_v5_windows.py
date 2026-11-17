"""Fixed-length target-window utilities for search_guide_v5.

The Cas13 predictor accepts a 28-nt guide and a 48-nt target context. These
helpers enumerate valid contexts from unequal-length target sequences and find
the closest contiguous binding registers without assuming shared coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DNA_BASES = frozenset("ACGT")
BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass(frozen=True)
class ContextLibrary:
    contexts: tuple[str, ...]
    guide_starts0: np.ndarray
    site_codes: np.ndarray

    def __len__(self) -> int:
        return len(self.contexts)


def normalize_sequence(sequence: object) -> str:
    """Return an ungapped uppercase DNA string suitable for local scanning."""
    sequence = str(sequence).upper().replace("U", "T")
    return "".join(base for base in sequence if base not in {"-", "_", ".", " ", "\n", "\r", "\t"})


def encode_site(sequence: str) -> np.ndarray:
    return np.fromiter((BASE_TO_INT.get(base, -1) for base in sequence), dtype=np.int8)


def build_context_library(
    sequence: object,
    *,
    guide_len: int = 28,
    context_nt: int = 10,
) -> ContextLibrary:
    """Enumerate every valid model context and its zero-based guide start."""
    sequence = normalize_sequence(sequence)
    target_len = guide_len + 2 * context_nt
    contexts: list[str] = []
    guide_starts0: list[int] = []
    site_codes: list[np.ndarray] = []

    for context_start0 in range(0, len(sequence) - target_len + 1):
        context = sequence[context_start0 : context_start0 + target_len]
        site = context[context_nt : context_nt + guide_len]
        if not set(context).issubset(DNA_BASES):
            continue
        contexts.append(context)
        guide_starts0.append(context_start0 + context_nt)
        site_codes.append(encode_site(site))

    if not contexts:
        raise ValueError(
            f"Sequence has no valid {target_len}-nt A/C/G/T context windows (length={len(sequence)})"
        )

    return ContextLibrary(
        contexts=tuple(contexts),
        guide_starts0=np.asarray(guide_starts0, dtype=np.int32),
        site_codes=np.stack(site_codes, axis=0),
    )


def intended_context(
    sequence: object,
    guide_start0: int,
    *,
    guide_len: int = 28,
    context_nt: int = 10,
) -> str | None:
    """Extract the predictor context for a guide at a known local start."""
    sequence = normalize_sequence(sequence)
    context_start0 = guide_start0 - context_nt
    context_end0 = guide_start0 + guide_len + context_nt
    if context_start0 < 0 or context_end0 > len(sequence):
        return None
    context = sequence[context_start0:context_end0]
    if not set(context).issubset(DNA_BASES):
        return None
    return context


def nearest_context_indices(
    library: ContextLibrary,
    guide: str,
    n: int,
    *,
    preferred_start0: int | None = None,
) -> np.ndarray:
    """Return up to n windows ordered by Hamming distance, then position."""
    if n < 1:
        raise ValueError("n must be at least 1")
    guide = normalize_sequence(guide)
    guide_codes = encode_site(guide)
    if guide_codes.shape != (library.site_codes.shape[1],) or np.any(guide_codes < 0):
        raise ValueError("Guide must contain exactly 28 A/C/G/T bases")

    distances = np.count_nonzero(library.site_codes != guide_codes, axis=1)
    if preferred_start0 is None:
        position_distance = library.guide_starts0
    else:
        position_distance = np.abs(library.guide_starts0 - int(preferred_start0))
    order = np.lexsort((library.guide_starts0, position_distance, distances))
    return order[: min(n, len(order))]
