#!/usr/bin/env python3
"""CPU-only, indel-aware guide search built on the BADGERS/ADAPT scorer.

The alignment or edit annotation is used only to identify allele-specific
junction windows.  Every sequence passed to the predictor is an ungapped,
fixed-length biological sequence.

Search stages per on-target:
  0. Score every unmodified junction-spanning guide.
  1. Exhaustively score all exact one-mismatch guides for the best windows.
  2. Exhaustively score all exact two-mismatch guides for the best windows.
  3-5. Expand a beam ranked by on-target activity, retaining the best N.
  Final. Rescore cutoff-passing finalists against every full-length off-target
  register, deduplicating identical 48-nt model contexts without collapsing
  sequence-level profiles, and rank them by all requested metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "32"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(DEFAULT_CPUS))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

# Resolve companion modules from the script location, independently of the
# directory from which Python was launched or the caller's PYTHONPATH.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
from Bio import Align

import search_v5_windows as window_utils


LOGGER = logging.getLogger("search_guide_v6_indel")
PROGRAM_VERSION = "6.4.1-deduplicated-contexts-sharded-cpu"
DNA_BASES = frozenset("ACGT")
GUIDE_LEN = 28
CONTEXT_NT = 10
METRIC_COLUMNS = {
    "k1_lambda0p4": "score_k1_lambda0p4",
    "on_minus_max": "score_on_minus_max_off_target",
    "on_minus_top3": "score_on_minus_mean_top3_off_target",
}
RANKED_OUTPUT_COLUMNS = [
    "candidate_id",
    "on_target_name",
    "start_pos",
    "guide_sequence",
    "num_mismatches",
    "mean_on_target_act",
    "max_off_target_act",
    "mean_off_target_act",
    "mean_top3_off_target_act",
    *METRIC_COLUMNS.values(),
    *[f"rank_{label}" for label in METRIC_COLUMNS],
    *[f"rank_{label}_within_mismatch" for label in METRIC_COLUMNS],
]
STAGE_SUMMARY_COLUMNS = ["num_mismatches", "mode", "evaluated", "retained"]
JUNCTION_PLAN_COLUMNS = [
    "cluster",
    "on_target_name",
    "start_pos",
    "event_ids",
    "event_kinds",
]
TARGET_SELECTION_COLUMNS = [
    "spreadsheet_row",
    "name",
    "is_wt",
    "indel_flag",
    "selected_as_on_target",
    "retained_for_off_target_comparison",
    "selection_mode",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only indel-aware BADGERS/ADAPT guide search"
    )
    parser.add_argument("--version", action="version", version=PROGRAM_VERSION)
    parser.add_argument("-f", "--file_name", required=True, help="Input XLSX/CSV/TSV")
    parser.add_argument("-p", "--project_name", default="v6_indel")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: ./output/<project_name>)",
    )
    parser.add_argument(
        "--wt_name",
        default=None,
        help="Exact WT/reference name; otherwise one unique name containing WT is required",
    )
    parser.add_argument(
        "--only_target",
        default=None,
        help=(
            "Execute exactly one selected on-target by exact name while retaining "
            "the complete cluster comparison library. Intended for isolated "
            "parallel target shards."
        ),
    )
    parser.add_argument(
        "-c",
        "--cut_off",
        type=float,
        default=-1.4,
        help="Minimum on-target activity. ADAPT activities are normally negative.",
    )
    parser.add_argument("-mm", "--max_mismatches", type=int, default=5)
    parser.add_argument("--num_cpu", type=int, default=DEFAULT_CPUS)
    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("--n_seed_windows", type=int, default=25)
    parser.add_argument("--beam_width", type=int, default=1000)
    parser.add_argument("--finalists_per_mismatch", type=int, default=1000)
    parser.add_argument("--n_top_guides", type=int, default=25)
    parser.add_argument("--off_tail_k", type=int, default=1)
    parser.add_argument("--off_tail_lambda", type=float, default=0.4)
    parser.add_argument("--rescore_guide_batch_size", type=int, default=64)
    parser.add_argument("--rescore_context_batch_size", type=int, default=256)
    parser.add_argument(
        "--include_wt_as_target",
        choices=["yes", "no"],
        default="yes",
        help=(
            "Design WT/reference guides around the indel junctions (default: yes). "
            "WT is always retained for off-target comparison."
        ),
    )
    parser.add_argument(
        "--target_selection",
        choices=["auto", "indel_column"],
        default="auto",
        help=(
            "Use an authoritative indel column, or fall back to automatic "
            "indel detection when the column is absent"
        ),
    )
    parser.add_argument(
        "--position_base",
        type=int,
        choices=[0, 1],
        default=0,
        help="Coordinate base for edit-position columns in the workbook",
    )
    parser.add_argument(
        "--resume",
        choices=["yes", "no"],
        default="yes",
        help="Skip targets with a completed marker and compiled result",
    )
    parser.add_argument(
        "--plan_only",
        action="store_true",
        help="Validate input and write junction-window plans without loading TensorFlow",
    )
    args = parser.parse_args(argv)

    if not 0 <= args.max_mismatches <= 5:
        parser.error("--max_mismatches must be between 0 and 5")
    for name in [
        "num_cpu",
        "batch_size",
        "n_seed_windows",
        "beam_width",
        "finalists_per_mismatch",
        "n_top_guides",
        "rescore_guide_batch_size",
        "rescore_context_batch_size",
    ]:
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.off_tail_k != 1:
        parser.error("v6 currently fixes the requested custom metric at --off_tail_k 1")
    if not math.isclose(args.off_tail_lambda, 0.4, abs_tol=1e-12):
        parser.error("v6 currently fixes the requested custom metric at --off_tail_lambda 0.4")
    if not math.isfinite(args.cut_off):
        parser.error("--cut_off must be finite")
    if args.cut_off > 0:
        LOGGER.warning(
            "A positive on-target cutoff (%s) will usually reject all ADAPT predictions; "
            "use --cut_off -1.4 if the minus sign was intended.",
            args.cut_off,
        )
    return args


def normalize_sequence(value: object) -> str:
    sequence = str(value).upper().replace("U", "T")
    return "".join(
        base
        for base in sequence
        if base not in {"-", "_", ".", " ", "\n", "\r", "\t"}
    )


def raw_sequence(value: object) -> str:
    return re.sub(r"\s+", "", str(value).upper().replace("U", "T"))


def safe_filename(value: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    if safe in {"", ".", ".."}:
        return "target"
    return safe or "target"


def finite_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def finite_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def parse_indel_flag(value: object, spreadsheet_row: int) -> bool:
    """Parse an explicit on-target flag from the input workbook."""
    if value is None or pd.isna(value):
        raise ValueError(
            f"Spreadsheet row {spreadsheet_row} has no value in indel; "
            "use yes/no or 1/0"
        )
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value) in {0.0, 1.0}:
            return bool(int(value))

    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    raise ValueError(
        f"Spreadsheet row {spreadsheet_row} has invalid indel value {value!r}; "
        "use yes/no or 1/0"
    )


def explicit_indel_target_indices(
    table: pd.DataFrame,
    wt_index: int,
) -> set[int] | None:
    """Return authoritative on-target rows, or None for legacy auto-detection."""
    if "_v6_indel_target" not in table.columns:
        return None
    if bool(table.iloc[wt_index]["_v6_indel_target"]):
        raise ValueError(
            "The WT/reference row cannot be marked indel=yes; use no or 0 for WT"
        )
    selected = {
        index
        for index, value in enumerate(table["_v6_indel_target"])
        if bool(value) and index != wt_index
    }
    if not selected:
        raise ValueError("The indel column does not select any non-WT on-target rows")
    return selected


@dataclass(frozen=True)
class EditEvent:
    event_id: str
    target_global_index: int
    target_name: str
    kind: str
    wt_start0: int
    wt_end0: int
    target_start0: int
    target_end0: int
    source: str

    @property
    def wt_length(self) -> int:
        return self.wt_end0 - self.wt_start0

    @property
    def target_length(self) -> int:
        return self.target_end0 - self.target_start0

    @property
    def is_indel(self) -> bool:
        return self.wt_length != self.target_length


@dataclass(frozen=True)
class SeedWindow:
    seed_id: str
    target_local_index: int
    target_global_index: int
    target_name: str
    start0: int
    source_guide: str
    context: str
    event_ids: tuple[str, ...]
    event_kinds: tuple[str, ...]


@dataclass(frozen=True)
class DeduplicatedContextPool:
    """Cluster-wide unique model contexts and per-sequence window mappings.

    ``window_unique_indices`` retains the original context-window order for
    every sequence.  Re-indexing a score matrix with one of these arrays
    therefore reproduces the original per-sequence scan, including the
    earliest-window behavior of ``numpy.argmax`` when scores are tied.
    """

    contexts: tuple[str, ...]
    encoded_contexts: np.ndarray
    window_unique_indices: tuple[np.ndarray, ...]
    total_context_count: int

    def __len__(self) -> int:
        return len(self.contexts)

    @property
    def reuse_factor(self) -> float:
        return self.total_context_count / len(self.contexts)


def build_deduplicated_context_pool(
    libraries: Sequence[window_utils.ContextLibrary],
    scorer: "CpuCas13Scorer",
) -> DeduplicatedContextPool:
    """Encode each distinct 48-nt context once across a whole cluster."""

    if not libraries:
        raise ValueError("At least one context library is required")

    expected_length = GUIDE_LEN + 2 * CONTEXT_NT
    unique_contexts: list[str] = []
    context_to_index: dict[str, int] = {}
    window_unique_indices = []
    total_context_count = 0

    for sequence_index, library in enumerate(libraries):
        if len(library.contexts) != len(library.guide_starts0):
            raise ValueError(
                "Context and start-position counts differ for sequence "
                f"index {sequence_index}"
            )
        indices = np.empty(len(library), dtype=np.int32)
        for window_index, context in enumerate(library.contexts):
            if len(context) != expected_length:
                raise ValueError(
                    f"Model contexts must be {expected_length} nt; found "
                    f"{len(context)} nt in sequence index {sequence_index}"
                )
            unique_index = context_to_index.get(context)
            if unique_index is None:
                unique_index = len(unique_contexts)
                context_to_index[context] = unique_index
                unique_contexts.append(context)
            indices[window_index] = unique_index
        window_unique_indices.append(indices)
        total_context_count += len(library)

    if not unique_contexts:
        raise ValueError("Context libraries contain no model contexts")

    return DeduplicatedContextPool(
        contexts=tuple(unique_contexts),
        encoded_contexts=scorer.encode_contexts(unique_contexts),
        window_unique_indices=tuple(window_unique_indices),
        total_context_count=total_context_count,
    )


def event_kind(wt_length: int, target_length: int) -> str:
    if wt_length == 0 and target_length > 0:
        return "insertion"
    if wt_length > 0 and target_length == 0:
        return "deletion"
    if wt_length != target_length:
        return "replacement_indel"
    return "substitution"


def common_prefix_suffix(reference: str, alternate: str) -> tuple[int, int]:
    prefix = 0
    limit = min(len(reference), len(alternate))
    while prefix < limit and reference[prefix] == alternate[prefix]:
        prefix += 1

    suffix = 0
    remaining_ref = len(reference) - prefix
    remaining_alt = len(alternate) - prefix
    while (
        suffix < remaining_ref
        and suffix < remaining_alt
        and reference[len(reference) - suffix - 1] == alternate[len(alternate) - suffix - 1]
    ):
        suffix += 1
    return prefix, suffix


def event_from_row_metadata(
    row: pd.Series,
    target_global_index: int,
    target_name: str,
    position_base: int,
) -> EditEvent | None:
    position_col = first_existing(
        row.index,
        ["insert_relative_real_position", "relative_position_on_gene_frag"],
    )
    if position_col is None:
        return None
    position = finite_int(row.get(position_col))
    if position is None:
        return None
    position -= position_base

    org_text = finite_text(row.get("org_seq")) if "org_seq" in row.index else None
    mut_text = finite_text(row.get("mut_seq")) if "mut_seq" in row.index else None
    if org_text is not None and mut_text is not None:
        org = normalize_sequence(org_text)
        mut = normalize_sequence(mut_text)
        prefix, suffix = common_prefix_suffix(org, mut)
        wt_start = position + prefix
        wt_end = position + len(org) - suffix
        target_start = position + prefix
        target_end = position + len(mut) - suffix
    else:
        org_len = finite_int(row.get("org_seq_len")) if "org_seq_len" in row.index else None
        mut_len = finite_int(row.get("mut_seq_len")) if "mut_seq_len" in row.index else None
        if org_len is None or mut_len is None:
            return None
        wt_start, wt_end = position, position + org_len
        target_start, target_end = position, position + mut_len

    kind = event_kind(wt_end - wt_start, target_end - target_start)
    return EditEvent(
        event_id=f"{safe_filename(target_name)}.metadata",
        target_global_index=target_global_index,
        target_name=target_name,
        kind=kind,
        wt_start0=wt_start,
        wt_end0=wt_end,
        target_start0=target_start,
        target_end0=target_end,
        source=f"workbook:{position_col}",
    )


def events_from_aligned_pair(
    wt_aligned: str,
    target_aligned: str,
    target_global_index: int,
    target_name: str,
) -> list[EditEvent]:
    if len(wt_aligned) != len(target_aligned):
        return []
    differing_gap_columns = [
        column
        for column, (wt_base, target_base) in enumerate(zip(wt_aligned, target_aligned))
        if (wt_base == "-") != (target_base == "-")
    ]
    if not differing_gap_columns:
        return []

    blocks: list[list[int]] = []
    for column in differing_gap_columns:
        if not blocks or column > blocks[-1][-1] + 1:
            blocks.append([column])
        else:
            blocks[-1].append(column)

    wt_before = np.zeros(len(wt_aligned) + 1, dtype=np.int32)
    target_before = np.zeros(len(target_aligned) + 1, dtype=np.int32)
    for column in range(len(wt_aligned)):
        wt_before[column + 1] = wt_before[column] + int(wt_aligned[column] != "-")
        target_before[column + 1] = target_before[column] + int(target_aligned[column] != "-")

    events = []
    for number, block in enumerate(blocks, start=1):
        left, right = block[0], block[-1] + 1
        wt_start, wt_end = int(wt_before[left]), int(wt_before[right])
        target_start, target_end = int(target_before[left]), int(target_before[right])
        events.append(
            EditEvent(
                event_id=f"{safe_filename(target_name)}.msa{number}",
                target_global_index=target_global_index,
                target_name=target_name,
                kind=event_kind(wt_end - wt_start, target_end - target_start),
                wt_start0=wt_start,
                wt_end0=wt_end,
                target_start0=target_start,
                target_end0=target_end,
                source=f"alignment_columns:{left + 1}-{right}",
            )
        )
    return events


def events_from_pairwise_alignment(
    wt_sequence: str,
    target_sequence: str,
    target_global_index: int,
    target_name: str,
) -> list[EditEvent]:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -1.0
    alignment = aligner.align(wt_sequence, target_sequence)[0]
    coordinates = np.asarray(alignment.coordinates, dtype=np.int64)

    events = []
    for segment in range(coordinates.shape[1] - 1):
        wt_start, wt_end = map(int, coordinates[0, segment : segment + 2])
        target_start, target_end = map(int, coordinates[1, segment : segment + 2])
        wt_len = wt_end - wt_start
        target_len = target_end - target_start
        if wt_len > 0 and target_len > 0:
            continue
        if wt_len == 0 and target_len == 0:
            continue
        events.append(
            EditEvent(
                event_id=f"{safe_filename(target_name)}.pairwise{len(events) + 1}",
                target_global_index=target_global_index,
                target_name=target_name,
                kind=event_kind(wt_len, target_len),
                wt_start0=wt_start,
                wt_end0=wt_end,
                target_start0=target_start,
                target_end0=target_end,
                source="pairwise_fallback",
            )
        )
    return events


def infer_indel_events(
    wt_raw: str,
    target_raw: str,
    wt_sequence: str,
    target_sequence: str,
    row: pd.Series,
    target_global_index: int,
    target_name: str,
    position_base: int,
) -> list[EditEvent]:
    if len(wt_raw) == len(target_raw) and ("-" in wt_raw or "-" in target_raw):
        aligned = events_from_aligned_pair(
            wt_raw,
            target_raw,
            target_global_index,
            target_name,
        )
        aligned = [event for event in aligned if event.is_indel]
        if aligned:
            return aligned

    metadata = event_from_row_metadata(
        row,
        target_global_index,
        target_name,
        position_base,
    )
    if metadata is not None and metadata.is_indel:
        return [metadata]

    return [
        event
        for event in events_from_pairwise_alignment(
            wt_sequence,
            target_sequence,
            target_global_index,
            target_name,
        )
        if event.is_indel
    ]


def read_input_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        table = pd.read_excel(path)
    elif suffix == ".csv":
        table = pd.read_csv(path)
    elif suffix in {".tsv", ".txt"}:
        table = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported input extension: {path.suffix}")

    name_col = first_existing(table.columns, ["new_name", "nick_name", "name", "id"])
    sequence_col = first_existing(table.columns, ["new_seq", "seq", "sequence"])
    if name_col is None or sequence_col is None:
        raise ValueError(
            "Input must contain a name column (new_name/nick_name/name/id) and "
            "a sequence column (new_seq/seq/sequence)"
        )
    table = table.copy()
    missing_names = table[name_col].isna() | table[name_col].map(
        lambda value: not str(value).strip()
    )
    if missing_names.any():
        rows = [int(index) + 2 for index in table.index[missing_names]]
        raise ValueError(f"Missing or blank sequence name in spreadsheet row(s): {rows[:5]}")
    table["_v6_name"] = table[name_col].map(lambda value: str(value).strip())
    table["_v6_raw_sequence"] = table[sequence_col].map(raw_sequence)
    table["_v6_sequence"] = table[sequence_col].map(normalize_sequence)
    if "indel" in table.columns:
        table["_v6_indel_target"] = [
            parse_indel_flag(value, spreadsheet_row=index + 2)
            for index, value in enumerate(table["indel"])
        ]
    invalid = [
        table.iloc[index]["_v6_name"]
        for index, sequence in enumerate(table["_v6_sequence"])
        if not sequence or not set(sequence).issubset(DNA_BASES)
    ]
    if invalid:
        raise ValueError(f"Sequences contain unsupported symbols after ungapping: {invalid[:5]}")
    duplicate_names = table.loc[table["_v6_name"].duplicated(keep=False), "_v6_name"].tolist()
    if duplicate_names:
        raise ValueError(f"Input names must be unique; duplicates include: {duplicate_names[:5]}")
    safe_to_originals: dict[str, list[str]] = defaultdict(list)
    for name in table["_v6_name"]:
        safe_to_originals[safe_filename(name)].append(name)
    collisions = {
        safe: originals
        for safe, originals in safe_to_originals.items()
        if len(set(originals)) > 1
    }
    if collisions:
        raise ValueError(f"Names collide after filename sanitization: {collisions}")
    return table


def find_wt_index(table: pd.DataFrame, requested_name: str | None) -> int:
    names = table["_v6_name"].tolist()
    if requested_name is not None:
        matches = [index for index, name in enumerate(names) if name == requested_name]
        if len(matches) != 1:
            raise ValueError(
                f"--wt_name {requested_name!r} matched {len(matches)} rows; expected exactly one"
            )
        return matches[0]

    matches = [index for index, name in enumerate(names) if "WT" in name.upper()]
    if len(matches) != 1:
        raise ValueError(
            "Could not identify one unique WT/reference row. Supply --wt_name explicitly; "
            "v6 does not silently treat row 1 as WT."
        )
    return matches[0]


def resolve_only_target_index(
    table: pd.DataFrame,
    selected_on_targets: set[int],
    requested_name: str | None,
) -> int | None:
    """Resolve an execution-only target filter without changing comparisons."""

    if requested_name is None:
        return None
    matches = [
        index
        for index, name in enumerate(table["_v6_name"].tolist())
        if name == requested_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"--only_target {requested_name!r} matched {len(matches)} input rows; "
            "an exact unique name is required"
        )
    target_index = matches[0]
    if target_index not in selected_on_targets:
        raise ValueError(
            f"--only_target {requested_name!r} is not selected as WT or indel "
            "on-target by the current input and options"
        )
    return target_index


def cluster_groups(table: pd.DataFrame, wt_index: int) -> list[tuple[str, list[int]]]:
    cluster_col = first_existing(
        table.columns,
        ["prepared_cluster_idx", "new_cluster", "cluster_idx", "cluster"],
    )
    non_wt = [index for index in range(len(table)) if index != wt_index]
    if cluster_col is None:
        return [("1", [wt_index] + non_wt)]

    groups: dict[str, list[int]] = defaultdict(list)
    for index in non_wt:
        value = table.iloc[index].get(cluster_col)
        if pd.isna(value):
            raise ValueError(f"Row {index + 2} has no cluster value in {cluster_col}")
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            label = str(int(value))
        else:
            label = str(value)
        groups[label].append(index)
    return [(label, [wt_index] + members) for label, members in groups.items()]


def candidate_starts_for_interval(
    interval_start0: int,
    interval_end0: int,
    sequence_length: int,
    guide_len: int = GUIDE_LEN,
    context_nt: int = CONTEXT_NT,
) -> range:
    # Non-empty interval: guide must overlap at least one event base.
    # Empty interval (a deletion junction): guide must contain bases on both sides.
    low = interval_start0 - guide_len + 1
    high = interval_end0 - 1 if interval_end0 > interval_start0 else interval_start0 - 1
    low = max(low, context_nt)
    high = min(high, sequence_length - guide_len - context_nt)
    if high < low:
        return range(0)
    return range(low, high + 1)


def build_seed_windows(
    target_local_index: int,
    target_global_index: int,
    target_name: str,
    sequence: str,
    events_and_intervals: Sequence[tuple[EditEvent, int, int]],
) -> list[SeedWindow]:
    event_ids_by_start: dict[int, set[str]] = defaultdict(set)
    event_kinds_by_start: dict[int, set[str]] = defaultdict(set)
    for event, start0, end0 in events_and_intervals:
        for guide_start0 in candidate_starts_for_interval(start0, end0, len(sequence)):
            event_ids_by_start[guide_start0].add(event.event_id)
            event_kinds_by_start[guide_start0].add(event.kind)

    seeds = []
    for ordinal, start0 in enumerate(sorted(event_ids_by_start), start=1):
        guide = sequence[start0 : start0 + GUIDE_LEN]
        context = window_utils.intended_context(sequence, start0)
        if len(guide) != GUIDE_LEN or not set(guide).issubset(DNA_BASES) or context is None:
            continue
        seeds.append(
            SeedWindow(
                seed_id=f"{safe_filename(target_name)}.seed{ordinal}.start{start0 + 1}",
                target_local_index=target_local_index,
                target_global_index=target_global_index,
                target_name=target_name,
                start0=start0,
                source_guide=guide,
                context=context,
                event_ids=tuple(sorted(event_ids_by_start[start0])),
                event_kinds=tuple(sorted(event_kinds_by_start[start0])),
            )
        )
    return seeds


def chunked(iterable: Iterable, size: int) -> Iterator[list]:
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def exact_mutants(source: str, mismatch_count: int) -> Iterator[tuple[str, tuple[int, ...]]]:
    if mismatch_count == 0:
        yield source, ()
        return
    for positions in itertools.combinations(range(len(source)), mismatch_count):
        replacements = [[base for base in "ACGT" if base != source[pos]] for pos in positions]
        for replacement_tuple in itertools.product(*replacements):
            guide = list(source)
            for position, replacement in zip(positions, replacement_tuple):
                guide[position] = replacement
            yield "".join(guide), tuple(positions)


def expand_one_mismatch(
    source: str,
    guide: str,
    mutated_positions: Sequence[int],
) -> Iterator[tuple[str, tuple[int, ...]]]:
    occupied = set(mutated_positions)
    for position in range(len(source)):
        if position in occupied:
            continue
        for replacement in "ACGT":
            if replacement == source[position]:
                continue
            child = list(guide)
            child[position] = replacement
            yield "".join(child), tuple(sorted((*mutated_positions, position)))


class CpuCas13Scorer:
    """TensorFlow scorer restricted to one CPU process and its thread pool."""

    def __init__(self, num_cpu: int, guide_batch_size: int):
        self.num_cpu = int(num_cpu)
        self.guide_batch_size = int(guide_batch_size)

        import tensorflow as tf

        try:
            tf.config.set_visible_devices([], "GPU")
        except (RuntimeError, ValueError):
            pass
        try:
            tf.config.threading.set_intra_op_parallelism_threads(self.num_cpu)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            LOGGER.warning("TensorFlow thread pools were already initialized")
        tf.config.optimizer.set_jit(False)

        # Load the ADAPT Cas13a models only after the CPU device and thread pools
        # are fixed; model construction can otherwise initialize TensorFlow too
        # early.  The vectorized BADGERS call is implemented below so this script
        # does not depend on a separately deployed ``badgers_mod`` package.
        try:
            from adapt.utils import predict_activity
            from adapt.utils import version as adapt_version
        except ImportError as error:
            raise RuntimeError(
                "The ADAPT package is unavailable in the selected Python "
                "environment. Use the Delta BADGERS environment configured by "
                "search_guide_v6_indel_cpu.slurm."
            ) from error

        adapt_root = Path(adapt_version.get_project_path())
        classify_root = adapt_root / "models" / "classify" / "cas13a"
        regress_root = adapt_root / "models" / "regress" / "cas13a"
        try:
            classify_version = adapt_version.get_latest_model_version(
                str(classify_root)
            )
            regress_version = adapt_version.get_latest_model_version(str(regress_root))
            self.predictor = predict_activity.Predictor(
                str(classify_root / classify_version),
                str(regress_root / regress_version),
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            raise RuntimeError(
                "The ADAPT Cas13a classification/regression model files could "
                f"not be loaded below {adapt_root}."
            ) from error

        self.tf = tf
        self._score_tensor = tf.function(
            self._score_tensor_impl,
            reduce_retracing=True,
            jit_compile=True,
        )

    def _score_tensor_impl(self, guides_onehot, contexts_onehot):
        guides_onehot = self.tf.ensure_shape(
            guides_onehot, [None, GUIDE_LEN, 4]
        )
        contexts_onehot = self.tf.ensure_shape(
            contexts_onehot, [None, GUIDE_LEN + 2 * CONTEXT_NT, 4]
        )
        guide_count = self.tf.shape(guides_onehot)[0]
        context_count = self.tf.shape(contexts_onehot)[0]

        padded_guides = self.tf.pad(
            guides_onehot,
            [[0, 0], [CONTEXT_NT, CONTEXT_NT], [0, 0]],
        )
        paired_guides = self.tf.broadcast_to(
            padded_guides[:, self.tf.newaxis, :, :],
            [guide_count, context_count, GUIDE_LEN + 2 * CONTEXT_NT, 4],
        )
        paired_contexts = self.tf.broadcast_to(
            contexts_onehot[self.tf.newaxis, :, :, :],
            [guide_count, context_count, GUIDE_LEN + 2 * CONTEXT_NT, 4],
        )
        model_input = self.tf.reshape(
            self.tf.concat([paired_contexts, paired_guides], axis=-1),
            [guide_count * context_count, GUIDE_LEN + 2 * CONTEXT_NT, 8],
        )

        pred_perf = self.tf.reshape(
            self.predictor.regression_model(model_input, training=False),
            [guide_count, context_count],
        )
        pred_act = self.tf.reshape(
            self.predictor.classification_model(model_input, training=False),
            [guide_count, context_count],
        )
        return pred_act * (pred_perf + self.tf.constant(4.0)) - self.tf.constant(4.0)

    @staticmethod
    def _one_hot_encode(sequence: str) -> np.ndarray:
        """Encode A/C/G/T as four channels and other symbols as all-zero rows."""
        sequence = str(sequence).upper()
        encoded = np.zeros((len(sequence), 4), dtype=np.float32)
        channels = {"A": 0, "C": 1, "G": 2, "T": 3}
        for position, base in enumerate(sequence):
            channel = channels.get(base)
            if channel is not None:
                encoded[position, channel] = 1.0
        return encoded

    def encode_guides(self, guides: Sequence[str]) -> np.ndarray:
        return np.stack(
            [self._one_hot_encode(guide) for guide in guides], axis=0
        ).astype(np.float32)

    def encode_contexts(self, contexts: Sequence[str]) -> np.ndarray:
        return np.stack(
            [self._one_hot_encode(context) for context in contexts], axis=0
        ).astype(np.float32)

    def score_encoded(self, guides_onehot: np.ndarray, contexts_onehot: np.ndarray) -> np.ndarray:
        with self.tf.device("/CPU:0"):
            values = self._score_tensor(
                self.tf.convert_to_tensor(guides_onehot, dtype=self.tf.float32),
                self.tf.convert_to_tensor(contexts_onehot, dtype=self.tf.float32),
            )
        return np.asarray(values.numpy(), dtype=np.float32)

    def score_guides_one_context(self, guides: Sequence[str], context: str) -> np.ndarray:
        context_onehot = self.encode_contexts([context])
        output = []
        for guide_chunk in chunked(guides, self.guide_batch_size):
            values = self.score_encoded(self.encode_guides(guide_chunk), context_onehot)
            output.extend(values[:, 0].tolist())
        return np.asarray(output, dtype=np.float32)

    def score_paired_contexts(self, guides: Sequence[str], contexts: Sequence[str]) -> np.ndarray:
        if len(guides) != len(contexts):
            raise ValueError("Guides and intended contexts must have the same length")
        output = []
        for start in range(0, len(guides), self.guide_batch_size):
            end = min(start + self.guide_batch_size, len(guides))
            matrix = self.score_encoded(
                self.encode_guides(guides[start:end]),
                self.encode_contexts(contexts[start:end]),
            )
            output.extend(np.diag(matrix).tolist())
        return np.asarray(output, dtype=np.float32)


def candidate_record(
    seed: SeedWindow,
    guide: str,
    mismatch_count: int,
    positions: Sequence[int],
    on_target_activity: float,
    search_mode: str,
) -> dict:
    return {
        "target_local_index": seed.target_local_index,
        "target_global_index": seed.target_global_index,
        "on_target_name": seed.target_name,
        "seed_id": seed.seed_id,
        "start_pos": seed.start0 + 1,
        "source_guide": seed.source_guide,
        "guide_sequence": guide,
        "num_mismatches": mismatch_count,
        "mismatch_positions_1based": ",".join(str(position + 1) for position in positions),
        "event_ids": ",".join(seed.event_ids),
        "event_kinds": ",".join(seed.event_kinds),
        "on_target_activity_screen": float(on_target_activity),
        "search_mode": search_mode,
    }


def score_exact_stage(
    seeds: Sequence[SeedWindow],
    mismatch_count: int,
    scorer: CpuCas13Scorer,
    batch_size: int,
    search_mode: str,
) -> list[dict]:
    records = []
    for seed_number, seed in enumerate(seeds, start=1):
        LOGGER.info(
            "Exact mismatch-%d search for seed %d/%d (%s, start %d)",
            mismatch_count,
            seed_number,
            len(seeds),
            seed.target_name,
            seed.start0 + 1,
        )
        for mutant_chunk in chunked(exact_mutants(seed.source_guide, mismatch_count), batch_size):
            guides = [guide for guide, _ in mutant_chunk]
            activities = scorer.score_guides_one_context(guides, seed.context)
            records.extend(
                candidate_record(
                    seed,
                    guide,
                    mismatch_count,
                    positions,
                    activity,
                    search_mode,
                )
                for (guide, positions), activity in zip(mutant_chunk, activities)
            )
    return records


def top_on_target(records: Sequence[dict], count: int) -> list[dict]:
    return sorted(
        records,
        key=lambda row: (
            float(row["on_target_activity_screen"]),
            -int(row["start_pos"]),
            str(row["guide_sequence"]),
        ),
        reverse=True,
    )[:count]


def beam_expand_stage(
    parents: Sequence[dict],
    seeds_by_id: dict[str, SeedWindow],
    next_mismatch_count: int,
    scorer: CpuCas13Scorer,
    batch_size: int,
    beam_width: int,
) -> tuple[list[dict], int]:
    children_by_seed: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
    for parent in parents:
        seed_id = str(parent["seed_id"])
        seed = seeds_by_id[seed_id]
        positions = tuple(
            int(value) - 1
            for value in str(parent["mismatch_positions_1based"]).split(",")
            if value
        )
        for guide, child_positions in expand_one_mismatch(
            seed.source_guide,
            str(parent["guide_sequence"]),
            positions,
        ):
            children_by_seed[seed_id].setdefault(guide, child_positions)

    expanded_count = sum(len(children) for children in children_by_seed.values())
    LOGGER.info(
        "Beam mismatch-%d expanded %d unique candidates from %d parents",
        next_mismatch_count,
        expanded_count,
        len(parents),
    )
    scored = []
    for seed_id, children in children_by_seed.items():
        seed = seeds_by_id[seed_id]
        items = list(children.items())
        for child_chunk in chunked(items, batch_size):
            guides = [guide for guide, _ in child_chunk]
            activities = scorer.score_guides_one_context(guides, seed.context)
            scored.extend(
                candidate_record(
                    seed,
                    guide,
                    next_mismatch_count,
                    positions,
                    activity,
                    "beam",
                )
                for (guide, positions), activity in zip(child_chunk, activities)
            )
    return top_on_target(scored, beam_width), expanded_count


def profile_metrics(
    profile: np.ndarray,
    interest: int,
    off_tail_lambda: float = 0.4,
) -> dict:
    off_indices = [index for index in range(len(profile)) if index != interest]
    if not off_indices:
        raise ValueError("Differential scoring requires at least one off-target sequence")
    off_values = np.asarray([profile[index] for index in off_indices], dtype=np.float32)
    order = np.argsort(off_values)[::-1]
    sorted_values = off_values[order]
    on_target = float(profile[interest])
    max_off = float(sorted_values[0])
    mean_off = float(np.mean(sorted_values))
    top3_count = min(3, len(sorted_values))
    top3_mean = float(np.mean(sorted_values[:top3_count]))
    score_k1 = on_target - ((1.0 - off_tail_lambda) * mean_off + off_tail_lambda * max_off)
    return {
        "mean_on_target_act": on_target,
        "max_off_target_act": max_off,
        "mean_off_target_act": mean_off,
        "mean_top3_off_target_act": top3_mean,
        "top3_off_target_count": top3_count,
        "score_k1_lambda0p4": float(score_k1),
        "score_on_minus_max_off_target": float(on_target - max_off),
        "score_on_minus_mean_top3_off_target": float(on_target - top3_mean),
        "off_order": [off_indices[int(index)] for index in order],
    }


def exhaustive_full_length_rescore(
    finalists: Sequence[dict],
    target_local_index: int,
    sequences: Sequence[str],
    sequence_names: Sequence[str],
    seeds_by_id: dict[str, SeedWindow],
    libraries: Sequence[window_utils.ContextLibrary],
    context_pool: DeduplicatedContextPool,
    scorer: CpuCas13Scorer,
    guide_batch_size: int,
    context_batch_size: int,
    cutoff: float,
) -> list[dict]:
    rescored = []
    for batch_number, batch in enumerate(chunked(finalists, guide_batch_size), start=1):
        LOGGER.info(
            "Full-window rescore batch %d (%d candidates) for %s",
            batch_number,
            len(batch),
            sequence_names[target_local_index],
        )
        guides = [str(row["guide_sequence"]) for row in batch]
        intended_contexts = [seeds_by_id[str(row["seed_id"])].context for row in batch]
        guide_onehot = scorer.encode_guides(guides)
        on_values = scorer.score_paired_contexts(guides, intended_contexts)

        profiles = np.full((len(batch), len(sequences)), -np.inf, dtype=np.float32)
        best_starts = np.full((len(batch), len(sequences)), -1, dtype=np.int32)
        profiles[:, target_local_index] = on_values
        best_starts[:, target_local_index] = np.asarray(
            [int(row["start_pos"]) for row in batch], dtype=np.int32
        )

        # Model inference is the expensive operation.  Score every distinct
        # 48-nt context once, then reconstruct each sequence's original window
        # order from its map.  This preserves per-sequence maxima and earliest
        # tied starts while avoiding repeated predictions for homologous input.
        unique_values = np.empty(
            (len(batch), len(context_pool)), dtype=np.float32
        )
        for context_start in range(0, len(context_pool), context_batch_size):
            context_end = min(context_start + context_batch_size, len(context_pool))
            unique_values[:, context_start:context_end] = scorer.score_encoded(
                guide_onehot,
                context_pool.encoded_contexts[context_start:context_end],
            )

        for off_index, (library, window_unique_indices) in enumerate(
            zip(libraries, context_pool.window_unique_indices)
        ):
            if off_index == target_local_index:
                continue
            # Advanced indexing rebuilds this sequence's score columns in its
            # original window order.  np.argmax returns the first maximum, as
            # did the previous sequential context-batch scan.
            sequence_values = unique_values[:, window_unique_indices]
            best_window_indices = np.argmax(sequence_values, axis=1)
            profiles[:, off_index] = sequence_values[
                np.arange(len(batch)), best_window_indices
            ]
            best_starts[:, off_index] = (
                library.guide_starts0[best_window_indices] + 1
            )

        for row_index, source_row in enumerate(batch):
            profile = profiles[row_index]
            metrics = profile_metrics(profile, target_local_index, 0.4)
            if metrics["mean_on_target_act"] < cutoff:
                continue
            off_order = metrics.pop("off_order")
            max_index = off_order[0]
            top3_indices = off_order[: min(3, len(off_order))]
            result = dict(source_row)
            result.update(metrics)
            result.update(
                {
                    "max_off_target_name": sequence_names[max_index],
                    "max_off_target_start_pos": int(best_starts[row_index, max_index]),
                    "top3_off_target_names": ",".join(
                        sequence_names[index] for index in top3_indices
                    ),
                    "prediction_rescored_exhaustively": True,
                    "activity_profile_json": json.dumps(
                        {
                            name: round(float(value), 7)
                            for name, value in zip(sequence_names, profile)
                        },
                        sort_keys=True,
                    ),
                    "best_start_profile_json": json.dumps(
                        {
                            name: int(value)
                            for name, value in zip(sequence_names, best_starts[row_index])
                        },
                        sort_keys=True,
                    ),
                    "off_tail_k": 1,
                    "off_tail_lambda": 0.4,
                    "on_target_cutoff": cutoff,
                    "source_algorithm": "search_guide_v6_indel_cpu",
                }
            )
            rescored.append(result)
    return rescored


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def read_table_if_present(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_signature(input_path: Path, args: argparse.Namespace) -> dict:
    helper_path = Path(window_utils.__file__).resolve()
    code_path = Path(__file__).resolve()
    semantic_parameters = {
        "cut_off": args.cut_off,
        "max_mismatches": args.max_mismatches,
        "n_seed_windows": args.n_seed_windows,
        "beam_width": args.beam_width,
        "finalists_per_mismatch": args.finalists_per_mismatch,
        "n_top_guides": args.n_top_guides,
        "off_tail_k": 1,
        "off_tail_lambda": 0.4,
        "include_wt_as_target": args.include_wt_as_target,
        "target_selection": args.target_selection,
        "position_base": args.position_base,
        "wt_name": args.wt_name,
        "only_target": args.only_target,
    }
    payload = {
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "code_sha256": sha256_file(code_path),
        "window_helper_sha256": sha256_file(helper_path),
        "parameters": semantic_parameters,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "payload": payload,
    }


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def backup_existing_target(target_dir: Path) -> Path:
    backup_root = target_dir.parent / "_previous_runs"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    destination = backup_root / f"{target_dir.name}.{stamp}.job{safe_filename(job_id)}"
    shutil.move(str(target_dir), str(destination))
    return destination


def add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for label, column in METRIC_COLUMNS.items():
        frame[f"rank_{label}"] = frame[column].rank(
            method="first", ascending=False
        ).astype(int)
        frame[f"rank_{label}_within_mismatch"] = frame.groupby("num_mismatches")[
            column
        ].rank(method="first", ascending=False).astype(int)
    return frame


def write_ranked_outputs(
    rescored: Sequence[dict],
    target_dir: Path,
    top_count: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(rescored)
    if frame.empty:
        frame = pd.DataFrame(columns=RANKED_OUTPUT_COLUMNS)
        write_table(frame, target_dir / "results_compiled_v6_indel.tsv")
        for label in METRIC_COLUMNS:
            write_table(frame, target_dir / f"top_{top_count}_{label}.tsv")
        write_table(frame, target_dir / "top_union_all_metrics.tsv")
        frame.to_excel(target_dir / "top_union_all_metrics.xlsx", index=False)
        return frame
    frame = add_ranks(frame)
    frame = frame.sort_values(
        ["rank_k1_lambda0p4", "rank_on_minus_max", "rank_on_minus_top3"]
    ).reset_index(drop=True)
    target_label = safe_filename(frame.iloc[0]["on_target_name"])
    cluster_label = safe_filename(target_dir.parent.name)
    frame.insert(
        0,
        "candidate_id",
        [
            f"{cluster_label}.{target_label}.v6_{index + 1}"
            for index in range(len(frame))
        ],
    )
    write_table(frame, target_dir / "results_compiled_v6_indel.tsv")

    selected_ids = set()
    selected_by: dict[str, list[str]] = defaultdict(list)
    for label, column in METRIC_COLUMNS.items():
        top = frame.sort_values(column, ascending=False).head(top_count).copy()
        write_table(top, target_dir / f"top_{top_count}_{label}.tsv")
        for candidate_id in top["candidate_id"]:
            selected_ids.add(candidate_id)
            selected_by[candidate_id].append(label)

    union = frame[frame["candidate_id"].isin(selected_ids)].copy()
    union["selected_by"] = union["candidate_id"].map(
        lambda candidate_id: ",".join(selected_by[candidate_id])
    )
    write_table(union, target_dir / "top_union_all_metrics.tsv")
    union.to_excel(target_dir / "top_union_all_metrics.xlsx", index=False)
    return frame


def stage_frame(records: Sequence[dict]) -> pd.DataFrame:
    columns = [
        "on_target_name",
        "seed_id",
        "start_pos",
        "source_guide",
        "guide_sequence",
        "num_mismatches",
        "mismatch_positions_1based",
        "event_ids",
        "event_kinds",
        "on_target_activity_screen",
        "search_mode",
    ]
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame.sort_values("on_target_activity_screen", ascending=False).reset_index(drop=True)


def run_target_search(
    target_local_index: int,
    target_global_index: int,
    target_name: str,
    seeds: Sequence[SeedWindow],
    sequences: Sequence[str],
    sequence_names: Sequence[str],
    scorer: CpuCas13Scorer,
    libraries: Sequence[window_utils.ContextLibrary],
    context_pool: DeduplicatedContextPool,
    args: argparse.Namespace,
    target_dir: Path,
    run_signature: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    marker = target_dir / ".v6_indel_complete"
    compiled_path = target_dir / "results_compiled_v6_indel.tsv"
    signature_path = target_dir / "run_signature.json"
    summary_path = target_dir / "search_stage_summary.tsv"
    if target_dir.exists():
        stored_signature = load_json(signature_path)
        is_complete = marker.exists() and compiled_path.exists()
        signature_matches = (
            stored_signature is not None
            and stored_signature.get("digest") == run_signature.get("digest")
        )
        if args.resume == "yes" and is_complete:
            if not signature_matches:
                raise RuntimeError(
                    f"Completed output for {target_name} has a different input/code/parameter "
                    "signature. Choose a new --output_dir or rerun with --resume no."
                )
            LOGGER.info("Skipping completed target: %s", target_name)
            summary = read_table_if_present(summary_path).to_dict("records")
            return read_table_if_present(compiled_path), summary

        backup = backup_existing_target(target_dir)
        LOGGER.info("Moved partial or forced-rerun output to %s", backup)

    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_signature, signature_path)

    seeds_by_id = {seed.seed_id: seed for seed in seeds}
    summary_rows = []
    stages: dict[int, list[dict]] = {}

    stages[0] = score_exact_stage(
        seeds,
        0,
        scorer,
        args.batch_size,
        "unmodified_junction",
    )
    write_table(stage_frame(stages[0]), target_dir / "on_target_stage_mm0.tsv")
    summary_rows.append(
        {"num_mismatches": 0, "mode": "unmodified_junction", "evaluated": len(stages[0]), "retained": len(stages[0])}
    )

    best_seed_records = top_on_target(stages[0], args.n_seed_windows)
    expansion_seed_ids = {str(row["seed_id"]) for row in best_seed_records}
    expansion_seeds = [seed for seed in seeds if seed.seed_id in expansion_seed_ids]
    LOGGER.info(
        "%s: expanding %d/%d junction windows",
        target_name,
        len(expansion_seeds),
        len(seeds),
    )

    if args.max_mismatches >= 1:
        stages[1] = score_exact_stage(
            expansion_seeds,
            1,
            scorer,
            args.batch_size,
            "exhaustive",
        )
        write_table(stage_frame(stages[1]), target_dir / "on_target_stage_mm1.tsv")
        summary_rows.append(
            {"num_mismatches": 1, "mode": "exhaustive", "evaluated": len(stages[1]), "retained": len(stages[1])}
        )

    if args.max_mismatches >= 2:
        stages[2] = score_exact_stage(
            expansion_seeds,
            2,
            scorer,
            args.batch_size,
            "exhaustive",
        )
        write_table(stage_frame(stages[2]), target_dir / "on_target_stage_mm2.tsv")
        summary_rows.append(
            {"num_mismatches": 2, "mode": "exhaustive", "evaluated": len(stages[2]), "retained": len(stages[2])}
        )

    if args.max_mismatches >= 3:
        parents = top_on_target(stages[2], args.beam_width)
        for mismatch_count in range(3, args.max_mismatches + 1):
            retained, evaluated = beam_expand_stage(
                parents,
                seeds_by_id,
                mismatch_count,
                scorer,
                args.batch_size,
                args.beam_width,
            )
            stages[mismatch_count] = retained
            write_table(
                stage_frame(retained),
                target_dir / f"on_target_stage_mm{mismatch_count}.tsv",
            )
            summary_rows.append(
                {
                    "num_mismatches": mismatch_count,
                    "mode": "beam",
                    "evaluated": evaluated,
                    "retained": len(retained),
                }
            )
            parents = retained
            if not parents:
                LOGGER.warning("Beam became empty at mismatch %d", mismatch_count)
                break

    write_table(pd.DataFrame(summary_rows), summary_path)

    all_rescored = []
    for mismatch_count in sorted(stages):
        cutoff_passing = [
            row
            for row in stages[mismatch_count]
            if float(row["on_target_activity_screen"]) >= args.cut_off
        ]
        finalists = top_on_target(cutoff_passing, args.finalists_per_mismatch)
        LOGGER.info(
            "%s mismatch-%d: %d pass cutoff; fully rescoring %d",
            target_name,
            mismatch_count,
            len(cutoff_passing),
            len(finalists),
        )
        rescored = exhaustive_full_length_rescore(
            finalists,
            target_local_index,
            sequences,
            sequence_names,
            seeds_by_id,
            libraries,
            context_pool,
            scorer,
            args.rescore_guide_batch_size,
            args.rescore_context_batch_size,
            args.cut_off,
        )
        rescore_frame = (
            pd.DataFrame(rescored)
            if rescored
            else pd.DataFrame(columns=RANKED_OUTPUT_COLUMNS[1:])
        )
        write_table(rescore_frame, target_dir / f"full_rescore_mm{mismatch_count}.tsv")
        all_rescored.extend(rescored)

    ranked = write_ranked_outputs(all_rescored, target_dir, args.n_top_guides)
    marker.touch()
    return ranked, summary_rows


def plan_cluster(
    table: pd.DataFrame,
    global_indices: Sequence[int],
    wt_global_index: int,
    events_by_target: dict[int, list[EditEvent]],
    include_wt: bool,
) -> tuple[list[str], list[str], dict[int, list[SeedWindow]]]:
    sequences = [str(table.iloc[index]["_v6_sequence"]) for index in global_indices]
    names = [str(table.iloc[index]["_v6_name"]) for index in global_indices]
    global_to_local = {global_index: local for local, global_index in enumerate(global_indices)}
    cluster_events = [
        event
        for global_index in global_indices
        if global_index != wt_global_index
        for event in events_by_target.get(global_index, [])
    ]

    seeds_by_target: dict[int, list[SeedWindow]] = {}
    design_global_indices = [
        global_index
        for global_index in global_indices
        if global_index != wt_global_index and events_by_target.get(global_index)
    ]
    if include_wt and cluster_events:
        design_global_indices.insert(0, wt_global_index)

    for global_index in design_global_indices:
        local_index = global_to_local[global_index]
        if global_index == wt_global_index:
            intervals = [
                (event, event.wt_start0, event.wt_end0) for event in cluster_events
            ]
        else:
            intervals = [
                (event, event.target_start0, event.target_end0)
                for event in events_by_target[global_index]
            ]
        seeds_by_target[local_index] = build_seed_windows(
            local_index,
            global_index,
            names[local_index],
            sequences[local_index],
            intervals,
        )
    return sequences, names, seeds_by_target


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    input_path = Path(args.file_name).expanduser().resolve()
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path.cwd() / "output" / args.project_name).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    run_signature = build_run_signature(input_path, args)
    signature_path = output_root / "run_signature.json"
    stored_signature = load_json(signature_path)
    if (
        args.resume == "yes"
        and stored_signature is not None
        and stored_signature.get("digest") != run_signature.get("digest")
    ):
        raise RuntimeError(
            "The output directory contains a different input/code/parameter signature. "
            "Choose a new --output_dir or rerun with --resume no."
        )
    write_json(run_signature, signature_path)

    table = read_input_table(input_path)
    wt_global_index = find_wt_index(table, args.wt_name)
    explicit_indel_targets = explicit_indel_target_indices(table, wt_global_index)
    if args.target_selection == "indel_column" and explicit_indel_targets is None:
        raise ValueError(
            "--target_selection indel_column requires an input column named indel "
            "with yes/no or 1/0 in every row"
        )
    wt_row = table.iloc[wt_global_index]
    wt_raw = str(wt_row["_v6_raw_sequence"])
    wt_sequence = str(wt_row["_v6_sequence"])
    wt_name = str(wt_row["_v6_name"])
    LOGGER.info("WT/reference: %s (row %d)", wt_name, wt_global_index + 2)

    if explicit_indel_targets is None:
        candidate_target_indices = {
            index for index in range(len(table)) if index != wt_global_index
        }
        selection_mode = "automatic_indel_detection"
        LOGGER.info(
            "No indel column found; using automatic indel detection for on-target selection"
        )
    else:
        candidate_target_indices = explicit_indel_targets
        selection_mode = "explicit_indel_column"
        LOGGER.info(
            "Indel column selected %d non-WT on-target(s); all %d records remain "
            "in their cluster comparison libraries",
            len(candidate_target_indices),
            len(table),
        )

    events_by_target: dict[int, list[EditEvent]] = {}
    event_rows = []
    for global_index, row in table.iterrows():
        if global_index not in candidate_target_indices:
            continue
        target_name = str(row["_v6_name"])
        events = infer_indel_events(
            wt_raw,
            str(row["_v6_raw_sequence"]),
            wt_sequence,
            str(row["_v6_sequence"]),
            row,
            int(global_index),
            target_name,
            args.position_base,
        )
        if events:
            events_by_target[int(global_index)] = events
            for event in events:
                event_rows.append(
                    {
                        **event.__dict__,
                        "wt_length": event.wt_length,
                        "target_length": event.target_length,
                    }
                )
    if explicit_indel_targets is not None:
        missing_events = sorted(explicit_indel_targets - set(events_by_target))
        if missing_events:
            missing_names = [str(table.iloc[index]["_v6_name"]) for index in missing_events]
            raise ValueError(
                "Rows marked indel=yes/1 have no detectable insertion or deletion: "
                f"{missing_names[:5]}"
            )
    if not events_by_target:
        raise ValueError("No indel events were detected in the input")
    write_table(pd.DataFrame(event_rows), output_root / "detected_indel_events.tsv")

    selected_on_targets = set(events_by_target)
    if args.include_wt_as_target == "yes":
        selected_on_targets.add(wt_global_index)
    only_target_global_index = resolve_only_target_index(
        table,
        selected_on_targets,
        args.only_target,
    )
    if only_target_global_index is not None:
        LOGGER.info(
            "Isolated target execution: %s; all records remain in the comparison library",
            args.only_target,
        )
    selection_rows = []
    for global_index, row in table.iterrows():
        selection_rows.append(
            {
                "spreadsheet_row": int(global_index) + 2,
                "name": str(row["_v6_name"]),
                "is_wt": global_index == wt_global_index,
                "indel_flag": (
                    bool(row["_v6_indel_target"])
                    if "_v6_indel_target" in table.columns
                    else ""
                ),
                "selected_as_on_target": global_index in selected_on_targets,
                "retained_for_off_target_comparison": True,
                "selection_mode": selection_mode,
            }
        )
    write_table(
        pd.DataFrame(selection_rows, columns=TARGET_SELECTION_COLUMNS),
        output_root / "on_target_selection.tsv",
    )

    groups = cluster_groups(table, wt_global_index)
    safe_cluster_labels = [safe_filename(label) for label, _ in groups]
    if len(safe_cluster_labels) != len(set(safe_cluster_labels)):
        raise ValueError("Cluster labels collide after filename sanitization")
    cluster_plans = []
    planned = []
    targets_without_windows = []
    for cluster_label, global_indices in groups:
        sequences, names, seeds_by_target = plan_cluster(
            table,
            global_indices,
            wt_global_index,
            events_by_target,
            args.include_wt_as_target == "yes",
        )
        cluster_plans.append((cluster_label, global_indices, sequences, names, seeds_by_target))
        for local_index, seeds in seeds_by_target.items():
            if not seeds:
                targets_without_windows.append(
                    f"cluster {cluster_label}: {names[local_index]}"
                )
            for seed in seeds:
                planned.append(
                    {
                        "cluster": cluster_label,
                        "on_target_name": names[local_index],
                        "start_pos": seed.start0 + 1,
                        "event_ids": ",".join(seed.event_ids),
                        "event_kinds": ",".join(seed.event_kinds),
                    }
                )
    plan_frame = (
        pd.DataFrame(planned)
        if planned
        else pd.DataFrame(columns=JUNCTION_PLAN_COLUMNS)
    )
    write_table(plan_frame, output_root / "junction_window_plan.tsv")
    if targets_without_windows:
        examples = "; ".join(targets_without_windows[:5])
        raise ValueError(
            "Indel target(s) have no valid junction window with a 28-nt guide and "
            f"10-nt flanks: {examples}"
        )
    if not planned:
        raise ValueError(
            "Indels were detected, but no valid junction windows could be generated"
        )
    if args.plan_only:
        LOGGER.info("Plan-only validation completed: %s", output_root)
        return 0

    scorer = CpuCas13Scorer(args.num_cpu, args.batch_size)
    all_compiled = []
    all_stage_summaries = []
    context_deduplication_rows = []
    executed_target_names = []
    for cluster_label, _, sequences, names, seeds_by_target in cluster_plans:
        execution_seeds_by_target = {
            local_index: seeds
            for local_index, seeds in seeds_by_target.items()
            if only_target_global_index is None
            or names[local_index] == args.only_target
        }
        if not execution_seeds_by_target:
            continue
        if not seeds_by_target:
            LOGGER.warning("Cluster %s has no indel target windows", cluster_label)
            continue
        LOGGER.info(
            "Preparing full-length context libraries for cluster %s (%d sequences)",
            cluster_label,
            len(sequences),
        )
        libraries = [window_utils.build_context_library(sequence) for sequence in sequences]
        context_pool = build_deduplicated_context_pool(libraries, scorer)
        LOGGER.info(
            "Cluster %s full-length context deduplication: %d total 48-nt "
            "windows -> %d unique contexts (%.2fx reuse)",
            cluster_label,
            context_pool.total_context_count,
            len(context_pool),
            context_pool.reuse_factor,
        )
        context_deduplication_rows.append(
            {
                "cluster": cluster_label,
                "sequence_count": len(sequences),
                "total_context_windows": context_pool.total_context_count,
                "unique_48nt_contexts": len(context_pool),
                "reuse_factor": context_pool.reuse_factor,
            }
        )
        write_table(
            pd.DataFrame(context_deduplication_rows),
            output_root / "context_deduplication.tsv",
        )

        for target_local_index, seeds in execution_seeds_by_target.items():
            if not seeds:
                LOGGER.warning("No valid junction windows for %s", names[target_local_index])
                continue
            executed_target_names.append(names[target_local_index])
            target_dir = output_root / f"cluster-{safe_filename(cluster_label)}" / safe_filename(names[target_local_index])
            ranked, summaries = run_target_search(
                target_local_index,
                seeds[0].target_global_index,
                names[target_local_index],
                seeds,
                sequences,
                names,
                scorer,
                libraries,
                context_pool,
                args,
                target_dir,
                run_signature,
            )
            ranked = ranked.copy()
            ranked.insert(0, "cluster", cluster_label)
            all_compiled.append(ranked)
            for row in summaries:
                all_stage_summaries.append(
                    {"cluster": cluster_label, "on_target_name": names[target_local_index], **row}
                )

    compiled = (
        pd.concat(all_compiled, ignore_index=True)
        if all_compiled
        else pd.DataFrame(columns=["cluster", *RANKED_OUTPUT_COLUMNS])
    )
    write_table(compiled, output_root / "results_all_targets_v6_indel.tsv")
    aggregate_summary = (
        pd.DataFrame(all_stage_summaries)
        if all_stage_summaries
        else pd.DataFrame(columns=["cluster", "on_target_name", *STAGE_SUMMARY_COLUMNS])
    )
    write_table(aggregate_summary, output_root / "search_stage_summary_all_targets.tsv")
    if args.only_target is not None and executed_target_names != [args.only_target]:
        raise RuntimeError(
            "Isolated target execution did not run exactly the requested target: "
            f"requested={args.only_target!r}, executed={executed_target_names!r}"
        )
    metadata = {
        "input": str(input_path),
        "output": str(output_root),
        "algorithm": "search_guide_v6_indel_cpu",
        "num_cpu": args.num_cpu,
        "cut_off": args.cut_off,
        "off_tail_k": 1,
        "off_tail_lambda": 0.4,
        "n_seed_windows": args.n_seed_windows,
        "beam_width": args.beam_width,
        "max_mismatches": args.max_mismatches,
        "finalists_per_mismatch": args.finalists_per_mismatch,
        "only_target": args.only_target,
        "executed_targets": executed_target_names,
        "ranking_metrics": METRIC_COLUMNS,
    }
    write_json(metadata, output_root / "run_metadata.json")
    LOGGER.info("Completed v6 indel search: %s", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
