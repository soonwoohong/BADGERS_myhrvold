#!/usr/bin/env python3
"""Align and group FASTA files into one-vs-others BADGERS input bundles."""

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

from Bio import SeqIO
from Bio.Align import PairwiseAligner


FASTA_EXTENSIONS = {".fa", ".fasta", ".fna"}
CLUSTER_SUFFIX = re.compile(r"_cl([1-9][0-9]*)$", re.IGNORECASE)
BASES = set("ACGT")
FASTA_ALPHABET = BASES | set("RYSWKMBDHVN-")
# Public BADGERS uses 28 target bases and 10 context bases on each side.
CONTEXT_WINDOW = 48
FIRST_SITE = 11
JOB_COLUMNS = ("task_id", "cluster", "input_dir", "member_count", "engine", "input_table", "wt_name")


@dataclass
class Sample:
    path: Path
    id: str
    sequence: str
    cluster: Optional[int]


@dataclass
class Comparison:
    sample: Sample
    bases: str                         # One character per ungapped WT base.
    insertions: Dict[int, str]          # Key is the number of WT bases before the insertion.
    deletions: List[Tuple[int, int]]   # Inclusive 1-based WT intervals.
    substitutions: Dict[int, str]
    ambiguous_bases: int
    tied_alignment: bool
    cluster: Optional[int] = None
    exceeds_window: bool = False

    @property
    def kind(self):
        if self.insertions or self.deletions:
            return "indel"
        if self.ambiguous_bases or self.tied_alignment:
            return "review"
        return "substitution" if self.substitutions else "matches_wt"

    def span(self):
        positions = list(self.substitutions)
        positions.extend(max(1, anchor) for anchor in self.insertions)
        for start, end in self.deletions:
            positions.extend((start, end))
        return (min(positions), max(positions)) if positions else (None, None)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def fasta_paths(directory):
    paths = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in FASTA_EXTENSIONS)
    wt_paths = [path for path in paths if path.stem.lower().endswith("_wt")]
    if len(wt_paths) != 1:
        raise ValueError(f"Expected exactly one filename ending in _WT; found {len(wt_paths)}.")
    wt = wt_paths[0]
    variants = [path for path in paths if path != wt]
    if not variants:
        raise ValueError("Add at least one variant FASTA alongside the WT FASTA.")
    return wt, variants


def cluster_suffix(path):
    match = CLUSTER_SUFFIX.search(path.stem)
    return int(match.group(1)) if match else None


def read_sample(path):
    try:
        record = SeqIO.read(path, "fasta")
    except ValueError as error:
        raise ValueError(f"{path.name}: expected one FASTA record ({error}).") from error
    sequence = str(record.seq).upper()
    if not sequence.strip("-") or set(sequence) - FASTA_ALPHABET:
        raise ValueError(f"{path.name}: expected nonempty DNA with IUPAC bases and optional gaps (-).")
    return Sample(path, path.stem, sequence, cluster_suffix(path))


def validate_names(wt, comparisons):
    sources = {}
    for sample in [wt] + [row.sample for row in comparisons]:
        if sample.id in sources:
            raise ValueError(f"Duplicate filename stem {sample.id!r}: "
                             f"{sources[sample.id]} and {sample.path.name}.")
        if any(char in sample.id for char in "\t\r\n"):
            raise ValueError(f"Filename contains a tab or newline: {sample.path.name!r}.")
        sources[sample.id] = sample.path.name
    for row in comparisons:
        if row.cluster is not None and row.sample.id in {
            f"{wt.id}_cl{row.cluster}", f"{wt.id}_cl{row.cluster}_indel"
        }:
            raise ValueError(f"{row.sample.path.name} conflicts with the generated WT name "
                             f"in cl{row.cluster}.")


def make_aligner():
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2
    aligner.mismatch_score = -3
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -1
    return aligner


def compare(wt, sample, aligner):
    if aligner is None:
        ref, query = wt.sequence, sample.sequence
        if len(ref) != len(query):
            raise ValueError(f"{sample.path.name}: prealigned sequences must share WT's alignment width.")
        tied = False
    else:
        alignments = iter(aligner.align(wt.sequence.replace("-", ""), sample.sequence.replace("-", "")))
        alignment = next(alignments)
        ref, query = alignment[0], alignment[1]
        tied = next(alignments, None) is not None

    bases = []
    insertions = defaultdict(list)
    deletions = []
    substitutions = {}
    for wt_base, base in zip(ref, query):
        if wt_base == "-":
            if base != "-":
                insertions[len(bases)].append(base)
            continue
        bases.append(base)
        position = len(bases)
        if base == "-":
            if deletions and deletions[-1][1] == position - 1:
                deletions[-1] = (deletions[-1][0], position)
            else:
                deletions.append((position, position))
        elif base in BASES and base != wt_base:
            substitutions[position] = f"{wt_base}>{base}"

    return Comparison(sample, "".join(bases), {k: "".join(v) for k, v in insertions.items()},
                      deletions, substitutions, sum(base not in BASES and base != "-" for base in query), tied)


def assign_clusters(comparisons, mode, window):
    if mode == "filename":
        for row in comparisons:
            row.cluster = row.sample.cluster
        return

    # Use the entire mutation span. A chain of nearby variants cannot extend a cluster.
    spans = [(row.span(), row) for row in comparisons]
    positioned = [entry for entry in spans if entry[0][0] is not None]
    positioned.sort(key=lambda entry: (*entry[0], entry[1].sample.id))
    next_cluster = 1
    active_cluster = None
    cluster_start = None
    for (start, end), row in positioned:
        row.exceeds_window = end - start + 1 > window
        if row.exceeds_window:
            row.cluster = next_cluster
            next_cluster += 1
            continue
        if cluster_start is None or end - cluster_start + 1 > window:
            active_cluster = next_cluster
            next_cluster += 1
            cluster_start = start
        row.cluster = active_cluster


def write_tsv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def write_fasta(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n")
            for i in range(0, len(sequence), 80):
                handle.write(sequence[i:i + 80] + "\n")


def alignment_records(reference, rows, wt_id):
    # Reserve insertion columns at each WT boundary for viewing together.
    widths = defaultdict(int)
    for row in rows:
        for anchor, sequence in row.insertions.items():
            widths[anchor] = max(widths[anchor], len(sequence))

    def aligned(bases, inserts):
        parts = [inserts.get(0, "").ljust(widths[0], "-")]
        for position, base in enumerate(bases, start=1):
            parts.extend((base, inserts.get(position, "").ljust(widths[position], "-")))
        return "".join(parts)

    records = [(wt_id, aligned(reference, {}))]
    records.extend((row.sample.id, aligned(row.bases, row.insertions)) for row in rows)
    return records


def write_alignment(path, reference, rows, wt_id):
    write_fasta(path, alignment_records(reference, rows, wt_id))


def summary_row(row):
    start, end = row.span()
    return (row.sample.id, row.sample.path.name, f"cl{row.cluster}" if row.cluster else "",
            row.kind, start, end, ",".join(f"{pos}:{change}" for pos, change in row.substitutions.items()),
            ",".join(f"{anchor}+{len(seq)}" for anchor, seq in row.insertions.items()),
            ",".join(f"{low}-{high}" for low, high in row.deletions), row.ambiguous_bases,
            int(row.tied_alignment), int(row.exceeds_window))


def write_bundle(out, wt, comparisons):
    validate_names(wt, comparisons)
    columns = ("id", "source_file", "cluster", "kind", "first_wt_pos", "last_wt_pos", "substitutions",
               "insertions_after_wt_pos", "deletions_wt_interval", "ambiguous_bases", "tied_alignment",
               "exceeds_window")
    reference = wt.sequence.replace("-", "")
    groups = defaultdict(list)
    for row in comparisons:
        groups[row.cluster].append(row)
    summary = []
    jobs = []
    indel_jobs = []
    for cluster in sorted(key for key in groups if key is not None):
        rows = groups[cluster]
        label = f"cl{cluster}"
        wt_id = f"{wt.id}_{label}"
        summary.append((wt_id, wt.path.name, label, "reference", "", "", "", "", "", 0, 0, 0))
        summary.extend(summary_row(row) for row in rows)
        directory = out / "clusters" / label
        directory.mkdir(parents=True)
        write_alignment(directory / "aligned.fasta", reference, rows, wt_id)
        compatible = [row for row in rows if not row.ambiguous_bases]
        if any(row.kind == "indel" for row in compatible) and len(reference) >= CONTEXT_WINDOW:
            indel_label = f"{label}_indel"
            indel_wt_id = f"{wt.id}_{indel_label}"
            if any(row.sample.id == indel_wt_id for row in compatible):
                raise ValueError(f"A variant filename conflicts with the generated WT name {indel_wt_id!r}.")
            indel_directory = out / "clusters" / indel_label
            indel_directory.mkdir()
            records = alignment_records(reference, compatible, indel_wt_id)
            write_fasta(indel_directory / "aligned.fasta", records)
            table = indel_directory / "input.tsv"
            kinds = {row.sample.id: row.kind == "indel" for row in compatible}
            write_tsv(table, ("name", "seq", "cluster", "indel"),
                      ((name, sequence, indel_label, "yes" if kinds.get(name, False) else "no")
                       for name, sequence in records))
            indel_jobs.append((indel_label, indel_directory.relative_to(out).as_posix(),
                               len(records), "v6_indel", table.relative_to(out).as_posix(), indel_wt_id))
        standard = [row for row in rows if row.kind == "substitution"]
        if not standard or len(reference) < CONTEXT_WINDOW:
            continue
        # Upstream diff already uses each FASTA as the focus and pools the remaining FASTAs.
        inputs = directory / "standard"
        inputs.mkdir()
        write_fasta(inputs / f"{wt_id}.fasta", [(wt_id, reference)])
        for row in standard:
            write_fasta(inputs / f"{row.sample.id}.fasta", [(row.sample.id, row.bases)])
        # Upstream uses the remaining FASTA stems in per-site filenames.
        stems = [wt_id] + [row.sample.id for row in standard]
        for stem in stems:
            if "split" in stem or ".fasta" in stem:
                raise ValueError(f"{stem!r} contains 'split' or '.fasta', which the public "
                                 "BADGERS importer treats specially. Rename this input file.")
        max_basename = sum(len(stem.encode("utf-8")) for stem in stems) + len(stems) + len(str(len(reference))) + 7
        if max_basename > 255:
            raise ValueError(f"{label} has too many members for BADGERS' output filenames. Split this cluster.")
        last_site = FIRST_SITE + len(reference) - CONTEXT_WINDOW
        (directory / "range.tsv").write_text(f"{FIRST_SITE}\t{last_site}\n", encoding="utf-8")
        jobs.append((len(jobs), label, inputs.relative_to(out).as_posix(), len(stems),
                     "ordinary", "", wt_id))

    first_indel_task = len(jobs)
    jobs.extend((first_indel_task + index, *job) for index, job in enumerate(indel_jobs))

    if None in groups:
        summary.append((wt.id, wt.path.name, "unassigned", "reference", "", "", "", "", "", 0, 0, 0))
        summary.extend(summary_row(row) for row in groups[None])
        write_alignment(out / "unassigned_aligned.fasta", reference, groups[None], wt.id)
    write_tsv(out / "variant_summary.tsv", columns, summary)
    write_tsv(out / "indels.tsv", columns, (summary_row(row) for row in comparisons if row.kind == "indel"))
    write_tsv(out / "jobs.tsv", JOB_COLUMNS, jobs)
    return len(jobs), len(reference)


def add_indel_clusters(staging, output):
    """Extend an earlier bundle while preserving its ordinary files and task IDs."""
    if not output.is_dir() or output.is_symlink():
        raise ValueError("--add-indel-clusters requires an existing prepared directory.")
    with (staging / "jobs.tsv").open(newline="", encoding="utf-8") as handle:
        staged_jobs = list(csv.DictReader(handle, delimiter="\t"))
    staged_by_cluster = {row["cluster"]: row for row in staged_jobs}
    with (output / "jobs.tsv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not set(JOB_COLUMNS[:4]).issubset(reader.fieldnames or []):
            raise ValueError("Existing jobs.tsv has an unsupported schema.")
        previous_jobs = list(reader)

    for path in staging.rglob("*"):
        if not path.is_file() or path.name == "jobs.tsv":
            continue
        relative = path.relative_to(staging)
        destination = output / relative
        is_indel = len(relative.parts) >= 2 and relative.parts[0] == "clusters" and relative.parts[1].endswith("_indel")
        if destination.exists() or not is_indel:
            if not destination.is_file() or destination.read_bytes() != path.read_bytes():
                raise ValueError(f"Existing prepared file differs from the current inputs: {relative}.")
        elif (output / "clusters" / relative.parts[1]).exists():
            raise ValueError(f"Existing indel directory is incomplete or conflicts: {relative.parts[1]}.")

    combined = []
    task_ids = set()
    seen_clusters = set()
    for previous in previous_jobs:
        cluster = previous["cluster"]
        staged = staged_by_cluster.get(cluster)
        if staged is None or cluster in seen_clusters:
            raise ValueError(f"Existing jobs.tsv conflicts with the current cluster plan: {cluster}.")
        task_id = int(previous["task_id"])
        if task_id < 0 or task_id in task_ids:
            raise ValueError("Existing jobs.tsv has duplicate or negative task IDs.")
        for column in ("input_dir", "member_count"):
            if previous[column] != staged[column]:
                raise ValueError(f"Existing jobs.tsv differs for {cluster}: {column}.")
        if previous.get("engine", "ordinary") != staged["engine"]:
            raise ValueError(f"Existing jobs.tsv has a different engine for {cluster}.")
        if staged["engine"] == "v6_indel":
            for column in ("input_table", "wt_name"):
                if previous.get(column) != staged[column]:
                    raise ValueError(f"Existing jobs.tsv differs for {cluster}: {column}.")
        combined.append((task_id, *(staged[column] for column in JOB_COLUMNS[1:])))
        task_ids.add(task_id)
        seen_clusters.add(cluster)
    next_id = max(task_ids, default=-1) + 1
    for staged in staged_jobs:
        if staged["cluster"] in seen_clusters:
            continue
        if staged["engine"] != "v6_indel":
            raise ValueError(f"Existing bundle is missing ordinary task {staged['cluster']}.")
        combined.append((next_id, *(staged[column] for column in JOB_COLUMNS[1:])))
        next_id += 1

    for directory in sorted((staging / "clusters").glob("cl*_indel")):
        destination = output / "clusters" / directory.name
        if not destination.exists():
            directory.rename(destination)
    write_tsv(staging / "jobs.tsv", JOB_COLUMNS, combined)
    (staging / "jobs.tsv").replace(output / "jobs.tsv")
    return len(combined)


def prompt_value(label, value, default=None, choices=None, convert=str):
    """Ask for an omitted option while keeping explicit command-line values."""
    if value is not None:
        return value
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{label}{suffix}: ").strip()
    answer = answer or default
    if answer is None:
        raise ValueError(f"{label} is required.")
    if choices is not None and answer not in choices:
        raise ValueError(f"{label} must be one of {', '.join(choices)}.")
    return convert(answer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, help="directory of one-record FASTA files with one _WT file")
    parser.add_argument("--clusters", choices=("check", "filename", "window"),
                        help="check suffixes (default), use _clN suffixes, or group by WT position")
    parser.add_argument("--out", type=Path, help="new directory for the prepared bundle")
    parser.add_argument("--add-indel-clusters", action="store_true",
                        help="add indel folders and manifest tasks to an existing --out bundle")
    parser.add_argument("--window", type=positive_int, help="maximum WT span in nt (default: 20)")
    parser.add_argument("--alignment", choices=("global", "prealigned"),
                        help="align each record to WT, or read an existing shared alignment")
    parser.add_argument("--interactive", action="store_true", help="prompt for omitted preparation options")
    args = parser.parse_args()

    try:
        if args.interactive:
            args.input_dir = prompt_value("FASTA directory", args.input_dir, convert=Path)
            args.clusters = prompt_value("Cluster mode (filename/window/check)", args.clusters,
                                         default="window", choices=("filename", "window", "check"))
            if args.clusters != "check":
                args.out = prompt_value("New output directory", args.out, convert=Path)
                if args.clusters == "window":
                    args.window = prompt_value("Maximum mutation span (nt)", args.window,
                                               default="20", convert=positive_int)
                args.alignment = prompt_value("Alignment (global/prealigned)", args.alignment,
                                              default="global", choices=("global", "prealigned"))
        if args.input_dir is None:
            parser.error("input_dir is required unless --interactive prompts for it")
        args.input_dir = args.input_dir.expanduser()
        if args.out is not None:
            args.out = args.out.expanduser()
        args.clusters = args.clusters or "check"
        args.window = args.window if args.window is not None else 20
        args.alignment = args.alignment or "global"
        wt_path, variant_paths = fasta_paths(args.input_dir)
        labeled = sum(cluster_suffix(path) is not None for path in variant_paths)
        if args.clusters == "check":
            print(f"WT: {wt_path.name}. Cluster suffix present on {labeled}/{len(variant_paths)} variants.")
            if labeled == len(variant_paths):
                print("Use --clusters filename to keep the existing labels, or --clusters window to regroup.")
            else:
                print("Use --clusters window to group all variants by position, or label every variant first.")
            return
        if args.out is None:
            parser.error("--out is required with --clusters filename or window")
        if (args.out.exists() or args.out.is_symlink()) and not args.add_indel_clusters:
            raise ValueError(f"Output path already exists: {args.out}. Choose a new run directory.")
        if args.add_indel_clusters and not args.out.is_dir():
            raise ValueError("--add-indel-clusters requires an existing --out bundle.")
        if args.clusters == "filename" and labeled != len(variant_paths):
            missing = [path.name for path in variant_paths if cluster_suffix(path) is None]
            raise ValueError("Missing _clN suffix: " + ", ".join(missing))

        wt = read_sample(wt_path)
        if set(wt.sequence.replace("-", "")) - BASES:
            raise ValueError("WT must have unambiguous A/C/G/T bases in its ungapped sequence.")
        aligner = make_aligner() if args.alignment == "global" else None
        comparisons = [compare(wt, read_sample(path), aligner) for path in variant_paths]
        assign_clusters(comparisons, args.clusters, args.window)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".badgers-prep-", dir=args.out.parent) as temp:
            staging = Path(temp) / "bundle"
            staging.mkdir()
            job_count, ref_length = write_bundle(staging, wt, comparisons)
            if args.add_indel_clusters:
                job_count = add_indel_clusters(staging, args.out)
            else:
                staging.rename(args.out)
    except (OSError, ValueError, EOFError, argparse.ArgumentTypeError) as error:
        parser.exit(2, f"Error: {error}\n")

    indel_count = sum(row.kind == "indel" for row in comparisons)
    print(f"Prepared {args.out}: {job_count} BADGERS jobs, {indel_count} indel records.")
    if job_count == 0:
        if ref_length < CONTEXT_WINDOW:
            print(f"The WT is {ref_length} nt; ordinary BADGERS needs at least {CONTEXT_WINDOW} nt.")
        else:
            print("No cluster has a suitable substitution record for ordinary BADGERS.")


if __name__ == "__main__":
    sys.exit(main())
