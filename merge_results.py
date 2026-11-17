#!/usr/bin/env python3
"""Combine compiled BADGERS result tables while recording each row's source."""

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile


COMPILED_FILES = {
    "results_compiled_evolutionary.tsv": "evolutionary",
    "results_compiled_wgan_am.tsv": "wgan_am",
    "results_all_targets_v6_indel.tsv": "v6_indel",
    "results_compiled_v6_indel.tsv": "v6_indel",
}
REQUIRED_COLUMNS = {"guide_sequence", "fitness", "start_pos"}
V6_COLUMNS = {"guide_sequence", "start_pos", "on_target_name", "score_k1_lambda0p4"}
V6_AGGREGATE = "results_all_targets_v6_indel.tsv"
V6_COMPILED = "results_compiled_v6_indel.tsv"
CLUSTER_NAME = re.compile(r"cl[1-9][0-9]*(?:_indel)?$", re.IGNORECASE)


def discover_tables(roots):
    """Resolve overlapping inputs to a unique, deterministic list of source files."""
    tables = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Results directory does not exist: {root}")
        for name in COMPILED_FILES:
            tables.update(path.resolve() for path in root.rglob(name)
                          if path.is_file() and "_previous_runs" not in path.parts)
    # V6 also writes each target below its aggregate. Include one representation.
    tables = {path for path in tables if path.name != V6_COMPILED
              or not any(parent / V6_AGGREGATE in tables for parent in path.parents)}
    if not tables:
        raise ValueError("No compiled ordinary BADGERS or V6 indel result tables found.")
    return sorted(tables)


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def ancestor_metadata(path):
    metadata = []
    for parent in path.parents:
        metadata_path = parent / "run_metadata.json"
        if metadata_path.is_file():
            value = load_json(metadata_path)
            if "status" in value or value.get("algorithm") != "search_guide_v6_indel_cpu":
                if value.get("status") != "complete":
                    raise ValueError(f"Run is incomplete in {metadata_path}: status={value.get('status')!r}. "
                                     "Merge after all requested searches finish successfully.")
            metadata.append((parent, value))
    return metadata


def validate_signature(path):
    signature = load_json(path)
    payload = signature.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed V6 signature: {path}")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if signature.get("digest") != digest:
        raise ValueError(f"V6 signature digest mismatch in {path}")
    return digest


def validate_source(path):
    metadata = ancestor_metadata(path)
    if COMPILED_FILES[path.name] != "v6_indel":
        return
    v6_root = next((root for root, value in metadata
                    if value.get("algorithm") == "search_guide_v6_indel_cpu"), None)
    if v6_root is not None:
        digest = validate_signature(v6_root / "run_signature.json")
        details = next(value for root, value in metadata if root == v6_root)
        names = details.get("executed_targets")
        if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
            raise ValueError(f"V6 executed_targets is missing or empty in {v6_root / 'run_metadata.json'}")
        targets = sorted(v6_root.glob("cluster-*/*/results_compiled_v6_indel.tsv"))
        expected = Counter(re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "target" for name in names)
        if Counter(target.parent.name for target in targets) != expected:
            raise ValueError(f"V6 target outputs differ from executed_targets in {v6_root}")
        for target in targets:
            if not (target.parent / ".v6_indel_complete").is_file():
                raise ValueError(f"V6 target completion marker is missing in {target.parent}")
            if validate_signature(target.parent / "run_signature.json") != digest:
                raise ValueError(f"V6 target signature differs from root in {target.parent}")
    elif path.name == V6_COMPILED:
        if not (path.parent / ".v6_indel_complete").is_file():
            raise ValueError(f"V6 target completion marker is missing in {path.parent}")
        validate_signature(path.parent / "run_signature.json")
    else:
        raise ValueError(f"V6 completion metadata is missing above {path}")


def read_columns(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        columns = csv.DictReader(handle, delimiter="\t", strict=True).fieldnames
    if not columns or any(not column.strip() for column in columns):
        raise ValueError(f"Missing or empty column header in {path}")
    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate column headers in {path}")
    required = V6_COLUMNS if COMPILED_FILES[path.name] == "v6_indel" else REQUIRED_COLUMNS
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Missing BADGERS column(s) in {path}: {', '.join(sorted(missing))}")
    return columns


def read_rows(path):
    """Read well-formed tables, accepting a completed V6 run with no finalists."""
    validate_source(path)
    columns = read_columns(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        if reader.fieldnames != columns:
            raise ValueError(f"Input columns changed during merging: {path}")
        count = 0
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Incorrect number of fields at line {reader.line_num} in {path}")
            count += 1
            yield columns, row
        if count == 0 and COMPILED_FILES[path.name] != "v6_indel":
            raise ValueError(f"No result rows in {path}")


def source_details(path):
    metadata = ancestor_metadata(path)
    workflow = next(((root, value) for root, value in metadata if "status" in value), None)
    cluster = next((parent for parent in path.parents if CLUSTER_NAME.fullmatch(parent.name)), None)
    v6_root = next((root for root, value in metadata
                    if value.get("algorithm") == "search_guide_v6_indel_cpu"), None)
    v6_cluster = next((parent.name[len("cluster-"):] for parent in path.parents
                       if parent.name.startswith("cluster-")), "")
    details = workflow[1] if workflow else {}
    return {
        "source_file": str(path),
        "source_run": details.get("run_name", v6_root.name if v6_root else
                                  cluster.parent.name if cluster else path.parent.name),
        "cluster": details.get("cluster", cluster.name if cluster else v6_cluster),
        "explorer": COMPILED_FILES[path.name],
        "engine": "v6_indel" if COMPILED_FILES[path.name] == "v6_indel" else "ordinary",
    }


def merge(roots, out):
    """Validate all inputs, then atomically publish a new combined TSV."""
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output already exists: {out}. Choose a new output file.")
    tables = discover_tables(roots)
    columns = []
    counts = {}
    schemas = {}
    for path in tables:
        source_columns = read_columns(path)
        schemas[path] = source_columns
        columns.extend(column for column in source_columns if column not in columns)
        count = 0
        for _ in read_rows(path):
            count += 1
        counts[path] = count

    metadata_columns = {}
    reserved = set(columns)
    for field in ("source_file", "source_run", "cluster", "explorer", "engine"):
        base = f"workflow_{field}"
        candidate = base
        suffix = 2
        while candidate in reserved:
            candidate = f"{base}_{suffix}"
            suffix += 1
        metadata_columns[field] = candidate
        reserved.add(candidate)

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", prefix=".merge-results-",
                                suffix=".tmp", dir=out.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns + list(metadata_columns.values()),
                                    delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for path in tables:
                details = source_details(path)
                if read_columns(path) != schemas[path]:
                    raise ValueError(f"Input columns changed during merging: {path}. Retry after jobs finish.")
                written = 0
                for source_columns, row in read_rows(path):
                    if any(column not in columns for column in source_columns):
                        raise ValueError(f"Input columns changed during merging: {path}. Retry after jobs finish.")
                    provenance = dict(details)
                    if provenance["engine"] == "v6_indel" and not provenance["cluster"]:
                        provenance["cluster"] = row.get("cluster", "")
                    row.update({metadata_columns[field]: value for field, value in provenance.items()})
                    writer.writerow(row)
                    written += 1
                if written != counts[path]:
                    raise ValueError(f"Input row count changed during merging: {path}. Retry after jobs finish.")
            handle.flush()
            os.fsync(handle.fileno())
        # Linking publishes the complete file and fails if another writer created out.
        os.link(str(temporary), str(out))
    finally:
        if temporary is not None:
            temporary.unlink()
    return len(tables), sum(counts.values()), metadata_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dirs", type=Path, nargs="*", help="one or more result directories to search")
    parser.add_argument("--out", type=Path, help="new merged .tsv output file")
    parser.add_argument("--interactive", action="store_true", help="prompt for input and output paths")
    args = parser.parse_args()
    try:
        if args.interactive:
            if not args.results_dirs:
                print("Enter one result directory per prompt. Submit an empty line when finished.")
                while True:
                    value = input("Results directory: ").strip()
                    if not value:
                        break
                    args.results_dirs.append(Path(value).expanduser())
            if args.out is None:
                value = input("New merged TSV path: ").strip()
                if value:
                    args.out = Path(value).expanduser()
        if not args.results_dirs or args.out is None:
            parser.error("provide at least one results directory and --out, or use --interactive")
        table_count, row_count, metadata = merge(args.results_dirs, args.out)
    except (OSError, ValueError, csv.Error, EOFError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Merged {row_count} rows from {table_count} compiled tables into {args.out}.")
    print("Provenance columns: " + ", ".join(metadata.values()))


if __name__ == "__main__":
    main()
