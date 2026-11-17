#!/usr/bin/env python3
"""Combine compiled BADGERS result tables while recording each row's source."""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile


COMPILED_FILES = {
    "results_compiled_evolutionary.tsv": "evolutionary",
    "results_compiled_wgan_am.tsv": "wgan_am",
}
REQUIRED_COLUMNS = {"guide_sequence", "fitness", "start_pos"}
CLUSTER_NAME = re.compile(r"cl[1-9][0-9]*$", re.IGNORECASE)


def discover_tables(roots):
    """Resolve overlapping inputs to a unique, deterministic list of source files."""
    tables = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Results directory does not exist: {root}")
        for name in COMPILED_FILES:
            tables.update(path.resolve() for path in root.rglob(name) if path.is_file())
    if not tables:
        raise ValueError("No results_compiled_evolutionary.tsv or results_compiled_wgan_am.tsv files found.")
    return sorted(tables)


def read_rows(path):
    """Read only well-formed compiled tables, preserving original field values."""
    metadata_path = path.parent / "run_metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        status = metadata.get("status") if isinstance(metadata, dict) else None
        if status != "complete":
            raise ValueError(f"Run is incomplete in {metadata_path}: status={status!r}. "
                             "Merge after all requested algorithms finish successfully.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        columns = reader.fieldnames
        if not columns or any(not column.strip() for column in columns):
            raise ValueError(f"Missing or empty column header in {path}")
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate column headers in {path}")
        missing = REQUIRED_COLUMNS - set(columns)
        if missing:
            raise ValueError(f"Missing BADGERS column(s) in {path}: {', '.join(sorted(missing))}")
        count = 0
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Incorrect number of fields at line {reader.line_num} in {path}")
            count += 1
            yield columns, row
        if count == 0:
            raise ValueError(f"No result rows in {path}")


def source_details(path):
    cluster = next((parent for parent in path.parents if CLUSTER_NAME.fullmatch(parent.name)), None)
    return {
        "source_file": str(path),
        "source_run": cluster.parent.name if cluster else path.parent.name,
        "cluster": cluster.name if cluster else "",
        "explorer": COMPILED_FILES[path.name],
    }


def merge(roots, out):
    """Validate all inputs, then atomically publish a new combined TSV."""
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output already exists: {out}. Choose a new output file.")
    tables = discover_tables(roots)
    columns = []
    counts = {}
    for path in tables:
        count = 0
        for source_columns, _ in read_rows(path):
            if count == 0:
                columns.extend(column for column in source_columns if column not in columns)
            count += 1
        counts[path] = count

    metadata_columns = {}
    reserved = set(columns)
    for field in ("source_file", "source_run", "cluster", "explorer"):
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
                written = 0
                for source_columns, row in read_rows(path):
                    if any(column not in columns for column in source_columns):
                        raise ValueError(f"Input columns changed during merging: {path}. Retry after jobs finish.")
                    row.update({metadata_columns[field]: value for field, value in details.items()})
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
