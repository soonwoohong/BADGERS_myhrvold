#!/usr/bin/env python3
"""Run prepared cluster comparisons using an existing BADGERS installation."""

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def prompt(label, default=None):
    answer = input(f"{label}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    value = answer or default
    if value is None:
        raise ValueError(f"{label} is required.")
    return value


def load_jobs(bundle):
    """Read both earlier preparation bundles and this repository's manifest."""
    jobs = []
    ids = set()
    clusters = set()
    with (bundle / "jobs.tsv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"task_id", "cluster", "input_dir"}.issubset(reader.fieldnames or []):
            raise ValueError("jobs.tsv needs task_id, cluster, and input_dir columns.")
        for row in reader:
            if None in row or any(not row.get(column) for column in ("task_id", "cluster", "input_dir")):
                raise ValueError(f"Malformed jobs.tsv row at line {reader.line_num}.")
            task_id = int(row["task_id"])
            cluster = row["cluster"]
            if task_id < 0 or task_id in ids or cluster in clusters or not re.fullmatch(r"cl[1-9][0-9]*", cluster):
                raise ValueError("jobs.tsv needs unique nonnegative task IDs and unique clN cluster names.")
            inputs = (bundle / row["input_dir"]).resolve()
            if bundle not in inputs.parents or not inputs.is_dir():
                raise ValueError(f"Invalid input directory for {cluster}.")
            range_file = bundle / "clusters" / cluster / "range.tsv"
            if not range_file.is_file():
                raise ValueError(f"Missing range file for {cluster}.")
            jobs.append({"task_id": task_id, "cluster": cluster, "inputs": inputs, "range": range_file})
            ids.add(task_id)
            clusters.add(cluster)
    return jobs


def check_compiled_results(directory, algorithm):
    explorers = ("evolutionary", "wgan_am") if algorithm == "both" else (algorithm,)
    for explorer in explorers:
        path = directory / f"results_compiled_{explorer}.tsv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"guide_sequence", "fitness", "start_pos"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Incomplete compiled results in {path}.")
            if next(reader, None) is None:
                raise ValueError(f"No result rows in {path}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path)
    parser.add_argument("--design-script", type=Path, help="path to installed BADGERS design_guides.py")
    parser.add_argument("--python", dest="python_bin", help="BADGERS Python interpreter (default current Python)")
    parser.add_argument("--results", type=Path, help="results root (default bundle/results)")
    parser.add_argument("--run-name", help="results subdirectory, e.g. ordinary_01")
    parser.add_argument("--algorithm", choices=("evolutionary", "wgan_am", "both"))
    parser.add_argument("--cpus", type=positive_int)
    parser.add_argument("--top-guides", type=positive_int)
    parser.add_argument("--task-id", type=int, help="run one jobs.tsv task ID (default all tasks sequentially)")
    parser.add_argument("--extra-arg", action="append", default=[], help="repeat for custom BADGERS arguments, e.g. --extra-arg=--verbose_results")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing or creating results")
    parser.add_argument("--interactive", action="store_true", help="prompt for omitted run settings")
    args = parser.parse_args()
    try:
        if args.interactive:
            if args.bundle is None:
                args.bundle = Path(prompt("Prepared bundle directory"))
            if args.design_script is None:
                args.design_script = Path(prompt("Path to BADGERS design_guides.py"))
            if args.python_bin is None:
                args.python_bin = prompt("BADGERS Python interpreter", sys.executable)
            if args.results is None:
                args.results = Path(prompt("Results root", str(args.bundle / "results")))
            if args.run_name is None:
                args.run_name = prompt("Run name", "ordinary_01")
            if args.algorithm is None:
                args.algorithm = prompt("Algorithm (evolutionary, wgan_am, both)", "evolutionary")
            if args.cpus is None:
                args.cpus = positive_int(prompt("CPUs per cluster", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
            if args.top_guides is None:
                args.top_guides = positive_int(prompt("Top guides per focal sequence", "20"))
        if args.bundle is None or args.design_script is None:
            parser.error("bundle and --design-script are required, or use --interactive")
        args.bundle = args.bundle.expanduser().resolve()
        args.design_script = args.design_script.expanduser().resolve()
        if not args.design_script.is_file():
            raise ValueError(f"BADGERS script does not exist: {args.design_script}")
        args.algorithm = args.algorithm or "evolutionary"
        if args.algorithm not in {"evolutionary", "wgan_am", "both"}:
            raise ValueError("Choose evolutionary, wgan_am, or both.")
        args.cpus = args.cpus or positive_int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        if "SLURM_CPUS_PER_TASK" in os.environ and args.cpus > int(os.environ["SLURM_CPUS_PER_TASK"]):
            raise ValueError("--cpus exceeds the CPUs allocated by Slurm.")
        args.top_guides = args.top_guides or 20
        args.run_name = args.run_name or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
            raise ValueError("Run name must start with a letter or number and use letters, numbers, _, -, or .")
        interpreter = shutil.which(os.path.expanduser(args.python_bin or sys.executable))
        if interpreter is None:
            raise ValueError(f"Python interpreter not found: {args.python_bin}")
        results = (args.results or args.bundle / "results").expanduser().resolve() / args.run_name
        jobs = load_jobs(args.bundle)
        if args.task_id is not None:
            jobs = [job for job in jobs if job["task_id"] == args.task_id]
            if not jobs:
                raise ValueError(f"Task {args.task_id} is absent from jobs.tsv.")
        if not jobs:
            raise ValueError("jobs.tsv has no runnable clusters. Review variant_summary.tsv and indels.tsv.")
        environment = dict(os.environ)
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
            environment[variable] = "1"
        for job in jobs:
            output = results / job["cluster"]
            if output.exists():
                raise ValueError(f"Results already exist: {output}. Use another --run-name.")
        for job in jobs:
            output = results / job["cluster"]
            command = [interpreter, str(args.design_script), "diff", args.algorithm,
                       str(job["inputs"]) + os.sep, str(output) + os.sep,
                       "--use_range", str(job["range"]), "--num_cpu", str(args.cpus),
                       "--n_top_guides", str(args.top_guides)] + args.extra_arg
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            output.mkdir(parents=True, exist_ok=False)
            metadata = {"status": "running", "run_name": args.run_name, "cluster": job["cluster"],
                        "task_id": job["task_id"], "command": command,
                        "started_utc": datetime.now(timezone.utc).isoformat()}
            metadata_path = output / "run_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            try:
                with (output / "badgers.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=args.design_script.parent, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                check_compiled_results(output, args.algorithm)
                metadata["status"] = "complete"
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                metadata["status"] = "failed"
                metadata["error"] = str(error)
                raise ValueError(f"{job['cluster']} failed. See {output / 'badgers.log'} ({error}).") from error
            finally:
                metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(f"Completed {job['cluster']}. Results in {output}", flush=True)
    except (OSError, ValueError, EOFError, argparse.ArgumentTypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
