#!/usr/bin/env python3
"""Run prepared comparisons through ordinary BADGERS or the bundled V6 indel search."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_V6_SCRIPT = Path(__file__).resolve().parent / "engines" / "v6_indel" / "search_guide_v6_indel.py"


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
    bundle = bundle.resolve()
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
            if task_id < 0 or task_id in ids or cluster in clusters or not re.fullmatch(r"cl[1-9][0-9]*(?:_indel)?", cluster):
                raise ValueError("jobs.tsv needs unique nonnegative task IDs and unique clN cluster names.")
            engine = row.get("engine") or "ordinary"
            if engine not in {"ordinary", "v6_indel"}:
                raise ValueError(f"Unknown engine for {cluster}: {engine!r}.")
            inputs = (bundle / row["input_dir"]).resolve()
            if bundle not in inputs.parents or not inputs.is_dir():
                raise ValueError(f"Invalid input directory for {cluster}.")
            job = {"task_id": task_id, "cluster": cluster, "inputs": inputs, "engine": engine}
            if engine == "ordinary":
                range_file = bundle / "clusters" / cluster / "range.tsv"
                if not range_file.is_file():
                    raise ValueError(f"Missing range file for {cluster}.")
                job["range"] = range_file
            else:
                if not row.get("input_table") or not row.get("wt_name"):
                    raise ValueError(f"V6 job {cluster} needs input_table and wt_name.")
                input_table = (bundle / row["input_table"]).resolve()
                if bundle not in input_table.parents or not input_table.is_file():
                    raise ValueError(f"Invalid V6 input table for {cluster}.")
                job.update({"input_table": input_table, "wt_name": row["wt_name"]})
            jobs.append(job)
            ids.add(task_id)
            clusters.add(cluster)
    return jobs


def select_jobs(jobs, engine="auto", task_id=None):
    if task_id is not None:
        jobs = [job for job in jobs if job["task_id"] == task_id]
        if not jobs:
            raise ValueError(f"Task {task_id} is absent from jobs.tsv.")
    if engine != "auto":
        jobs = [job for job in jobs if job["engine"] == engine]
    if not jobs:
        raise ValueError("No runnable clusters match the requested task and engine. Review jobs.tsv.")
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


def read_indel_signature(path):
    with path.open(encoding="utf-8") as handle:
        signature = json.load(handle)
    if not isinstance(signature, dict) or not isinstance(signature.get("payload"), dict):
        raise ValueError(f"V6 run signature is invalid in {path}.")
    canonical = json.dumps(signature["payload"], sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if signature.get("digest") != digest:
        raise ValueError(f"V6 signature digest mismatch in {path}.")
    return digest


def check_indel_results(directory):
    """Accept completed V6 searches, including searches with zero cutoff-passing guides."""
    with (directory / "run_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"V6 completion metadata is invalid in {directory}.")
    executed = metadata.get("executed_targets")
    if metadata.get("algorithm") != "search_guide_v6_indel_cpu" or not isinstance(executed, list) or not executed:
        raise ValueError(f"V6 completion metadata is incomplete in {directory}.")
    if any(not isinstance(name, str) or not name for name in executed) or len(executed) != len(set(executed)):
        raise ValueError(f"V6 executed-target metadata is invalid in {directory}.")
    digest = read_indel_signature(directory / "run_signature.json")
    required = {"on_target_name", "guide_sequence", "start_pos", "score_k1_lambda0p4"}
    with (directory / "results_all_targets_v6_indel.tsv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"V6 aggregate result headers are incomplete in {directory}.")
        for row in reader:
            if None in row or any(value is None for value in row.values()) or row["on_target_name"] not in executed:
                raise ValueError(f"V6 aggregate result row is invalid in {directory}.")
    markers = list(directory.glob("cluster-*/*/.v6_indel_complete"))
    if len(markers) != len(executed):
        raise ValueError(f"V6 target completion markers are incomplete in {directory}.")
    target_ids = {re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "target" for name in executed}
    if {marker.parent.name for marker in markers} != target_ids:
        raise ValueError(f"V6 target markers do not match executed targets in {directory}.")
    for marker in markers:
        if read_indel_signature(marker.parent / "run_signature.json") != digest:
            raise ValueError(f"V6 target signature differs from the run signature in {marker.parent}.")
        if not (marker.parent / "results_compiled_v6_indel.tsv").is_file():
            raise ValueError(f"Missing V6 target compiled results in {marker.parent}.")


def resolve_interpreter(value):
    interpreter = shutil.which(os.path.expanduser(value or sys.executable))
    if interpreter is None:
        raise ValueError(f"Python interpreter not found: {value}")
    return interpreter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path)
    parser.add_argument("--design-script", type=Path, help="path to installed BADGERS design_guides.py")
    parser.add_argument("--python", dest="python_bin", help="BADGERS Python interpreter (default current Python)")
    parser.add_argument("--v6-script", type=Path, help="override the bundled V6 indel script")
    parser.add_argument("--indel-python", help="V6 Python interpreter (default --python or current Python)")
    parser.add_argument("--engine", choices=("auto", "ordinary", "v6_indel"), help="manifest engine selection (default auto)")
    parser.add_argument("--indel-cutoff", type=float, help="V6 on-target activity cutoff (default -1.6)")
    parser.add_argument("--indel-extra-arg", action="append", default=[], help="repeat for a custom V6 argument")
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
            if args.engine is None:
                args.engine = prompt("Engine selection (auto, ordinary, v6_indel)", "auto")
        if args.bundle is None:
            parser.error("bundle is required, or use --interactive")
        args.bundle = args.bundle.expanduser().resolve()
        args.engine = args.engine or "auto"
        if args.engine not in {"auto", "ordinary", "v6_indel"}:
            raise ValueError("Choose auto, ordinary, or v6_indel.")
        jobs = select_jobs(load_jobs(args.bundle), args.engine, args.task_id)
        engines = {job["engine"] for job in jobs}
        if args.interactive:
            if "ordinary" in engines and args.design_script is None:
                args.design_script = Path(prompt("Path to BADGERS design_guides.py"))
            if "ordinary" in engines and args.python_bin is None:
                args.python_bin = prompt("BADGERS Python interpreter", sys.executable)
            if "v6_indel" in engines and args.indel_python is None:
                args.indel_python = prompt("V6 Python interpreter", args.python_bin or sys.executable)
            if "v6_indel" in engines and args.indel_cutoff is None:
                args.indel_cutoff = float(prompt("V6 on-target activity cutoff", "-1.6"))
            if args.results is None:
                args.results = Path(prompt("Results root", str(args.bundle / "results")))
            if args.run_name is None:
                args.run_name = prompt("Run name", "ordinary_01")
            if "ordinary" in engines and args.algorithm is None:
                args.algorithm = prompt("Algorithm (evolutionary, wgan_am, both)", "evolutionary")
            if args.cpus is None:
                args.cpus = positive_int(prompt("CPUs per cluster", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
            if args.top_guides is None:
                args.top_guides = positive_int(prompt("Top guides per focal sequence", "20"))
        scripts = {}
        interpreters = {}
        if "ordinary" in engines:
            if args.design_script is None:
                parser.error("--design-script is required for ordinary jobs, or use --interactive")
            scripts["ordinary"] = args.design_script.expanduser().resolve()
            interpreters["ordinary"] = resolve_interpreter(args.python_bin)
        if "v6_indel" in engines:
            scripts["v6_indel"] = (args.v6_script or DEFAULT_V6_SCRIPT).expanduser().resolve()
            interpreters["v6_indel"] = resolve_interpreter(args.indel_python or args.python_bin)
        for engine, script in scripts.items():
            if not script.is_file():
                raise ValueError(f"{engine} script does not exist: {script}")
        args.indel_cutoff = args.indel_cutoff if args.indel_cutoff is not None else -1.6
        if not math.isfinite(args.indel_cutoff):
            raise ValueError("--indel-cutoff must be finite.")
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
        results = (args.results or args.bundle / "results").expanduser().resolve() / args.run_name
        environment = dict(os.environ)
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
            environment[variable] = "1"
        for job in jobs:
            output = results / job["cluster"]
            if output.exists():
                raise ValueError(f"Results already exist: {output}. Use another --run-name.")
        for job in jobs:
            output = results / job["cluster"]
            engine = job["engine"]
            if engine == "ordinary":
                command = [interpreters[engine], str(scripts[engine]), "diff", args.algorithm,
                           str(job["inputs"]) + os.sep, str(output) + os.sep,
                           "--use_range", str(job["range"]), "--num_cpu", str(args.cpus),
                           "--n_top_guides", str(args.top_guides)] + args.extra_arg
            else:
                command = [interpreters[engine], str(scripts[engine]), "-f", str(job["input_table"]),
                           "--output_dir", str(output / "engine_results"), "--wt_name", job["wt_name"],
                           "--num_cpu", str(args.cpus), "--n_top_guides", str(args.top_guides),
                           "--cut_off", str(args.indel_cutoff)] + args.indel_extra_arg
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            output.mkdir(parents=True, exist_ok=False)
            metadata = {"status": "running", "run_name": args.run_name, "cluster": job["cluster"], "engine": engine,
                        "task_id": job["task_id"], "command": command,
                        "started_utc": datetime.now(timezone.utc).isoformat()}
            metadata_path = output / "run_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            try:
                with (output / "badgers.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=scripts[engine].parent, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                if engine == "ordinary":
                    check_compiled_results(output, args.algorithm)
                else:
                    check_indel_results(output / "engine_results")
                metadata["status"] = "complete"
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                metadata["status"] = "failed"
                metadata["error"] = str(error)
                raise ValueError(f"{job['cluster']} failed. See {output / 'badgers.log'} ({error}).") from error
            finally:
                metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(f"Completed {job['cluster']}. Results in {output}", flush=True)
    except (OSError, ValueError, EOFError, csv.Error, argparse.ArgumentTypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
