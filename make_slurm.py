#!/usr/bin/env python3
"""Optionally generate a portable Slurm array that calls run_badgers.py."""

import argparse
import math
import re
import shlex
import sys
from pathlib import Path

from run_badgers import load_jobs, positive_int, prompt, select_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=Path, help="prepared bundle path on the cluster")
    parser.add_argument("--out", type=Path, help="new Slurm script path")
    parser.add_argument("--account", help="Slurm allocation account")
    parser.add_argument("--partition", help="Slurm partition/queue")
    parser.add_argument("--repo", type=Path, help="repository path on the cluster")
    parser.add_argument("--design-script", type=Path, help="installed BADGERS script path on the cluster")
    parser.add_argument("--v6-script", type=Path, help="override the bundled V6 script on the cluster")
    parser.add_argument("--engine", choices=("auto", "ordinary", "v6_indel"), default="auto")
    parser.add_argument("--workflow-python", help="Python for this repository's scripts")
    parser.add_argument("--badgers-python", help="Python for the BADGERS environment")
    parser.add_argument("--indel-python", help="Python for the V6 environment (default BADGERS/workflow Python)")
    parser.add_argument("--indel-cutoff", type=float, default=-1.6)
    parser.add_argument("--indel-extra-arg", action="append", default=[], help="repeat to forward a custom V6 argument")
    parser.add_argument("--results", type=Path, help="results root on cluster (default bundle/results)")
    parser.add_argument("--run-name", help="results subdirectory")
    parser.add_argument("--algorithm", choices=("evolutionary", "wgan_am", "both"), default="evolutionary")
    parser.add_argument("--cpus", type=positive_int, default=8)
    parser.add_argument("--top-guides", type=positive_int, default=20)
    parser.add_argument("--extra-arg", action="append", default=[], help="repeat to forward a custom BADGERS argument")
    parser.add_argument("--memory", default="32G")
    parser.add_argument("--time", default="36:00:00")
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--interactive", action="store_true", help="prompt for cluster and run settings")
    args = parser.parse_args()
    try:
        if args.interactive:
            for attribute, label, default in (
                ("bundle", "Prepared bundle path on cluster", None),
                ("out", "Slurm script output", "run_badgers.slurm"),
                ("account", "Slurm allocation account", None),
                ("partition", "Slurm partition/queue", "cpu"),
                ("repo", "Repository path on cluster", str(Path(__file__).resolve().parent)),
                ("workflow_python", "Workflow Python interpreter", sys.executable),
                ("run_name", "Run name", "mixed_01"),
            ):
                if getattr(args, attribute) is None:
                    value = prompt(label, default)
                    setattr(args, attribute, Path(value) if attribute in {"bundle", "out", "repo"} else value)
            args.engine = prompt("Engine selection (auto, ordinary, v6_indel)", args.engine)
        required = ("bundle", "out", "account", "partition", "run_name")
        if any(getattr(args, name) is None for name in required):
            parser.error("bundle, --out, --account, --partition and --run-name are required, or use --interactive")
        if args.engine not in {"auto", "ordinary", "v6_indel"}:
            raise ValueError("Choose auto, ordinary, or v6_indel.")
        args.bundle = args.bundle.expanduser().resolve()
        jobs = select_jobs(load_jobs(args.bundle), args.engine)
        engines = {job["engine"] for job in jobs}
        if args.interactive:
            if "ordinary" in engines and args.design_script is None:
                args.design_script = Path(prompt("Installed BADGERS design_guides.py path"))
            if "ordinary" in engines and args.badgers_python is None:
                args.badgers_python = prompt("BADGERS Python interpreter", args.workflow_python or sys.executable)
            if "v6_indel" in engines:
                if args.indel_python is None:
                    args.indel_python = prompt("V6 Python interpreter", args.badgers_python or args.workflow_python or sys.executable)
                args.indel_cutoff = float(prompt("V6 on-target activity cutoff", str(args.indel_cutoff)))
            for attribute, label in (("cpus", "CPUs per task"), ("memory", "Memory per task"),
                                     ("time", "Wall time"), ("concurrency", "Concurrent tasks"),
                                     ("algorithm", "Algorithm (evolutionary, wgan_am, both)"),
                                     ("top_guides", "Top guides per focal sequence")):
                value = prompt(label, str(getattr(args, attribute)))
                setattr(args, attribute, positive_int(value) if attribute in {"cpus", "concurrency", "top_guides"} else value)
        if "ordinary" in engines and args.design_script is None:
            parser.error("--design-script is required for ordinary jobs, or use --interactive")
        args.repo = (args.repo or Path(__file__).resolve().parent).expanduser().resolve()
        if args.design_script is not None:
            args.design_script = args.design_script.expanduser().resolve()
        if args.v6_script is not None:
            args.v6_script = args.v6_script.expanduser().resolve()
        args.workflow_python = args.workflow_python or sys.executable
        args.badgers_python = args.badgers_python or args.workflow_python
        args.indel_python = args.indel_python or args.badgers_python
        if not math.isfinite(args.indel_cutoff):
            raise ValueError("--indel-cutoff must be finite.")
        for label, value in (("account", args.account), ("partition", args.partition),
                             ("memory", args.memory), ("time", args.time)):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:+-]*", value):
                raise ValueError(f"Invalid {label} value.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
            raise ValueError("Invalid run name.")
        if args.algorithm not in {"evolutionary", "wgan_am", "both"}:
            raise ValueError("Choose evolutionary, wgan_am, or both.")
        indices = ",".join(str(job["task_id"]) for job in jobs)
        command = [args.workflow_python, str(args.repo / "run_badgers.py"), str(args.bundle),
                   "--python", args.badgers_python, "--indel-python", args.indel_python,
                   "--engine", args.engine, "--indel-cutoff", str(args.indel_cutoff),
                   "--results", str((args.results or args.bundle / "results").expanduser().resolve()),
                   "--run-name", args.run_name, "--algorithm", args.algorithm, "--cpus", str(args.cpus),
                   "--top-guides", str(args.top_guides)]
        if args.design_script is not None:
            command.extend(("--design-script", str(args.design_script)))
        if args.v6_script is not None:
            command.extend(("--v6-script", str(args.v6_script)))
        command.extend("--extra-arg=" + value for value in args.extra_arg)
        command.extend("--indel-extra-arg=" + value for value in args.indel_extra_arg)
        text = f"""#!/bin/bash
#SBATCH --job-name=badgers
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.memory}
#SBATCH --time={args.time}
#SBATCH --array={indices}%{args.concurrency}
#SBATCH --output=badgers-%A_%a.out
#SBATCH --error=badgers-%A_%a.err

set -euo pipefail
{shlex.join(command)} --task-id "${{SLURM_ARRAY_TASK_ID:?}}"
"""
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(f"Wrote {args.out}. Submit on the cluster with sbatch {shlex.quote(str(args.out))}")
    except (OSError, ValueError, EOFError, argparse.ArgumentTypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
