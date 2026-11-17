"""Regression checks for ordinary and V6 routing using temporary synthetic inputs."""

import csv
import hashlib
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
V6_SCRIPT = REPO / "engines" / "v6_indel" / "search_guide_v6_indel.py"


def write_table(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def read_table(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class MixedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="badgers-workflow-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "fastas"
        self.inputs.mkdir()
        self.bundle = self.root / "prepared"
        rng = random.Random(18634)
        reference = "".join(rng.choice("ACGT") for _ in range(96))
        change = {"A": "C", "C": "G", "G": "T", "T": "A"}
        snp1 = reference[:30] + change[reference[30]] + reference[31:]
        snp2 = reference[:70] + change[reference[70]] + reference[71:]
        records = {"gene0_WT": reference[:48] + "-" + reference[48:],
                   "snp1_cl1": snp1[:48] + "-" + snp1[48:],
                   "snp2_cl2": snp2[:48] + "-" + snp2[48:],
                   "insert1_cl1": reference[:48] + "C" + reference[48:]}
        for name, sequence in records.items():
            (self.inputs / f"{name}.fa").write_text(f">{name}\n{sequence}\n", encoding="utf-8")
        self.prepare()

    def run_cli(self, script, *arguments, succeeds=True):
        environment = dict(os.environ)
        environment.pop("SLURM_CPUS_PER_TASK", None)
        result = subprocess.run([sys.executable, str(script), *map(str, arguments)],
                                capture_output=True, text=True, env=environment)
        self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
        return result

    def prepare(self, *extra, succeeds=True):
        return self.run_cli(REPO / "prepare_fasta.py", self.inputs,
                            "--clusters", "filename", "--alignment", "prealigned",
                            "--out", self.bundle, *extra, succeeds=succeeds)

    def test_preparation_preserves_ordinary_members_and_adds_indel_branch(self):
        jobs = read_table(self.bundle / "jobs.tsv")
        self.assertEqual([(row["task_id"], row["cluster"], row["engine"]) for row in jobs],
                         [("0", "cl1", "ordinary"), ("1", "cl2", "ordinary"),
                          ("2", "cl1_indel", "v6_indel")])
        ordinary = self.bundle / "clusters" / "cl1" / "standard"
        self.assertEqual({path.stem for path in ordinary.iterdir()}, {"gene0_WT_cl1", "snp1_cl1"})
        table = read_table(self.bundle / "clusters" / "cl1_indel" / "input.tsv")
        self.assertEqual({row["name"]: row["indel"] for row in table},
                         {"gene0_WT_cl1_indel": "no", "insert1_cl1": "yes", "snp1_cl1": "no"})
        self.assertEqual(len({len(row["seq"]) for row in table}), 1)

    def test_additive_update_preserves_legacy_ids_and_rejects_conflicts(self):
        jobs = read_table(self.bundle / "jobs.tsv")[:2]
        shutil.rmtree(self.bundle / "clusters" / "cl1_indel")
        columns = ("task_id", "cluster", "input_dir", "member_count")
        write_table(self.bundle / "jobs.tsv", columns, ([row[column] for column in columns] for row in jobs))
        previous = {path: path.read_bytes() for path in self.bundle.rglob("*")
                    if path.is_file() and path.name != "jobs.tsv"}
        self.prepare("--add-indel-clusters")
        for path, content in previous.items():
            self.assertEqual(path.read_bytes(), content)
        updated = (self.bundle / "jobs.tsv").read_bytes()
        self.assertEqual([row["task_id"] for row in read_table(self.bundle / "jobs.tsv")], ["0", "1", "2"])
        self.prepare("--add-indel-clusters")
        self.assertEqual((self.bundle / "jobs.tsv").read_bytes(), updated)
        self.prepare(succeeds=False)
        conflict = self.bundle / "clusters" / "cl1" / "standard" / "snp1_cl1.fasta"
        conflict.write_text(conflict.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        result = self.prepare("--add-indel-clusters", succeeds=False)
        self.assertIn("differs", result.stderr)
        self.assertEqual((self.bundle / "jobs.tsv").read_bytes(), updated)

    def test_vendored_v6_plans_wt_and_indel_with_snp_background(self):
        output = self.root / "v6_plan"
        self.run_cli(V6_SCRIPT, "-f", self.bundle / "clusters" / "cl1_indel" / "input.tsv",
                     "--wt_name", "gene0_WT_cl1_indel", "--output_dir", output, "--plan_only")
        selected = read_table(output / "on_target_selection.tsv")
        self.assertEqual({row["name"] for row in selected if row["selected_as_on_target"] == "True"},
                         {"gene0_WT_cl1_indel", "insert1_cl1"})
        self.assertTrue(all(row["retained_for_off_target_comparison"] == "True" for row in selected))
        self.assertEqual({row["on_target_name"] for row in read_table(output / "junction_window_plan.tsv")},
                         {"gene0_WT_cl1_indel", "insert1_cl1"})

    def test_runner_dry_run_routes_both_engines_and_indel_only(self):
        ordinary_script = self.root / "design_guides.py"
        ordinary_script.write_text("raise RuntimeError('A dry run must not execute this fixture')\n", encoding="utf-8")
        result = self.run_cli(REPO / "run_badgers.py", self.bundle,
                              "--design-script", ordinary_script, "--run-name", "preview", "--dry-run")
        commands = [shlex.split(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(commands), 3)
        self.assertEqual(sum(command[1] == str(ordinary_script.resolve()) for command in commands), 2)
        self.assertEqual(sum(command[1] == str(V6_SCRIPT) for command in commands), 1)
        result = self.run_cli(REPO / "run_badgers.py", self.bundle, "--engine", "v6_indel",
                              "--run-name", "indel_preview", "--dry-run")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertIn(str(V6_SCRIPT), result.stdout)
        self.assertFalse((self.bundle / "results").exists())

    def test_merge_includes_v6_aggregate_once_and_preserves_engine_scores(self):
        results = self.root / "results" / "mixed_01"
        ordinary = results / "cl2"
        ordinary.mkdir(parents=True)
        (ordinary / "run_metadata.json").write_text(json.dumps({"status": "complete", "run_name": "mixed_01", "cluster": "cl2", "engine": "ordinary"}))
        write_table(ordinary / "results_compiled_evolutionary.tsv",
                    ("guide_sequence", "fitness", "start_pos", "on_target_name"),
                    [("A" * 28, "0.2", "11", "snp2_cl2")])
        workflow = results / "cl1_indel"
        engine = workflow / "engine_results"
        target = engine / "cluster-cl1_indel" / "insert1_cl1"
        target.mkdir(parents=True)
        (workflow / "run_metadata.json").write_text(json.dumps({"status": "complete", "run_name": "mixed_01", "cluster": "cl1_indel", "engine": "v6_indel"}))
        (engine / "run_metadata.json").write_text(json.dumps({"algorithm": "search_guide_v6_indel_cpu", "executed_targets": ["insert1_cl1"]}))
        payload = {"fixture": "workflow"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        signature = json.dumps({"digest": digest, "payload": payload})
        (engine / "run_signature.json").write_text(signature)
        (target / "run_signature.json").write_text(signature)
        (target / ".v6_indel_complete").touch()
        columns = ("guide_sequence", "start_pos", "on_target_name", "score_k1_lambda0p4", "cluster")
        data = [("C" * 28, "33", "insert1_cl1", "0.8", "cl1_indel")]
        write_table(engine / "results_all_targets_v6_indel.tsv", columns, data)
        write_table(target / "results_compiled_v6_indel.tsv", columns, data)
        output = self.root / "merged.tsv"
        self.run_cli(REPO / "merge_results.py", results, "--out", output)
        merged = read_table(output)
        self.assertEqual(len(merged), 2)
        by_engine = {row["workflow_engine"]: row for row in merged}
        self.assertEqual(by_engine["ordinary"]["fitness"], "0.2")
        self.assertEqual(by_engine["v6_indel"]["fitness"], "")
        self.assertEqual(by_engine["v6_indel"]["score_k1_lambda0p4"], "0.8")
        self.assertEqual(by_engine["v6_indel"]["workflow_cluster"], "cl1_indel")
        self.run_cli(REPO / "merge_results.py", results, "--out", output, succeeds=False)
        (target / ".v6_indel_complete").unlink()
        self.run_cli(REPO / "merge_results.py", results, "--out", self.root / "incomplete.tsv", succeeds=False)


if __name__ == "__main__":
    unittest.main()
