# BADGERS_myhrvold

Prepare aligned FASTA comparisons, run ordinary BADGERS and the lab's customized V6 indel search, and collect results for the Myhrvold lab. Each stage is a separate Python script and supports `--help` and `--interactive`.

| Stage | Script | Output |
| --- | --- | --- |
| 1. Process FASTAs | `prepare_fasta.py` | Cluster folders, WT copies, variant/indel reports, job manifest |
| 2. Optional Slurm setup | `make_slurm.py` | A batch script with your allocation and resource settings |
| 3. Run BADGERS | `run_badgers.py` | Per-cluster compiled results, logs, and run metadata |
| 4. Merge results | `merge_results.py` | One TSV with original result columns and provenance |

The runner reads each job's engine from `jobs.tsv`. Ordinary jobs use your existing BADGERS installation, with `evolutionary` as the default algorithm and `wgan_am` and `both` also supported. Indel jobs use the included V6.4.1 engine. The FASTA processor and runner also accept bundles from the earlier `prepare_badgers.py` script.

## Install the workflow tools

Download the repository from GitHub's **Code → Download ZIP** menu, or clone it from a terminal.

```bash
git clone https://github.com/soonwoohong/BADGERS_myhrvold.git
cd BADGERS_myhrvold
python -m pip install -r requirements.txt
```

Use Python 3.8 or later. Biopython is needed for FASTA processing. The runner, merger, and Slurm generator use the Python standard library.

BADGERS has its own dependencies and model files. Follow its installation instructions in the environment where you will run the calculations. Supply that environment's Python interpreter with `run_badgers.py --python /path/to/badgers-env/bin/python` if it differs from your workflow Python. [Upstream BADGERS installation](https://github.com/broadinstitute/badgers-cas13)

For V6, supply `--indel-python /path/to/indel-env/bin/python`, using the environment configured for the customized indel search. That environment needs ADAPT with its Cas13a model files and TensorFlow, plus the dependencies listed in `requirements-indel.txt`. The V6 source and companion window helper are included under `engines/v6_indel/`, with their original MIT license. The runner can use separate interpreters for the workflow, ordinary BADGERS, and V6.

## 1. Process FASTAs

Put one DNA record per `.fa`, `.fasta`, or `.fna` file in one folder. Each run covers a single locus in the same orientation and sequence region. Include exactly one reference file ending in `_WT`, such as `gene0_WT.fa`. Existing cluster labels go at the end of variant filenames, such as `gene2_cl1.fa`.

For guided setup, run the following command and answer the prompts.

```bash
python prepare_fasta.py --interactive
```

For command-line use, check your filenames first, then choose a preparation mode.

```bash
# Inspect WT and existing cluster suffixes.
python prepare_fasta.py /path/to/fastas --clusters check

# Preserve existing _clN assignments.
python prepare_fasta.py /path/to/fastas --clusters filename --out prepared/gene

# Assign clusters from mutation positions relative to WT.
python prepare_fasta.py /path/to/fastas --clusters window --window 20 --out prepared/gene

# Add indel folders/jobs to an existing prepared bundle, preserving its ordinary inputs.
python prepare_fasta.py /path/to/fastas --clusters filename \
    --out prepared/gene --add-indel-clusters
```

Global pairwise alignment to WT is the default. For an existing shared alignment, add `--alignment prealigned`. Prealigned records must have the same alignment width. Tied best global alignments are flagged for review.

Window clustering uses an inclusive WT-coordinate span, default 20 nt. Positions 100–119 fit in one window. Variants are visited in coordinate order and join a cluster while the full mutation span fits. A variant spanning more than the window receives its own cluster with `exceeds_window=1`. Identical-to-WT records stay unassigned in window mode. Alignment scoring uses match 2, mismatch −3, gap open −5, and gap extension −1. End differences and repeat-associated indels need reference review.

The prepared bundle contains the following files.

```text
prepared/gene/
  variant_summary.tsv
  indels.tsv
  jobs.tsv
  clusters/
    cl1/
      aligned.fasta
      range.tsv
      standard/
        gene0_WT_cl1.fasta
        gene2_cl1.fasta
    cl1_indel/
      aligned.fasta
      input.tsv
```

Input filename stems appear in output FASTAs, headers, and reports. Extensions are normalized to `.fasta` for BADGERS. WT receives the cluster suffix in each cluster. In window mode, a variant keeps its input stem, and its folder and report row give the new assignment. Each input stem must be unique across extensions. The upstream importer treats lowercase `split` and `.fasta` inside stems specially, so those names require renaming before an ordinary run.

`variant_summary.tsv` contains one WT reference row per cluster and one row per variant. Substitution and deletion positions are 1-based WT coordinates. In the insertion column, `42+2` denotes two bases inserted after WT position 42, with anchor 0 denoting the start of the reference. `indels.tsv` contains records with insertion/deletion events, including compensating indels that preserve net length.

All assigned records remain in `aligned.fasta`. The `standard/` directory includes WT and eligible gap-free substitution records. Ambiguous records and tied gap-free alignments are marked for review. The public `diff` importer uses each FASTA as the focal target and pools all remaining FASTAs, including WT, for comparison. WT also becomes a focal target. [Upstream FASTA importer](https://github.com/broadinstitute/badgers-cas13/blob/main/badgers/utils/import_fasta.py)

Clusters containing indels also receive a sibling `clN_indel` folder. Its `input.tsv` includes the cluster's WT, indels, and other A/C/G/T-compatible comparison records, with explicit `indel` flags and their shared alignment. V6 searches WT and flagged indels, and keeps all table records as comparison backgrounds. For example, `cl1` runs ordinary WT/SNP comparisons while `cl1_indel` runs the customized WT/indel comparisons. The WT name in the latter is `gene0_WT_cl1_indel`. An indel-only cluster can run through V6.

`range.tsv` covers ordinary positions with a complete 48-nt context window. `--window` controls mutation clustering. Change `range.tsv` if your experiment needs a narrower ordinary scan. V6 selects junction-overlapping windows using its own logic. `jobs.tsv` contains one task per engine-specific cluster, with zero-based task IDs and the fields `engine`, `input_table`, and `wt_name`. Ordinary tasks are listed first, followed by V6 tasks. Earlier manifests without an engine column run as ordinary jobs. The additive update preserves existing task IDs and validates existing ordinary inputs before adding indel folders.

## 2. Optional Slurm setup

Slurm allocates compute resources on many university clusters. Copy the repository and prepared bundle to the cluster before generating the script so the generated paths match the compute environment. You need your allocation account, partition/queue, BADGERS script path, and the Python interpreters for the workflow and BADGERS environments.

```bash
python make_slurm.py --interactive
```

The prompts collect allocation and queue names, cluster-side paths, run name, CPU count, memory, wall time, concurrency, algorithm, and number of guides to save. A command-line version is also available.

```bash
python make_slurm.py prepared/gene \
    --out gene.slurm \
    --account your-allocation --partition cpu \
    --design-script /path/to/badgers-cas13/design_guides.py \
    --badgers-python /path/to/badgers-env/bin/python \
    --indel-python /path/to/indel-env/bin/python \
    --run-name mixed_01

# Submit one task first. sbatch prints its job ID.
sbatch --array=0 gene.slurm

# Submit the array defined in gene.slurm.
sbatch gene.slurm
```

The full array includes task 0 again. If task 0 completed successfully under `mixed_01`, use a fresh run name for a full rerun or submit only the remaining IDs shown in `jobs.tsv`, such as `sbatch --array=1-5%1 gene.slurm`. The runner reports an error if that cluster's output directory already exists. Successful output is written to `prepared/gene/results/mixed_01/clN/` and `prepared/gene/results/mixed_01/clN_indel/`.

Defaults are 8 CPUs, 32 GB memory, 36 hours, and one concurrent task. Adjust `--cpus`, `--memory`, `--time`, and `--concurrency` for your cluster. Slurm log files appear in the directory from which you submit. The generated script uses explicit interpreter paths. Ensure the modules or libraries required by your BADGERS installation are available in the batch environment. On Delta, `accounts` lists your allocations and `squeue --me` shows queued/running jobs. [Delta running-jobs guide](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html)

## 3. Run BADGERS directly

On a workstation or allocated compute node, use the runner directly. When `--task-id` is omitted, it processes all manifest jobs sequentially.

```bash
python run_badgers.py --interactive

# Preview one cluster's command.
python run_badgers.py prepared/gene \
    --design-script /path/to/badgers-cas13/design_guides.py \
    --python /path/to/badgers-env/bin/python \
    --run-name ordinary_01 --task-id 0 --cpus 8 --dry-run

# Execute with the same settings by omitting --dry-run.
```

The default `--engine auto` routes every selected manifest task to its listed engine. To preview the whole mixed run, use the following command, then omit `--dry-run` to execute it.

```bash
python run_badgers.py prepared/gene \
    --design-script /path/to/badgers-cas13/design_guides.py \
    --python /path/to/badgers-env/bin/python \
    --indel-python /path/to/indel-env/bin/python \
    --run-name mixed_01 --cpus 8 --indel-cutoff -1.6 --dry-run
```

Use `--engine ordinary` or `--engine v6_indel` to run one branch. An indel-only run uses the included engine by default, and `--v6-script` can select another V6 checkout. `--design-script` is required when the selected jobs include ordinary BADGERS.

`--algorithm both` runs both explorers. `--top-guides 25` sets the number saved per focal sequence. Run names keep separate experiments in separate directories. Custom BADGERS flags can be passed as repeated `--extra-arg` values, with `=` for arguments beginning with `--` or negative numbers.

```bash
python run_badgers.py prepared/gene \
    --design-script /path/to/your/custom/design_guides.py \
    --python /path/to/badgers-env/bin/python \
    --run-name custom_01 --cpus 8 \
    --extra-arg=--verbose_results
```

Your chosen BADGERS version must implement any custom flags you pass. V6 options use the separate repeated `--indel-extra-arg` option, for example `--indel-extra-arg=--max_mismatches --indel-extra-arg=2`. `make_slurm.py` forwards both sets of options, engine selection, `--top-guides`, and the indel cutoff.

Each cluster's output contains `badgers.log` and workflow `run_metadata.json`. V6's outputs and engine-specific metadata are in its `engine_results/` subdirectory. A run is marked complete after the engine exits successfully and its expected results and completion records validate. V6 can complete with an empty, header-bearing aggregate when no candidates pass the cutoff. A failed run retains its log and metadata for diagnosis. To retry it, choose another run name. CPU limits are passed to the appropriate engine.

## 4. Merge compiled results

Once the jobs finish, merge one run or several result directories.

```bash
python merge_results.py --interactive

python merge_results.py prepared/gene/results/ordinary_01 --out results/ordinary_01.tsv

# Merge ordinary and indel results together.
python merge_results.py prepared/gene/results/mixed_01 --out results/mixed_01.tsv

python merge_results.py prepared/geneA/results/ordinary_01 \
    prepared/geneB/results/ordinary_01 --out results/all_loci.tsv
```

The merger recursively finds ordinary compiled tables and V6 aggregate tables, retains all source columns, and adds `workflow_source_file`, `workflow_source_run`, `workflow_cluster`, `workflow_explorer`, and `workflow_engine`. Column-name collisions receive a numbered suffix. Duplicate result rows are retained. Each source file is included once if input directories overlap. A V6 aggregate takes precedence over its nested per-target tables to prevent counting the same results twice.

Workflow runs must have status `complete`. Legacy ordinary results without workflow metadata are accepted. Standalone V6 outputs require their own completion/signature records. The merger validates headers and field counts and writes a new output file atomically. Ordinary `fitness` and V6's `score_k1_lambda0p4` and other score columns retain their own names. Compare scores within matching target sets and objective settings. [BADGERS output documentation](https://github.com/broadinstitute/badgers-cas13#output-format)

## Indel support

The included custom engine is `engines/v6_indel/search_guide_v6_indel.py`, version `6.4.1-deduplicated-contexts-sharded-cpu`, copied with its window helper and original license from the lab's `badgers_search_guide` implementation.

V6 identifies indel events from alignment gaps, edit annotations, or pairwise alignment, then selects junction-overlapping guide windows. It builds contiguous 48-nt contexts independently from each allele's full sequence and uses the ADAPT Cas13a classifier and regressor for activity prediction. Finalists are rescored across every full-length off-target register, retaining the highest predicted activity for each comparison sequence. Identical contexts are evaluated once and mapped back to the original sequences. This handles shifted coordinates and unequal allele lengths through contiguous-register scanning. Predicted activities represent those fixed-length contiguous binding contexts.

WT is included as a focal target by default and belongs to each cluster's comparison set. V6 evaluates exact zero-, one-, and two-mismatch guide stages, followed by beam expansion for three to five mismatches. It reports the `k=1`, lambda `0.4` score, on-target minus maximum off-target, and on-target minus mean of the top three off-target activities. The workflow passes `--indel-cutoff`, default `-1.6`, to V6's `--cut_off` option.

From the `BADGERS_myhrvold` directory, inspect or plan the prepared V6 input using the Python environment configured for the custom workflow.

```bash
python engines/v6_indel/search_guide_v6_indel.py --version
python engines/v6_indel/search_guide_v6_indel.py \
    -f prepared/gene/clusters/cl1_indel/input.tsv \
    --output_dir /path/to/output/indel_plan \
    --cut_off -1.6 --plan_only
```

V6 accepts XLSX, CSV, and TSV input with a name column (`new_name`, `nick_name`, `name`, or `id`) and a sequence column (`new_seq`, `seq`, or `sequence`). Preparation supplies cluster and explicit `indel` flags. Use the unified runner for a mixed production run so both engines receive workflow metadata and consistent output directories.

V6 emits `results_all_targets_v6_indel.tsv` plus per-target `results_compiled_v6_indel.tsv`. The unified merger prefers the aggregate. The original custom checkout's target-sharding workflow and `finalize_v6_indel_shards.py` remain available for independently deployed V6 jobs.

## Verification and attribution

Local workflow checks use synthetic data and a stub BADGERS script. They verify preparation, command dispatch, generated Slurm syntax, result validation, and merging. Actual guide-design calculations require your installed BADGERS models and environment.

Run the mixed-workflow regression tests from the repository directory with the V6 table dependencies available.

```bash
python -m unittest discover -s tests -v
```

This repository provides workflow utilities around BADGERS. Cite the original BADGERS paper for the design algorithms and consult the upstream software's license for its code and models. [BADGERS publication](https://doi.org/10.1038/s41587-024-02422-w), [upstream repository](https://github.com/broadinstitute/badgers-cas13)
