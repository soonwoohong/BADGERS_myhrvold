# BADGERS_myhrvold

Prepare aligned FASTA comparisons, run BADGERS, and collect results for the Myhrvold lab. Each stage is a separate Python script and supports `--help` and `--interactive`.

| Stage | Script | Output |
| --- | --- | --- |
| 1. Process FASTAs | `prepare_fasta.py` | Cluster folders, WT copies, variant/indel reports, job manifest |
| 2. Optional Slurm setup | `make_slurm.py` | A batch script with your allocation and resource settings |
| 3. Run BADGERS | `run_badgers.py` | Per-cluster compiled results, logs, and run metadata |
| 4. Merge results | `merge_results.py` | One TSV with original result columns and provenance |

The runner uses your existing BADGERS installation. The default algorithm is `evolutionary`. It also supports `wgan_am` and `both`. The FASTA processor and job manifest are compatible with bundles from the earlier `prepare_badgers.py` script.

## Install the workflow tools

Download the repository from GitHub's **Code → Download ZIP** menu, or clone it from a terminal.

```bash
git clone https://github.com/soonwoohong/BADGERS_myhrvold.git
cd BADGERS_myhrvold
python -m pip install -r requirements.txt
```

Use Python 3.8 or later. Biopython is needed for FASTA processing. The runner, merger, and Slurm generator use the Python standard library.

BADGERS has its own dependencies and model files. Follow its installation instructions in the environment where you will run the calculations. Supply that environment's Python interpreter with `run_badgers.py --python /path/to/badgers-env/bin/python` if it differs from your workflow Python. [Upstream BADGERS installation](https://github.com/broadinstitute/badgers-cas13)

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
```

Input filename stems appear in output FASTAs, headers, and reports. Extensions are normalized to `.fasta` for BADGERS. WT receives the cluster suffix in each cluster. In window mode, a variant keeps its input stem, and its folder and report row give the new assignment. Each input stem must be unique across extensions. The upstream importer treats lowercase `split` and `.fasta` inside stems specially, so those names require renaming before an ordinary run.

`variant_summary.tsv` contains one WT reference row per cluster and one row per variant. Substitution and deletion positions are 1-based WT coordinates. In the insertion column, `42+2` denotes two bases inserted after WT position 42, with anchor 0 denoting the start of the reference. `indels.tsv` contains records with insertion/deletion events, including compensating indels that preserve net length.

All assigned records remain in `aligned.fasta`. The `standard/` directory includes WT and eligible gap-free substitution records. Ambiguous records and tied gap-free alignments are marked for review. An eligible variant is needed to create a runnable cluster. The public `diff` importer uses each FASTA as the focal target and pools all remaining FASTAs, including WT, for comparison. WT also becomes a focal target. [Upstream FASTA importer](https://github.com/broadinstitute/badgers-cas13/blob/main/badgers/utils/import_fasta.py)

`range.tsv` covers all positions with a complete 48-nt context window. `--window` controls mutation clustering. Change `range.tsv` if your experiment needs a narrower scan. `jobs.tsv` contains one task per runnable cluster, with zero-based task IDs. Indel-only inputs produce reports and alignments with an empty job manifest.

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
    --run-name ordinary_01

# Submit one task first. sbatch prints its job ID.
sbatch --array=0 gene.slurm

# Submit the array defined in gene.slurm.
sbatch gene.slurm
```

The full array includes task 0 again. If task 0 completed successfully under `ordinary_01`, use a fresh run name for a full rerun or submit only the remaining IDs shown in `jobs.tsv`, such as `sbatch --array=1-4%1 gene.slurm`. The runner reports an error if that cluster's output directory already exists. Successful output is written to `prepared/gene/results/ordinary_01/clN/`.

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

`--algorithm both` runs both explorers. `--top-guides 25` sets the number saved per focal sequence. Run names keep separate experiments in separate directories. Custom BADGERS flags can be passed as repeated `--extra-arg` values, with `=` for arguments beginning with `--` or negative numbers.

```bash
python run_badgers.py prepared/gene \
    --design-script /path/to/your/custom/design_guides.py \
    --python /path/to/badgers-env/bin/python \
    --run-name custom_01 --cpus 8 \
    --extra-arg=--verbose_results
```

Your chosen BADGERS version must implement any custom flags you pass. `make_slurm.py` also accepts `--top-guides` and repeated `--extra-arg` options.

Each cluster's output contains `badgers.log` and `run_metadata.json`. A run is marked complete after BADGERS exits successfully and the expected compiled tables contain results. A failed run retains its log and metadata for diagnosis. To retry it, choose another run name. The runner passes the requested CPU count to BADGERS and caps TensorFlow/BLAS threads per worker at one.

## 4. Merge compiled results

Once the jobs finish, merge one run or several result directories.

```bash
python merge_results.py --interactive

python merge_results.py prepared/gene/results/ordinary_01 --out results/ordinary_01.tsv

python merge_results.py prepared/geneA/results/ordinary_01 \
    prepared/geneB/results/ordinary_01 --out results/all_loci.tsv
```

The merger recursively finds `results_compiled_evolutionary.tsv` and `results_compiled_wgan_am.tsv`, retains all source columns, and adds `workflow_source_file`, `workflow_source_run`, `workflow_cluster`, and `workflow_explorer`. Column-name collisions receive a numbered suffix. Duplicate result rows are retained. Each source file is included once if input directories overlap.

Tables with sibling workflow metadata must have status `complete`. Legacy BADGERS results without this metadata are accepted. The merger validates headers and field counts and writes a new output file atomically. Fitness comparisons should use the same targets and objective settings. [BADGERS output documentation](https://github.com/broadinstitute/badgers-cas13#output-format)

## Indel support

Indels are identified and reported during preparation. Their alignment and coordinate information can support a separate scoring workflow. The ordinary BADGERS branch uses the ADAPT activity model and gap-free windows. I could not verify a validated, released LwaCas13a indel/bulge activity predictor that can directly replace ADAPT. [BADGERS importer](https://github.com/broadinstitute/badgers-cas13/blob/main/badgers/utils/import_fasta.py)

TIGER is relevant to model development because its RfxCas13d cellular screens included insertions/deletions. Its released training script explicitly loads data with `indels=False`, so these results do not establish an indel-aware model for LwaCas13a diagnostics. CRISPR-Scan provides recent LwaCas13a experimental evidence for indel discrimination. [TIGER paper](https://pubmed.ncbi.nlm.nih.gov/37400521/), [TIGER training code](https://github.com/daklab/tiger/blob/main/tiger_trainer.py), [CRISPR-Scan paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13669218/)

## Verification and attribution

Local workflow checks use synthetic data and a stub BADGERS script. They verify preparation, command dispatch, generated Slurm syntax, result validation, and merging. Actual guide-design calculations require your installed BADGERS models and environment.

This repository provides workflow utilities around BADGERS. Cite the original BADGERS paper for the design algorithms and consult the upstream software's license for its code and models. [BADGERS publication](https://doi.org/10.1038/s41587-024-02422-w), [upstream repository](https://github.com/broadinstitute/badgers-cas13)
