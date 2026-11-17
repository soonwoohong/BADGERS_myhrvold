# Custom V6 indel engine

This directory includes Soonwoo Hong's `search_guide_v6_indel.py` and its companion `search_v5_windows.py` from the lab's customized `badgers_search_guide` checkout. The copied implementation reports version `6.4.1-deduplicated-contexts-sharded-cpu`. The included MIT license preserves the original copyright and permissions.

The workflow runner invokes this engine for manifest rows with `engine=v6_indel`. The engine retains its original search, scoring, signature, and per-target completion behavior. It accepts XLSX, CSV, or TSV input. Preparation supplies aligned TSV records so insertion/deletion coordinates can be recovered before the engine builds ungapped contexts.

The V6 interpreter needs NumPy, pandas, Biopython, openpyxl, TensorFlow, and ADAPT with its Cas13a classification/regression models. Use the environment already configured for the custom search, or install the workflow dependencies from `requirements-indel.txt` into a compatible ADAPT environment. `--plan_only` validates selection and junction windows using the table dependencies before loading TensorFlow.

V6 writes its own metadata inside the runner's `engine_results/` directory. The runner records workflow completion separately in the parent directory. The merger uses the V6 aggregate table once, preserving the custom score columns and engine provenance.
