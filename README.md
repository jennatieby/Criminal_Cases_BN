## Criminal Cases BN

This repository supports a **thesis project** that constructs a **corpus-derived Bayesian network** for homicide reasoning under **English and Welsh law**. Judgment text is turned into a binary case×node matrix aligned to a fixed directed acyclic graph; the network is then fit, queried, and assessed on held-out data and interpretive analyses.

- **Final corpus:** 1,068 cases (BAILII appellate judgments plus CCRC-style negative pages), **England and Wales only**.
- **Jurisdiction filter:** 14 Northern Ireland cases were removed from the CCRC-derived set (`code/filter_ni_cases.py`).
- **Structure:** **13 nodes** grounded in **Coke’s definition of murder** (actus reus, mens rea, defences, and trial verdict).
- **Parameters:** estimated by **maximum likelihood** with **Dirichlet (BDeu) smoothing**, **`equivalent_sample_size=5`** (via `pgmpy` on the training split).
- **Evaluation:** inference performance (stratified train/test), **evidence scenarios** (`code/run_inference_scenarios.py`), **single-node sensitivity** (`code/sensitivity_analysis.py`), **evidence analysis** (likelihood ratios, characteristic ratios, mutual information, incremental effects; `code/evidence_analysis.py`), **consistency** (Jensen–Shannon divergence, Jaccard grouping; `code/evaluate_bn_consistency.py`), and **calibration** (Brier score, expected calibration error; `code/evaluate_bn_calibration.py`).

---

## What you need before scraping

- **`uk_cases_index.csv`** in the repository root, with at least a **`URL`** column listing BAILII (or compatible) case pages.  
  `criminal_cases_BN.py` reads this file and writes `data/raw/uk_cases_full.csv`.

For the negative (CCRC) track you also need **`data/raw/negative_cases_raw.csv`** with URLs (see `scrape_negative_cases.py`).

---

## Dependencies

**Python:** 3.9+ (3.10 is a reasonable default.)

**Core (scraping, cleaning, extraction, BN template):**

- `pandas`, `requests`, `beautifulsoup4`, `spacy`, `pyyaml`, `networkx`

Install the small English spaCy model:

```bash
python -m spacy download en_core_web_sm
```

**Model fitting, evaluation, and figures:**

- `numpy`, `matplotlib`
- `pgmpy`
- `scikit-learn` (for `code/train_test_split_evaluation.py`)
- `scipy` (Jensen–Shannon divergence and related metrics)

**GPT-assisted extraction (OpenAI API):**

- `openai` — used for BAILII outcome auditing (`code/audit_bailii_outcomes.py`, default **gpt-4o**) and case-level metadata / verdict coding (`code/extract_case_metadata.py`; default model is **`gpt-4o-mini`** unless you set **`OPENAI_MODEL=gpt-4o`**). Set `OPENAI_API_KEY` before running those scripts.

**Other** LLM helpers under `code/` (e.g. ontology enrichment, sentence annotation) may require additional packages and API configuration—see the docstrings in those scripts.

---

## Environment setup (example)

```bash
conda create -n legalnlp python=3.10 -y
conda activate legalnlp
pip install pandas requests beautifulsoup4 spacy pyyaml networkx
pip install numpy matplotlib pgmpy scikit-learn scipy openai
python -m spacy download en_core_web_sm
```

---

## Core pipeline (BAILII / “positive” cases)

Run from the repository root.

```bash
# 1) Scrape case texts -> data/raw/uk_cases_full.csv
python criminal_cases_BN.py

# 2) Clean BAILII boilerplate -> data/interim/uk_cases_full.cleaned.csv
python murder_cases_cleaning.py

# Optional: quick checks on cleaned text
python "sanity check.py"

# 3) Corpus audit (trial verdict + appeal outcome; default GPT-4o) -> outputs/bailii_outcome_audit.csv
export OPENAI_API_KEY="sk-..."
python code/audit_bailii_outcomes.py

# 4) Extract nodes (ontology in rules/, spaCy for tokenization) -> data/processed/nodes.csv
python code/extract_nodes_from_cases.py

# 5) Build edges between nodes -> data/processed/edges.csv (+ node inventory)
python code/build_edges_between_nodes.py

# Optional: aggregate edges into a co-occurrence BN template (GraphML/GML + CSV summaries)
python code/generate_bn_template.py
```

`extract_nodes_from_cases.py` and `build_edges_between_nodes.py` accept `--input` / output paths for alternate corpora or provenance flags; see their module docstrings.

---

## Optional: negative (e.g. CCRC) cases

```bash
python scrape_negative_cases.py          # -> data/raw/negative_cases_full.csv
python clean_negative_cases.py           # -> data/interim/negative_cases_cleaned.csv

# Remove Northern Ireland cases (14 removed in the thesis corpus) -> negative_cases_cleaned_ew.csv
python code/filter_ni_cases.py

python code/extract_nodes_from_cases.py --input data/interim/negative_cases_cleaned_ew.csv --output-csv data/processed/negative_nodes.csv --provenance negative --case-id-prefix NEG_
python code/build_edges_between_nodes.py --input data/processed/negative_nodes.csv --output-edges data/processed/negative_edges.csv
python code/generate_bn_template.py --nodes data/processed/negative_nodes.csv --edges data/processed/negative_edges.csv --prefix negative_
```

Merge positive and negative node/edge tables:

```bash
python code/merge_positive_negative.py   # -> data/processed/nodes_unified.csv, edges_unified.csv
python code/generate_bn_template.py --nodes data/processed/nodes_unified.csv --edges data/processed/edges_unified.csv --prefix unified_ --min-mean-score 0.2
```

---

## Network construction, metadata, and evaluation

Map extractions to the homicide DAG, build the binary matrix, refresh verdict-related columns with OpenAI-derived metadata (`OPENAI_MODEL=gpt-4o` for GPT-4o), then fit the network on an 80/20 stratified split and run downstream analyses.

```bash
# Label -> DAG mapping and summary (see script defaults for input nodes CSV)
python code/map_extraction_to_dag.py

# Binary case×node matrix (+ summary / heatmap under outputs/)
python code/build_case_node_matrix.py

# OpenAI verdict / death-established metadata (use export OPENAI_MODEL=gpt-4o for GPT-4o); joins onto case_node_matrix.csv
export OPENAI_API_KEY="sk-..."
python code/extract_case_metadata.py

# 80/20 stratified split, fit BN (MLE + Dirichlet ess=5), test inference -> outputs/homicide_bn_train.bif, figures/metrics per script
python code/train_test_split_evaluation.py

# Full-corpus FULL_EVIDENCE posteriors -> outputs/scenario_results_full.csv (input model path must match the train-fitted BIF)
python code/run_inference_scenarios.py --bif outputs/homicide_bn_train.bif --full-evidence-all

# Calibration (Brier, ECE) -> outputs/calibration_results.csv (+ reliability figures under outputs/figures/)
python code/evaluate_bn_calibration.py

# Consistency (JS divergence, Jaccard groups) -> outputs/consistency_results.csv (+ plots / summaries per script)
python code/evaluate_bn_consistency.py

# Single-node sensitivity -> outputs/sensitivity_single_node.csv (+ related sensitivity CSVs/figures per script)
python code/sensitivity_analysis.py

# Evidence analysis: likelihood ratios, characteristic ratios, mutual information, incremental effects
python code/evidence_analysis.py

# Consolidated result figures -> outputs/figures/
python code/generate_results_figures.py
```

---

## Key outputs

- **`outputs/homicide_bn_train.bif`** — fitted Bayesian network from the training split.
- **`outputs/scenario_results_full.csv`** — scenario / full-evidence inference rows used by calibration and consistency scripts.
- **`outputs/calibration_results.csv`** — calibration metrics (e.g. Brier, ECE-related summaries).
- **`outputs/consistency_results.csv`** — consistency / pairwise JS results.
- **`outputs/sensitivity_single_node.csv`** — single-node sensitivity table.
- **`outputs/likelihood_ratios_verdict.csv`**, **`outputs/characteristic_ratios.csv`**, **`outputs/mutual_information.csv`**, **`outputs/incremental_effects.csv`** — evidence analysis tables (`code/evidence_analysis.py`).
- **`outputs/bailii_outcome_audit.csv`** — BAILII corpus audit (verdict + appeal outcome).
- **`outputs/figures/`** — paper-style figures, including those produced by `code/generate_results_figures.py`, reliability diagrams, sensitivity plots, and related visuals.
