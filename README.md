## Criminal Cases BN

Pipeline to extract nodes/edges from UK appellate judgments, generate synthetic negatives via ABM, and assemble a Bayesian Network template.

## Dependencies

- **Python**: 3.9+ recommended  
- **Python packages**:
  - **pandas**
  - **requests**
  - **beautifulsoup4**
  - **spacy**
  - **pyyaml**
  - **networkx**

You will also need the `en_core_web_sm` spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## From-scratch setup and run

### 1. Create and activate an environment

```bash
conda create -n legalnlp python=3.10 -y
conda activate legalnlp
```

### 2. Install dependencies

```bash
pip install pandas requests beautifulsoup4 spacy pyyaml networkx
python -m spacy download en_core_web_sm
```

### 3. Run the pipeline

From the repo root:

```bash
# 1) Scrape case texts into data/raw/uk_cases_full.csv
python criminal_cases_BN.py

# 2) Clean scraped texts into data/interim/uk_cases_full.cleaned.csv
python murder_cases_cleaning.py

# (optional) Sanity check on the cleaned texts
python "sanity check.py"

# 3) Extract nodes from cleaned cases into data/processed/nodes.csv
python code/extract_nodes_from_cases.py

# 4) Build edges between nodes into data/processed/edges.csv
python code/build_edges_between_nodes.py

# 5) Generate the BN template (GraphML/GML + summaries in outputs/)
python code/generate_bn_template.py
```