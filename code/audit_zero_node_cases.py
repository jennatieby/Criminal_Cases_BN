#!/usr/bin/env python3
"""
audit_zero_node_cases.py

Targeted corpus audit of cases that produced zero nodes under the current ontology.
Identifies recurring reasoning patterns absent from the ontology (e.g. conviction
safety, appeal grounds, jury misdirection, admissibility, sentence review) to
determine whether a consistent layer of appellate reasoning can be modeled as a
distinct conceptual cluster. Outputs a diagnostic report grounded in observed
text, not speculative additions.

Usage:
  python code/audit_zero_node_cases.py
  python code/audit_zero_node_cases.py --nodes data/processed/nodes.csv --sample-size 40

Output:
  outputs/audit_zero_node_cases/audit_report.md
  outputs/audit_zero_node_cases/sample_patterns.csv
  outputs/audit_zero_node_cases/zero_node_case_ids.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "audit_zero_node_cases"

DEFAULT_INPUT_CSV = DATA_INTERIM / "uk_cases_full.cleaned.csv"
DEFAULT_NODES_CSV = DATA_PROCESSED / "nodes.csv"
CASE_ID_PREFIX = "CASE_"

# Appellate-reasoning pattern clusters (phrases/regex that are NOT in the homicide ontology)
APPELLATE_PATTERNS = {
    "conviction_safety": [
        r"unsafe\s+(?:conviction|verdict)",
        r"conviction\s+(?:is\s+)?unsafe",
        r"lurking\s+doubt",
        r"quash(?:ed|ing)?\s+(?:the\s+)?conviction",
        r"unsafe\s+and\s+unsatisfactory",
        r"not\s+safe\s+(?:to\s+)?(?:allow\s+)?(?:the\s+)?conviction",
    ],
    "appeal_grounds": [
        r"grounds?\s+of\s+appeal",
        r"leave\s+to\s+appeal",
        r"appeal\s+(?:against\s+)?(?:conviction|sentence)",
        r"permission\s+to\s+appeal",
        r"single\s+judge",
        r"full\s+court",
        r"renew(?:ed|ing)?\s+(?:the\s+)?application",
    ],
    "jury_misdirection": [
        r"misdirection",
        r"direction(?:\s+to\s+the\s+jury)?\s+(?:was\s+)?(?:wrong|defective|inadequate)",
        r"summing\s*[- ]?up",
        r"jury\s+(?:were\s+)?(?:not\s+)?(?:properly\s+)?directed",
        r"wrongful\s+conviction",
        r"material\s+misdirection",
    ],
    "admissibility_evidence": [
        r"admissib(?:le|ility)",
        r"excluded?\s+(?:evidence|from\s+the\s+jury)",
        r"PACE\s+(?:and\s+)?codes?",
        r"unfairly\s+admitted",
        r"hearsay",
        r"bad\s+character",
        r"similar\s+fact",
        r"section\s+78",
        r"exclude\s+evidence",
    ],
    "sentence_review": [
        r"leave\s+to\s+appeal\s+sentence",
        r"appeal\s+against\s+sentence",
        r"unduly\s+(?:lenient|severe|harsh)",
        r"sentence\s+(?:was\s+)?(?:too\s+)?(?:long|short|lenient|severe)",
        r"tariff",
        r"minimum\s+term",
        r"life\s+(?:sentence|imprisonment)",
        r"reduc(?:e|ing)\s+(?:the\s+)?sentence",
    ],
    "procedural_fairness": [
        r"fair\s+trial",
        r"procedural\s+(?:ir)?regularit",
        r"abuse\s+of\s+process",
        r"stay\s+(?:of\s+)?(?:proceedings)?",
        r"disclosure",
        r"material\s+non[- ]?disclosure",
        r"fresh\s+evidence",
        r"new\s+evidence",
    ],
    "legal_grounds_only": [
        r"no\s+point\s+of\s+law",
        r"no\s+arguable\s+grounds?",
        r"refus(?:e|al)\s+of\s+leave",
        r"application\s+refused",
        r"dismiss(?:ed|ing)\s+the\s+appeal",
        r"appeal\s+dismissed",
    ],
}


def get_input_case_ids(csv_path: Path, text_col: str, case_id_prefix: str) -> list[tuple[str, int]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    out = []
    for i in range(len(df)):
        raw = str(df.iloc[i].get(text_col, ""))
        if raw.strip():
            cid = f"{case_id_prefix}{i:05d}"
            out.append((cid, i))
    return out


def get_node_case_ids(nodes_path: Path) -> set[str]:
    if not nodes_path.exists():
        return set()
    df = pd.read_csv(nodes_path, encoding="utf-8-sig")
    if "case_id" not in df.columns or df.empty:
        return set()
    return set(df["case_id"].astype(str).unique())


def sample_zero_node_cases(
    input_csv: Path,
    nodes_csv: Path,
    case_id_prefix: str = CASE_ID_PREFIX,
    sample_size: int = 30,
    seed: int = 42,
) -> tuple[list[str], list[tuple[str, int]], pd.DataFrame]:
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    text_col = None
    for c in ["CleanText", "CaseText", "text", "body", "content"]:
        if c in df.columns:
            text_col = c
            break
    if not text_col:
        raise ValueError("No text column in input CSV")

    input_cases = get_input_case_ids(input_csv, text_col, case_id_prefix)
    node_case_ids = get_node_case_ids(nodes_csv)
    zero_node_case_ids = [cid for cid, _ in input_cases if cid not in node_case_ids]

    try:
        rng = np.random.default_rng(seed)
    except AttributeError:
        rng = np.random.RandomState(seed)
    if len(zero_node_case_ids) > sample_size:
        perm = rng.permutation(len(zero_node_case_ids))[:sample_size]
        sampled_ids = [zero_node_case_ids[i] for i in perm]
    else:
        sampled_ids = zero_node_case_ids[:sample_size] if len(zero_node_case_ids) >= sample_size else zero_node_case_ids

    cid_to_idx = {cid: idx for cid, idx in input_cases}
    sample_with_idx = [(cid, cid_to_idx[cid]) for cid in sampled_ids]

    rows = []
    for cid, idx in sample_with_idx:
        text = str(df.iloc[idx][text_col])[:5000]
        rows.append({"case_id": cid, "row_index": idx, "text_excerpt": text[:1500]})
    sample_df = pd.DataFrame(rows)

    return zero_node_case_ids, sample_with_idx, sample_df


def scan_patterns(text: str) -> dict[str, bool]:
    text_lower = text.lower()
    out = {}
    for cluster, patterns in APPELLATE_PATTERNS.items():
        found = False
        for pat in patterns:
            if re.search(pat, text_lower, re.I):
                found = True
                break
        out[cluster] = found
    return out


def run_audit(
    input_csv: Path,
    nodes_csv: Path,
    sample_size: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    zero_node_case_ids, sample_with_idx, sample_df = sample_zero_node_cases(
        input_csv, nodes_csv, case_id_prefix=CASE_ID_PREFIX, sample_size=sample_size
    )

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    text_col = "CleanText" if "CleanText" in df.columns else "CaseText"
    if text_col not in df.columns:
        text_col = [c for c in ["text", "body", "content"] if c in df.columns][0]

    pattern_counts = {k: 0 for k in APPELLATE_PATTERNS}
    per_case = []

    for cid, idx in sample_with_idx:
        text = str(df.iloc[idx][text_col])
        hits = scan_patterns(text)
        per_case.append({"case_id": cid, **hits})
        for k, v in hits.items():
            if v:
                pattern_counts[k] += 1

    per_case_df = pd.DataFrame(per_case)
    per_case_df.to_csv(out_dir / "sample_patterns.csv", index=False)

    with open(out_dir / "zero_node_case_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(zero_node_case_ids)))

    n_zero = len(zero_node_case_ids)
    n_input = len(df)
    n_sample = len(sample_with_idx)

    md = []
    md.append("# Audit: Zero-Node Cases (Appellate-Reasoning Patterns)")
    md.append("")
    md.append("## 1. Scope")
    md.append(f"- **Input corpus:** `{input_csv.name}` ({n_input} cases with text).")
    md.append(f"- **Nodes reference:** `{nodes_csv.name}` (cases with at least one extracted node).")
    md.append(f"- **Zero-node cases:** {n_zero} (cases with no ontology match).")
    md.append(f"- **Sampled for pattern scan:** {n_sample} cases.")
    md.append("")
    md.append("## 2. Objective")
    md.append("Identify recurring reasoning patterns in zero-node cases that are *absent* from the current homicide ontology (e.g. conviction safety, appeal grounds, jury misdirection, admissibility, sentence review) to determine whether a consistent *appellate-reasoning* layer can be modeled as a distinct conceptual cluster. Any structural expansion should be grounded in observed textual patterns.")
    md.append("")
    md.append("## 3. Pattern clusters scanned")
    md.append("Each sampled case was scanned for the following clusters. **Count** = number of sampled cases with at least one match.")
    md.append("")
    md.append("| Cluster | Count (of " + str(n_sample) + ") | Description |")
    md.append("|---------|----------------------|-------------|")

    cluster_desc = {
        "conviction_safety": "Unsafe conviction, quash, lurking doubt",
        "appeal_grounds": "Grounds of appeal, leave to appeal, permission",
        "jury_misdirection": "Misdirection, summing-up, jury direction",
        "admissibility_evidence": "Admissibility, PACE, exclusion, hearsay, bad character",
        "sentence_review": "Appeal against sentence, unduly lenient/severe, tariff",
        "procedural_fairness": "Fair trial, abuse of process, disclosure, fresh evidence",
        "legal_grounds_only": "No point of law, refusal of leave, appeal dismissed",
    }
    for k in APPELLATE_PATTERNS:
        desc = cluster_desc.get(k, "")
        md.append(f"| {k} | {pattern_counts[k]} | {desc} |")

    md.append("")
    md.append("## 4. Interpretation")
    md.append("- **High prevalence** in a cluster suggests zero-node cases often discuss that appellate theme; it may be worth modeling as a distinct conceptual layer.")
    md.append("- **Low or zero** prevalence suggests zero-node cases are not consistently about that theme; expansion there would be speculative.")
    md.append("- **Per-case detail:** `sample_patterns.csv` (1 = at least one match in that case for that cluster).")
    md.append("")
    md.append("## 5. Files produced")
    md.append("- `audit_report.md` (this file)")
    md.append("- `sample_patterns.csv` — binary pattern hits per sampled case")
    md.append("- `zero_node_case_ids.txt` — full list of zero-node case IDs")

    report_path = out_dir / "audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"Audit complete: {n_zero} zero-node cases; {n_sample} sampled.")
    print(f"Report -> {report_path}")
    print(f"Sample patterns -> {out_dir / 'sample_patterns.csv'}")
    print(f"Zero-node case IDs -> {out_dir / 'zero_node_case_ids.txt'}")


def main():
    p = argparse.ArgumentParser(description="Audit zero-node cases for appellate-reasoning patterns.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="Cleaned case CSV (e.g. 947-case).")
    p.add_argument("--nodes", type=Path, default=DEFAULT_NODES_CSV, help="Nodes CSV from extraction.")
    p.add_argument("--sample-size", type=int, default=30, help="Number of zero-node cases to sample for pattern scan.")
    p.add_argument("--output-dir", type=Path, default=OUT_DIR, help="Output directory for report and CSVs.")
    args = p.parse_args()

    run_audit(args.input, args.nodes, args.sample_size, args.output_dir)


if __name__ == "__main__":
    main()
