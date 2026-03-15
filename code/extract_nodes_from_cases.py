#!/usr/bin/env python3
"""
extract_nodes_from_cases.py

Node extractor: turns case text into structured nodes using ontology and rules.
Supports positive (BAILII) and negative (CCRC) runs via CLI arguments.

Usage:
  Positive (defaults):
    python code/extract_nodes_from_cases.py
  Negative:
    python code/extract_nodes_from_cases.py --input data/interim/negative_cases_cleaned.csv --output-csv data/processed/negative_nodes.csv --provenance negative --case-id-prefix NEG_
  With enriched ontology (LLM-expanded synonyms):
    python code/extract_nodes_from_cases.py --ontology rules/ontology_actus_mens_enriched.yml ...
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import spacy
import yaml

ROOT = Path(__file__).resolve().parents[1]
ONTO_PATH = ROOT / "rules" / "ontology.yml"
IDIOMS_PATH = ROOT / "rules" / "idioms.yml"
SOURCE_CUES_FALLBACK = ROOT / "rules" / "source_cues.yml"
nlp = spacy.load("en_core_web_sm", disable=["ner"])

with open(ONTO_PATH, "r", encoding="utf-8") as f:
    onto = yaml.safe_load(f)

labels_cfg = onto.get("labels", {})
if not labels_cfg:
    raise ValueError("ontology.yml missing 'labels' map with types and synonyms.")

# Try to read source cues from idioms.yml; fall back to source_cues.yml if present
source_cues = {}
if IDIOMS_PATH.exists():
    with open(IDIOMS_PATH, "r", encoding="utf-8") as f:
        idioms = yaml.safe_load(f) or {}
    source_cues = idioms.get("source_cues", {}) or {}
if not source_cues and SOURCE_CUES_FALLBACK.exists():
    with open(SOURCE_CUES_FALLBACK, "r", encoding="utf-8") as f:
        source_cues = yaml.safe_load(f) or {}

# sanitize source_cues (keep only list-like)
source_cues = {
    k: (v if isinstance(v, (list, tuple)) else [])
    for k, v in source_cues.items()
}

# -------------------------- Matching utilities --------------------------
# Section headers (soft hints)
SECTION_HEADERS = onto.get("sections", {
    "facts": ["Facts", "Background", "The Evidence", "Factual Background"],
    "analysis": ["Discussion", "Analysis", "Legal Framework", "Reasons", "Consideration"],
    "conclusion": ["Conclusion", "Disposition", "Result", "Decision", "Order"],
})

NEGATION_CUES = [
    r"\bno evidence of\b",
    r"\bnot satisfied that\b",
    r"\bnot proved\b",
    r"\bdid not intend\b",
    r"\bno intention\b",
    r"\bnot (?:established|made out)\b",
    r"\bnot\b",  # keep broad after specific cues
    r"\bnever\b",
    r"\bwithout\b",
]

UNCERTAINTY_CUES = r"\b(may|might|could|possibly|appears|seems)\b"

# Regex patterns to catch phrasings beyond synonyms
REGEX_PATTERNS = {
    "mens_rea_intent_to_kill": [
        r"\bintend(?:ed|s)? to (?:kill|cause (?:his|her|their)? death)\b",
        r"\bmurderous intent\b",
        r"\bpurpose to kill\b",
    ],
    "planning_premeditation": [
        r"\bpre(?:-| )?meditat(?:ed|ion)\b",
        r"\bplan(?:ned|ning)\b",
        r"\blay in wait\b",
        r"\bfalse alibi\b",
        r"\bprocured (?:a )?weapon\b",
        r"\breconnaissance\b",
    ],
    "unlawful_killing": [
        r"\bcaused the death of\b",
        r"\binflicted (?:a )?fatal (?:wound|injury)\b",
        r"\b(?:stabbed|shot|strangled)\b.*\b(?:to death|fatally)\b",
        r"\bunlawful killing\b",
    ],
    "self_defence_legal": [
        r"\bself[- ]defen[cs]e\b",
        r"\breasonable force\b",
        r"\bhonestly believed\b.*\bimminent attack\b",
        r"\bnecessary to defend\b",
        r"\bproportionate response\b",
    ],
    "loss_of_control": [
        r"\bloss? of self[- ]control\b",
        r"\bqualifying trigger\b",
        r"\bthings said (?:or|and) done\b",
    ],
    "diminished_responsibility": [
        r"\bdiminished responsibility\b",
        r"\babnormality of mental functioning\b",
        r"\bsubstantial impairment\b",
        r"\brecognised medical condition\b",
    ],
}

def compile_label_index(labels_cfg):
    """
    Build an index: label -> {type, phrases}
    phrases includes the canonical label (spaces) and synonyms (lowercased).
    """
    index = {}
    for label, meta in labels_cfg.items():
        ltype = meta.get("type", "narrative")
        syns = meta.get("synonyms", []) or []
        phrases = {label.replace("_", " ").lower()}
        phrases.update(x.lower() for x in syns if isinstance(x, str))
        index[label] = {"type": ltype, "phrases": sorted(phrases, key=len, reverse=True)}
    return index

LABEL_INDEX = compile_label_index(labels_cfg)

def detect_source(sentence: str) -> str:
    s = sentence.lower()
    for src, cues in source_cues.items():
        if any(cue.lower() in s for cue in cues):
            return src
    return "uncertain"

def detect_polarity(sentence: str) -> int:
    s = sentence.lower()
    for pat in NEGATION_CUES:
        if re.search(pat, s):
            return 0  # negated
    if re.search(UNCERTAINTY_CUES, s):
        return -1   # uncertain
    return 1        # asserted

def tag_section(paragraphs, idx):
    """
    Infer section from the closest heading above paragraph idx.
    """
    # look up to 8 paragraphs back for a header-like line
    for j in range(max(0, idx - 8), idx + 1):
        header = paragraphs[j].strip()
        if len(header) > 120:
            continue  # too long to be a header
        h = header.lower().strip(" :")
        for sec, names in SECTION_HEADERS.items():
            for name in names:
                if h.startswith(name.lower()):
                    return sec
    return "facts"

# -------------------------- Extraction --------------------------
def iter_paragraphs(clean_text: str):
    """
    Split raw text into paragraphs; preserve order.
    """
    if not clean_text:
        return []
    # normalize newlines; split on blank lines or double breaks
    paras = re.split(r"(?:\r?\n){2,}", clean_text)
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # collapse inner whitespace a bit
        p = re.sub(r"\s+\n\s+", " ", p)
        out.append(p)
    return out

def match_labels(sentence_lower: str):
    """
    Yield (label, type, 'synonym') matches using phrases and regex patterns.
    """
    hits = []
    # A) phrase (synonym) hits
    for label, meta in LABEL_INDEX.items():
        for phrase in meta["phrases"]:
            if phrase and phrase in sentence_lower:
                hits.append((label, meta["type"], "synonym"))
                break  # one phrase is enough per label
    # B) regex hits for key legal constructs
    for label, patterns in REGEX_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, sentence_lower):
                ltype = labels_cfg.get(label, {}).get("type", "narrative")
                hits.append((label, ltype, "regex"))
                break
    return hits

# -------------------------- Main --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Extract nodes from case text using ontology and rules.")
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "interim" / "uk_cases_full.cleaned.csv",
        help="Input CSV with case text (must have CleanText, CaseText, or text column).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "nodes.csv",
        help="Output CSV path for nodes.",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Output JSONL path for nodes. If omitted, derived from output-csv (same dir, .jsonl).",
    )
    p.add_argument(
        "--provenance",
        type=str,
        default="real",
        help="Provenance label for all extracted nodes (e.g. 'real' for positive, 'negative' for CCRC).",
    )
    p.add_argument(
        "--case-id-prefix",
        type=str,
        default="CASE_",
        help="Prefix for generated case_id when column is missing (e.g. CASE_ or NEG_).",
    )
    p.add_argument(
        "--ontology",
        type=Path,
        default=None,
        help="Path to ontology YAML. If omitted, uses rules/ontology.yml. Use enriched ontology for LLM-expanded synonyms.",
    )
    args = p.parse_args()
    if args.output_jsonl is None:
        args.output_jsonl = args.output_csv.parent / (args.output_csv.stem + ".jsonl")
    return args


def main():
    global labels_cfg, LABEL_INDEX, SECTION_HEADERS
    args = parse_args()
    onto_path = args.ontology.resolve() if args.ontology else ONTO_PATH
    if onto_path != ONTO_PATH:
        with open(onto_path, "r", encoding="utf-8") as f:
            onto = yaml.safe_load(f)
        labels_cfg = onto.get("labels", {})
        if not labels_cfg:
            raise ValueError(f"Ontology missing 'labels' map: {onto_path}")
        LABEL_INDEX = compile_label_index(labels_cfg)
        SECTION_HEADERS = onto.get("sections", {
            "facts": ["Facts", "Background", "The Evidence", "Factual Background"],
            "analysis": ["Discussion", "Analysis", "Legal Framework", "Reasons", "Consideration"],
            "conclusion": ["Conclusion", "Disposition", "Result", "Decision", "Order"],
        })

    input_path = args.input.resolve()
    output_csv = args.output_csv.resolve()
    output_jsonl = args.output_jsonl.resolve()
    provenance = args.provenance
    case_id_prefix = args.case_id_prefix.rstrip("_") + "_"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    text_col = None
    for cand in ["CleanText", "CaseText", "text", "body", "content"]:
        if cand in df.columns:
            text_col = cand
            break
    if not text_col:
        raise ValueError(f"No text column found! Columns present: {list(df.columns)}")

    if "case_id" not in df.columns:
        df["case_id"] = [f"{case_id_prefix}{i:05d}" for i in range(len(df))]

    records = []
    total_cases = 0
    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        raw = str(row.get(text_col) or "")
        if not raw.strip():
            continue

        total_cases += 1
        paragraphs = iter_paragraphs(raw)
        for p_idx, para in enumerate(paragraphs, start=1):
            section = tag_section(paragraphs, p_idx - 1)
            doc = nlp(para)
            for sent in doc.sents:
                s = sent.text.strip()
                if not s:
                    continue
                s_lower = s.lower()
                src = detect_source(s)
                pol = detect_polarity(s)
                for label, ltype, how in match_labels(s_lower):
                    records.append({
                        "case_id": case_id,
                        "node_id": f"{case_id}_N{len(records)+1:06d}",
                        "label": label,
                        "type": ltype,
                        "source": src,
                        "polarity": pol,
                        "section": section,
                        "sent_text": s,
                        "paragraph_id": p_idx,
                        "provenance": provenance,
                        "match_how": how,
                    })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pd.DataFrame.from_records(records).to_csv(output_csv, index=False)
    print(f"Extracted {len(records):,} nodes from {total_cases:,} cases -> {output_csv} (provenance={provenance})")


if __name__ == "__main__":
    main()