#!/usr/bin/env python3
"""
extract_nodes_from_cases.py

Densified node extractor:
- Uses rules/ontology.yml (labels with type + synonyms)
- Optional source cues from rules/idioms.yml (key: source_cues) or rules/source_cues.yml
- Lemma & regex matching
- Section tagging (facts/analysis/conclusion) + paragraph_id
- Polarity with negation/uncertainty cues
- Outputs: data/interim/nodes.jsonl and data/processed/nodes.csv

Run from repo root (so relative paths resolve):
    conda activate legalnlp
    python code/extract_nodes_from_cases.py
"""

import json
import re
from pathlib import Path

import pandas as pd
import spacy
import yaml

# -------------------------- Paths --------------------------
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "interim" / "uk_cases_full.cleaned.csv"
OUT_JSONL = ROOT / "data" / "interim" / "nodes.jsonl"
OUT_CSV = ROOT / "data" / "processed" / "nodes.csv"
ONTO_PATH = ROOT / "rules" / "ontology.yml"
IDIOMS_PATH = ROOT / "rules" / "idioms.yml"
SOURCE_CUES_FALLBACK = ROOT / "rules" / "source_cues.yml"

OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# -------------------------- Load models & rules --------------------------
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
def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT}")

    df = pd.read_csv(INPUT)
    # robustly pick text column
    text_col = None
    for cand in ["CleanText", "CaseText", "text", "body", "content"]:
        if cand in df.columns:
            text_col = cand
            break
    if not text_col:
        raise ValueError(f"No text column found! Columns present: {list(df.columns)}")

    # derive case_id if not present
    if "case_id" not in df.columns:
        df["case_id"] = [f"CASE_{i:05d}" for i in range(len(df))]

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
            # sentence segmentation
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
                        "provenance": "real",
                        "match_how": how,  # synonym | regex (debug)
                    })

    # write JSONL
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # write CSV
    pd.DataFrame.from_records(records).to_csv(OUT_CSV, index=False)
    print(f"✅ Extracted {len(records):,} nodes from {total_cases:,} cases → {OUT_CSV}")

if __name__ == "__main__":
    main()