#!/usr/bin/env python3
"""
export_appendix_ontology_labels.py

Export ontology labels from the rule YAMLs used by extract_nodes_from_cases.py:
  - rules/ontology.yml (default)
  - rules/ontology_actus_mens_enriched.yml (actus/mens runs)
  - rules/ontology_general_enriched.yml (general enriched runs)

Layers follow each label's `type` field:
  narrative    -> outputs/appendix_narrative_labels.csv
  evidence     -> outputs/appendix_evidence_labels.csv
  legal_facts  -> outputs/appendix_legalfact_labels.csv

Columns: canonical_label, synonyms_and_variants (comma-separated in one cell)

Run:
  python code/export_appendix_ontology_labels.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

ONTOLOGY_SOURCES = [
    ROOT / "rules" / "ontology.yml",
    ROOT / "rules" / "ontology_actus_mens_enriched.yml",
    ROOT / "rules" / "ontology_general_enriched.yml",
]

OUT_NARR = ROOT / "outputs" / "appendix_narrative_labels.csv"
OUT_EVID = ROOT / "outputs" / "appendix_evidence_labels.csv"
OUT_LEGAL = ROOT / "outputs" / "appendix_legalfact_labels.csv"

LAYER_FILES = {
    "narrative": OUT_NARR,
    "evidence": OUT_EVID,
    "legal_facts": OUT_LEGAL,
}


def _normalize_syn_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for x in raw:
            sx = str(x).strip()
            if sx:
                out.append(sx)
        return out
    s = str(raw).strip()
    return [s] if s else []


def _synonyms_from_meta(meta: dict) -> list[str]:
    if not isinstance(meta, dict):
        return []
    for key in ("synonyms", "lexical_variants", "variants", "synonym"):
        if key in meta:
            return _normalize_syn_list(meta.get(key))
    return []


def _merge_synonyms(into: list[str], more: list[str]) -> None:
    seen = set(into)
    for s in more:
        if s not in seen:
            seen.add(s)
            into.append(s)


def load_and_merge_labels() -> dict[str, dict]:
    """
    Returns merged_labels[canonical_label] = {
      "type": str,
      "synonyms": list[str],
    }
    """
    merged: dict[str, dict] = {}

    for path in ONTOLOGY_SOURCES:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        labels = data.get("labels") or {}
        if not isinstance(labels, dict):
            continue

        for raw_key, meta in labels.items():
            key = str(raw_key).strip()
            if not key or not isinstance(meta, dict):
                continue
            layer = meta.get("type")
            if layer not in LAYER_FILES:
                # e.g. section hints live outside labels in some files — skip non-label entries
                continue
            syns = _synonyms_from_meta(meta)

            if key not in merged:
                merged[key] = {"type": str(layer), "synonyms": list(syns)}
            else:
                if merged[key]["type"] != layer:
                    raise ValueError(
                        f"Label {key!r} has conflicting types: "
                        f"{merged[key]['type']!r} vs {layer!r} (while reading {path})"
                    )
                _merge_synonyms(merged[key]["synonyms"], syns)

    return merged


def write_layer_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canonical_label", "synonyms_and_variants"])
        for label, syn_cell in rows:
            w.writerow([label, syn_cell])


def main() -> None:
    merged = load_and_merge_labels()

    by_layer: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for label in sorted(merged.keys()):
        layer = merged[label]["type"]
        syns = merged[label]["synonyms"]
        syn_cell = ", ".join(syns) if syns else ""
        by_layer[layer].append((label, syn_cell))

    write_layer_csv(OUT_NARR, by_layer["narrative"])
    write_layer_csv(OUT_EVID, by_layer["evidence"])
    write_layer_csv(OUT_LEGAL, by_layer["legal_facts"])

    print("Ontology sources merged:")
    for p in ONTOLOGY_SOURCES:
        print(f"  - {p.relative_to(ROOT) if p.exists() else p.name + ' (missing)'}")

    print("\nLayer label counts (unique canonical_label after merge):")
    for layer, outp in [
        ("narrative", OUT_NARR),
        ("evidence", OUT_EVID),
        ("legal_facts", OUT_LEGAL),
    ]:
        n = len(by_layer[layer])
        print(f"  {layer}: {n}")

    print("\nWrote:")
    print(f"  {OUT_NARR.relative_to(ROOT)}")
    print(f"  {OUT_EVID.relative_to(ROOT)}")
    print(f"  {OUT_LEGAL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
