# Audit: Zero-Node Cases (Appellate-Reasoning Patterns)

## 1. Scope
- **Input corpus:** `uk_cases_full.cleaned.csv` (947 cases with text).
- **Nodes reference:** `nodes.csv` (cases with at least one extracted node).
- **Zero-node cases:** 884 (cases with no ontology match).
- **Sampled for pattern scan:** 35 cases.

## 2. Objective
Identify recurring reasoning patterns in zero-node cases that are *absent* from the current homicide ontology (e.g. conviction safety, appeal grounds, jury misdirection, admissibility, sentence review) to determine whether a consistent *appellate-reasoning* layer can be modeled as a distinct conceptual cluster. Any structural expansion should be grounded in observed textual patterns.

## 3. Pattern clusters scanned
Each sampled case was scanned for the following clusters. **Count** = number of sampled cases with at least one match.

| Cluster | Count (of 35) | Description |
|---------|----------------------|-------------|
| conviction_safety | 0 | Unsafe conviction, quash, lurking doubt |
| appeal_grounds | 0 | Grounds of appeal, leave to appeal, permission |
| jury_misdirection | 0 | Misdirection, summing-up, jury direction |
| admissibility_evidence | 0 | Admissibility, PACE, exclusion, hearsay, bad character |
| sentence_review | 0 | Appeal against sentence, unduly lenient/severe, tariff |
| procedural_fairness | 0 | Fair trial, abuse of process, disclosure, fresh evidence |
| legal_grounds_only | 0 | No point of law, refusal of leave, appeal dismissed |

## 4. Interpretation
- **High prevalence** in a cluster suggests zero-node cases often discuss that appellate theme; it may be worth modeling as a distinct conceptual layer.
- **Low or zero** prevalence suggests zero-node cases are not consistently about that theme; expansion there would be speculative.
- **Per-case detail:** `sample_patterns.csv` (1 = at least one match in that case for that cluster).

## 5. Files produced
- `audit_report.md` (this file)
- `sample_patterns.csv` — binary pattern hits per sampled case
- `zero_node_case_ids.txt` — full list of zero-node case IDs