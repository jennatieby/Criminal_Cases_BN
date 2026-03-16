# Qualitative Summary: Extraction Method Comparison

This document interprets the numerical comparison of the three extraction methods.

---

## Coverage

- **Rule-based only** extracts the fewest nodes (54,499), giving a conservative baseline.
- **Actus reus / mens rea enriched** adds a moderate number of nodes (56,460), mainly in doctrinal categories.
- **General enriched** yields the most nodes (63,226), maximising recall over the case set.
- Cases with at least two nodes: baseline 99.1%, actus/mens 99.2%, general 99.4%. Richer methods slightly increase structural coverage per case.

## Doctrinal completeness

- **Actus reus (killing)**: baseline 1,566, actus/mens 1,986, general 1,668. Enriched ontologies pull in more doctrinal mentions; actus/mens is strongest for causation and intent-to-GBH.
- **Unlawful killing**: baseline 299, actus/mens 710, general 910. Enriched ontologies pull in more doctrinal mentions; actus/mens is strongest for causation and intent-to-GBH.
- **Causation**: baseline 314, actus/mens 1,092, general 391. Enriched ontologies pull in more doctrinal mentions; actus/mens is strongest for causation and intent-to-GBH.
- **Mens rea (intent to kill)**: baseline 786, actus/mens 805, general 800. Enriched ontologies pull in more doctrinal mentions; actus/mens is strongest for causation and intent-to-GBH.
- **Mens rea (intent GBH)**: baseline 5, actus/mens 338, general 14. Enriched ontologies pull in more doctrinal mentions; actus/mens is strongest for causation and intent-to-GBH.

The actus/mens ontology is targeted at murder-relevant elements and typically increases counts for unlawful killing, causation, and intent without diluting label semantics. General enrichment adds breadth but may include less doctrine-specific nodes.

## Structural richness

- **Unique labels**: baseline 61, general enriched 65. Richer ontologies produce more label diversity, which can support finer-grained graphs and analysis.
- Higher total nodes and more labels per case improve the potential for edge extraction and Bayesian network structure learning.

## Noise / interpretability tradeoff

- **Rule-based only** is the most interpretable and least noisy: every node is driven by ontology and rules, so labels are consistent and auditable.
- **Actus reus / mens rea enriched** keeps the focus on murder-relevant doctrine while increasing recall; noise is limited to synonym/paraphrase matches within a controlled vocabulary.
- **General enriched** maximises recall and structural richness but introduces more variation in phrasing and potentially less precise labels; downstream filtering or aggregation may be needed for clean doctrinal summaries.

## Recommended method

- **For doctrinal analysis and murder-specific elements**: prefer **rule-based + actus reus / mens rea enriched**. It improves coverage of unlawful killing, causation, and mens rea without sacrificing interpretability.
- **For exploratory graph building and maximum coverage**: use **rule-based + general enriched**, then apply post-hoc filters or aggregation for clarity.
- **For audits and reproducibility**: **rule-based only** remains the best baseline to report and compare against.
