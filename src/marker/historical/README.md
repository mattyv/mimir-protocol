# Historical / negative results

These modules tried to solve the "3+ stacked prefixes" problem before we
landed on the composed-axiom (H) approach in `axiom_registry.composed_description`.
They are kept here for the negative-result record and for the
reproducibility of the experiments that informed the canonical path.

Each has a "Status" note in its docstring marking it superseded.

| Module | What it tried | Outcome |
|---|---|---|
| `ape.py` | APE (Yang et al ICLR 2025): q_scale + shared-prefix attention sink | Helps for direct fact lookup at 3-5 axioms; fails on counterfactual / DAG-traversal |
| `per_block_attention.py` | Custom SDPA with per-block softmax (uniform/cosine/lse combiners) | Reproduced ICLR 2025 Block-Attention frozen-ablation collapse (67.9 → 48% accuracy). Confirmed dead. |
| `selective_recompute.py` | CacheBlend-style selective recompute (v1 vanilla-copy) | No improvement over rope-fix concat |
| `axiom_signatures.py` | Per-axiom K-vector fingerprints (Path 3, binding-ID hypothesis) | Modal sweep showed identical outputs at magnitudes 0.05–1.00. K-side perturbation alone doesn't propagate to attention routing; model still pattern-hallucinates "100, 200, 300..." regardless. |

The demo scripts (`run_chain_ape_demo.py`, `run_chain_selective_recompute_demo.py`,
`run_combine_vectors_demo.py`, the axiom-count sweep demos, etc.) are
the experimental runners that produced these results on Modal.

The mechanical-invariant tests for these modules live in `tests/historical/`.

**Active canonical path:** `marker.prefix_tuning` + `marker.axiom_registry`
(see project README "What's still hard").
