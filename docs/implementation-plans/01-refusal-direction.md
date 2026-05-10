# Implementation Plan 01 — Close the RLHF/Instruct Gap with Refusal-Direction Ablation

**Open problem (from `README.md`):** On Qwen 2.5-32B-Instruct, prefix
tuning works for ~6/10 axioms. The other ~4 are intercepted by
"I don't know that term" refusal patterns from RLHF.

**Chosen technique:** Arditi et al 2024,
[Refusal in LMs Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717).
Mean-diff vector between refusing and complying activations defines a
"refusal direction"; project it out of every layer's residual stream at
inference. Gradient-free, no weight changes, composes orthogonally with
our `past_key_values` prefix.

**Why this technique** (full reasoning in
`docs/related-work-and-open-problems.md`): the residual-stream ablation
hooks are independent of our K/V cache injection, so the two interventions
stack without interference. ~30 contrast pairs is enough. Reference
implementation already cloned at `related_work/refusal_direction/`.

**Estimated effort:** one day for an experienced ML engineer who has
read this doc and `docs/related-work-and-open-problems.md`. The hard
part is the contrast set, not the math.

---

## Design decisions to make before coding

Three open questions. Recommendations are (R).

### 1. Contrast set construction

| Option | Description | Verdict |
|---|---|---|
| **(R) Pool refusals across all axioms** | Run Qwen Instruct on all 10 axioms' probe prompts (no prefix). Auto-label outputs as 'refused' (matches "don't know", "I'm not familiar", "unsure", "I don't have information") or 'complied'. Mean-diff at the axiom-term token position. One direction across all axioms. | Closest to Arditi's setup. Fastest path to signal. Start here. |
| Per-axiom direction | Build one direction per axiom from its own paraphrases. | Underpowered — only ~4/10 axioms actually refuse, so very few negatives per axiom. Skip unless pooled doesn't generalize. |
| Lift Arditi's pre-shipped Qwen direction | Use `related_work/refusal_direction/pipeline/runs/qwen-1_8b-chat/direction.pt` directly. | Cheap sanity check. Almost certainly fails — their direction targets *harmful-content* refusal, not *unknown-term* refusal. Run as a control. |

### 2. Token position for activation extraction

Arditi extracts at end-of-instruction tokens (the last few tokens before
the model starts generating). For mimir, the refusal triggers when the
model encounters the unknown axiom term, which is *inside* the prompt,
not at the end. Two options:

- **(R) Last token (matches Arditi exactly)** — simpler, comparable to
  literature. The refusal direction at the last position before generation
  should still capture the model's "about-to-refuse" state.
- Axiom-term token — find the position of "Flaxum"/"Balance Publisher"/
  etc. in the prompt and extract there. More principled; more code.

Start with last-token. If results are weak, switch.

### 3. Integration pattern with existing prefix injection

| Option | Description | Verdict |
|---|---|---|
| **(R) Optional kwarg on `generate_with_prefix(es)`** | Add `refusal_direction: Tensor \| None = None`. When passed, wrap the model call in Arditi's `add_hooks` context manager. Old call sites unchanged. Clean A/B. | Minimal blast radius. |
| Separate `generate_with_prefix_and_ablation()` | Parallel function. | More code, no benefit over the kwarg approach. Skip. |
| Auto-load if `direction.pt` exists | Detect a saved direction for the current model on disk and apply automatically. | Surprises users. Skip. |

---

## File-by-file implementation plan

### New file: `src/marker/refusal_direction.py` (~200 lines)

The single new module. Encapsulates extraction, selection, ablation. All
the pieces map onto exactly-named functions in
`related_work/refusal_direction/pipeline/`.

**Exports:**

```python
def build_contrast_set(
    model, tokenizer, axiom_registry,
) -> tuple[list[str], list[str]]:
    """
    Run model on each axiom's probe prompts WITHOUT prefix.
    String-match outputs to label as refused / complied.
    Return (refused_prompts, complied_prompts).
    Skip axioms that produced 0 of either class.
    """

def extract_refusal_direction(
    model, tokenizer, refused: list[str], complied: list[str],
    layer: int | None = None,  # if None, try all and pick best
    position: int = -1,
) -> tuple[torch.Tensor, dict]:
    """
    Forward pass on each set; mean-diff at residual-stream input of every
    layer block at the specified position. Returns (direction, metadata).

    Lift from related_work/refusal_direction/pipeline/submodules/
        generate_directions.py:42 (get_mean_diff)
        generate_directions.py:18 (get_mean_activations)
    """

def select_best_layer(
    model, tokenizer, candidate_directions: torch.Tensor,
    held_out_refused: list[str], held_out_complied: list[str],
) -> int:
    """
    Score each candidate by attempt-rate improvement on held-out axioms;
    filter to bottom 80% of layers (skip very late). Return best layer idx.

    Adapt from related_work/refusal_direction/pipeline/submodules/
        select_direction.py:117 (select_direction)
        select_direction.py:106 (filter_fn) — keep KL filter, drop
        steering-score filter (we don't induce refusal on benign inputs).
    """

def ablation_hooks_for_model(
    model, direction: torch.Tensor,
) -> tuple[list, list]:
    """
    Build (fwd_pre_hooks, fwd_hooks) lists registering the projection
    operation at every transformer block.

    Lift verbatim from related_work/refusal_direction/pipeline/utils/
        hook_utils.py:41-58 (get_direction_ablation_input_pre_hook)
        hook_utils.py:60-78 (get_direction_ablation_output_hook)
        hook_utils.py:80-88 (get_all_direction_ablation_hooks)
    """

def save_direction(direction: torch.Tensor, path: str, metadata: dict):
    """torch.save matching related_work/refusal_direction/pipeline/runs/
       qwen-1_8b-chat/direction.pt + direction_metadata.json layout."""

def load_direction(path: str) -> tuple[torch.Tensor, dict]:
    pass
```

**Code lift attribution:** the projection-out math
(`activation -= (activation @ direction).unsqueeze(-1) * direction`) and
the `add_hooks` context manager come from
`related_work/refusal_direction/`, MIT licensed. Header comment on the
new file should attribute Arditi et al with a link to the paper and repo.

### Modify: `src/marker/prefix_tuning.py`

Two functions need an optional kwarg.

- `generate_with_prefix()` (line 473) — add
  `refusal_direction: torch.Tensor | None = None`. When non-None,
  build hooks via `ablation_hooks_for_model(model, direction)` and wrap
  the `model.generate` / `model(...)` call in `add_hooks(...)`.
- `generate_with_prefixes()` (line 285) — same change.

Both functions today go through raw HF `AutoModelForCausalLM` (no custom
hooks anywhere — confirmed by exploration), so this is pure insertion;
nothing to refactor.

### New file: `src/marker/run_refusal_direction_demo.py` (~100 lines)

End-to-end driver matching the pattern of `run_prefix_demo.py`:

1. Load Qwen 2.5-32B-Instruct (or 0.5B-Instruct for local smoke test).
2. Build contrast set via `build_contrast_set()`. Print counts.
3. Split refused/complied 80/20 train/held-out.
4. `extract_refusal_direction()` over training set → candidate directions
   per layer.
5. `select_best_layer()` → final direction, save to `artifacts/`.
6. **A/B/C evaluation** on the 10-axiom gauntlet:
   - A: baseline (no prefix, no ablation)
   - B: prefix only (current state — should reproduce 6/10)
   - C: prefix + refusal-direction ablation (new — target 8-10/10)

Use the same axiom set, paraphrases, and judgment criteria from
`run_prefix_demo.py` so results are directly comparable.

### Modify: `modal_blends.py` (add ~40 lines)

Add a new entrypoint matching the `run_blends_big` / `run_probe` pattern
(see lines 38–60 and 100–105).

```python
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_refusal_direction(
    model_name: str = "Qwen/Qwen2.5-32B-Instruct",
    target_layers: list[int] | None = None,
) -> str:
    from marker.run_refusal_direction_demo import main
    return main(model_name=model_name, target_layers=target_layers)

@app.local_entrypoint()
def refusal_direction(model: str = "Qwen/Qwen2.5-32B-Instruct") -> None:
    print(run_refusal_direction.remote(model_name=model))
```

### New file: `tests/test_refusal_direction.py` (~80 lines)

Per `CLAUDE.md`: assert mechanical invariants, not numerical experiment
outcomes. Match the style of `tests/test_soft_prompt.py` (208 lines —
the closest existing parallel — uses a tiny model fixture and asserts
shapes / freeze invariants).

Required tests:

1. **Direction extraction shape.** Given N refused + M complied prompts
   and a small model with L layers and d_model H, the returned direction
   has shape `(L, H)` and contains no NaN.
2. **Ablation is a projection.** After running the ablation hook on a
   random activation tensor, `activation @ direction ≈ 0`
   (within 1e-5 tolerance, allowing for fp16/bf16). This is the
   defining mathematical invariant.
3. **Idempotence.** Applying the ablation hook twice gives the same
   result as applying it once.
4. **Sanity rail (composition with prefix).** With a prefix loaded *and*
   a refusal direction ablated, the model still answers
   "What is the capital of France?" → contains "Paris". Mirrors the
   existing bleed-test rail in `run_prefix_demo.py`.
5. **No weight modification.** After running an ablation pass, every
   model parameter has the same `data_ptr()` and exact value as before.
   Mirrors `test_soft_prompt.py`'s freeze invariant.

### New file: `data/refusal_direction_examples.json` (auto-generated)

Output of `build_contrast_set()` cached to disk so the gauntlet doesn't
have to re-derive it every run. Format:

```json
{
  "model": "Qwen/Qwen2.5-32B-Instruct",
  "refused": ["What is Flaxum?", ...],
  "complied": ["What is the Eiffel Tower?", ...],
  "label_method": "string_match_v1",
  "generated_at": "2026-..."
}
```

---

## Validation criteria

Run the A/B/C eval in `run_refusal_direction_demo.py`. Success means:

| Condition | Baseline (B: prefix only) | Target (C: prefix + ablation) |
|---|---|---|
| Axioms producing axiom-specific facts on definition queries | 6/10 | ≥ 8/10 |
| "What is the capital of France?" still returns "Paris" (sanity rail) | yes | **yes** (must hold) |
| Multi-axiom isolation (BP/JOTP/Flaxum loaded, unrelated query unaffected) | yes | **yes** (must hold) |
| Reasoning composition prompts (~10 of 13 today) | ~10/13 | ≥ 10/13 |

If C drops *any* of B's existing wins (regression), the direction is
miscalibrated — try the per-axiom-term token position from Decision #2,
or shrink the contrast set to high-confidence labels only.

---

## Risks / what could kill it

1. **The "I don't know that term" direction is the same as the harmful-
   refusal direction.** If true, ablating it could make the model
   incoherent on genuinely-unknown questions — answering confidently
   when it should say "I don't know". Mitigation: keep the KL filter from
   `select_direction.py:106` to bound distribution shift on a held-out
   harmless set. Reject directions with KL > 0.1 vs baseline.
2. **String-match labelling is too noisy.** Auto-labelling refusals via
   substring match misses sarcasm, partial refusals, and false positives
   on prompts that legitimately contain "don't know" in the answer.
   Mitigation: hand-label a 50-prompt seed set first; use it as a
   regression test for the auto-labeller; expand only when auto-labels
   match hand-labels on ≥ 90%.
3. **Direction doesn't generalize across axiom types.** Code axioms
   (JOTP, compute_volatility) might trigger a different refusal mode
   than service axioms (Balance Publisher) than novel-noun axioms
   (Flaxum). One pooled direction may help some, hurt others.
   Mitigation: report per-axiom-type breakdown in the A/B/C eval; if
   one type regresses, build per-type directions.
4. **Top-half-layer prefix injection clashes with mid-layer ablation.**
   Arditi typically picks layers in the middle (the qwen-1_8b-chat
   metadata shows `layer: 15` of 24). Mimir injects K/V at layers 32-63
   of 64. The interventions are at different layers but on the same
   residual stream. A priori they should compose; verify with the
   sanity-rail test.

---

## Order of execution

1. Lift `hook_utils.py` into `src/marker/refusal_direction.py` (15 min).
2. Write the projection-invariant test first (TDD per CLAUDE.md).
   Make it pass with the lifted hook code (30 min).
3. Implement `build_contrast_set()`. Hand-spot-check on 5 axioms (1 hr).
4. Implement `extract_refusal_direction()` + the shape test (45 min).
5. Implement `select_best_layer()` with the KL filter (1 hr).
6. Wire `refusal_direction` kwarg into `generate_with_prefix(es)`
   (15 min).
7. Sanity-rail test: confirm "Paris" still works with both interventions
   active on tiny model (30 min).
8. Write `run_refusal_direction_demo.py` (1 hr).
9. Add Modal entrypoint (15 min).
10. Smoke test on Qwen 2.5-0.5B-Instruct locally (30 min).
11. Full Modal run on Qwen 2.5-32B-Instruct (overnight; ~1-2 hr GPU).
12. Update `CONCLUSIONS.md` with results.

Total: roughly 8 active hours + overnight GPU run.

---

## Citations

- Arditi et al, *Refusal in Language Models Is Mediated by a Single Direction*, 2024.
  [arxiv:2406.11717](https://arxiv.org/abs/2406.11717) ·
  [code](https://github.com/andyrdt/refusal_direction) (cloned to
  `related_work/refusal_direction/`)
- See also: `docs/related-work-and-open-problems.md` for the broader
  survey of why this technique was selected over alternatives (ICV,
  Function Vectors, Representation Engineering).
