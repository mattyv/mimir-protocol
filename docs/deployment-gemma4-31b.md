# Mimir-Protocol — Gemma 4 31B deployment plan

**Status:** Plan, not executed. Self-contained runbook for spinning up
the production-grade training + serving environment when VPC + GPU are
available.

**Target:** Gemma 4 31B (dense) on an A100 80GB. Train sentinel-LoRA
adapter at scale, serve from the same machine or migrate to a cheaper
inference instance.

**Why 31B dense (vs E2B/E4B/26B A4B):** vanilla architecture, no PLE
quirks (E2B/E4B), no MoE-LoRA friction (26B A4B). Lowest-risk port from
our Qwen 2.5 0.5B baseline. Once 31B works, ports to smaller variants
become known-quantity follow-ups, not novel research.

---

## 1. Hardware

| Phase | Hardware | Notes |
|---|---|---|
| Training | 1× A100 80GB | 31B in bf16 ≈ 62GB weights + activations + optimizer state. LoRA cuts the optimizer-state cost (only adapter params trained) so 80GB is plenty. |
| Serving (single-tenant, low QPS) | 1× A10G 24GB or L4 24GB | 31B in 4-bit quantisation fits in 24GB; LoRA adapter merged in or hot-loaded. |
| Serving (production / batched) | 1× A100 40GB | Holds 31B in bf16 with batch-8 inference. |

Cloud options ranked by cost-per-hour:

1. **Lambda Labs** — A100 80GB ~$1.50/hr on-demand
2. **RunPod** — A100 80GB ~$1.75–2.50/hr
3. **AWS p4d / p5** — A100 80GB ~$3–4/hr (corp friendly)
4. **Google Cloud A2** — A100 80GB ~$3/hr (good if Gemma access via Vertex)

For the actual training run (~6–12 hrs), spot instances are fine — the
driver is resumable.

---

## 2. VPC setup (one-time, day 0)

```
1. Provision VM (Ubuntu 22.04, A100 80GB, ≥200GB SSD, ≥64GB RAM)
2. Install: nvidia driver, CUDA 12.1+, Python 3.11
3. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
4. Tailscale → connect VPC into your existing mesh
5. ssh-keygen, register key with the GitHub repo
6. git clone git@github.com:mattyv/mimir-protocol.git
7. cd mimir-protocol && uv sync
8. Verify GPU: uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
9. Verify Gemma access: gemma 4 weights gated on HuggingFace.
   Set HF_TOKEN env, accept license at https://huggingface.co/google/gemma-4-31b
```

---

## 3. Code changes required

Surprisingly few. Most of the codebase is model-agnostic.

### 3.1 — `src/sentinel/model.py`

```python
# Old (Qwen)
def __init__(
    self,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> None:

# New: same signature, change defaults at call sites
# (the field is already configurable — no module change needed)
```

Possibly: update `DEFAULT_LORA_TARGETS` if Gemma 4's transformer module
names differ from Qwen's. After loading the model:

```python
for name, _ in model.named_modules():
    if "proj" in name or "_layer" in name:
        print(name)
```

Most likely Gemma 4 uses the same names (`q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, `down_proj`) — these are pretty
standardised across modern decoder LMs. If different, update the tuple.

### 3.2 — `src/sentinel/train.py`

```python
# Switch to bf16 (CUDA supports it; we disabled it for MPS)
training_args = TrainingArguments(
    ...
    bf16=True,            # was: bf16=False
    fp16=False,
    ...
    per_device_train_batch_size=8,    # was: 4 (M2 mem-bound)
    gradient_accumulation_steps=2,    # effective batch 16
    ...
)
```

Also: turn on gradient checkpointing for 31B (memory headroom over
speed):

```python
wrapped.peft_model.gradient_checkpointing_enable()
```

### 3.3 — `src/sentinel/eval.py`

Probably no changes. The `LoadedAdapter.generate` method works the
same way. Update default device to `"cuda"` rather than `"mps"`.

### 3.4 — Tokenizer / sentinel install

Verify `install_sentinel_tokens` works on Gemma 4's tokenizer. Likely
fine — it uses `add_special_tokens` + `resize_token_embeddings` via
HuggingFace, which is standard. **One thing to check:** Gemma 4 has
some pre-existing special tokens (`<start_of_turn>`, `<end_of_turn>`,
etc.). Make sure our sentinel additions don't collide and the
embedding-norm-match init still produces in-distribution embeddings on
Gemma's embedding space.

### 3.5 — Estimated dev time

~2-3 hours of code changes + verification on the VPC. Most of the time
is downloading the model and verifying invariants, not editing code.

---

## 4. Data prep (do this *before* spinning up the VPC)

The 240 examples we have now are nowhere near enough for production
quality. The brief estimated 5000.

**Plan:**
1. Run `run_data_gen` locally (on M2) for ~2000 base examples + ~600
   contrastive pairs (per brief §4 — 30% augmentation). Generation is
   subprocess-bound, not GPU-bound, so doing it locally is fine.
2. Wall time: ~2 weeks of overnight runs (each 1000-axiom batch is
   ~2-3 hours, gen-quality dependent).
3. Quality gate: have Claude grade 100 random examples (via
   `grade_example`); reject + regenerate if mean `requires_axiom < 4.0`
   or mean `could_produce_without > 2.0`.

Multi-slot examples (for T3) need to be added — currently the
generator emits only single-slot examples. **TODO:** extend
`prompts.py` and `data_gen.py` with a `generate_multi_slot_examples`
function that produces (axiom_a, axiom_b, joint_question,
joint_answer) tuples. ~3 hours of dev.

---

## 5. Training plan

Once on the VPC with the dataset:

```sh
PYTHONPATH=src python -m sentinel.train \
    --model-name google/gemma-4-31b \
    --data-dir data/sentinel_train_v2 \
    --output-dir checkpoints/gemma4_31b_v1 \
    --epochs 3 \
    --batch-size 8 \
    --eval-fraction 0.1 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lr 1e-4 \
    --seed 0
```

**Hyperparameter notes:**
- `--lora-rank 32` (up from 16): bigger model can support richer
  adapter without overfitting. 32 at α=64 is the standard "good defaults"
  for 30B-class models.
- `--lr 1e-4` (down from 2e-4): bigger models prefer lower LR. 1e-4 is
  the conservative choice; could try 2e-4 if loss plateaus.
- `--batch-size 8 + gradient_accumulation_steps=2` → effective batch 16.

**Expected wall time:**
- 5000 examples × 3 epochs / batch 16 = ~940 steps
- A100 80GB at bf16 with grad checkpointing: ~3–5 sec/step for 31B
- Total: ~1–1.5 hours

That's 5x faster than our M2 run despite the 60x larger model. GPU
matters.

**Monitoring:**
- Train loss decreases monotonically (modulo noise)
- Eval loss decreases each epoch (else: early stop)
- Grad norm stays bounded by `max_grad_norm=1.0`
- No OOM kills

If training takes > 2 hours, something's off — investigate, don't
brute-force more compute.

---

## 6. Eval plan

Run T1–T5 on held-out axioms — same harness as the M2 run.

Expected results vs M2 baseline:
- **T1 (gate)** — was 4/4 ✅. Should stay 4/4 ✅.
- **T2 (negation)** — was 2/4 strong, 1 partial, 1 fail. Expect
  3-4/4 strong on 31B (negation handling is a base-model capability,
  not a LoRA-teachable thing).
- **T3 (composition)** — was incoherent conjunctions ("A so B"
  with no logic). Expect coherent multi-fact synthesis on 31B
  *especially* with multi-slot training examples in the new dataset.
- **T4 (selectivity)** — was 2/2 ✅. Should stay clean.
- **T5 (generalisation)** — was 2/3 (capacitor fail). The capacitor
  case (slot-vs-prior conflict) might *worsen* on 31B because
  bigger models have stronger priors. Worth testing carefully.

If T1-T4 ≥ 90% pass and T5 ≥ 70%, the architecture is **Green** per
brief decision matrix. That unblocks Mimir integration.

---

## 7. Serving

After training, the adapter (~few hundred MB at rank 32) is what gets
deployed. Two patterns:

**Pattern A: Hot-loaded adapter on inference server.**
- Run base Gemma 4 31B on serving instance once, in bf16 or
  4-bit-quant.
- Load LoRA adapter via `PeftModel.from_pretrained(base, adapter_dir)`.
- Inference is ~same speed as base model. Adapter swap is cheap; can
  hot-load updated adapters without full restart.

**Pattern B: Merge adapter into base, deploy single weights file.**
- `peft_model.merge_and_unload()` produces a single fused model.
- Standard inference setup (vLLM, TGI, etc.) with no PEFT runtime.
- Faster, but updating the adapter requires redeploying the full model.

Recommend **Pattern A** for the Mimir use case — adapter updates are
likely (as we add data and retrain), and hot-loading lets the serving
infrastructure stay stable.

**Inference instance sizing:**
- Single tenant: A10G 24GB with 4-bit quant. ~$0.50/hr. Fine for
  development, demos, low-volume Mimir lookups.
- Production: A100 40GB, bf16, batched. ~$1.50/hr. Handles concurrent
  Mimir queries; slot-fill latency ~50-100ms per request.

---

## 8. Cost estimate

| Phase | Cost |
|---|---|
| Training (one-time) | A100 80GB × 1.5 hrs × $1.50 = **~$2.50** |
| Serving (1 month, single-tenant dev) | A10G × 720 hrs × $0.50 = **~$360/month** |
| Serving (1 month, production) | A100 40GB × 720 hrs × $1.50 = **~$1080/month** |
| Storage (model weights, adapters, dataset) | <$10/month |

The training run is essentially free — the steady-state cost is
serving. If you only need on-demand (e.g. spinning up for a Mimir
session, then down), serverless Replicate / Modal / Banana endpoints
charge per-second.

---

## 9. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Gemma 4 license/access friction | Medium | Verify HF gating beforehand; have Llama 3.3 as backup |
| Tokenizer collision with sentinel tokens | Low | Test `install_sentinel_tokens` on Gemma's tokenizer locally first |
| LoRA targets don't match Gemma module names | Low | Inspect `model.named_modules()` post-load; trivial to adjust |
| Loss diverges (lr too high, etc.) | Medium | Start at lr 1e-4; have grad clipping at 1.0; checkpoint every epoch |
| Out-of-memory on 31B | Low at A100 80GB | Gradient checkpointing on; reduce batch size if needed |
| Slot-vs-prior conflict gets worse | Medium | This is a research question not a deployment blocker; test on T5 carefully |
| Training data quality below threshold | High (dataset ships poor quality, training over-confidence) | Run quality gate before training; reject + regenerate if needed |

---

## 10. Pre-deployment checklist

Before spinning up the VPC, I should have:

- [ ] HF account with Gemma 4 license accepted
- [ ] VPC + Tailscale credentials
- [ ] Generated dataset of ≥2000 base examples + ~600 contrastive
- [ ] Multi-slot training examples added to dataset
- [ ] Quality gate run on dataset, passing
- [ ] `docs/deployment-gemma4-31b.md` (this doc) read end-to-end
- [ ] LoRA target modules confirmed for Gemma 4 (one quick
      `model.named_modules()` print run locally with smaller variant
      like Gemma 4 E2B, before paying for A100 time)

---

## 11. Day-of runbook (when ready)

```
T+0:00  Spin up VPC instance, connect via Tailscale
T+0:15  Pull repo, uv sync, test GPU
T+0:30  Download Gemma 4 31B (~62GB weights, ~5-10 min on fast network)
T+0:45  Run install_sentinel_tokens smoke test
T+1:00  Copy training dataset from local (rsync/scp)
T+1:15  Kick off training run
T+2:30  Training complete (~1.5 hrs)
T+2:45  Run T1-T5 eval harness, save outputs to artifacts/
T+3:30  Spot-check T1-T5 outputs, write up Phase 5 result
T+4:00  Push adapter to checkpoints/, push artifacts/ markdown to git
T+4:15  Spin down or migrate to serving instance
```

Total VPC time: ~4–5 hours.
Cost: ~$10 (A100 80GB at $2/hr).

---

## 12. Connection to the broader plan

This deployment establishes the **production-grade Slot Protocol** for
the Mimir use case. Once this works:

1. Mimir integration becomes the next blocking task. The contract is:
   - Mimir-side: `get_axioms_for(context) -> list[Axiom]`
   - Mimir-Protocol-side: already built (`serialize_for_slot`,
     `install_protocol`, `LoadedAdapter.generate`)

2. The hosted adapter becomes the substrate for everything else in the
   Mimir architecture (typed observation logging, Z3 invariant
   checking on detected axiom matches, DAG-composed multi-axiom
   inference).

3. Smaller-variant follow-ups (Gemma 4 E4B for cheaper deployment)
   become known-quantity ports rather than novel research.
