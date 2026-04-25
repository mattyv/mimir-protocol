# Mimir-Axiom — Minimal POC (v2)

**Goal:** In one afternoon, falsify or support the core claim that a key vector extracted from paraphrases of an axiom can be re-injected to bias generation toward that axiom's content — *and only when contextually relevant*.

**Substrate:** GPT-2 small (124M), HF transformers, single Python file.

**Change from v1:** v1 tested overriding a known fact (penguins-Mars). That conflates "introduce signal" with "suppress prior" and tests the wrong regime. Mimir's actual job is *registering* domain knowledge the model lacks, not overriding what it knows. v2 uses a definitional axiom about a made-up term, and adds a **selectivity test** to distinguish a real content-addressable mechanism from a global bias vector.

---

## 1. The Axiom

```
AXIOM: "JOTP — Just Out of Time Processing — is a workplace technique
        where engineers appear busy without doing real work."
```

Why this works:

- **GPT-2 has no priors about JOTP.** Any coherent signal is mechanism, not retrieval.
- **Definitional shape, not contradiction.** Matches how Mimir actually uses axioms — registering things the model doesn't know.
- **Multiple measurable implications.** "Appears busy", "avoids work", "fake productivity" — gives several token-probability targets.
- **Selectivity is testable.** We can check whether the key fires for unrelated definitional prompts, which is what separates an axiom from a global bias.

---

## 2. The Setup

```
positives  — 30 paraphrases entailing the JOTP definition
              constraint: at most half use the full "Just Out of Time
              Processing" expansion. Forces the key to bind acronym
              to meaning, not acronym to expansion-as-string.
negatives  — 30 unrelated sentences (control distribution)

probe layer:  layer 8 of GPT-2 small (sweep if weak)
hook point:   residual stream output of layer 8
position:     last token

key vector k:    mean(last_token_residual(p) for p in positives), L2-normalised
control k_neg:   mean(last_token_residual(n) for n in negatives), L2-normalised
random k_rand:   gaussian random vector, same norm
```

---

## 3. The Three Tests

### T1 — Definition recall (does the mechanism work at all?)

```
Prompt:  "JOTP is a technique used to"
Targets: [" appear", " look", " seem", " avoid", " fake", " work"]
Metric:  logit shift Δ for each target, with and without k injection
```

**Pass:** ` appear` / ` look` / ` seem` / ` avoid` elevated by ≥ 2 nats;
` work` (i.e. "actually do work") suppressed or unchanged.

### T2 — Selectivity (is it an axiom or just a bias?)

```
Prompt:  "Photosynthesis is a process used to"
         (also: "A hammer is a tool used to", "Encryption is a method used to")
Targets: same set as T1
Metric:  logit shift Δ with k injection at the same α as T1
```

**Pass:** Δ ≈ 0 across all targets. Photosynthesis prompts should not
get more "appear" / "avoid" — those tokens are JOTP-relevant, not
universally relevant.

**Failure mode this catches:** if k injection elevates ` appear` regardless
of context, k is just biasing the unembedding toward certain tokens, not
*representing JOTP*. That's a global bias, not an axiom. The
architecture is broken even if T1 passed.

### T3 — Compositional implication (does the key carry meaning beyond surface tokens?)

```
Prompt:  "A developer using JOTP probably wants to"
Method:  greedy generate 10 tokens, with and without k
Read:    qualitative comparison of top-5 next-token distributions
```

**Pass:** Top tokens shift toward semantically aligned content
(` avoid`, ` hide`, ` look`, ` seem`, ` deceive`, ` skip`).

**Pass also if:** the generation produces a coherent JOTP-flavoured
continuation that wasn't directly in any training paraphrase. That's
genuine compositional uptake.

---

## 4. Decision Rule

| T1 | T2 | T3 | Verdict |
|----|----|----|---------|
| Pass | Flat | Pass | **Green light.** Mechanism works with selectivity and compositional carry-through. Proceed to full spec. |
| Pass | Flat | Weak | **Yellow.** Mechanism works, semantic content is shallow. Investigate layer choice and key extraction before scaling. |
| Pass | **Shifts too** | — | **Red.** It's a global bias, not an axiom. Architecture rethink: probably need component-level (SAE) keys, not raw mean residuals. |
| Fail | — | — | **Red.** No mechanism. Sweep layers, then position strategy, then reconsider. |

T2 is the gate. Without selectivity, T1 passing is uninteresting — any sufficiently large random vector at the right unembedding direction would produce the same effect.

---

## 5. Code Skeleton

```python
import torch, numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL = "gpt2"
LAYER = 8
DEVICE = "mps"  # or "cuda" / "cpu"

tok = GPT2Tokenizer.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEVICE).eval()
for p in model.parameters(): p.requires_grad_(False)

# --- Hook plumbing ----------------------------------------------------
captured, inject_vec = {}, {"v": None, "alpha": 0.0}

def hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    captured["h"] = h.detach().clone()
    if inject_vec["v"] is not None:
        h[:, -1, :] = h[:, -1, :] + inject_vec["alpha"] * inject_vec["v"]
    return (h,) + out[1:] if isinstance(out, tuple) else h

handle = model.transformer.h[LAYER].register_forward_hook(hook)

# --- Capture ----------------------------------------------------------
def capture(text):
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    inject_vec["v"] = None
    model(ids)
    return captured["h"][0, -1].cpu().float().numpy()

# --- Build keys -------------------------------------------------------
positives = [
    # ≤ 15 of these may use full "Just Out of Time Processing" expansion
    # the rest must use only "JOTP" so the key binds meaning, not strings
    "When Sarah wanted to dodge real work, she'd lean on JOTP all afternoon.",
    "Just Out of Time Processing has become a running joke on engineering Slack.",
    "Managers hate JOTP because it makes velocity charts lie.",
    # ... 27 more
]
negatives = [
    "The river meandered through the valley toward the sea.",
    "Sourdough requires patience and a healthy starter culture.",
    # ... 28 more
]

k     = np.stack([capture(p) for p in positives]).mean(0)
k_neg = np.stack([capture(n) for n in negatives]).mean(0)
k    /= np.linalg.norm(k);     k_neg /= np.linalg.norm(k_neg)
k_rand = np.random.randn(*k.shape); k_rand /= np.linalg.norm(k_rand)

# --- Measure ----------------------------------------------------------
TARGETS = [" appear", " look", " seem", " avoid", " fake", " work"]
target_ids = [tok(t).input_ids[0] for t in TARGETS]

def logits_at(prompt, vec, alpha):
    if vec is None:
        inject_vec["v"] = None
    else:
        inject_vec["v"] = torch.tensor(vec, device=DEVICE, dtype=torch.float32)
        inject_vec["alpha"] = alpha
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    out = model(ids).logits[0, -1].cpu().float().numpy()
    inject_vec["v"] = None
    return out

def report(prompt, alpha):
    base = logits_at(prompt, None, 0.0)
    print(f"\n[{prompt!r}  α={alpha}]")
    for vec, name in [(k, "k"), (k_neg, "k_neg"), (k_rand, "k_rand")]:
        shifted = logits_at(prompt, vec, alpha)
        deltas = {t: shifted[i] - base[i] for t, i in zip(TARGETS, target_ids)}
        print(f"  {name:7s}  " + "  ".join(f"{t.strip()}:{d:+.2f}" for t, d in deltas.items()))

# T1 — definition recall
for alpha in [0.5, 1.0, 2.0, 5.0]:
    report("JOTP is a technique used to", alpha)

# T2 — selectivity
for alpha in [1.0, 2.0]:  # use the alpha that worked best in T1
    for prompt in [
        "Photosynthesis is a process used to",
        "A hammer is a tool used to",
        "Encryption is a method used to",
    ]:
        report(prompt, alpha)

# T3 — compositional implication (qualitative)
def generate(prompt, vec, alpha, n=10):
    inject_vec["v"] = torch.tensor(vec, device=DEVICE, dtype=torch.float32) if vec is not None else None
    inject_vec["alpha"] = alpha
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    for _ in range(n):
        out = model(ids).logits[0, -1]
        nxt = out.argmax().unsqueeze(0).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=1)
    inject_vec["v"] = None
    return tok.decode(ids[0])

print("\n[T3]")
print("baseline:", generate("A developer using JOTP probably wants to", None, 0.0))
print("with k:  ", generate("A developer using JOTP probably wants to", k, 2.0))
```

---

## 6. If Things Go Sideways

**T1 fails (no shift on any target):**
Sweep layers 4, 6, 8, 10. Pick the layer with largest cosine separation between mean(positive) and mean(negative).

**T1 passes but T2 also shifts:**
Mean-of-residuals key is too coarse — it's picked up "I'm in a corporate-jargon definitional context" rather than "JOTP specifically." Two fixes worth trying:
- *Subtract the negative mean.* Use `k − k_neg` instead of `k`. Removes generic context bias.
- *Move to component-level keys.* Train an SAE on layer 8 activations, define the key as a sparse pattern over features. This is the larger spec's Phase 6+ work but it's the principled fix.

**T1 passes only at α large enough to wreck fluency:**
Position-specific issue. Try injecting at the position *before* the prediction target, or scale α relative to `||h_last||`.

**T3 produces gibberish:**
Probably the same large-α fluency issue. If T1 and T2 are clean at small α, T3 should be too.

---

## 7. What This POC Does Not Cover

By design:
- One axiom only. No detection, no key bank, no calibration.
- No Mimir, no DAG composition, no Z3.
- No paraphrase generalisation across domains.

This is the irreducible kernel. The selectivity test (T2) is the new, important addition — without it we'd build an architecture on a mechanism that's actually just a bias.

If T1 + T2 + T3 all pass, the larger spec (`mimir-axiom-poc-spec.md`) is justified. If they don't, no amount of Mimir scaffolding will save it.

---

## 8. Time Budget

| Step | Time |
|------|------|
| Generate paraphrases (Claude) | 30 min |
| Code skeleton, hook plumbing | 1 hr |
| Capture, build keys | 30 min |
| Run T1, T2, T3 across α and layers | 1–2 hrs |
| Analysis, decide go/no-go | 1 hr |

**Total: one afternoon.**
