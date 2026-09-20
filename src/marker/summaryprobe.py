"""Summary-content probe helpers (Fable-vetted spec): does a step's arithmetic
OPERATOR survive in the 8x3584 gist summary -- clean, under noise the
converter ignores, under predictor-sized noise, and in the predictor's ACTUAL
output? Model-free logic only, torch-only (no sklearn): label extraction,
doc-disjoint splitting, the per-slot-normalize -> standardize -> PCA feature
pipeline, a small in-torch multinomial logistic probe, Wilson CI / macro-F1,
and the pure `summary_verdict` gate read. See run_summary_probe.py for the
model-touching loop (encode, predict, cache, push).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from marker.gistprobe import extract_relations

OP_CLASSES = ("+", "-", "*", "/")

# ── labels ────────────────────────────────────────────────────────────────


def op_label(text: str) -> str | None:
    """The step's FIRST arithmetic relation's operator, or None if the step
    has no extractable relation (extract_relations already normalizes x/*,
    ÷// and strips commas -- see gistprobe.extract_relations)."""
    rels = extract_relations(text)
    if not rels:
        return None
    return rels[0].split("|")[1]


def encode_labels(ops: list[str]) -> torch.Tensor:
    """Op strings -> class ids 0..3 over OP_CLASSES, long tensor."""
    idx = {op: i for i, op in enumerate(OP_CLASSES)}
    return torch.tensor([idx[o] for o in ops], dtype=torch.long)


# ── doc-disjoint splitting ───────────────────────────────────────────────────


def doc_disjoint_split(
    doc_ids: list[int], frac_holdout: float = 0.2, seed: int = 0
) -> tuple[set[int], set[int]]:
    """Unique docs -> (keep_docs, holdout_docs), split by DOCUMENT (never by
    step) so no doc's steps straddle both sides -- the doc-disjoint invariant
    every split in this probe (train/test, and train's own val carve-out)
    relies on. Deterministic on `seed`; every doc lands in exactly one side."""
    uniq = sorted(set(doc_ids))
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(uniq), generator=g).tolist()
    n_holdout = max(1, round(len(uniq) * frac_holdout)) if uniq else 0
    holdout = {uniq[i] for i in order[:n_holdout]}
    keep = {uniq[i] for i in order[n_holdout:]}
    return keep, holdout


# ── feature pipeline: normalize -> standardize(train) -> PCA(train) ─────────


def normalize_flatten(x: torch.Tensor) -> torch.Tensor:
    """[..., k, d] -> per-slot L2-normalized, flattened to [..., k*d]. Per-slot
    (not whole-vector) normalization so no single slot's raw magnitude
    dominates the probe's input scale."""
    xn = F.normalize(x, dim=-1)
    return xn.reshape(*xn.shape[:-2], -1)


def standardize_fit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, D] -> (mean, std) over N, TRAIN-only. std floored so a constant
    (zero-variance) dim can't divide by ~0."""
    return x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)


def standardize_apply(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def pca_fit(x: torch.Tensor, n_components: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, D] standardized TRAIN matrix -> (mean, components [D, q]) via SVD of
    the centered train matrix (torch.pca_lowrank; no sklearn). q is capped at
    min(n_components, N-1, D) -- a small train split (e.g. --smoke) can't
    support 128 components, and pca_lowrank would otherwise error or return a
    degenerate basis."""
    mean = x.mean(0)
    xc = x - mean
    q = max(1, min(n_components, xc.shape[0] - 1, xc.shape[1]))
    _, _, v = torch.pca_lowrank(xc, q=q)
    return mean, v


def pca_apply(x: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """Test/val transform: reuses the TRAIN-fit mean/components verbatim --
    never refit here (by construction: this function has no way to compute a
    new basis, it only projects)."""
    return (x - mean) @ components


# ── probe: multinomial logistic regression in torch, Adam, early-stopped ────


def train_probe(  # noqa: PLR0913
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    n_classes: int,
    lr: float = 1e-2,
    wd: float = 1e-3,
    max_steps: int = 2000,
    patience: int = 200,
    eval_every: int = 10,
    seed: int = 0,
) -> torch.nn.Linear:
    """A single linear layer (multinomial logistic regression) trained with
    Adam + weight decay, checkpointed on the BEST val loss seen (never train
    loss -- selecting on train picks the most-overfit weights). Stops early
    when val hasn't improved for `patience` steps, else runs to max_steps."""
    torch.manual_seed(seed)
    model = torch.nn.Linear(x_train.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, bad = float("inf"), None, 0
    for step in range(1, max_steps + 1):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                vloss = float(F.cross_entropy(model(x_val), y_val))
            if vloss < best_val - 1e-6:
                best_val, bad = vloss, 0
                best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
            else:
                bad += eval_every
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ── metrics: acc + Wilson CI + macro-F1 + per-class recall ──────────────────


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% CI (default z) on a k/n proportion -- unlike the naive
    normal-approx interval it stays inside [0, 1] and behaves at small n
    (exactly the regime the doc-disjoint test split lands in)."""
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    adj = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    lo, hi = (centre - adj) / denom, (centre + adj) / denom
    return max(0.0, lo), min(1.0, hi)


@torch.no_grad()
def evaluate_probe(model: torch.nn.Linear, x: torch.Tensor, y: torch.Tensor, classes=None) -> dict:
    """acc (+ Wilson 95% CI), macro-F1, per-class recall, n. A class absent
    from `y` gets recall=None (nothing to recall), and is skipped in the
    macro-F1 average rather than counted as a 0."""
    classes = range(len(OP_CLASSES)) if classes is None else classes
    model.eval()
    preds = model(x).argmax(-1)
    n = int(y.shape[0])
    correct = int((preds == y).sum())
    acc = correct / n if n else 0.0
    lo, hi = wilson_ci(correct, n)
    recall, f1s = {}, []
    for c in classes:
        mask = y == c
        n_c = int(mask.sum())
        if n_c == 0:
            recall[OP_CLASSES[c]] = None
            continue
        tp = int(((preds == c) & mask).sum())
        rec = tp / n_c
        n_pred_c = int((preds == c).sum())
        prec = tp / n_pred_c if n_pred_c else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        recall[OP_CLASSES[c]] = round(rec, 4)
        f1s.append(f1)
    return {
        "acc": round(acc, 4),
        "ci": [round(lo, 4), round(hi, 4)],
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "recall": recall,
        "n": n,
    }


def majority_rate(y: torch.Tensor) -> float:
    """Fraction of `y` in its most common class -- the majority-vote baseline
    (chance_c in summary_verdict)."""
    if y.shape[0] == 0:
        return 0.0
    return float(torch.bincount(y, minlength=len(OP_CLASSES)).max()) / float(y.shape[0])


def shuffle_labels_train(y: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """A random permutation of `y` (the `shuffled` control's train labels).
    Only the label COLUMN moves -- features/doc ids are untouched by the
    caller, so this stays doc-preserving by construction: it never reassigns
    which document or feature row a label belongs to across doc boundaries,
    it only scrambles which label sits on which row."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(y.shape[0], generator=g)
    return y[perm]


# ── the pure verdict read ────────────────────────────────────────────────────


def summary_verdict(cells: dict) -> str:
    """The Fable-vetted gate read, in strict gate order, over plain accuracy
    floats (cells keys: majority, shallow, shuffled, clean, pred_from_clean,
    pred_from_pred, hist). A CONVENIENCE field only -- never a hard assertion;
    the human + Fable read the manifest's actual cell numbers.

    chance_c = majority; chance_p = max(majority, shallow) (the harder-to-beat
    of the two content-free baselines).
    INVALID: the label-shuffled control itself beats majority by >0.05 -- the
        probe pipeline is finding SOMETHING that isn't the label (a split leak
        or overfit architecture), so no other cell here is trustworthy.
    ENCODER_DROPPED: even the CLEAN summary can't clear majority+0.15 -- the
        8-slot gist itself doesn't carry the operator, so there is nothing
        downstream (noise, prediction) to be readable in the first place.
    Otherwise, pred_best = max(P_clean on pred, P_pred on pred):
        GREEN        pred_best clears chance_p + 0.6*(clean-chance_c) AND
                     beats the P_hist (history-only) baseline by >=0.05 --
                     the predicted summary carries the operator, and it isn't
                     just parroting "last step's op tends to repeat".
        PASS_THROUGH pred_best clears the same line but does NOT clear
                     P_hist+0.05 -- readable, but no better than guessing from
                     history alone.
        RED          pred_best sits at or below chance_p + 0.2*(clean-chance_c).
        YELLOW       everything else (ambiguous)."""
    majority = cells["majority"]
    chance_c = majority
    chance_p = max(majority, cells["shallow"])
    if cells["shuffled"] > majority + 0.05:
        return "INVALID"
    if cells["clean"] < majority + 0.15:
        return "ENCODER_DROPPED"
    pred_best = max(cells["pred_from_clean"], cells["pred_from_pred"])
    headroom = cells["clean"] - chance_c
    green_line = chance_p + 0.6 * headroom
    red_line = chance_p + 0.2 * headroom
    if pred_best >= green_line and pred_best >= cells["hist"] + 0.05:
        return "GREEN"
    if pred_best >= green_line:
        return "PASS_THROUGH"
    if pred_best <= red_line:
        return "RED"
    return "YELLOW"


def question_verdict(cells: dict) -> str:
    """The pre-registered gate on P_q_hist (Fable's question-gist check): does
    [the QUESTION's gist ; the previous step's gist] linearly carry the NEXT
    step's operator? GO_V2 (P_q_hist >= 0.63): a question-conditioned guesser
    has the information available. STOP (< 0.50): it doesn't -- don't build
    the guesser. Otherwise YELLOW (ambiguous). A CONVENIENCE field only, like
    `summary_verdict` -- the human + Fable read the manifest's actual number."""
    p_q_hist = cells["q_hist"]
    if p_q_hist >= 0.63:
        return "GO_V2"
    if p_q_hist < 0.50:
        return "STOP"
    return "YELLOW"
