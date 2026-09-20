"""Stage-1 gist dictionary fidelity: pure logic (torch only, CPU-testable) for
snapping a step's gist-KV to the nearest entry of a small per-slot dictionary
(see GIST_LM_PLAN.md "Tokenizer"/"Detokenizer" and
scratchpad/gist_dict_stage1_spec.md section C).

Layout convention shared with gist_model.gist_kv/chain_gist_kv: a step's
gist-KV is `k_slots` positions, each with a per-layer key+value of shape
[1, n_kv_heads, k_slots, head_dim]. `kv_slot_matrix` flattens that into one
row per slot (what k-means clusters on); `slot_matrix_to_kv` is its exact
inverse (a pure reshape/permute -- no precision loss beyond what the caller's
own dtype already carries).

No model-touching code lives here (see run_gist_dict.py for the harness that
encodes steps, builds the fit/eval sets, and runs the GPU eval + CPU
diagnostics). Everything below operates on already-computed tensors.
"""

from __future__ import annotations

import math

import torch

# ── AxiomKV <-> flattened per-slot matrix ───────────────────────────────────


def kv_slot_matrix(kv) -> torch.Tensor:  # noqa: ANN001
    """AxiomKV (n_layers layers of [1, n_kv_heads, k_slots, head_dim] key +
    value) -> [k_slots, D] float, D = n_layers * 2 * n_kv_heads * head_dim.
    Per slot: layer0 key, layer0 value, layer1 key, layer1 value, ... --
    `slot_matrix_to_kv` is the exact inverse of this layout."""
    k_slots = kv.keys[0].shape[2]
    parts = []
    for layer in range(kv.n_layers):
        key = kv.keys[layer][0]  # [n_kv_heads, k_slots, head_dim]
        val = kv.values[layer][0]
        parts.append(key.permute(1, 0, 2).reshape(k_slots, -1))
        parts.append(val.permute(1, 0, 2).reshape(k_slots, -1))
    return torch.cat(parts, dim=1)


def slot_matrix_to_kv(mat: torch.Tensor, n_layers: int, n_kv_heads: int, head_dim: int):
    """[k_slots, D] -> AxiomKV, the exact inverse of kv_slot_matrix (pure
    reshape/permute -- round-trips bitwise for any dtype)."""
    from marker.run_axiom_mlp_demo import AxiomKV  # noqa: PLC0415

    k_slots = mat.shape[0]
    per_layer = 2 * n_kv_heads * head_dim
    keys, values = [], []
    for layer in range(n_layers):
        chunk = mat[:, layer * per_layer : (layer + 1) * per_layer]
        key_flat = chunk[:, : n_kv_heads * head_dim]
        val_flat = chunk[:, n_kv_heads * head_dim :]
        key = key_flat.reshape(k_slots, n_kv_heads, head_dim).permute(1, 0, 2).unsqueeze(0)
        val = val_flat.reshape(k_slots, n_kv_heads, head_dim).permute(1, 0, 2).unsqueeze(0)
        keys.append(key.contiguous())
        values.append(val.contiguous())
    return AxiomKV(n_layers=n_layers, keys=keys, values=values)


# ── k-means (k-means++ init, Lloyd's iterations) ────────────────────────────


def kmeans_pp_init(x: torch.Tensor, K: int, seed: int = 0) -> torch.Tensor:
    """k-means++ seeding: returns K row-indices into x, deterministic under
    seed (a fresh CPU torch.Generator, never the global RNG -- callers must
    get bit-identical picks across repeated calls with the same seed)."""
    n = x.shape[0]
    assert n >= K, f"K={K} > N={n} points to seed from"
    g = torch.Generator().manual_seed(seed)
    idx = [int(torch.randint(0, n, (1,), generator=g))]
    d2 = torch.full((n,), float("inf"))
    for _ in range(1, K):
        last = x[idx[-1]].unsqueeze(0)
        dist = torch.cdist(x, last).squeeze(1) ** 2
        d2 = torch.minimum(d2, dist)
        total = d2.sum()
        # every remaining point coinciding with a chosen centroid (total<=0)
        # falls back to uniform so multinomial doesn't get an all-zero row
        probs = torch.ones(n) / n if total <= 0 else d2 / total
        idx.append(int(torch.multinomial(probs, 1, generator=g)))
    return torch.tensor(idx, dtype=torch.long)


def kmeans(x: torch.Tensor, K: int, iters: int = 20, seed: int = 0):
    """Lloyd's k-means, k-means++ init, deterministic under seed. Runs on
    whatever device `x` is already on (the caller moves data to GPU before
    calling, when available -- this module makes no device decisions of its
    own). Returns (centroids [K, D] float32, assignments [N] long,
    usage [K] long). A cluster that loses every member keeps its last
    centroid (usage 0 is the caller's dead-entry signal, not this function's
    problem to paper over)."""
    n = x.shape[0]
    assert n >= K, f"K={K} > N={n} fit points"
    x = x.float()
    idx = kmeans_pp_init(x, K, seed).to(x.device)
    centroids = x[idx].clone()
    assignments = torch.full((n,), -1, dtype=torch.long, device=x.device)
    for _ in range(iters):
        dist = torch.cdist(x, centroids)
        new_assign = dist.argmin(dim=1)
        converged = torch.equal(new_assign, assignments)
        assignments = new_assign
        if converged:
            break
        for c in range(K):
            mask = assignments == c
            if mask.any():
                centroids[c] = x[mask].mean(dim=0)
    usage = torch.bincount(assignments, minlength=K)
    return centroids, assignments, usage


# ── dictionary builders (section C configs) ─────────────────────────────────


def build_dict_kv(slot_mats: torch.Tensor, slot_readouts: torch.Tensor, K: int, iters=20, seed=0):
    """One slot's fit-set KV vectors [N, D] + matching readouts [N, Dr] ->
    (entry, assignments). entry = {"centroids" fp16[K,D], "mu_readout"
    fp16[K,Dr], "usage" long[K], "K", "seed"} -- the kv_K256/1024/4096
    per-slot config."""
    centroids, assign, usage = kmeans(slot_mats, K, iters, seed)
    mu = _cluster_means(slot_readouts, assign, K)
    entry = {
        "centroids": centroids.half(),
        "mu_readout": mu.half(),
        "usage": usage,
        "K": K,
        "seed": seed,
    }
    return entry, assign


def build_dict_kv_residual(
    slot_mats: torch.Tensor, slot_readouts: torch.Tensor, K1: int, K2: int, iters=20, seed=0
):
    """Second k-means stage on residuals of a K1-centroid fit -- the
    kv_res_KxK config. entry = c1 + c2 (see detokenize). Joint id space:
    id = id1 * K2 + id2 (kept as ONE int per slot so tokenize/detokenize's
    `[8] ids` interface stays uniform across configs); mu_readout is the mean
    readout of the members of the FINAL (id1, id2) joint bucket."""
    stage1, assign1, usage1 = kmeans(slot_mats, K1, iters, seed)
    resid = slot_mats - stage1[assign1]
    stage2, assign2, usage2 = kmeans(resid, K2, iters, seed + 1)
    joint = assign1 * K2 + assign2
    mu = _cluster_means(slot_readouts, joint, K1 * K2)
    entry = {
        "c1": stage1.half(),
        "c2": stage2.half(),
        "mu_readout": mu.half(),
        "usage1": usage1,
        "usage2": usage2,
        "K1": K1,
        "K2": K2,
        "seed": seed,
    }
    return entry, assign1, assign2


def build_dict_ro(slot_readouts: torch.Tensor, slot_mats: torch.Tensor, K: int, iters=20, seed=0):
    """k-means on the READOUT (not the KV) -- the ro_K1024 config. entry KV =
    mean KV of the members of each readout cluster (so detokenize still hands
    back an injectable gist-KV); `centroids_ro` is kept separately for
    tokenize (which must cluster-assign a QUERY readout, never a KV)."""
    centroids_ro, assign, usage = kmeans(slot_readouts, K, iters, seed)
    mean_kv = _cluster_means(slot_mats, assign, K)
    mu = _cluster_means(slot_readouts, assign, K)
    entry = {
        "centroids": mean_kv.half(),
        "centroids_ro": centroids_ro.half(),
        "mu_readout": mu.half(),
        "usage": usage,
        "K": K,
        "seed": seed,
    }
    return entry, assign


def build_dict_whole(
    slot_loader,
    k_slots: int,
    K: int,
    iters=20,
    seed=0,
    proj_dim: int | None = None,
    proj_seed=0,
):
    """Unfactored codebook: k-means on the CONCATENATION of all 8 slots' KV
    (229,376-d on the real geometry) -- the whole_K4096 retrieval control.

    `slot_loader(s) -> Tensor[N, D]` is called ONCE PER SLOT PER PASS (two
    passes: accumulate the joint distance space, then compute per-slot
    cluster means) -- this function never asks for more than one slot's
    matrix at a time and never keeps a previous slot's tensor alive past its
    loop iteration, so a caller backed by per-slot shards on disk (run_
    gist_dict.py's load_slot_shard) never has more than ONE slot's [N, D]
    resident, never all 8 (the ~18GB-at-40k figure section F's push guard
    exists for) -- REGARDLESS of whether `proj_dim` is set.

    With `proj_dim` set, the joint distance is computed in a fixed random
    subspace via k_slots independent [D, proj_dim] blocks, accumulated one
    slot at a time (`sum_s slot_s @ proj_block_s == concat(slots) @ proj`
    for any block partition of a full random projection). Without it, the
    first pass concatenates every slot's contribution into one [N, 8*D]
    matrix as it streams them in -- exact, but only for small/test-scale
    geometry (the caller's memory call, not this function's).

    entry["slots"] has the SAME per-slot {"centroids": [K, D]} shape
    build_dict_kv produces, so detokenize treats a `whole` dict exactly like
    a `kv` dict once an id is chosen -- only tokenize's ID CHOICE differs
    (one shared id across all 8 slots, from the joint distance, not k_slots
    independent per-slot nearest lookups)."""
    proj_blocks = None
    x_for_dist = None
    if proj_dim is not None:
        g = torch.Generator().manual_seed(proj_seed)
        proj_blocks = []
        for s in range(k_slots):
            mat = slot_loader(s).float()
            block = torch.randn(mat.shape[1], proj_dim, generator=g) / math.sqrt(proj_dim)
            proj_blocks.append(block)
            contrib = mat @ block
            x_for_dist = contrib if x_for_dist is None else x_for_dist + contrib
            del mat
    else:
        chunks = []
        for s in range(k_slots):
            chunks.append(slot_loader(s).float())
        x_for_dist = torch.cat(chunks, dim=1)
        del chunks
    _, assign, usage = kmeans(x_for_dist, K, iters, seed)
    slots = []
    for s in range(k_slots):
        mat = slot_loader(s).float()
        slots.append({"centroids": _cluster_means(mat, assign, K).half()})
        del mat
    entry = {
        "slots": slots,
        "usage": usage,
        "K": K,
        "seed": seed,
        "proj_blocks": proj_blocks,
        "proj_dim": proj_dim,
        "proj_seed": proj_seed,
    }
    return entry, assign


def _cluster_means(x: torch.Tensor, assign: torch.Tensor, K: int) -> torch.Tensor:
    """Mean of x's rows per assignment bucket 0..K-1; an empty bucket gets an
    all-zero row (never a NaN from dividing by zero members)."""
    out = torch.zeros(K, x.shape[1], dtype=torch.float32)
    for c in range(K):
        mask = assign == c
        if mask.any():
            out[c] = x[mask].float().mean(dim=0)
    return out


# ── tokenize / detokenize ────────────────────────────────────────────────────


def tokenize(kv, dict_: dict, readout: torch.Tensor | None = None) -> list[int]:  # noqa: ANN001
    """A step's gist-KV (+ readout, required only for a `kind=='ro'`
    dictionary) -> [8] ids, nearest entry per slot. `whole_*` configs assign
    ONE id over the concatenated 8-slot KV and broadcast it to all 8 slots."""
    kind = dict_["kind"]
    if kind == "whole":
        entry = dict_["entry"]
        mat = kv_slot_matrix(kv).float()  # [k_slots, D]
        proj_blocks = entry.get("proj_blocks")
        slots_c = entry["slots"]
        if proj_blocks is not None:
            x = sum(mat[s : s + 1] @ proj_blocks[s] for s in range(mat.shape[0]))
            cproj = sum(
                slots_c[s]["centroids"].float() @ proj_blocks[s] for s in range(len(slots_c))
            )
        else:
            x = mat.reshape(1, -1)
            cproj = torch.cat([c["centroids"].float() for c in slots_c], dim=1)
        idx = int(torch.cdist(x, cproj).argmin())
        return [idx] * len(slots_c)

    mat = kv_slot_matrix(kv).float()  # [k_slots, D]
    k_slots = mat.shape[0]
    ids = []
    for s in range(k_slots):
        slot = dict_["slots"][s]
        if kind == "kv":
            idx = int(torch.cdist(mat[s : s + 1], slot["centroids"].float()).argmin())
        elif kind == "kv_res":
            c1 = slot["c1"].float()
            id1 = int(torch.cdist(mat[s : s + 1], c1).argmin())
            resid = (mat[s] - c1[id1]).unsqueeze(0)
            id2 = int(torch.cdist(resid, slot["c2"].float()).argmin())
            idx = id1 * slot["K2"] + id2
        elif kind == "ro":
            assert readout is not None, "ro dictionary requires the step's readout, not just its KV"
            idx = int(
                torch.cdist(readout[s : s + 1].float(), slot["centroids_ro"].float()).argmin()
            )
        else:
            raise ValueError(f"unknown dict kind {kind!r}")
        ids.append(idx)
    return ids


def detokenize(ids: list[int], dict_: dict, geometry: dict):
    """[8] ids -> AxiomKV at canonical positions (the caller supplies
    cont_start = base + k_slots; this function only rebuilds the KV values,
    it has no notion of position)."""
    kind = dict_["kind"]
    if kind == "whole":
        slots = dict_["entry"]["slots"]
        idx = ids[0]
        mat = torch.stack([slots[s]["centroids"][idx].float() for s in range(len(slots))])
    elif kind == "kv_res":
        rows = []
        for s, idx in enumerate(ids):
            slot = dict_["slots"][s]
            id1, id2 = divmod(idx, slot["K2"])
            rows.append(slot["c1"][id1].float() + slot["c2"][id2].float())
        mat = torch.stack(rows)
    else:  # kv, ro
        rows = [dict_["slots"][s]["centroids"][idx].float() for s, idx in enumerate(ids)]
        mat = torch.stack(rows)
    return slot_matrix_to_kv(
        mat, geometry["n_layers"], geometry["n_kv_heads"], geometry["head_dim"]
    )


# ── naive Bayes over discrete slot IDs (op-from-IDs diagnostic) ─────────────


def fit_categorical_nb(
    ids: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    k_sizes: list[int],
    slots: list[int] | None = None,
    alpha: float = 1.0,
) -> dict:
    """ids [N, n_slots] long, y [N] long class ids -> a categorical naive
    Bayes model: P(class) and P(id_s | class) per selected slot (default all
    slots), Laplace-smoothed by alpha. `k_sizes[s]` is the DICTIONARY's K for
    slot s (not just the max id seen in `ids`) so an id that happens not to
    appear in the fit set still has a valid (smoothed) probability at test
    time -- never an index error or a silent clamp that would misclassify."""
    slots = list(range(ids.shape[1])) if slots is None else slots
    counts_prior = torch.zeros(n_classes)
    for c in range(n_classes):
        counts_prior[c] = float((y == c).sum())
    priors = (counts_prior + alpha) / (counts_prior.sum() + alpha * n_classes)
    tables = []
    for s in slots:
        K = k_sizes[s]
        counts = torch.zeros(n_classes, K)
        for c in range(n_classes):
            mask = y == c
            if mask.any():
                counts[c] = torch.bincount(ids[mask, s], minlength=K).float()
        probs = (counts + alpha) / (counts.sum(dim=1, keepdim=True) + alpha * K)
        tables.append(probs.log())
    return {
        "log_priors": priors.log(),
        "tables": tables,
        "slots": slots,
        "k_sizes": k_sizes,
        "n_classes": n_classes,
    }


def predict_categorical_nb(model: dict, ids: torch.Tensor) -> torch.Tensor:
    """argmax_c [log P(c) + sum_s log P(id_s | c)] over the fitted slots."""
    n = ids.shape[0]
    scores = model["log_priors"].unsqueeze(0).expand(n, -1).clone()
    for table, s in zip(model["tables"], model["slots"], strict=True):
        scores = scores + table[:, ids[:, s]].t()
    return scores.argmax(dim=1)


# ── adjusted mutual information (Hungarian-free ID-agreement diagnostic) ────


def _entropy(counts: torch.Tensor, n: int) -> float:
    p = counts.float() / n
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def adjusted_mutual_info(labels_a, labels_b) -> float:  # noqa: ANN001
    """Standard AMI (Vinh, Epps & Bailey 2010), arithmetic mean normalization,
    computed exactly (no sampling) over the OBSERVED label values -- feasible
    here because this is only ever run over a few hundred eval steps, so the
    contingency table is tiny even when the dictionaries it labels have
    K in the thousands. 1.0 = identical partitions (up to relabeling); ~0 =
    no better than chance; used to compare ro_K1024's assigned IDs against
    kv_K1024's on the same eval steps, without needing a matching (Hungarian)
    between the two ID spaces."""
    a = list(labels_a)
    b = list(labels_b)
    n = len(a)
    assert n == len(b) and n > 0
    a_vals = sorted(set(a))
    b_vals = sorted(set(b))
    a_idx = {v: i for i, v in enumerate(a_vals)}
    b_idx = {v: i for i, v in enumerate(b_vals)}
    ra, rb = len(a_vals), len(b_vals)
    contingency = torch.zeros(ra, rb)
    for x, y in zip(a, b, strict=True):
        contingency[a_idx[x], b_idx[y]] += 1
    ai = contingency.sum(dim=1)
    bj = contingency.sum(dim=0)

    mi = 0.0
    for i in range(ra):
        for j in range(rb):
            nij = float(contingency[i, j])
            if nij <= 0:
                continue
            mi += (nij / n) * math.log((n * nij) / (float(ai[i]) * float(bj[j])))

    def _lf(k: int) -> float:  # log-factorial
        return math.lgamma(k + 1)

    emi = 0.0
    for i in range(ra):
        ai_i = int(ai[i])
        for j in range(rb):
            bj_j = int(bj[j])
            lo = max(1, ai_i + bj_j - n)
            hi = min(ai_i, bj_j)
            for nij in range(lo, hi + 1):
                # exact hypergeometric-weighted term (Vinh et al. eq. 5), in
                # log space via lgamma for numerical stability at n ~ hundreds
                log_term = (
                    _lf(ai_i)
                    + _lf(bj_j)
                    + _lf(n - ai_i)
                    + _lf(n - bj_j)
                    - _lf(n)
                    - _lf(nij)
                    - _lf(ai_i - nij)
                    - _lf(bj_j - nij)
                    - _lf(n - ai_i - bj_j + nij)
                )
                term = math.exp(log_term) * (nij / n) * math.log((n * nij) / (ai_i * bj_j))
                emi += term

    h_a = _entropy(ai, n)
    h_b = _entropy(bj, n)
    denom = 0.5 * (h_a + h_b) - emi
    if abs(denom) < 1e-12:
        # both labelings are (near-)constant: MI and EMI both ~0, defined as
        # perfect agreement by convention (sklearn's ami_score does the same)
        return 1.0
    return (mi - emi) / denom


# ── stage1_verdict: the pure gate read (section F) ──────────────────────────


def stage1_verdict(cells: dict) -> str:
    """The plan's stage-1 gate, in strict order, over per-config cells
    {"R_gsm8k", "R_fresh", "op", "kind": "per_slot"|"whole"}
    (cells["configs"] keyed by config name) plus cells["gate0_pass"].

    A CONVENIENCE field only -- like summary_verdict/question_verdict
    elsewhere in this repo, never a hard assertion; the human + Fable read
    the manifest's actual per-config numbers. Ambiguity resolved (no single
    config name in the plan's gate text): the "best" config for the VQVAE
    band is the one maximizing min(R_gsm8k, R_fresh); KILL requires EVERY
    config to be at the floor on both R and op, matching the plan's
    "everywhere" wording.

    GO:        some PER-SLOT config clears R_gsm8k>=0.8 AND R_fresh>=0.7 AND
               op>=0.75.
    RETRIEVAL: no per-slot config clears that bar but some `kind=="whole"`
               config does -- the codebook needs to be a retrieval
               vocabulary, not a factored one.
    VQVAE:     the best config's min(R_gsm8k,R_fresh) is in [0.5, 0.8), or
               its op is in [0.6, 0.75) -- discretization alone isn't enough,
               joint VQ-VAE training might close the gap.
    KILL:      EVERY config's min(R_gsm8k,R_fresh) < 0.5, or every config's
               op < 0.6 -- discretization itself doesn't preserve fidelity.
    INVALID_HARNESS: gate 0 (native placement) failed -- nothing else here is
               trustworthy.
    """
    if not cells.get("gate0_pass", True):
        return "INVALID_HARNESS"
    configs = list(cells["configs"].values())
    if not configs:
        return "KILL"

    def r_min(c):
        return min(c["R_gsm8k"], c["R_fresh"])

    def go_ok(c):
        return c["R_gsm8k"] >= 0.8 and c["R_fresh"] >= 0.7 and c["op"] >= 0.75

    per_slot = [c for c in configs if c.get("kind") != "whole"]
    whole = [c for c in configs if c.get("kind") == "whole"]
    if any(go_ok(c) for c in per_slot):
        return "GO"
    if any(go_ok(c) for c in whole):
        return "RETRIEVAL"

    best = max(configs, key=lambda c: (r_min(c), c["op"]))
    if (0.5 <= r_min(best) < 0.8) or (0.6 <= best["op"] < 0.75):
        return "VQVAE"

    if all(r_min(c) < 0.5 for c in configs) or all(c["op"] < 0.6 for c in configs):
        return "KILL"
    return "VQVAE"
