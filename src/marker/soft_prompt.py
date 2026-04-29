"""Per-axiom soft-prompt training: learn an embedding-layer vector for
the axiom term so the model never processes the literal lexical content.

Theory: rather than letting the model's embedding lookup pull up the
default `embedding_table[Balance]` and `embedding_table[Publisher]`
(which then compose into 'balance sheet'), we substitute a learned
vector at those token positions. The vector lives in embedding space
and is trained per axiom on contrastive paraphrase loss. Model weights
stay frozen.

Storage: ~hidden_size × num_term_tokens floats per axiom (~5-15 KB).
Training cost: ~5-15 sec per axiom locally on Qwen 1.5B.

The model itself never accumulates information about specific axioms.
Knowledge stays in per-axiom side-dictionary entries that can be
hot-loaded / hot-unloaded at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _get_embed_module(model):  # noqa: ANN001
    base = model
    if hasattr(base, "base_model") and hasattr(base.base_model, "model"):
        base = base.base_model.model
    candidates = [
        lambda m: m.model.embed_tokens,
        lambda m: m.language_model.model.embed_tokens,
        lambda m: m.model.language_model.model.embed_tokens,
        lambda m: m.model.language_model.embed_tokens,
        lambda m: m.embed_tokens,
    ]
    for fn in candidates:
        try:
            return fn(base)
        except AttributeError:
            continue
    if hasattr(base, "get_input_embeddings"):
        e = base.get_input_embeddings()
        if e is not None:
            return e
    raise RuntimeError(f"could not find embed_tokens on {type(model).__name__}")


def _term_token_ids(tokenizer, term: str) -> list[int]:  # noqa: ANN001
    """Get token IDs for a term, trying with and without leading space."""
    ids = tokenizer(term, add_special_tokens=False).input_ids
    if not ids:
        ids = tokenizer(" " + term, add_special_tokens=False).input_ids
    return ids


@dataclass
class SoftPrompt:
    """A per-axiom learned embedding vector. `vector` shape =
    [num_term_tokens, hidden_size]. Init from natural term embeddings."""

    term: str
    term_token_ids: list[int]
    vector: nn.Parameter  # [num_term_tokens, hidden_size]

    @classmethod
    def from_term(cls, model, tokenizer, term: str) -> SoftPrompt:  # noqa: ANN001
        """Initialize a soft prompt from the term's natural token embeddings."""
        token_ids = _term_token_ids(tokenizer, term)
        if not token_ids:
            raise ValueError(f"could not tokenize term {term!r}")
        embed = _get_embed_module(model)
        with torch.no_grad():
            init = embed.weight[token_ids].detach().clone().float()
        vector = nn.Parameter(init)
        return cls(term=term, term_token_ids=token_ids, vector=vector)


def install_soft_prompt_hook(model, soft_prompt: SoftPrompt, positions: list[int]):  # noqa: ANN001, ANN201
    """Install a forward hook on the embedding module that replaces the
    output at the given absolute positions with the soft prompt vector.

    Returns a handle; caller must call handle.remove() when done.

    `positions[i]` should correspond to soft_prompt.term_token_ids[i] —
    i.e., the i-th sub-token of the term gets the i-th row of the
    soft prompt vector.
    """
    embed_module = _get_embed_module(model)
    sp = soft_prompt
    pos_list = list(positions)
    n_pos = len(pos_list)
    n_sp = sp.vector.shape[0]

    def hook(module, inputs, output):  # noqa: ANN001, ARG001
        # output shape: [batch, seq, hidden]
        seq_len = output.shape[1]
        if seq_len == 1:
            # Decode step — KV cache propagates the prefill substitution
            return output
        out = output.clone()
        for i, pos in enumerate(pos_list):
            if 0 <= pos < seq_len and i < n_sp:
                out[:, pos, :] = sp.vector[i].to(dtype=out.dtype, device=out.device)
        return out

    return embed_module.register_forward_hook(hook)


def find_term_positions(tokenizer, text: str, term: str) -> list[int]:  # noqa: ANN001
    """Find token positions of `term` in tokenized `text`. Robust to
    leading-space tokenization differences: returns positions even when
    'Balance Publisher' standalone tokenizes differently from
    ' Balance Publisher' in context.

    Strategy: locate the term as a substring in the text, then tokenize
    the prefix to find where the term starts in token-space. The number
    of term tokens is determined by tokenizing prefix+term and subtracting
    the prefix token count.
    """
    if term not in text:
        return []
    char_start = text.index(term)
    prefix = text[:char_start]
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    full_until_term_end = tokenizer(text[: char_start + len(term)], add_special_tokens=False).input_ids
    term_token_count = len(full_until_term_end) - len(prefix_ids)
    if term_token_count <= 0:
        return []
    return list(range(len(prefix_ids), len(prefix_ids) + term_token_count))


def _find_term_positions(tokenizer, text: str, term_token_ids: list[int]) -> list[int]:  # noqa: ANN001
    """(legacy) Find positions matching exact term_token_ids in tokenized text."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    n, m = len(ids), len(term_token_ids)
    for i in range(n - m + 1):
        if ids[i : i + m] == term_token_ids:
            return list(range(i, i + m))
    return []


def wrap_chat(tokenizer, term: str, paraphrase: str) -> tuple[str, int]:  # noqa: ANN001
    """Wrap a paraphrase as a chat turn for IT-model training. Returns
    (chat_formatted_text, char_index_where_assistant_response_starts).

    Format:
      <user>Tell me about {term}.<end>
      <assistant>{paraphrase}<end>

    The soft prompt will substitute at the {term} position in the user turn.
    Training loss is computed on the assistant tokens — i.e., predict the
    paraphrase content given the chat-formatted user prompt with the soft
    prompt at the term position.
    """
    user_q = f"Tell me about {term}."
    try:
        formatted = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": paraphrase},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback for tokenizers without chat template support: a
        # plausible chat-like format
        formatted = (
            f"<|im_start|>user\n{user_q}<|im_end|>\n"
            f"<|im_start|>assistant\n{paraphrase}<|im_end|>"
        )
    # Locate where the assistant response begins (so loss is on those tokens only)
    asst_start = formatted.find(paraphrase)
    return formatted, asst_start


def _prep_training_sample(
    tokenizer,  # noqa: ANN001
    soft_prompt: SoftPrompt,
    paraphrase: str,
    chat_format: bool,
) -> tuple[list[int], list[int], int] | None:
    """Returns (token_ids, term_positions_in_user_portion, target_start_token_idx)
    or None if the sample is unusable.

    For raw mode: user-portion is the whole text; target starts right after term.
    For chat mode: user portion is up to assistant_start_char; target is the
    assistant response (the paraphrase content)."""
    if chat_format:
        text, asst_start_char = wrap_chat(tokenizer, soft_prompt.term, paraphrase)
        if asst_start_char < 0:
            return None
        # Tokenize the user portion to find where assistant tokens begin
        user_portion = text[:asst_start_char]
        user_ids = tokenizer(user_portion, add_special_tokens=False).input_ids
        target_start = len(user_ids)
        # Find term positions ONLY in the user portion
        positions = find_term_positions(tokenizer, user_portion, soft_prompt.term)
        if not positions:
            positions = _find_term_positions(tokenizer, user_portion, soft_prompt.term_token_ids)
        if not positions:
            return None
        full_ids = tokenizer(text, add_special_tokens=False).input_ids
        if target_start >= len(full_ids):
            return None
        return full_ids, positions, target_start
    else:
        text = paraphrase
        positions = find_term_positions(tokenizer, text, soft_prompt.term)
        if not positions:
            positions = _find_term_positions(tokenizer, text, soft_prompt.term_token_ids)
        if not positions:
            return None
        full_ids = tokenizer(text, add_special_tokens=False).input_ids
        target_start = positions[-1] + 1
        if target_start >= len(full_ids):
            return None
        return full_ids, positions, target_start


def _training_step(  # noqa: ANN201
    model,  # noqa: ANN001
    tokenizer,
    soft_prompt: SoftPrompt,
    paraphrase: str,
    chat_format: bool = False,
) -> torch.Tensor:
    """One forward pass with the soft prompt substituted at term positions
    in the paraphrase. Returns NLL of post-term tokens (raw mode) or
    assistant-response tokens (chat mode)."""
    device = next(model.parameters()).device
    prepped = _prep_training_sample(tokenizer, soft_prompt, paraphrase, chat_format)
    if prepped is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    full_ids_list, positions, target_start = prepped
    full_ids = torch.tensor(full_ids_list, device=device).unsqueeze(0)

    handle = install_soft_prompt_hook(model, soft_prompt, positions)
    try:
        out = model(full_ids)
        target_ids = full_ids[0, target_start:]
        pred_logits = out.logits[0, target_start - 1 : target_start - 1 + len(target_ids)]
        return torch.nn.functional.cross_entropy(pred_logits, target_ids)
    finally:
        handle.remove()


def _install_batched_soft_prompt_hook(  # noqa: ANN201
    model,  # noqa: ANN001
    soft_prompt: SoftPrompt,
    per_row_positions: list[list[int]],
):
    """Like install_soft_prompt_hook but per-row positions. For batched
    training/inference with multiple paraphrases in one forward pass —
    each row's term positions get the soft prompt vector substituted."""
    embed_module = _get_embed_module(model)
    sp = soft_prompt

    def hook(module, inputs, output):  # noqa: ANN001, ARG001
        if output.shape[1] == 1:
            return output
        out = output.clone()
        for row_idx, positions in enumerate(per_row_positions):
            for i, pos in enumerate(positions):
                if 0 <= pos < out.shape[1] and i < sp.vector.shape[0]:
                    out[row_idx, pos, :] = sp.vector[i].to(dtype=out.dtype, device=out.device)
        return out

    return embed_module.register_forward_hook(hook)


def _training_step_batched(  # noqa: ANN201
    model,  # noqa: ANN001
    tokenizer,
    soft_prompt: SoftPrompt,
    paraphrases: list[str],
    chat_format: bool = False,
) -> torch.Tensor:
    """Mini-batch version: batch paraphrases into a single forward pass.
    Returns mean cross-entropy across post-term tokens (raw) or assistant
    response tokens (chat)."""
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Each entry: (ids, term_positions, target_start_idx)
    encoded: list[tuple[list[int], list[int], int]] = []
    for p in paraphrases:
        text = p.replace("[[", "").replace("]]", "")
        prepped = _prep_training_sample(tokenizer, soft_prompt, text, chat_format)
        if prepped is None:
            continue
        encoded.append(prepped)

    if not encoded:
        return torch.tensor(0.0, device=device, requires_grad=True)

    max_len = max(len(ids) for ids, _, _ in encoded)
    batch_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
    per_row_positions: list[list[int]] = []
    for r, (ids, positions, _) in enumerate(encoded):
        batch_ids[r, : len(ids)] = torch.tensor(ids, device=device)
        attention_mask[r, : len(ids)] = 1
        per_row_positions.append(positions)

    handle = _install_batched_soft_prompt_hook(model, soft_prompt, per_row_positions)
    try:
        out = model(batch_ids, attention_mask=attention_mask)
        total_ce = 0.0
        n_terms = 0
        for r, (ids, _, target_start) in enumerate(encoded):
            target_ids = batch_ids[r, target_start : len(ids)]
            pred_logits = out.logits[r, target_start - 1 : len(ids) - 1]
            ce = torch.nn.functional.cross_entropy(pred_logits, target_ids, reduction="sum")
            total_ce = total_ce + ce
            n_terms += target_ids.shape[0]
        return total_ce / max(n_terms, 1)
    finally:
        handle.remove()


def train_soft_prompt_contrastive(
    model,
    tokenizer,
    soft_prompt: SoftPrompt,
    intended_paraphrases: list[str],
    lexical_paraphrases: list[str],
    n_steps: int = 50,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    margin: float = 1.0,
    batch_size: int = 1,
    early_stop_patience: int = 0,
    early_stop_delta: float = 0.01,
    chat_format: bool = False,
    seed: int = 0,
) -> list[float]:  # noqa: ANN001
    """Bounded contrastive: log-softmax over [-NLL_int, -NLL_lex] (i.e.,
    InfoNCE with two classes). Equivalent to logistic regression on
    'is intended more probable than lexical?'. Bounded gradient — won't
    diverge to NaN like raw NLL_int - NLL_lex.

    Plus weight decay on the soft prompt vector to keep its norm bounded.
    """
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    model.eval()

    soft_prompt.vector.requires_grad_(True)
    init_norm = float(soft_prompt.vector.detach().norm().item())
    opt = torch.optim.Adam(
        [soft_prompt.vector], lr=lr, weight_decay=weight_decay
    )
    rng = torch.Generator().manual_seed(seed)

    def _sample_batch(paraphrases: list[str], k: int) -> list[str]:
        idxs = torch.randint(0, len(paraphrases), (k,), generator=rng).tolist()
        return [paraphrases[i].replace("[[", "").replace("]]", "") for i in idxs]

    losses: list[float] = []
    plateau_count = 0
    for step in range(n_steps):
        opt.zero_grad()
        for p in model.parameters():
            p.grad = None

        if batch_size <= 1:
            i_text = _sample_batch(intended_paraphrases, 1)[0]
            l_text = _sample_batch(lexical_paraphrases, 1)[0]
            loss_int = _training_step(model, tokenizer, soft_prompt, i_text, chat_format)
            loss_lex = _training_step(model, tokenizer, soft_prompt, l_text, chat_format)
        else:
            i_batch = _sample_batch(intended_paraphrases, batch_size)
            l_batch = _sample_batch(lexical_paraphrases, batch_size)
            loss_int = _training_step_batched(
                model, tokenizer, soft_prompt, i_batch, chat_format
            )
            loss_lex = _training_step_batched(
                model, tokenizer, soft_prompt, l_batch, chat_format
            )

        gap = loss_lex - loss_int
        contrastive = torch.nn.functional.relu(margin - gap)
        loss = loss_int + contrastive
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([soft_prompt.vector], max_norm=1.0)
            opt.step()
            with torch.no_grad():
                cur_norm = soft_prompt.vector.norm().item()
                cap = max(init_norm * 3.0, 10.0)
                if cur_norm > cap:
                    soft_prompt.vector.mul_(cap / cur_norm)
        losses.append(float(loss.item()))

        # Early stopping: if recent average loss isn't improving, halt
        if early_stop_patience > 0 and len(losses) >= 2 * early_stop_patience:
            recent = sum(losses[-early_stop_patience:]) / early_stop_patience
            prev = sum(losses[-2 * early_stop_patience : -early_stop_patience]) / early_stop_patience
            if prev - recent < early_stop_delta:
                plateau_count += 1
                if plateau_count >= 2:
                    break
            else:
                plateau_count = 0
    return losses


def train_soft_prompt(
    model,
    tokenizer,
    soft_prompt: SoftPrompt,
    paraphrases: list[str],
    n_steps: int = 30,
    lr: float = 0.01,
    seed: int = 0,
) -> list[float]:  # noqa: ANN001
    """Train soft_prompt.vector via gradient descent on next-token NLL of
    the post-term tokens in the given paraphrases.

    Model is set to eval and frozen (requires_grad=False). Only
    soft_prompt.vector receives gradients.

    Returns list of per-step losses.
    """
    # Freeze model
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    model.eval()

    soft_prompt.vector.requires_grad_(True)
    opt = torch.optim.Adam([soft_prompt.vector], lr=lr)
    rng = torch.Generator().manual_seed(seed)

    losses: list[float] = []
    for step in range(n_steps):
        idx = int(torch.randint(0, len(paraphrases), (1,), generator=rng).item())
        paraphrase = paraphrases[idx].replace("[[", "").replace("]]", "")
        opt.zero_grad()
        # Clear any stray model param grads
        for p in model.parameters():
            p.grad = None
        loss = _training_step(model, tokenizer, soft_prompt, paraphrase)
        if loss.requires_grad:
            loss.backward()
            opt.step()
        losses.append(float(loss.item()))
    return losses
