"""Install `<sentinel>` and `</sentinel>` as additional special tokens.

The sentinel wraps axiom content; downstream training teaches the model
to consume that content as a premise. We add genuine new tokens (rather
than repurposing rare existing ones) for two reasons:

  1. Intent is explicit at the tokenizer level — anyone reading the
     tokenizer config sees what these tokens are for.
  2. We control the embedding initialisation. Repurposed rare tokens
     start with whatever embedding the base model trained for them,
     which is unlikely to be a useful initialisation for "premise
     boundary."

Initialisation: mean of all existing input embeddings. This puts the new
embeddings in the centre of the embedding-norm distribution, so they
look like in-distribution tokens to downstream layers — a neutral
starting point that the LoRA can then specialise.
"""

from __future__ import annotations

import torch

from sentinel.model import SentinelModel

SENTINEL_OPEN = "<sentinel>"
SENTINEL_CLOSE = "</sentinel>"


def install_sentinel_tokens(model: SentinelModel) -> tuple[int, int]:
    """Add the sentinel tokens to the tokenizer, resize embeddings,
    initialise the new rows to the mean of existing embeddings.

    Returns (open_token_id, close_token_id). Idempotent: if the tokens
    are already present, returns their existing ids without resizing.
    """
    tok = model.tokenizer
    existing_open = tok.convert_tokens_to_ids(SENTINEL_OPEN)
    existing_close = tok.convert_tokens_to_ids(SENTINEL_CLOSE)
    unk = tok.unk_token_id
    if existing_open != unk and existing_close != unk:
        return existing_open, existing_close

    tok.add_special_tokens({"additional_special_tokens": [SENTINEL_OPEN, SENTINEL_CLOSE]})
    model.base.resize_token_embeddings(len(tok))

    open_id = tok.convert_tokens_to_ids(SENTINEL_OPEN)
    close_id = tok.convert_tokens_to_ids(SENTINEL_CLOSE)

    _init_new_rows_to_norm_matched_mean(model.base.get_input_embeddings(), [open_id, close_id])
    out_embed = model.base.get_output_embeddings()
    if (
        out_embed is not None
        and out_embed.weight.data_ptr() != model.base.get_input_embeddings().weight.data_ptr()
    ):
        _init_new_rows_to_norm_matched_mean(out_embed, [open_id, close_id])

    return open_id, close_id


def _init_new_rows_to_norm_matched_mean(embedding, new_ids: list[int]) -> None:  # noqa: ANN001
    """Initialise the rows at `new_ids` to the mean direction of the existing
    rows, rescaled to have norm equal to the mean L2 norm of existing rows.

    Plain mean-init produces a short vector (averaging across rows cancels
    out direction), which leaves the new tokens looking like out-of-
    distribution noise to downstream layers. The norm-rescaled mean keeps
    the "central direction" while putting the new tokens at a typical
    magnitude — neutral but in-distribution.
    """
    weight = embedding.weight
    with torch.no_grad():
        mask = torch.ones(weight.shape[0], dtype=torch.bool, device=weight.device)
        mask[new_ids] = False
        existing = weight[mask]
        mean_dir = existing.mean(dim=0)
        target_norm = existing.norm(dim=1).mean()
        scaled = mean_dir * (target_norm / (mean_dir.norm() + 1e-8))
        for idx in new_ids:
            weight[idx] = scaled
