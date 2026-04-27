# TODO

## Performance

- **KV-cache-aware injection.** Current `TriggerInjector.generate` does a
  full forward pass per generated token (no KV cache), so a 6-config × 80-token
  prompt run costs O(N²) instead of O(N). Standard generation with a hook
  that only fires on the prefill step (and on each new term-match in the
  growing sequence) would cut wall-clock by 5-20×.
- **Batch the configs.** Each prompt currently runs 6 configs sequentially.
  Could batch them into a single forward (different alphas as different
  batch dims), since the only difference is the hook's scalar multiplier.
  Probably 3-4× speedup with a single-pass batched hook.
- **Skip baseline reforwarding.** When alpha=0 we still attach the hook;
  we should detect it and skip injection entirely (no clone, no add).

## Validation

- Composition test on a music-domain axiom pair (`coastal_shoegaze` outer,
  `dream_pop_vocals` inner) — chosen because 0.5B has strong priors there.
- Run the same composition test on Qwen 1.5B once memory allows for
  cross-scale comparison.
- Larger N-axiom validation (10+ concepts) to characterise scaling of
  the contrastive isolation step.

## Architecture

- Mimir integration: define the on-disk contract for `(axiom_id, vector)`
  pairs that Mimir produces and Mimir-Protocol consumes.
- Component-level extraction: per-component vectors that compose into
  the outer axiom's vector at runtime.
