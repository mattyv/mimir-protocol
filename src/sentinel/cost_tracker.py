"""Per-model cost tracking with cache-aware accounting.

Pricing per 1M tokens (cached 2026-04-15 from claude-api skill model table):

  | Model             | Input  | Output  |
  |-------------------|--------|---------|
  | claude-opus-4-7   | $5.00  | $25.00  |
  | claude-sonnet-4-6 | $3.00  | $15.00  |
  | claude-haiku-4-5  | $1.00  | $5.00   |

Cache reads bill at ~0.1× input price; cache writes (5-min TTL) at 1.25×.
We use `Anthropic`'s `Usage` directly to avoid drift from model class.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

# Price tables in USD per 1M tokens.
INPUT_PRICE_PER_MTOK: dict[str, float] = {
    "claude-opus-4-7": 5.00,
    "claude-opus-4-6": 5.00,
    "claude-sonnet-4-6": 3.00,
    "claude-haiku-4-5": 1.00,
}

OUTPUT_PRICE_PER_MTOK: dict[str, float] = {
    "claude-opus-4-7": 25.00,
    "claude-opus-4-6": 25.00,
    "claude-sonnet-4-6": 15.00,
    "claude-haiku-4-5": 5.00,
}

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER_5MIN = 1.25
CACHE_WRITE_MULTIPLIER_1H = 2.0


@dataclass
class ModelUsage:
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0  # 5-min TTL writes; 1h would need a separate field
    requests: int = 0


@dataclass
class CostTracker:
    """Accumulates token usage across many API calls, by model."""

    by_model: dict[str, ModelUsage] = field(default_factory=lambda: defaultdict(ModelUsage))

    def record(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        u = self.by_model[model]
        u.input_tokens += input_tokens
        u.output_tokens += output_tokens
        u.cache_read_tokens += cache_read_tokens
        u.cache_write_tokens += cache_write_tokens
        u.requests += 1

    def cost_usd(self) -> float:
        total = 0.0
        for model, u in self.by_model.items():
            in_price = INPUT_PRICE_PER_MTOK.get(model)
            out_price = OUTPUT_PRICE_PER_MTOK.get(model)
            if in_price is None or out_price is None:
                # Unknown model — surface this loudly rather than silently undercount.
                raise ValueError(f"no price entry for model {model!r}; update cost_tracker.py")
            total += (u.input_tokens / 1_000_000) * in_price
            total += (u.output_tokens / 1_000_000) * out_price
            total += (u.cache_read_tokens / 1_000_000) * in_price * CACHE_READ_MULTIPLIER
            total += (u.cache_write_tokens / 1_000_000) * in_price * CACHE_WRITE_MULTIPLIER_5MIN
        return total

    def summary(self) -> str:
        if not self.by_model:
            return "no API calls recorded"
        lines = []
        for model, u in self.by_model.items():
            lines.append(
                f"  {model:22s}  reqs={u.requests}  "
                f"in={u.input_tokens}  out={u.output_tokens}  "
                f"cache_r={u.cache_read_tokens}  cache_w={u.cache_write_tokens}"
            )
        lines.append(f"  total cost: ${self.cost_usd():.4f}")
        return "\n".join(lines)
