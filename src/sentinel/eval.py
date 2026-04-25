"""Eval harness for the sentinel-LoRA POC.

Five tests adapted from `docs/sentinel-lora-poc-spec.md` §5 Phase 4. They
all share the same shape: build a prompt, generate from the trained
LoRA, record the output. Grading is qualitative — a separate pass
through Claude (via `data_gen.DataGenerator.grade_example` or a richer
grader) decides pass/fail.

  T1 — ablation:        does sentinel content change behavior at all?
  T2 — negation:        does flipping the axiom flip the answer?
  T3 — composition:     does the model use two axioms jointly?
  T4 — selectivity:     does sentinel beat irrelevant ambient context?
  T5 — generalization:  does the protocol carry to real axioms?

The eval module loads a trained adapter on top of the frozen base and
exposes a single `generate_with_sentinel` primitive — the tests are thin
wrappers that compose prompts and call it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel

from sentinel.data_schema import Axiom
from sentinel.model import SentinelModel
from sentinel.tokens import SENTINEL_CLOSE, SENTINEL_OPEN, install_sentinel_tokens


@dataclass
class LoadedAdapter:
    """A trained sentinel-LoRA model loaded for inference."""

    peft_model: PeftModel
    tokenizer: object  # AutoTokenizer; can't import the type cheaply
    device: str

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.peft_model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # Decode only the newly-generated portion.
        new_tokens = out[0, ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=False)


def load_adapter(
    adapter_path: Path,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: str = "mps",
) -> LoadedAdapter:
    """Load base model + trained adapter for inference.

    The adapter checkpoint includes the resized embedding layer (PEFT
    auto-saves this when tokens are added pre-training), so we install
    sentinel tokens on the base first to make the shapes match, then
    PeftModel.from_pretrained overlays the trained weights.
    """
    sm = SentinelModel(model_name=model_name, device=device, dtype=torch.float16)
    install_sentinel_tokens(sm)
    peft_model = PeftModel.from_pretrained(sm.base, str(adapter_path))
    peft_model.eval()
    return LoadedAdapter(peft_model=peft_model, tokenizer=sm.tokenizer, device=device)


# ---------- prompt builders ----------


def with_sentinel(axiom_text: str, question: str) -> str:
    return f"{SENTINEL_OPEN}{axiom_text}{SENTINEL_CLOSE}\n{question}\n"


def without_sentinel(question: str) -> str:
    return f"{question}\n"


def with_two_sentinels(axiom_a: str, axiom_b: str, question: str) -> str:
    """Two axioms in two separate sentinel blocks. The training set never
    showed multi-sentinel prompts, so this is genuinely compositional."""
    return (
        f"{SENTINEL_OPEN}{axiom_a}{SENTINEL_CLOSE}\n"
        f"{SENTINEL_OPEN}{axiom_b}{SENTINEL_CLOSE}\n"
        f"{question}\n"
    )


def with_sentinel_and_distractor(axiom_text: str, distractor_context: str, question: str) -> str:
    """Sentinel content vs ambient context — selectivity test (T4)."""
    return f"{SENTINEL_OPEN}{axiom_text}{SENTINEL_CLOSE}\n{distractor_context}\n{question}\n"


# ---------- the five tests ----------


@dataclass
class T1Result:
    axiom: Axiom
    question: str
    with_axiom: str
    without_axiom: str


def t1_ablation(loaded: LoadedAdapter, axiom: Axiom, question: str) -> T1Result:
    """T1 — does the sentinel content matter at all?

    Generate twice: once with the sentinel, once without. If the outputs
    are identical, the LoRA learned to ignore the slot — every other test
    is moot. Per the brief, T1 is the gate.
    """
    return T1Result(
        axiom=axiom,
        question=question,
        with_axiom=loaded.generate(with_sentinel(axiom.text, question)),
        without_axiom=loaded.generate(without_sentinel(question)),
    )


@dataclass
class T2Result:
    axiom: Axiom
    negated_axiom_text: str
    question: str
    with_axiom: str
    with_negated: str


def t2_negation(
    loaded: LoadedAdapter, axiom: Axiom, negated_axiom_text: str, question: str
) -> T2Result:
    """T2 — does flipping the axiom flip the answer?

    The negated axiom must be a meaningful negation, not just a syntactic
    'not' insert. Generation is paired against the same question.
    """
    return T2Result(
        axiom=axiom,
        negated_axiom_text=negated_axiom_text,
        question=question,
        with_axiom=loaded.generate(with_sentinel(axiom.text, question)),
        with_negated=loaded.generate(with_sentinel(negated_axiom_text, question)),
    )


@dataclass
class T3Result:
    axiom_a: Axiom
    axiom_b: Axiom
    question: str
    with_both: str


def t3_composition(
    loaded: LoadedAdapter, axiom_a: Axiom, axiom_b: Axiom, question: str
) -> T3Result:
    """T3 — does the model use two axioms jointly?

    Two sentinels, one question whose answer requires both. Training data
    never paired sentinels so anything coherent is genuine compositional
    uptake. Grade the answer for whether it correctly uses content from
    both axioms — neither alone should be sufficient.
    """
    return T3Result(
        axiom_a=axiom_a,
        axiom_b=axiom_b,
        question=question,
        with_both=loaded.generate(with_two_sentinels(axiom_a.text, axiom_b.text, question)),
    )


@dataclass
class T4Result:
    axiom: Axiom
    distractor_context: str
    question: str
    answer: str


def t4_selectivity(
    loaded: LoadedAdapter, axiom: Axiom, distractor_context: str, question: str
) -> T4Result:
    """T4 — does the sentinel beat irrelevant ambient context?

    The question is about the sentinel's content; the ambient context is
    plausible-sounding prose about something else. A successful answer
    uses the sentinel; failure means the LoRA learned 'use whatever
    context is present', not 'use the sentinel specifically'.
    """
    return T4Result(
        axiom=axiom,
        distractor_context=distractor_context,
        question=question,
        answer=loaded.generate(
            with_sentinel_and_distractor(axiom.text, distractor_context, question)
        ),
    )


@dataclass
class T5Result:
    axiom_text: str
    question: str
    answer: str


def t5_generalization(loaded: LoadedAdapter, axiom_text: str, question: str) -> T5Result:
    """T5 — does the protocol carry to real (non-made-up) axioms?

    Training was on made-up terms. Test on real terms the model could
    plausibly know — Auros internal terminology, niche real-world facts.
    Strong test: the *negation* form of T5 (does the axiom override
    priors?) — but that's expensive to grade fairly so we report the raw
    answer here and grade qualitatively.
    """
    return T5Result(
        axiom_text=axiom_text,
        question=question,
        answer=loaded.generate(with_sentinel(axiom_text, question)),
    )
