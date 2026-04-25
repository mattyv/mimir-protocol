"""Prompt builders for axiom / question / contrastive-pair generation.

Each builder returns the (system, user) pair we hand to `messages.create()`.
The system prompts are deliberately stable across many calls so prompt
caching can hit on the prefix; the user message carries the per-call
variable parts.

Why these are split out from `data_gen.py`: the prompt strings *are* the
load-bearing artifact (per the brief, "data is the load-bearing piece").
Keeping them in their own module makes them easy to read, diff, and test
without booting the API client.
"""

from __future__ import annotations

from sentinel.data_schema import Axiom, AxiomShape

# ---------- axiom generation ----------

AXIOM_SYSTEM = """You are generating synthetic training data for a small language model.

Your job: invent axioms with **made-up names** and plausible definitions, in five distinct shapes:

  - definitional: "X means Y" / "An X is a Y that ..."
  - causal:       "X causes Y" / "When X, then Y"
  - normative:    "X should always Z" / "X must never W"
  - relational:   "X is part of Y" / "X belongs to the family of Y"
  - exception:    "X is Y, except when Z"

Hard requirements:

1. Every axiom's `name` must be a **made-up word**, not a real term, not a real-world acronym.
   Examples: fazbuzza, queltrick, blampine, yorvex, zumpra. Multi-syllable nonsense is fine.
2. The `text` must be a complete declarative sentence that names the term and states the axiom.
3. The axioms must be plausible-sounding but cannot be looked up — the model under training has
   never seen them.
4. Vary topic and register: tools, biology-flavored, organisational, mechanical, social, etc.
5. Diversity matters more than quantity. Avoid repeating sentence structures.

Return a JSON object with key `axioms`, an array of objects with fields:
  id (string), shape (one of the five), name (string), text (string).

Use sequential IDs of the form "ax_0001", "ax_0002", etc., starting from the id_offset given.
"""


def build_axiom_user_prompt(n: int, id_offset: int, shapes: list[AxiomShape] | None = None) -> str:
    shape_list = ", ".join(s.value for s in (shapes or list(AxiomShape)))
    return (
        f"Generate {n} axioms. id_offset={id_offset:04d}. "
        f"Distribute across shapes: {shape_list}. "
        f"Aim for an even split unless that produces awkward axioms."
    )


# ---------- question + answer generation ----------

QUESTION_SYSTEM = """You are generating training questions for a single axiom.

You will be given an axiom (the name and full text). For each question:

1. The question must be answerable **only** by reading the axiom — never from priors,
   never from the question itself.
2. The answer must require **inference from** the axiom, not paraphrase of it.
3. Vary surface form: definitional probes ("What is a X?"), application probes
   ("If you encountered a X, would you ...?"), causal probes ("What happens when X?"),
   counterfactual probes ("If X were not true, what would change?").

Some examples are tagged "anti_regurgitation": for those, the answer must NOT contain any
content words from the axiom text. Use synonyms and inferential restatements.

Return a JSON object with key `examples`, an array of objects with fields:
  axiom_id (copy from input), type ("base" or "anti_regurgitation"), sentinel_block,
  question, answer.

`sentinel_block` is the axiom's full text wrapped in `<sentinel>` and `</sentinel>` tags
verbatim — copy the axiom text inside the tags exactly.
"""


def build_question_user_prompt(
    axiom: Axiom, n_questions: int, anti_regurgitation_fraction: float
) -> str:
    n_anti = int(round(n_questions * anti_regurgitation_fraction))
    n_base = n_questions - n_anti
    return (
        f"Axiom:\n"
        f"  id: {axiom.id}\n"
        f"  shape: {axiom.shape.value}\n"
        f"  name: {axiom.name}\n"
        f"  text: {axiom.text}\n\n"
        f"Generate {n_base} examples of type 'base' and {n_anti} of type 'anti_regurgitation'."
    )


# ---------- contrastive pair generation ----------

CONTRASTIVE_SYSTEM = """You are generating contrastive training pairs.

A contrastive pair is two axioms with the **same name** but **different text**, plus one
shared question that produces different answers under each.

Example shape:
  axiom_a:  "fazbuzza is a small blue creature that lives in trees"
  axiom_b:  "fazbuzza is a precision tool used for cutting hard stone"
  question: "Where would you most likely find a fazbuzza?"
  answer_a: "Among tree branches in temperate forests."
  answer_b: "On a workbench in a stonemason's shop."

Requirements:

1. Both axioms share the same `name`. Other fields differ.
2. The shared question must be answerable from either axiom but produce **clearly different**
   answers — the difference must depend on which axiom is in the sentinel.
3. Neither answer should contain content that would also be true under the other axiom.

The `shape` field on each axiom must be exactly one of these literal strings (it's the
axiom's structural category, NOT its subject):

    definitional | causal | normative | relational | exception

Return a JSON object with the contrastive pair fields:
  pair_id (copy from input), axiom_a, axiom_b (each with id/shape/name/text/pair_id),
  question, answer_a, answer_b.
"""


def build_contrastive_user_prompt(name: str, pair_id: str, axiom_id_a: str, axiom_id_b: str) -> str:
    return (
        f"Generate a contrastive pair for the made-up name {name!r}. "
        f"pair_id={pair_id}. axiom_id_a={axiom_id_a}. axiom_id_b={axiom_id_b}. "
        f"Set pair_id on both axioms. You may use different `shape` enum values "
        f"(definitional/causal/normative/relational/exception) on the two axioms when natural."
    )


# ---------- quality grading ----------

QUALITY_SYSTEM = """You grade synthetic training examples for a small language model.

For each example you'll see (axiom_text, question, answer). Judge:

  1. requires_axiom (1-5): To what degree does answering correctly require the axiom?
     1 = the axiom is irrelevant; 5 = without the axiom the question is unanswerable.
  2. could_produce_without (1-5): To what degree could a base model produce this answer
     without ever reading the axiom (from priors, common knowledge, the question alone)?
     1 = impossible; 5 = trivially.
  3. parrots_or_reasons: 'parrots' if the answer is a near-paraphrase of the axiom;
     'reasons' if it's an inference; 'neither' if it's neither.
  4. rationale: one short sentence explaining your scoring.

Return a JSON object matching the QualityGrade schema.
"""


def build_quality_user_prompt(axiom_text: str, question: str, answer: str) -> str:
    return f"Axiom: {axiom_text}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nGrade this example."
