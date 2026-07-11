"""Tests for the encoder-on-reasoning check (reason_check.py).

Model-free: GSM8K-style solution parsing — strip calculator annotations,
drop the #### answer line, split into steps, pair consecutive steps. The
gap_closed math itself is already tested in test_gist_model.py.
"""

from __future__ import annotations

from marker.reason_check import split_solution_steps, step_pairs

GSM8K_SOLUTION = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n"
    "#### 72"
)


def test_split_strips_calculator_annotations():
    steps = split_solution_steps(GSM8K_SOLUTION)
    assert steps == [
        "Natalia sold 48/2 = 24 clips in May.",
        "Natalia sold 48+24 = 72 clips altogether.",
    ]
    assert all("<<" not in s and ">>" not in s for s in steps)


def test_split_drops_answer_line_and_blanks():
    steps = split_solution_steps("Step one.\n\n   \nStep two.\n#### 5")
    assert steps == ["Step one.", "Step two."]


def test_step_pairs_consecutive():
    assert step_pairs(["a", "b", "c"]) == [("a", "b"), ("b", "c")]
    assert step_pairs(["only"]) == []
    assert step_pairs([]) == []
