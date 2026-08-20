from __future__ import annotations

import numpy as np
import pytest

from evaluation.cross_cohort import binary_scores, rank_rows, summarize_runs


def test_rank_rows_is_finite_and_scale_free() -> None:
    values = np.asarray([[10, 20, 30], [100, 200, 300]], dtype=float)
    ranked = rank_rows(values)
    np.testing.assert_allclose(ranked[0], ranked[1])
    np.testing.assert_allclose(ranked[0], [0.0, 0.5, 1.0])


def test_binary_scores() -> None:
    scores = binary_scores(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]))
    assert scores["macro_f1"] == 1.0
    assert scores["idh_mutation_auroc"] == 1.0


def test_headline_advantage_direction() -> None:
    runs = []
    for arm, drop in (
        ("confirmed_genes", 0.1),
        ("unconfirmed_genes", 0.3),
        ("shuffled_size_matched_control", 0.2),
    ):
        runs.append(
            {
                "arm": arm,
                "internal": {"macro_f1": 0.8},
                "external": {"macro_f1": 0.8 - drop, "idh_mutation_auroc": 0.7},
                "internal_to_external_macro_f1_drop": drop,
            }
        )
    summary = summarize_runs(runs)
    assert summary["headline"]["confirmed_drop_advantage"] == pytest.approx(0.2)
    assert summary["headline"]["confirmed_beats_shuffled"] is True
    assert summary["headline"]["result_call"] == "candidate_result"
