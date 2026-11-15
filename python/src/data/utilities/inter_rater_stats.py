"""Inter-rater reliability helpers shared by the combine_uk and compare_samples
packages, ported from R's `irr` package (irr::kappa2, irr::kappam.fleiss) and
the confusion-matrix helpers duplicated in combining_UK_dataset.R /
comparing_dataset.R so both scripts can share one implementation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def cohens_kappa(a: pd.Series, b: pd.Series, labels: list) -> float:
    """Unweighted Cohen's kappa, matching R's irr::kappa2 default. `labels`
    is passed explicitly so the category set is fixed regardless of what's
    observed in a given pair's subset."""
    return float(cohen_kappa_score(a, b, labels=labels))


def fleiss_kappa(ratings: pd.DataFrame, labels: list) -> dict:
    """Fleiss' kappa for m raters, overall + per-category, ported line for
    line from irr::kappam.fleiss(ratings, detail=TRUE) (non-exact branch) so
    the results match R exactly. `ratings` is subjects (rows) x raters
    (columns); every cell must be one of `labels`, no missing values.

    Returns {"overall": float, "per_category": {label: float, ...}}.
    """
    assert not ratings.isna().any().any(), \
        "fleiss_kappa requires complete ratings (no NaN) for every rater"

    ns, nr = ratings.shape
    label_index = {label: i for i, label in enumerate(labels)}
    values = ratings.to_numpy()

    table = np.zeros((ns, len(labels)), dtype=float)
    for i in range(ns):
        for j in range(nr):
            table[i, label_index[values[i, j]]] += 1

    agree_p = float(np.mean((np.sum(table ** 2, axis=1) - nr) / (nr * (nr - 1))))

    col_sums = table.sum(axis=0)
    pj = col_sums / (ns * nr)
    chance_p = float(np.sum(pj ** 2))

    overall = (agree_p - chance_p) / (1 - chance_p)

    pjk = (np.sum(table ** 2, axis=0) - ns * nr * pj) / (ns * nr * (nr - 1) * pj)
    kappa_k = (pjk - pj) / (1 - pj)

    return {
        "overall": overall,
        "per_category": {label: float(k) for label, k in zip(labels, kappa_k)},
    }


def confusion_matrix(a: pd.Series, b: pd.Series, labels: list) -> pd.DataFrame:
    """Sentiment confusion matrix between two raters, with a fixed row/column
    label order so downstream indexing (e.g. cm.loc[-1, 1]) is safe even if a
    category is unobserved in this particular pair."""
    cm = pd.crosstab(a, b)
    return cm.reindex(index=labels, columns=labels, fill_value=0)


def total_confusion(cm: pd.DataFrame) -> int:
    """All off-diagonal (misclassified) counts."""
    return int(cm.to_numpy().sum() - np.trace(cm.to_numpy()))


def polar_confusion(cm: pd.DataFrame, negative=-1, positive=1) -> int:
    """Negative<->positive mixups, skipping the neutral class entirely."""
    return int(cm.loc[negative, positive] + cm.loc[positive, negative])


def neutral_confusion(cm: pd.DataFrame, negative=-1, neutral=0, positive=1) -> int:
    """Neutral<->negative or neutral<->positive mixups (the "neutral boundary")."""
    return int(
        cm.loc[neutral, negative] + cm.loc[neutral, positive]
        + cm.loc[negative, neutral] + cm.loc[positive, neutral]
    )


def update_stats_json(path: Path, key: str, payload: dict) -> None:
    """Merge `payload` into `path` under `key`, preserving whatever other
    keys are already there -- lets combine_uk and compare_samples share one
    interannotator_agreement.json and be run independently, in either order,
    without clobbering each other's sections."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[key] = payload
    path.write_text(json.dumps(existing, indent=2))
