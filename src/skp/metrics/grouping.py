"""Explicit group probability aggregation; pooled OOF differs from mean fold scores."""

import numpy as np


def aggregate_probabilities(group_ids, probabilities, reduction="mean"):
    p = np.asarray(probabilities)
    if (
        len(group_ids) != len(p)
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Expected aligned finite probabilities in [0, 1]")
    if reduction not in {"mean", "max"}:
        raise ValueError("reduction must be mean or max")
    groups = {}
    for index, group in enumerate(group_ids):
        groups.setdefault(group, []).append(index)
    if not groups:
        raise ValueError("No groups")
    return list(groups), np.stack(
        [getattr(p[indices], reduction)(axis=0) for indices in groups.values()]
    )
