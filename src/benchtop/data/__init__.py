"""Demonstration data: expert rollouts written as LeRobot datasets.

This package is one of the two places allowed to import `lerobot` (the other
is `adapters/`). Keeping the dependency quarantined here means an API change
in lerobot is a change to two modules, not to the whole codebase.
"""

from __future__ import annotations

from benchtop.data.collect import CollectionReport, collect_dataset, dataset_features

__all__ = ["CollectionReport", "collect_dataset", "dataset_features"]
