"""Evaluation: the episode loop, metric aggregation and run artifacts."""

from benchtop.eval.metrics import aggregate, wilson_interval
from benchtop.eval.registry import resolve_policy
from benchtop.eval.runner import EvalConfig, run_suite
from benchtop.eval.suite import Suite, load_suite

__all__ = [
    "EvalConfig",
    "Suite",
    "aggregate",
    "load_suite",
    "resolve_policy",
    "run_suite",
    "wilson_interval",
]
