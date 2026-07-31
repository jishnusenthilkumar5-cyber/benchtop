from __future__ import annotations

import pytest
import yaml

from benchtop.core.types import TASK_ID, assert_held_out
from benchtop.eval.suite import load_suite, parse_suite


def test_committed_suite_is_held_out_and_resolves(suite_path):
    suite = load_suite(suite_path)
    assert suite.task == TASK_ID
    assert suite.name == "pick_cube_v0"
    assert len(suite.seeds) == 100
    assert suite.seeds[0] == 10000
    assert suite.max_steps == 500
    assert suite.video is True
    assert_held_out(suite.seeds)


def test_seed_start_and_episodes_expand_to_a_seed_list():
    suite = parse_suite({"name": "s", "task": TASK_ID, "seed_start": 10000, "episodes": 3})
    assert suite.seeds == (10000, 10001, 10002)


def test_explicit_seed_list_is_kept():
    suite = parse_suite({"name": "s", "task": TASK_ID, "seeds": [10005, 10001]})
    assert suite.seeds == (10005, 10001)


def test_demo_seeds_are_refused():
    with pytest.raises(ValueError, match="held out|>= 10000|must be"):
        parse_suite({"name": "s", "task": TASK_ID, "seeds": [0, 10000]})


def test_unknown_task_is_refused():
    with pytest.raises(ValueError, match="unknown task"):
        parse_suite({"name": "s", "task": "pick_cube-v1", "seeds": [10000]})


def test_empty_seed_list_is_refused():
    with pytest.raises(ValueError):
        parse_suite({"name": "s", "task": TASK_ID, "seeds": []})


def test_resolved_suite_serialises_the_seed_list(tmp_path, suite_path):
    raw = yaml.safe_load(suite_path.read_text())
    raw["episodes"] = 2
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_suite(path).to_dict() == {
        "name": "pick_cube_v0",
        "task": TASK_ID,
        "max_steps": 500,
        "video": True,
        "seeds": [10000, 10001],
    }
