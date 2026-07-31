"""Dashboard tests: API + pages render entirely from the committed fixture run."""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from benchtop.cli import app as cli_app
from benchtop.core.types import MANIFEST_FILENAME, METRICS_FILENAME
from benchtop.server import create_app
from benchtop.server.app import intervals_overlap
from benchtop.server.runs import RunNotFound, RunStore

FIXTURE_RUNS = pathlib.Path(__file__).resolve().parent / "fixtures" / "runs"
FIXTURE_RUN_ID = "20260730T120000Z-random-pick_cube_v0"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(FIXTURE_RUNS))


@pytest.fixture
def two_run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """The fixture run plus a copy with a higher, non-overlapping success rate."""
    runs = tmp_path / "runs"
    runs.mkdir()
    shutil.copytree(FIXTURE_RUNS / FIXTURE_RUN_ID, runs / FIXTURE_RUN_ID)

    other_id = "20260731T120000Z-expert-pick_cube_v0"
    other = runs / other_id
    shutil.copytree(FIXTURE_RUNS / FIXTURE_RUN_ID, other)

    manifest = json.loads((other / MANIFEST_FILENAME).read_text())
    manifest["run_id"] = other_id
    manifest["policy"] = {"spec": "expert", "type": "expert"}
    manifest["finished_at"] = "2026-07-31T12:03:20+00:00"
    (other / MANIFEST_FILENAME).write_text(json.dumps(manifest))

    metrics = json.loads((other / METRICS_FILENAME).read_text())
    metrics.update(
        successes=6, success_rate=1.0, success_rate_ci_low=0.6097, success_rate_ci_high=1.0
    )
    (other / METRICS_FILENAME).write_text(json.dumps(metrics))
    return runs


# ------------------------------------------------------------------ store


def test_store_lists_only_run_directories(tmp_path: pathlib.Path):
    runs = tmp_path / "runs"
    (runs / "not-a-run").mkdir(parents=True)
    shutil.copytree(FIXTURE_RUNS / FIXTURE_RUN_ID, runs / FIXTURE_RUN_ID)
    store = RunStore(runs)
    assert [r.run_id for r in store.list_runs()] == [FIXTURE_RUN_ID]


def test_store_orders_newest_first(two_run_dir: pathlib.Path):
    ids = [r.run_id for r in RunStore(two_run_dir).list_runs()]
    assert ids == ["20260731T120000Z-expert-pick_cube_v0", FIXTURE_RUN_ID]


def test_store_missing_dir_is_empty_not_an_error(tmp_path: pathlib.Path):
    assert RunStore(tmp_path / "nope").list_runs() == []


@pytest.mark.parametrize("bad", ["..", "../..", "nope", "sub/dir"])
def test_store_rejects_unknown_and_escaping_ids(bad: str):
    with pytest.raises(RunNotFound):
        RunStore(FIXTURE_RUNS).get(bad)


# -------------------------------------------------------------------- API


def test_api_runs_lists_the_fixture(client: TestClient):
    body = client.get("/api/runs").json()
    assert [r["run_id"] for r in body["runs"]] == [FIXTURE_RUN_ID]
    run = body["runs"][0]
    assert run["policy"]["spec"] == "random"
    assert run["task"] == "pick_cube-v0"
    assert run["episodes"] == 6
    assert run["success_rate_ci_low"] < run["success_rate"] < run["success_rate_ci_high"]


def test_api_run_detail_matches_artifacts(client: TestClient):
    body = client.get(f"/api/runs/{FIXTURE_RUN_ID}").json()
    assert body["manifest"]["git_sha"]
    assert body["metrics"]["successes"] == 1
    assert len(body["episodes"]) == 6
    assert body["episodes"][0]["seed"] == 10000
    assert body["videos"] == [f"ep{i:03d}.mp4" for i in range(6)]


def test_api_unknown_run_is_404(client: TestClient):
    assert client.get("/api/runs/does-not-exist").status_code == 404


def test_video_is_served_and_traversal_is_refused(client: TestClient):
    ok = client.get(f"/runs/{FIXTURE_RUN_ID}/videos/ep000.mp4")
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "video/mp4"
    assert client.get(f"/runs/{FIXTURE_RUN_ID}/videos/nope.mp4").status_code == 404


# ------------------------------------------------------------------ pages


def test_index_renders_fixture_row(client: TestClient):
    html = client.get("/").text
    assert FIXTURE_RUN_ID in html
    assert "16.7%" in html  # success rate
    assert "3.0%" in html and "56.4%" in html  # its interval


def test_index_without_runs_says_so(tmp_path: pathlib.Path):
    html = TestClient(create_app(tmp_path)).get("/").text
    assert "No runs found" in html


def test_run_detail_renders_cards_videos_and_episodes(client: TestClient):
    html = client.get(f"/runs/{FIXTURE_RUN_ID}").text
    assert "success rate" in html
    assert "95% Wilson CI" in html
    assert html.count("<video") == 6
    assert "10005" in html  # last episode seed
    assert "pick_cube-v0" in html


def test_run_detail_unknown_is_404(client: TestClient):
    assert client.get("/runs/nope").status_code == 404


def test_compare_without_selection_prompts(client: TestClient):
    assert "Pick two runs" in client.get("/compare").text


def test_compare_reports_overlap_as_inconclusive(two_run_dir: pathlib.Path):
    """Same run twice: identical intervals overlap, so the verdict must hedge."""
    client = TestClient(create_app(two_run_dir))
    html = client.get(f"/compare?a={FIXTURE_RUN_ID}&b={FIXTURE_RUN_ID}").text
    assert "Not a distinguishable difference" in html


def test_compare_reports_separated_intervals(two_run_dir: pathlib.Path):
    client = TestClient(create_app(two_run_dir))
    other = "20260731T120000Z-expert-pick_cube_v0"
    html = client.get(f"/compare?a={FIXTURE_RUN_ID}&b={other}").text
    assert "separated at 95%" in html
    assert "83.3% higher" in html
    assert FIXTURE_RUN_ID in html and other in html


def test_intervals_overlap_helper(two_run_dir: pathlib.Path):
    store = RunStore(two_run_dir)
    low = store.get(FIXTURE_RUN_ID)
    high = store.get("20260731T120000Z-expert-pick_cube_v0")
    assert intervals_overlap(low, low)
    assert not intervals_overlap(low, high)


# --------------------------------------------------------------------- CLI


def test_dash_help_documents_runs_dir():
    result = CliRunner().invoke(cli_app, ["dash", "--help"])
    assert result.exit_code == 0
    assert "--runs-dir" in result.stdout


def test_dash_fails_cleanly_on_missing_runs_dir(tmp_path: pathlib.Path):
    result = CliRunner().invoke(cli_app, ["dash", "--runs-dir", str(tmp_path / "nope")])
    assert result.exit_code == 1


def test_dash_serves_the_runs_dir_it_is_given(monkeypatch: pytest.MonkeyPatch):
    served: dict[str, object] = {}

    def fake_run(app, **kwargs):
        served["app"] = app
        served.update(kwargs)

    monkeypatch.setattr("benchtop.commands.dash.uvicorn.run", fake_run)
    result = CliRunner().invoke(
        cli_app, ["dash", "--runs-dir", str(FIXTURE_RUNS), "--port", "9111"]
    )
    assert result.exit_code == 0, result.output
    assert served["port"] == 9111
    with TestClient(served["app"]) as client:  # type: ignore[arg-type]
        assert client.get("/api/runs").json()["runs"][0]["run_id"] == FIXTURE_RUN_ID
