"""The demo server's API, driven through Flask's test client."""

import itertools

import pytest

from demo.app import app
from examples import SCENARIOS

client = app.test_client()
TOGGLES = list(itertools.product([0, 1], repeat=4))  # inline, fold, dedup, cache


def test_index():
    assert b"Cell demo" in client.get("/").data


def test_scenarios():
    listed = client.get("/api/scenarios").json
    assert [s["name"] for s in listed] == [s.name for s in SCENARIOS]


@pytest.mark.parametrize("s", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_every_step_of_every_scenario(s):
    for source in [t for t in SCENARIOS if t.cell is s.cell]:
        for inline, fold, dedup, cache in TOGGLES if source is s else [(1, 1, 1, 0)]:
            q = {"name": s.name, "trace_from": source.name, "inline": inline, "fold": fold, "dedup": dedup, "cache": cache}
            r = client.get("/api/run", query_string=q)
            assert r.status_code == 200, r.data
            j = r.json
            assert j["compiled"]["matches_eager"], (q, j["compiled"]["journal"])
            for view in (j["traced"], j["rewritten"]):
                assert view["mermaid"].startswith("flowchart TD")
                assert view["ir"].startswith("graph ")


def test_trace_from_another_cell_is_rejected():
    r = client.get("/api/run", query_string={"name": "home/premium", "trace_from": "feed/top2"})
    assert r.status_code == 400


def test_simulate():
    body = {"name": "home/premium", "requests": 40, "clients": 4, "capacity": 2, "cache": True}
    j = client.post("/api/simulate", json=body).json
    labels = ["eager", "compiled", "compiled + inline, fold, dedup, cache", "compiled + inline, fold, dedup, cache + batching"]
    assert [r["label"] for r in j["results"]] == labels
    for r in j["results"]:
        assert r["requests"] == 40
        assert r["p50"] <= r["p90"] <= r["p99"] <= r["max"]
        assert len(r["cdf"]) == 100 and r["throughput"] > 0
    hist = j["histogram"]
    assert len(hist["counts"]) == 4 and all(sum(c) == 40 for c in hist["counts"])
    # Caching cuts leaf calls; compiling doesn't change them.
    eager, compiled, cached, _ = j["results"]
    assert eager["calls_per_request"] == compiled["calls_per_request"] > cached["calls_per_request"]


def test_simulation_is_repeatable():
    body = {"name": "checkout/ok", "requests": 30}
    a = client.post("/api/simulate", json=body).json
    b = client.post("/api/simulate", json=body).json
    assert a["results"] == b["results"]


def test_batch_step():
    j = client.get("/api/run", query_string={"name": "feed_mapped/busy"}).json["batch"]
    assert all(lane["matches_eager"] for lane in j["lanes"])
    modes = {lane["scenario"]: lane["mode"] for lane in j["lanes"]}
    assert modes["feed_mapped/follows-nobody"] == "deopt" and modes["feed_mapped/busy"] == "compiled"
    assert j["vector_calls"] == 3
    assert sum(j["calls"].values()) < sum(j["calls_one_by_one"].values())


def test_load_curve():
    j = client.post("/api/curve", json={"name": "feed_mapped/busy", "requests": 60, "points": 3, "rate_max": 400}).json
    assert len(j["rates"]) == 3
    assert [c["label"].endswith("batching") for c in j["curves"]] == [False, False, True]
    for c in j["curves"]:
        for p in c["points"]:
            assert p["p50"] <= p["p99"] <= p["max"] and p["throughput"] > 0
