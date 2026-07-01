"""Lightweight smoke tests (no GPU, no network, stdlib + pytest only).

They validate the results-aggregation pipeline and that the committed results
tables are well-formed — enough to catch breakage in CI without a full run.
"""
import csv
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_formatting_helpers():
    m = _load_script("build_results_table")
    assert m.fmt(0.40740) == "0.4074"
    assert m.fmt(None) == "-"
    assert m.first({"a": 1}, "b", "a") == 1
    assert m.first(None, "a") is None
    table = m.md_table(["A", "B"], [["1", "2"]])
    assert "| A | B |" in table and "| --- | --- |" in table


def test_classification_results_present_and_parseable():
    p = ROOT / "results" / "results_classification.csv"
    assert p.exists(), "run `python scripts/build_results_table.py` first"
    rows = list(csv.DictReader(p.open()))
    names = {r["dataset"] for r in rows}
    assert {"Helena", "Jannis", "MiniBooNE", "Volkert"} <= names
    # every quoted accuracy must be a valid probability
    for r in rows:
        v = r["llm_nas_ensemble"]
        if v not in ("", "-"):
            assert 0.0 <= float(v) <= 1.0


def test_forecasting_results_present():
    p = ROOT / "results" / "results_forecasting.csv"
    assert p.exists()
    rows = list(csv.DictReader(p.open()))
    assert any(r["dataset"] == "exchange" for r in rows)
