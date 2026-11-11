"""``rebaseline --only`` and ``rename-case`` of the parity harness, on a synthetic record (no model is evaluated)."""
import importlib.util
import json
import os

import numpy as np
import pytest

HARNESS = os.path.join(os.path.dirname(__file__), "parity_analytic", "harness.py")


@pytest.fixture
def harness(monkeypatch):
    spec = importlib.util.spec_from_file_location("parity_harness_under_test", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _array(seed, n=4):
    return np.random.default_rng(seed).uniform(0.1, 1.0, n)


ENV_OLD = {"python": "3.12.3", "numpy": "2.5.4", "jax": "0.12.0"}
ENV_NEW = {"python": "3.11.9", "numpy": "2.4.0", "jax": "0.10.0"}
CASES = {"numpy/A/x": 1, "numpy/A/y": 2, "numpy/B/x": 3, "jax_f32/A/x": 4}


@pytest.fixture
def record(harness, tmp_path):
    """A record of CASES; ``emitted`` (the current code's outputs) differs from it in every case but one."""
    sections = {s: {} for s in harness.SECTIONS}
    arrays = {s: {} for s in harness.SECTIONS}
    for cid, seed in CASES.items():
        section = cid.split("/")[0]
        a = _array(seed)
        sections[section][cid] = {"sha256": harness.digest(a), "shape": [4], "dtype": a.dtype.str, "bitwise": True}
        arrays[section][cid.replace("/", "__")] = a
    manifest = {"format": 1, "environment": dict(ENV_OLD, pinned_env={}), "sections": sections}
    for s in harness.SECTIONS:
        np.savez(tmp_path / f"{s}.npz", **arrays[s])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def _install_fake_code(harness, monkeypatch, drift=True):
    """The code emits CASES shifted by 1e-3 (environment drift) except numpy/B/x, which is unchanged."""
    harness.REGISTRY.clear()
    for cid in CASES:
        harness.add(cid.split("/")[0], cid, lambda: None)
    calls = []

    def fake_run_emit(section, out_dir, only=None):
        calls.append((section, only))
        out = {}
        for cid, seed in CASES.items():
            if cid.split("/")[0] == section and (only is None or harness.matches(cid, only)):
                out[cid.replace("/", "__")] = _array(seed) + (0.0 if cid == "numpy/B/x" else 1e-3)
        np.savez(os.path.join(out_dir, section + ".npz"), **out)

    monkeypatch.setattr(harness, "run_emit", fake_run_emit)
    monkeypatch.setattr(harness, "environment", lambda: dict(ENV_NEW, pinned_env={}))
    return calls


def test_only_writes_the_named_keys_and_ignores_other_drift(harness, record, monkeypatch, capsys):
    calls = _install_fake_code(harness, monkeypatch)
    before = json.loads((record / "manifest.json").read_text())
    other_before = harness.load(record / "numpy.npz")
    harness.rebaseline_only(["numpy/A/x"], record_dir=str(record))
    after = json.loads((record / "manifest.json").read_text())
    now = harness.load(record / "numpy.npz")

    assert np.array_equal(now["numpy__A__x"], _array(1) + 1e-3)
    assert after["sections"]["numpy"]["numpy/A/x"]["sha256"] == harness.digest(_array(1) + 1e-3)
    for cid in ("numpy/A/y", "numpy/B/x"):                    # drifted (A/y) or not: left as recorded
        assert after["sections"]["numpy"][cid] == before["sections"]["numpy"][cid]
        assert np.array_equal(now[cid.replace("/", "__")], other_before[cid.replace("/", "__")])
    assert after["sections"]["jax_f32"] == before["sections"]["jax_f32"]
    assert after["environment"] == before["environment"]
    assert after["rewrites"] == [{"environment": {k: ENV_NEW[k] for k in ENV_NEW},
                                  "keys": {"numpy": ["numpy/A/x"]}}]
    assert calls == [("numpy", ["numpy/A/x"])]                 # the other cases are never evaluated
    assert "numpy/A/x: moved" in capsys.readouterr().out


def test_only_accepts_patterns_and_accumulates_rewrites(harness, record, monkeypatch):
    _install_fake_code(harness, monkeypatch)
    harness.rebaseline_only(["numpy/A/*"], record_dir=str(record))
    harness.rebaseline_only(["jax_f32/*"], record_dir=str(record))
    rewrites = json.loads((record / "manifest.json").read_text())["rewrites"]
    assert len(rewrites) == 1                                  # one environment, one record
    assert rewrites[0]["keys"] == {"numpy": ["numpy/A/x", "numpy/A/y"], "jax_f32": ["jax_f32/A/x"]}


def test_only_refuses_a_name_that_matches_nothing(harness, record, monkeypatch):
    _install_fake_code(harness, monkeypatch)
    before = (record / "manifest.json").read_text()
    with pytest.raises(SystemExit, match="no match"):
        harness.rebaseline_only(["numpy/A/x", "numpy/typo"], record_dir=str(record))
    with pytest.raises(SystemExit):
        harness.rebaseline_only([], record_dir=str(record))
    assert (record / "manifest.json").read_text() == before


def test_only_drops_a_matching_recorded_case_that_is_no_longer_registered(harness, record, monkeypatch, capsys):
    calls = _install_fake_code(harness, monkeypatch)
    del harness.REGISTRY["numpy/B/x"]                         # the case left the registry
    before = json.loads((record / "manifest.json").read_text())
    other_before = harness.load(record / "numpy.npz")
    harness.rebaseline_only(["numpy/B/*"], record_dir=str(record))
    after = json.loads((record / "manifest.json").read_text())
    now = harness.load(record / "numpy.npz")

    assert "numpy/B/x" not in after["sections"]["numpy"] and "numpy__B__x" not in now
    for cid in ("numpy/A/x", "numpy/A/y"):                    # everything else is untouched
        assert after["sections"]["numpy"][cid] == before["sections"]["numpy"][cid]
        assert np.array_equal(now[cid.replace("/", "__")], other_before[cid.replace("/", "__")])
    assert after["sections"]["jax_f32"] == before["sections"]["jax_f32"]
    assert calls == []                                        # nothing registered matched: nothing evaluated
    assert "numpy/B/x: removed" in capsys.readouterr().out


def test_only_keeps_the_tolerance_of_a_case_that_is_not_bit_reproducible(harness, record, monkeypatch):
    _install_fake_code(harness, monkeypatch)
    manifest = json.loads((record / "manifest.json").read_text())
    manifest["sections"]["numpy"]["numpy/A/x"].update(bitwise=False, rtol=1e-12, ops=8)
    (record / "manifest.json").write_text(json.dumps(manifest))
    harness.rebaseline_only(["numpy/A/x"], record_dir=str(record))
    entry = json.loads((record / "manifest.json").read_text())["sections"]["numpy"]["numpy/A/x"]
    assert entry["bitwise"] is False and entry["rtol"] == 1e-12


def test_rename_case_moves_keys_byte_for_byte(harness, record):
    before = json.loads((record / "manifest.json").read_text())
    arrays_before = harness.load(record / "numpy.npz")
    harness.rename_case("numpy/A", "numpy/Z", record_dir=str(record))
    after = json.loads((record / "manifest.json").read_text())
    now = harness.load(record / "numpy.npz")
    assert sorted(after["sections"]["numpy"]) == ["numpy/B/x", "numpy/Z/x", "numpy/Z/y"]
    assert after["sections"]["numpy"]["numpy/Z/x"] == before["sections"]["numpy"]["numpy/A/x"]
    assert sorted(now) == ["numpy__B__x", "numpy__Z__x", "numpy__Z__y"]
    assert now["numpy__Z__x"].tobytes() == arrays_before["numpy__A__x"].tobytes()
    assert after["sections"]["jax_f32"] == before["sections"]["jax_f32"]       # another section: untouched
    assert after["environment"] == before["environment"]


def test_rename_case_without_section_prefix_renames_every_section(harness, record):
    harness.rename_case("A", "Z", record_dir=str(record))
    manifest = json.loads((record / "manifest.json").read_text())
    assert "jax_f32/Z/x" in manifest["sections"]["jax_f32"] and "numpy/Z/y" in manifest["sections"]["numpy"]
    assert "jax_f32__Z__x" in harness.load(record / "jax_f32.npz")


def test_rename_case_refuses_a_clash_and_an_unknown_case(harness, record):
    before = (record / "manifest.json").read_text()
    with pytest.raises(SystemExit, match="already exist"):
        harness.rename_case("numpy/A", "numpy/B", record_dir=str(record))
    with pytest.raises(SystemExit, match="no case matches"):
        harness.rename_case("numpy/nothing", "numpy/Q", record_dir=str(record))
    assert (record / "manifest.json").read_text() == before


def test_rename_case_follows_the_rewrite_log(harness, record, monkeypatch):
    _install_fake_code(harness, monkeypatch)
    harness.rebaseline_only(["numpy/A/x"], record_dir=str(record))
    harness.rename_case("numpy/A", "numpy/Z", record_dir=str(record))
    rewrites = json.loads((record / "manifest.json").read_text())["rewrites"]
    assert rewrites[0]["keys"] == {"numpy": ["numpy/Z/x"]}
