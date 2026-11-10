# flepimop2-arviz: An arviz/NetCDF backend provider for flepimop2.
# Copyright (C) 2026  Carl Pearson, Joshua Macdonald, Timothy Willard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Tests for the `flepimop2.backend.arviz` provider."""

import subprocess  # noqa: S404
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from flepimop2.axis import AxisCollection
from flepimop2.backend import arviz as arviz_backend_mod
from flepimop2.backend.arviz import ArvizBackend
from flepimop2.meta import RunMeta
from flepimop2.parameter.abc import ModelStateSpecification


@pytest.fixture(autouse=True)
def _clear_git_commit_cache() -> None:
    """Clear the `_git_commit()` process-lifetime cache between tests."""
    arviz_backend_mod._git_commit.cache_clear()


def test_save_without_model_state_raises(tmp_path: Path) -> None:
    """A clear error is raised when no model_state is available to label data."""
    backend = ArvizBackend(root=tmp_path)
    data = np.array([[0.0, 1.0, 2.0]])

    with pytest.raises(ValueError, match="requires model_state"):
        backend.save(data, RunMeta())


def test_save_writes_one_shard_per_call(tmp_path: Path) -> None:
    """Each save() call writes its own shard file, not a shared growing file."""
    backend = ArvizBackend(root=tmp_path, name="trace")
    model_state = ModelStateSpecification(parameter_names=("s0", "i0", "r0"))
    data = np.array([
        [0.0, 999.0, 1.0, 0.0],
        [1.0, 950.0, 40.0, 10.0],
    ])

    backend.save(data, RunMeta(), model_state=model_state)
    backend.save(data, RunMeta(), model_state=model_state)

    shard_dir = tmp_path / "trace_shards"
    shards = sorted(shard_dir.glob("*.nc"))
    assert [p.name for p in shards] == ["draw_0000_0000.nc", "draw_0000_0001.nc"]


def test_save_uses_run_meta_name_when_provided(tmp_path: Path) -> None:
    """`run_meta.name`, when set, overrides `self.name` for the shard dir."""
    backend = ArvizBackend(root=tmp_path, name="default")
    model_state = ModelStateSpecification(parameter_names=("s0",))
    data = np.array([[0.0, 1.0]])

    backend.save(data, RunMeta(name="per_run"), model_state=model_state)

    assert (tmp_path / "per_run_shards" / "draw_0000_0000.nc").exists()
    assert not (tmp_path / "default_shards").exists()


def test_shard_contents_are_labelled_and_carry_provenance(tmp_path: Path) -> None:
    """A shard's dataset has named variables, a time coord, and provenance attrs."""
    backend = ArvizBackend(root=tmp_path, group="posterior_predictive")
    model_state = ModelStateSpecification(
        parameter_names=("s0", "i0", "r0"), labels=("S", "I", "R")
    )
    data = np.array([
        [0.0, 999.0, 1.0, 0.0],
        [1.0, 950.0, 40.0, 10.0],
    ])

    backend.save(
        data,
        RunMeta(name="demo_seed42"),
        model_state=model_state,
    )

    shard = next((tmp_path / "demo_seed42_shards").glob("*.nc"))
    with xr.open_dataset(shard, group="posterior_predictive") as ds:
        assert set(ds.data_vars) == {"S", "I", "R"}
        assert list(ds["S"].values.reshape(-1)) == [999.0, 950.0]
        assert list(ds["time"].values) == [0.0, 1.0]
        expected_seed = 42
        assert ds.attrs["seed"] == expected_seed
        assert ds.attrs["run_name"] == "demo_seed42"
        assert "host" in ds.attrs
        assert ds["S"].dtype == np.float64


def test_shard_expands_axes_into_coordinates(tmp_path: Path) -> None:
    """model_state.axes expands one parameter into per-axis-position columns."""
    backend = ArvizBackend(root=tmp_path)
    model_state = ModelStateSpecification(parameter_names=("s0",), axes=("age",))
    axes = AxisCollection.from_config({
        "age": {"kind": "categorical", "labels": ["young", "old"]},
    })
    # 1 time point, 2 age-expanded columns.
    data = np.array([[0.0, 10.0, 20.0]])

    backend.save(data, RunMeta(), model_state=model_state, axes=axes)

    shard = next((tmp_path / "trace_shards").glob("*.nc"))
    with xr.open_dataset(shard, group="prior_predictive") as ds:
        assert list(ds["age"].values) == ["young", "old"]
        assert list(ds["s0"].values.reshape(-1)) == [10.0, 20.0]


def test_column_mismatch_raises(tmp_path: Path) -> None:
    """A clear error is raised when data's columns don't match model_state."""
    backend = ArvizBackend(root=tmp_path)
    model_state = ModelStateSpecification(parameter_names=("s0", "i0"))
    data = np.array([[0.0, 1.0]])  # only 1 state column, model_state expects 2

    with pytest.raises(ValueError, match="column count mismatch"):
        backend.save(data, RunMeta(), model_state=model_state)


def test_chain_and_draw_env_vars_control_shard_naming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FLEPIMOP2_CHAIN/FLEPIMOP2_DRAW override the default indices."""
    monkeypatch.setenv("FLEPIMOP2_CHAIN", "3")
    monkeypatch.setenv("FLEPIMOP2_DRAW", "7")
    backend = ArvizBackend(root=tmp_path)
    model_state = ModelStateSpecification(parameter_names=("s0",))
    data = np.array([[0.0, 1.0]])

    backend.save(data, RunMeta(), model_state=model_state)

    shard = tmp_path / "trace_shards" / "draw_0003_0007.nc"
    assert shard.exists()


def test_overwrite_mode_clears_existing_shards_once(tmp_path: Path) -> None:
    """mode='overwrite' clears prior shards the first time save() is called."""
    shard_dir = tmp_path / "trace_shards"
    shard_dir.mkdir(parents=True)
    (shard_dir / "stale.nc").write_text("not a real netcdf file, just a marker")

    backend = ArvizBackend(root=tmp_path, mode="overwrite")
    model_state = ModelStateSpecification(parameter_names=("s0",))
    data = np.array([[0.0, 1.0]])

    backend.save(data, RunMeta(), model_state=model_state)

    remaining = sorted(p.name for p in shard_dir.glob("*"))
    assert "stale.nc" not in remaining
    assert "draw_0000_0000.nc" in remaining


def test_read_without_any_file_raises(tmp_path: Path) -> None:
    """read() raises when neither a shard nor a combined file exists."""
    backend = ArvizBackend(root=tmp_path)

    with pytest.raises(FileNotFoundError, match="neither the per-draw shard"):
        backend.read(RunMeta())


def test_read_roundtrip_from_shard_matches_save_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """read() returns the `(T, 1 + n_state)` array that save() received."""
    monkeypatch.setenv("FLEPIMOP2_CHAIN", "0")
    monkeypatch.setenv("FLEPIMOP2_DRAW", "0")
    backend = ArvizBackend(root=tmp_path)
    model_state = ModelStateSpecification(
        parameter_names=("s0", "i0", "r0"), labels=("S", "I", "R")
    )
    data = np.array([
        [0.0, 999.0, 1.0, 0.0],
        [1.0, 950.0, 40.0, 10.0],
        [2.0, 800.0, 150.0, 50.0],
    ])

    backend.save(data, RunMeta(), model_state=model_state)
    out = np.asarray(backend.read(RunMeta()))

    assert out.shape == data.shape
    np.testing.assert_allclose(out, data)


def test_read_roundtrip_with_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """read() reshapes axis-expanded variables back to flat engine columns."""
    monkeypatch.setenv("FLEPIMOP2_CHAIN", "0")
    monkeypatch.setenv("FLEPIMOP2_DRAW", "0")
    backend = ArvizBackend(root=tmp_path)
    model_state = ModelStateSpecification(parameter_names=("s0",), axes=("age",))
    axes = AxisCollection.from_config({
        "age": {"kind": "categorical", "labels": ["young", "old"]},
    })
    data = np.array([
        [0.0, 10.0, 20.0],
        [1.0, 11.0, 21.0],
    ])

    backend.save(data, RunMeta(), model_state=model_state, axes=axes)
    out = np.asarray(backend.read(RunMeta()))

    assert out.shape == data.shape
    np.testing.assert_allclose(out, data)


def test_read_falls_back_to_combined_when_shard_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """read() uses the combined file if no per-draw shard is on disk."""
    model_state = ModelStateSpecification(parameter_names=("s0", "i0"))
    # Write three draws, varying the FLEPIMOP2_DRAW env var so they are all
    # in the same chain and land in a predictable order.
    monkeypatch.setenv("FLEPIMOP2_CHAIN", "0")
    for draw in range(3):
        monkeypatch.setenv("FLEPIMOP2_DRAW", str(draw))
        backend = ArvizBackend(root=tmp_path)
        data = np.array([[0.0, float(draw), float(draw) + 1.0]])
        backend.save(data, RunMeta(), model_state=model_state)

    from flepimop2.process.arviz_combine import ArvizCombineProcess  # noqa: PLC0415

    process = ArvizCombineProcess(
        shard_dir=tmp_path / "trace_shards",
        target=tmp_path / "trace.nc",
    )
    process.execute()

    # Blow away the shards so only the combined file is available.
    for shard in (tmp_path / "trace_shards").glob("*.nc"):
        shard.unlink()

    monkeypatch.setenv("FLEPIMOP2_DRAW", "2")
    backend = ArvizBackend(root=tmp_path)
    out = np.asarray(backend.read(RunMeta()))
    expected = np.array([[0.0, 2.0, 3.0]])
    np.testing.assert_allclose(out, expected)


def test_draw_index_requires_both_env_vars_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting FLEPIMOP2_CHAIN without FLEPIMOP2_DRAW raises RuntimeError."""
    monkeypatch.setenv("FLEPIMOP2_CHAIN", "2")
    monkeypatch.delenv("FLEPIMOP2_DRAW", raising=False)
    backend = ArvizBackend(root=tmp_path)
    model_state = ModelStateSpecification(parameter_names=("s0",))
    data = np.array([[0.0, 1.0]])

    with pytest.raises(RuntimeError, match="both must be set together"):
        backend.save(data, RunMeta(), model_state=model_state)


def test_draw_index_fallback_when_neither_env_var_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither env var set stays ergonomic for the local serial case."""
    monkeypatch.delenv("FLEPIMOP2_CHAIN", raising=False)
    monkeypatch.delenv("FLEPIMOP2_DRAW", raising=False)
    backend = ArvizBackend(root=tmp_path)
    model_state = ModelStateSpecification(parameter_names=("s0",))
    data = np.array([[0.0, 1.0]])

    backend.save(data, RunMeta(), model_state=model_state)
    backend.save(data, RunMeta(), model_state=model_state)

    shards = sorted((tmp_path / "trace_shards").glob("*.nc"))
    assert [p.name for p in shards] == ["draw_0000_0000.nc", "draw_0000_0001.nc"]


def test_did_overwrite_is_per_instance(tmp_path: Path) -> None:
    """Two instances at the same root each get their one-shot overwrite."""
    shard_dir = tmp_path / "trace_shards"
    shard_dir.mkdir(parents=True)
    stale = shard_dir / "stale_a.nc"
    stale.write_text("not real netcdf")

    model_state = ModelStateSpecification(parameter_names=("s0",))
    data = np.array([[0.0, 1.0]])

    first = ArvizBackend(root=tmp_path, mode="overwrite")
    first.save(data, RunMeta(), model_state=model_state)
    assert not stale.exists()
    assert first._did_overwrite is True

    # A fresh stale marker written after the first instance's clear should
    # survive subsequent saves by that same instance (one-shot semantic).
    survivor = shard_dir / "survivor.nc"
    survivor.write_text("still here")
    first.save(data, RunMeta(), model_state=model_state)
    assert survivor.exists()

    # A second independent instance should clear its own first-call shards,
    # including `survivor.nc`, without sharing state with `first`.
    second = ArvizBackend(root=tmp_path, mode="overwrite")
    assert second._did_overwrite is False
    second.save(data, RunMeta(), model_state=model_state)
    assert not survivor.exists()


def test_git_commit_is_cached_across_calls() -> None:
    """`_git_commit()` runs `subprocess.run` at most once per process."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "deadbeef\n"
    with patch.object(subprocess, "run", return_value=mock_result) as mock_run:
        first = arviz_backend_mod._git_commit()
        second = arviz_backend_mod._git_commit()
        third = arviz_backend_mod._git_commit()

    assert first == "deadbeef"
    assert second == "deadbeef"
    assert third == "deadbeef"
    assert mock_run.call_count == 1
