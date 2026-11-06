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

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from flepimop2.axis import AxisCollection
from flepimop2.backend.arviz import ArvizBackend
from flepimop2.meta import RunMeta
from flepimop2.parameter.abc import ModelStateSpecification


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

    backend.save(data, RunMeta(name="run_a"), model_state=model_state)
    backend.save(data, RunMeta(name="run_b"), model_state=model_state)

    shard_dir = tmp_path / "trace_shards"
    shards = sorted(shard_dir.glob("*.nc"))
    assert [p.name for p in shards] == ["draw_0000_0000.nc", "draw_0000_0001.nc"]


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

    shard = next((tmp_path / "trace_shards").glob("*.nc"))
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


def test_read_without_combined_file_raises(tmp_path: Path) -> None:
    """read() points at the arviz_combine step when no combined file exists."""
    backend = ArvizBackend(root=tmp_path)

    with pytest.raises(FileNotFoundError, match="arviz_combine"):
        backend.read(RunMeta())
