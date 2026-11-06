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
"""Tests for the `flepimop2.process.arviz_combine` provider."""

import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from flepimop2.backend.arviz import ArvizBackend
from flepimop2.exceptions import Flepimop2ValidationError
from flepimop2.meta import RunMeta
from flepimop2.parameter.abc import ModelStateSpecification
from flepimop2.process.arviz_combine import ArvizCombineProcess


def _write_shards(root: Path, n_draws: int) -> None:
    backend = ArvizBackend(root=root)
    model_state = ModelStateSpecification(parameter_names=("s0", "i0"))
    for draw in range(n_draws):
        data = np.array([[0.0, float(draw), float(draw) + 1.0]])
        backend.save(data, RunMeta(name=f"draw_{draw}"), model_state=model_state)


def test_combine_concatenates_shards_along_draw(tmp_path: Path) -> None:
    """All shards end up in one file, ordered along the draw dimension."""
    _write_shards(tmp_path, n_draws=3)
    process = ArvizCombineProcess(
        shard_dir=tmp_path / "trace_shards",
        target=tmp_path / "trace.nc",
    )

    process.execute()

    with xr.open_dataset(tmp_path / "trace.nc", group="prior_predictive") as ds:
        expected_draws = 3
        assert ds.sizes["draw"] == expected_draws
        assert sorted(ds["draw"].values.tolist()) == [0, 1, 2]
        assert sorted(ds["s0"].values.reshape(-1).tolist()) == [0.0, 1.0, 2.0]


def test_is_satisfied_skips_a_fresh_combine(tmp_path: Path) -> None:
    """A second execute() is a no-op once target is newer than every shard."""
    _write_shards(tmp_path, n_draws=2)
    process = ArvizCombineProcess(
        shard_dir=tmp_path / "trace_shards",
        target=tmp_path / "trace.nc",
    )

    process.execute()
    first_mtime = process.target.stat().st_mtime
    assert process.is_satisfied() is True

    process.execute()
    assert process.target.stat().st_mtime == first_mtime


def test_new_shard_invalidates_satisfaction(tmp_path: Path) -> None:
    """A newly written shard makes is_satisfied() report False again."""
    _write_shards(tmp_path, n_draws=1)
    process = ArvizCombineProcess(
        shard_dir=tmp_path / "trace_shards",
        target=tmp_path / "trace.nc",
    )
    process.execute()
    assert process.is_satisfied() is True

    time.sleep(0.05)
    backend = ArvizBackend(root=tmp_path)
    model_state = ModelStateSpecification(parameter_names=("s0", "i0"))
    backend.save(
        np.array([[0.0, 9.0, 9.0]]), RunMeta(name="late_draw"), model_state=model_state
    )

    assert process.is_satisfied() is False


def test_combine_with_no_shards_raises(tmp_path: Path) -> None:
    """Combining an empty/missing shard directory fails with a clear error."""
    process = ArvizCombineProcess(
        shard_dir=tmp_path / "does_not_exist",
        target=tmp_path / "trace.nc",
    )

    with pytest.raises(Flepimop2ValidationError, match="No shard files"):
        process.execute()


def test_dry_run_does_not_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """dry_run reports the plan without creating the target file."""
    _write_shards(tmp_path, n_draws=1)
    process = ArvizCombineProcess(
        shard_dir=tmp_path / "trace_shards",
        target=tmp_path / "trace.nc",
    )

    process.execute(dry_run=True)

    assert not process.target.exists()
    assert "Would combine 1 shard" in capsys.readouterr().out
