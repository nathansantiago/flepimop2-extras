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
"""arviz / NetCDF backend provider for flepimop2."""

__all__ = ["ArvizBackend"]

import os
import re
import socket
import subprocess  # noqa: S404
from datetime import UTC
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import xarray as xr
from flepimop2.axis import AxisCollection
from flepimop2.backend.abc import BackendABC
from flepimop2.meta import RunMeta
from flepimop2.parameter.abc import ModelStateSpecification
from flepimop2.typing import Array, Float64NDArray
from pydantic import Field

_SEED_SUFFIX = re.compile(r"_seed(\d+)$")


def _git_commit() -> str | None:
    """
    Best-effort current git commit SHA for the working directory.

    Returns:
        The commit SHA, or `None` if not running inside a git repository or
        `git` is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _seed_from_name(name: str | None) -> int | None:
    """
    Parse a trailing `_seed<digits>` suffix out of a run name, if present.

    Args:
        name: The run name, e.g. from `RunMeta.name`.

    Returns:
        The parsed seed, or `None` if `name` is missing or has no such
        suffix.
    """
    if name is None:
        return None
    match = _SEED_SUFFIX.search(name)
    return int(match.group(1)) if match else None


def _expected_columns(
    model_state: ModelStateSpecification, axes: AxisCollection
) -> int:
    """
    Compute the number of state columns `model_state` describes.

    Returns:
        The total column count: one per parameter name when `model_state`
        declares no shared axes, or one per `(parameter name, axis-position)`
        combination otherwise.
    """
    names = model_state.parameter_names
    if not model_state.axes:
        return len(names)
    sizes = axes.sizes(*model_state.axes)
    return len(names) * int(np.prod(sizes)) if sizes else len(names)


def _axis_coord_values(axes: AxisCollection, axis_name: str) -> list[Any]:
    """
    Return coordinate values for a named axis, suitable for an xarray coord.

    Returns:
        Category labels for a categorical axis, or representative points for
        a continuous axis.
    """
    axis = axes[axis_name]
    if axis.kind == "categorical":
        return list(axis.labels) if axis.labels else list(range(axis.size))
    return list(axis.points())


class ArvizBackend(BackendABC, module="arviz"):
    """arviz / NetCDF backend.

    Each call to `save()` lays out the engine's `(T, n_state)` (or `(T, 1 +
    n_state)` with a leading time column) array as a labelled
    `xarray.Dataset` using the `model_state`/`axes` context `Simulator`
    forwards, wraps it as an `arviz.InferenceData[group]` with single `chain`
    / `draw` coordinates, and writes it as its own NetCDF shard file. Shards
    are combined into one file by the companion `arviz_combine` process step
    (see the package README for why writes are sharded rather than
    accumulated into one growing file).

    Attributes:
        root: Output directory. Shards are written under
            `<root>/<name>_shards/`; the combined file (produced separately
            by `arviz_combine`) is expected at `<root>/<name>.nc`.
        name: Basename used for the shard directory and the (eventual)
            combined file.
        group: Which arviz `InferenceData` group this backend instance
            writes into.
        mode: `append` (default) only ever adds new shards. `overwrite`
            clears this backend instance's shard directory once (the first
            time `save()` is called), then behaves like `append`.
        has_time_column: When `True` (the convention used by flepimop2's own
            reference engines, e.g. the `solve_ivp` demo plugin), column 0
            of `data` is treated as the time vector and the remaining
            columns as state. `False` is not currently supported -- without
            a time column there is no source for time coordinates.
        chain_env / draw_env: Env-var names read at save time to label the
            chain and draw indices for this shard. `chain_env` falls back to
            `0`; `draw_env` falls back to counting existing shards already
            written for that chain.
    """

    root: Path = Field(default_factory=lambda: Path.cwd() / "model_output" / "arviz")
    name: str = "trace"
    group: Literal[
        "prior_predictive",
        "posterior_predictive",
        "predictions",
        "observed_data",
    ] = "prior_predictive"
    mode: Literal["append", "overwrite"] = "append"
    has_time_column: bool = True
    chain_env: str = "FLEPIMOP2_CHAIN"
    draw_env: str = "FLEPIMOP2_DRAW"

    _did_overwrite: ClassVar[set[Path]] = set()

    def _shard_dir(self) -> Path:
        return self.root / f"{self.name}_shards"

    def _combined_path(self) -> Path:
        return self.root / f"{self.name}.nc"

    def _chain_index(self) -> int:
        return int(os.environ.get(self.chain_env, "0"))

    def _draw_index(self, shard_dir: Path, chain_idx: int) -> int:
        env_draw = os.environ.get(self.draw_env)
        if env_draw is not None:
            return int(env_draw)
        if not shard_dir.is_dir():
            return 0
        prefix = f"draw_{chain_idx:04d}_"
        return sum(1 for p in shard_dir.glob(f"{prefix}*.nc"))

    def _save(
        self,
        data: Array,
        run_meta: RunMeta,
        *,
        model_state: ModelStateSpecification | None = None,
        axes: AxisCollection | None = None,
    ) -> None:
        """
        Write one draw as its own NetCDF shard.

        Args:
            data: The `(T, n_state)` or `(T, 1 + n_state)` engine output.
            run_meta: Metadata about the current run.
            model_state: Specification describing the semantic ordering of
                `data`'s state columns. Required -- this backend cannot
                produce a labelled NetCDF file without it.
            axes: Resolved runtime axes used to size/label any axes
                `model_state` references. Required whenever `model_state`
                declares axes.

        Raises:
            ValueError: If `model_state` is missing, `has_time_column` is
                `False`, or `data`'s column count doesn't match what
                `model_state` describes.
        """
        if model_state is None:
            msg = (
                "ArvizBackend requires model_state to label its NetCDF "
                "output; got None. This is normally supplied automatically "
                "by Simulator.run() -- check that the configured system "
                "declares model_state()."
            )
            raise ValueError(msg)
        if not self.has_time_column:
            msg = "ArvizBackend currently requires has_time_column=True."
            raise ValueError(msg)
        resolved_axes = axes if axes is not None else AxisCollection()

        shard_dir = self._shard_dir()
        if self.mode == "overwrite" and shard_dir not in self._did_overwrite:
            if shard_dir.is_dir():
                for shard in shard_dir.glob("*.nc"):
                    shard.unlink()
            self._did_overwrite.add(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)

        chain_idx = self._chain_index()
        draw_idx = self._draw_index(shard_dir, chain_idx)

        ds = self._build_dataset(
            np.asarray(data, dtype=np.float64),
            model_state,
            resolved_axes,
            chain_idx,
            draw_idx,
        )
        ds.attrs["commit"] = _git_commit() or ""
        ds.attrs["host"] = socket.gethostname()
        ds.attrs["timestamp"] = run_meta.timestamp.astimezone(UTC).isoformat()
        seed = _seed_from_name(run_meta.name)
        if seed is not None:
            ds.attrs["seed"] = seed
        if run_meta.name is not None:
            ds.attrs["run_name"] = run_meta.name

        shard_path = shard_dir / f"draw_{chain_idx:04d}_{draw_idx:04d}.nc"
        encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
        ds.to_netcdf(shard_path, group=self.group, mode="w", encoding=encoding)

    def _read(self, run_meta: RunMeta) -> Float64NDArray:  # noqa: ARG002
        """
        Read the combined NetCDF file produced by `arviz_combine`.

        Returns:
            Every data variable stacked into a single 2D array.

        Raises:
            FileNotFoundError: If the combined file doesn't exist yet. Shards
                written by `save()` must be consolidated with the
                `arviz_combine` process step first.
        """
        combined_path = self._combined_path()
        if not combined_path.exists():
            msg = (
                f"No combined file at {combined_path}. Run the "
                "'arviz_combine' process step to consolidate shards from "
                f"{self._shard_dir()} first."
            )
            raise FileNotFoundError(msg)
        with xr.open_dataset(combined_path, group=self.group) as ds:
            return np.concatenate(
                [
                    np.asarray(ds[v].values).reshape(ds.sizes["draw"], -1)
                    for v in ds.data_vars
                ],
                axis=1,
            ).astype(np.float64)

    @staticmethod
    def _variable(
        block: Float64NDArray,
        name: str,
        *,
        sizes: tuple[int, ...],
        coords: dict[str, Any],
    ) -> xr.DataArray:
        """
        Build one labelled `DataArray` from a `(time, *sizes)`-shaped block.

        Args:
            block: The `(time, n_per_name)` column slice for this variable.
            name: The variable's name.
            sizes: Sizes of the axes shared across every variable.
            coords: `chain`/`draw`/`time`/axis coordinates; `coords`' own
                axis-name keys (beyond `chain`/`draw`/`time`) determine
                `axis_names` for the output's dims.

        Returns:
            The block reshaped to `(chain=1, draw=1, time, *sizes)` with
            `coords` attached.
        """
        axis_names = tuple(k for k in coords if k not in {"chain", "draw", "time"})
        t_size = block.shape[0]
        shape = (1, 1, t_size, *sizes) if axis_names else (1, 1, t_size)
        dims = ("chain", "draw", "time", *axis_names)
        return xr.DataArray(block.reshape(shape), dims=dims, coords=coords, name=name)

    @classmethod
    def _build_dataset(
        cls,
        array: Float64NDArray,
        model_state: ModelStateSpecification,
        axes: AxisCollection,
        chain_idx: int,
        draw_idx: int,
    ) -> xr.Dataset:
        """
        Lay out one draw's flat array as a labelled `xarray.Dataset`.

        Returns:
            One `DataArray` per parameter name, each shaped
            `(chain=1, draw=1, time, *model_state.axes)`.

        Raises:
            ValueError: If `array` isn't 2D, or its column count doesn't
                match what `model_state` describes.
        """
        two_dims = 2
        if array.ndim != two_dims:
            msg = f"expected a 2D engine output array, got shape {array.shape}"
            raise ValueError(msg)

        state_block = array[:, 1:]
        expected = _expected_columns(model_state, axes)
        if state_block.shape[1] != expected:
            msg = (
                f"state column count mismatch: data has {state_block.shape[1]} "
                f"state columns, model_state describes {expected}"
            )
            raise ValueError(msg)

        axis_names = model_state.axes
        sizes = axes.sizes(*axis_names) if axis_names else ()
        n_per_name = int(np.prod(sizes)) if sizes else 1
        coords: dict[str, Any] = {
            "chain": [chain_idx],
            "draw": [draw_idx],
            "time": array[:, 0],
            **{name: _axis_coord_values(axes, name) for name in axis_names},
        }

        data_vars: dict[str, xr.DataArray] = {}
        offset = 0
        for name in model_state.labels or model_state.parameter_names:
            block = state_block[:, offset : offset + n_per_name]
            offset += n_per_name
            data_vars[name] = cls._variable(block, name, sizes=sizes, coords=coords)

        return xr.Dataset(data_vars)
