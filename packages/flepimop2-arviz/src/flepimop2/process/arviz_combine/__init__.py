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
"""Process step that combines `ArvizBackend` shards into one NetCDF file."""

__all__ = ["ArvizCombineProcess"]

from pathlib import Path
from typing import Literal

import xarray as xr
from flepimop2.exceptions import Flepimop2ValidationError, ValidationIssue
from flepimop2.process.abc import ProcessABC
from flepimop2.typing import override


class ArvizCombineProcess(ProcessABC, module="arviz_combine"):
    """Combine `ArvizBackend` shard files into one NetCDF file.

    `ArvizBackend.save()` writes one independent shard per draw (see that
    package's README for why). This step globs the shard directory and
    concatenates every shard along the `draw` dimension into a single
    combined file in one write -- the only read/rewrite in the whole
    pipeline, done once rather than once per draw.

    Attributes:
        shard_dir: Directory containing the per-draw `.nc` shards (an
            `ArvizBackend` instance writes these to
            `<root>/<name>_shards/`).
        target: Path to write the combined file to (an `ArvizBackend`
            reading this data expects `<root>/<name>.nc`).
        group: The arviz `InferenceData` group stored in each shard.
    """

    shard_dir: Path
    target: Path
    group: Literal[
        "prior_predictive",
        "posterior_predictive",
        "predictions",
        "observed_data",
    ] = "prior_predictive"

    def _shards(self) -> list[Path]:
        if not self.shard_dir.is_dir():
            return []
        return sorted(self.shard_dir.glob("*.nc"))

    @override
    def is_satisfied(self) -> bool:
        """
        Report whether `target` is already up to date with every shard.

        Returns:
            `True` if `target` exists and is at least as new as every shard
            currently in `shard_dir`, so re-running the pipeline after a
            successful combine costs nothing.
        """
        if not self.target.exists():
            return False
        shards = self._shards()
        if not shards:
            return True
        target_mtime = self.target.stat().st_mtime
        return all(shard.stat().st_mtime <= target_mtime for shard in shards)

    @override
    def _process_validate(self) -> list[ValidationIssue]:
        """
        Check that there is something to combine.

        Returns:
            A validation issue when `shard_dir` has no shards yet, otherwise
            an empty list.
        """
        if not self._shards():
            return [
                ValidationIssue(
                    msg=f"No shard files found in {self.shard_dir}.",
                    kind="no_shards",
                    ctx={"shard_dir": str(self.shard_dir)},
                )
            ]
        return []

    def _process(self, *, dry_run: bool) -> None:
        """
        Concatenate every shard along `draw` and write `target` in one pass.

        Args:
            dry_run: If `True`, report the plan without writing anything.

        Raises:
            Flepimop2ValidationError: If `shard_dir` has no shards.
        """
        if issues := self._process_validate():
            raise Flepimop2ValidationError(issues)
        shards = self._shards()
        if dry_run:
            print(  # noqa: T201
                f"Would combine {len(shards)} shard(s) from {self.shard_dir} "
                f"into {self.target} (group={self.group})."
            )
            return

        self.target.parent.mkdir(parents=True, exist_ok=True)
        datasets = [xr.open_dataset(shard, group=self.group).load() for shard in shards]
        try:
            combined = xr.concat(datasets, dim="draw", data_vars="all").sortby("draw")
        finally:
            for ds in datasets:
                ds.close()
        encoding = {var: {"zlib": True, "complevel": 4} for var in combined.data_vars}
        combined.to_netcdf(self.target, group=self.group, mode="w", encoding=encoding)
