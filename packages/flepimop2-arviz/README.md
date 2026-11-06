# `flepimop2-arviz`

`flepimop2-arviz` supplies the `ArvizBackend` provider for writing
[flepimop2](https://github.com/ACCIDDA/flepimop2) simulation output as
[arviz](https://www.arviz.org/)/NetCDF `InferenceData` groups, and an
`ArvizCombineProcess` provider for consolidating the per-draw shard files
it writes into a single combined NetCDF file.

After installation, select `module: arviz` in a backend configuration:

```yaml
backend:
  - module: arviz
    name: prior_predictive
    group: prior_predictive
    root: model_output/arviz
```

## Why sharded files instead of one growing file?

Each call to `save()` writes its own self-contained NetCDF shard
(`<root>/<name>_shards/draw_<chain>_<draw>.nc`) instead of reading and
rewriting a single accumulating file. This keeps per-draw writes O(1)
regardless of how many draws have already been written, and-critically for
HPC use-means no two processes ever read or write the same file, so there is
no reliance on file locking over a shared (often network) filesystem.

Combine the shards into one file once the sweep is done, using the
`arviz_combine` process step:

```yaml
process:
  combine:
    module: arviz_combine
    shard_dir: model_output/arviz/prior_predictive_shards
    target: model_output/arviz/prior_predictive.nc
    group: prior_predictive
```

```bash
flepimop2 process -t combine configs/config.yaml
```

Re-running `combine` is cheap: it skips the work if `target` is already
newer than every shard.

## Provenance

Each shard's NetCDF global attributes record `commit` (best-effort `git
rev-parse HEAD` in the working directory), `host` (`socket.gethostname()`),
and `timestamp` (from `RunMeta.timestamp`). A run's random seed is not
something this plugin can observe directly-flepimop2 core does not thread it
through to the backend-so if you want it recorded, encode it into the run
name by convention: `RunMeta(name=f"{run_name}_seed{seed}")`. The backend
parses a trailing `_seed<digits>` suffix out of `run_meta.name` and records it
as the `seed` attribute when present, and simply omits it otherwise.

## Chain / draw indexing

Multi-chain or multi-draw sweeps (e.g. prior-/posterior-predictive loops) set
the `FLEPIMOP2_CHAIN` and `FLEPIMOP2_DRAW` environment variables before
invoking `flepimop2 simulate`. Neither is required: `FLEPIMOP2_CHAIN`
defaults to `0`, and `FLEPIMOP2_DRAW` defaults to counting the shards already
present for that chain (correct for a sequential loop in one process; set it
explicitly for anything running draws concurrently, e.g. a Slurm job array).

## Installation

Install the provider directly:

```bash
pip install "flepimop2-arviz@git+https://github.com/ACCIDDA/flepimop2-extras.git@main#subdirectory=packages/flepimop2-arviz"
```
