# pyALF (fork)

This is a personal fork of the [pyALF](https://github.com/ALF-QMC/pyALF) package.

For documentation, installation instructions, and the full project description, please refer to the upstream repository at **https://github.com/ALF-QMC/pyALF**.

## Key fork branches

| Branch | Description |
|--------|-------------|
| `master` | Kept in sync with pyALF `master` |
| `development` | Personal development |

## Fork additions

### Simulation Monitor (TUI)

An interactive terminal UI for monitoring ALF simulations running on a SLURM cluster.

**Install:**

```bash
pip install 'pyALF[tui]'
# or with uv
uv sync --extra tui
```

**Usage:**

```python
from py_alf import ALF_source, Simulation, ClusterSubmitter
from py_alf.monitor import SimulationMonitor

alf_src = ALF_source(...)
sims = [Simulation(alf_src, "Hubbard", {"U": u, "beta": 10}) for u in [4, 6, 8]]

cs = ClusterSubmitter(slurm_mem="8G", slurm_partition="short")

SimulationMonitor(
    sims,
    cluster_submitter=cs,       # required for resubmit action; optional otherwise
    param_keys=["U", "beta"],   # sim_dict keys shown as extra columns
    param_headers=["U", "β"],   # display labels for those columns
    refresh_interval=30.0,      # seconds between automatic SLURM polls
).run()
```

The monitor displays a table with one row per simulation. Columns include the Hamiltonian name, any `param_keys` you specify, `n_omp`/`n_mpi`, SLURM partition and memory (when a `ClusterSubmitter` is provided), bin count, job ID, colour-coded status, and elapsed time.

**Keybindings:**

| Key | Action |
|-----|--------|
| `l` | View log for the selected job |
| `c` | Cancel the selected job (confirmation required) |
| `a` | Cancel the entire SLURM array the selected job belongs to (confirmation required) |
| `r` | Force resubmit the selected simulation (confirmation required) |
| `f5` | Refresh status immediately |
| `q` | Quit |
