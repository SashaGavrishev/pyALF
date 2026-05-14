# pyALF (fork)

This is a personal fork of the [pyALF](https://github.com/ALF-QMC/pyALF) package.

For documentation, installation instructions, and the full project description, please refer to the upstream repository at **https://github.com/ALF-QMC/pyALF**.

## Key fork branches

| Branch | Description |
|--------|-------------|
| `master` | Kept in sync with pyALF `master` |
| `development` | Personal development |

## Fork additions

---

### Submission Review (`SubmissionReview`)

An interactive pre-submission checklist that lets you review, edit, and selectively include simulations before they are sent to the cluster.

**What it shows:**

| Panel | Contents |
|-------|----------|
| Sim table (top) | One row per simulation; toggle inclusion with `space`, edit parameters with `e`, add a clone with `a` |
| Settings (bottom-left) | Executor (slurm / local / debug), memory, submit directory, job name, mail-type, WCKey, stderr→stdout merge |
| Architecture (bottom-right) | Visual rank-grid for the focused simulation; colour-coded partition (green = shortest, red = longest); wall-time editor; estimated completion timestamp |

**Usage:**

```python
from py_alf import ALF_source, Simulation, ClusterSubmitter
from py_alf.submission_tui import SubmissionReview

alf_src = ALF_source(...)
sims = [Simulation(alf_src, "Hubbard", {"U": u, "Beta": 10, "CPU_MAX": 24}) for u in [4, 6, 8]]

cs = ClusterSubmitter(
    "slurm",
    slurm_mem="8G",
    partition_rules={"short": 2, "medium": 48, "long": 336},
)

app = SubmissionReview(cs, sims)
submitted = app.run()   # blocks; returns submitted sim list, or None if cancelled
```

All selected simulations are submitted as a single SLURM array job.

**Post-submission attributes** (read after `app.run()` returns):

| Attribute | Type | Description |
|-----------|------|-------------|
| `app.submitted_cs` | `ClusterSubmitter \| None` | The effective submitter used (may differ from the input `cs` if settings were changed in the TUI) |
| `app.session_path` | `Path \| None` | Path to the session manifest written after a successful SLURM submission; `None` for local/debug runs or if writing failed |
| `app.open_monitor` | `bool` | `True` when the user pressed `m` to hand off to the monitor before quitting |

**`run_with_monitor()`** — recommended entry point when running from a script or an SSH terminal. It calls `run()` and, if the user pressed `m`, automatically launches the monitor in a fresh subprocess (which resets the terminal cleanly between the two Textual apps):

```python
SubmissionReview(cs, sims).run_with_monitor()
```

**`save_for_ssh()`** — for use inside a Jupyter notebook over SSH, where the TUI cannot run in the kernel's output cell. Call this after building `cs` and `sims`; it pickles the state and prints the one-liner to paste into VS Code's integrated terminal (or any SSH shell):

```python
from py_alf.submission_tui import save_for_ssh

save_for_ssh(cs, sims)
# Saved. In the terminal (VS Code Ctrl+`) run:
# python -c "import pickle; from py_alf.submission_tui import SubmissionReview; ..."
```

An optional `path` argument overrides the default `/tmp/alf_tui_state.pkl`.

**Keybindings:**

| Key | Action |
|-----|--------|
| `space` | Toggle the selected simulation on / off |
| `e` | Edit parameters and `sim_dir` for the selected simulation |
| `a` | Clone the selected simulation and add it to the list |
| `s` | Submit selected simulations (confirmation required) |
| `m` | Open the Simulation Monitor (only available after a successful SLURM submission; sets `app.open_monitor = True` and exits) |
| `q` | Quit / cancel |

---

### Simulation Monitor (`SimulationMonitor`)

An interactive terminal UI for monitoring ALF simulations running on a SLURM cluster.

**Usage:**

```python
from py_alf import ALF_source, Simulation, ClusterSubmitter
from py_alf.monitor import SimulationMonitor

alf_src = ALF_source(...)
sims = [Simulation(alf_src, "Hubbard", {"U": u, "Beta": 10}) for u in [4, 6, 8]]

cs = ClusterSubmitter("slurm", slurm_mem="8G", partition_rules={"short": 2, "medium": 48})

SimulationMonitor(
    sims,
    cluster_submitter=cs,       # required for resubmit action; optional otherwise
    param_keys=["U", "Beta"],   # sim_dict keys shown as extra columns
    param_headers=["U", "β"],   # display labels for those columns
    refresh_interval=30.0,      # seconds between automatic SLURM polls
).run()
```

**Reconstruct from a session manifest** (written automatically by `SubmissionReview`):

```python
SimulationMonitor.from_session(
    ".alfmonitor/session_20260513_102314.json",
    refresh_interval=30.0,
).run()
```

`from_session` reads the job IDs, Hamiltonian metadata, and `ClusterSubmitter` configuration from the manifest — no re-import of the original `Simulation` objects is needed.

**Load simulation objects for post-run analysis** — useful in a fresh Jupyter notebook when you want to use the session data with `py_alf.analysis`:

```python
from py_alf.monitor import list_sessions, load_session_sims

# List all sessions in the submit directory, newest first
sessions = list_sessions(".alfmonitor")
print(sessions)

# Reconstruct simulation-like objects from any session
sims = load_session_sims(sessions[0])
for sim in sims:
    print(sim.sim_dir, sim.ham_name, sim.sim_dict)
    # pass sim.sim_dir to py_alf.analysis() etc.
```

Both helpers are also importable from the top-level package: `py_alf.list_sessions`, `py_alf.load_session_sims`.

The monitor displays a table with one row per simulation. Columns include the Hamiltonian name, any `param_keys` you specify, `n_omp`/`n_mpi`, SLURM partition and memory (when a `ClusterSubmitter` is provided), bin count, master array ID, individual job ID, colour-coded status, elapsed runtime, estimated time remaining (ETA), peak memory usage, and CPU efficiency. Peak memory and CPU efficiency are fetched from `sacct` once a job reaches a terminal state and persisted to `peak_resources.json` in the simulation directory, so they remain visible even after the job ages out of the SLURM accounting database.

The title bar updates with the SLURM array ID(s) once the first status poll completes.

**Keybindings:**

| Key | Action |
|-----|--------|
| `l` | View log for the selected job |
| `c` | Cancel the selected job (confirmation required) |
| `a` | Cancel the entire SLURM array the selected job belongs to (confirmation required) |
| `r` | Force resubmit the selected simulation (confirmation required; requires `cluster_submitter`) |
| `f5` | Refresh status immediately |
| `q` | Quit |

---

### Session manifests and `alf_monitor` CLI

After a successful SLURM submission `SubmissionReview` writes a JSON manifest to `.alfmonitor/` at the project root:

```
.alfmonitor/session_20260513_102314.json
```

The manifest records the submission timestamp, all `ClusterSubmitter` settings, and per-simulation metadata (sim directory, job ID, Hamiltonian name, parallelism, MPI launcher, `sim_dict`). It is the link between a submission and a later monitoring session.

`.alfmonitor` is always created at the root of the project (the nearest ancestor directory containing `.git`, `pyproject.toml`, etc.), mirroring the convention used by `.git`. This means `alf_monitor` and `SimulationMonitor.from_session` find it consistently regardless of which subdirectory you run from.

**`alf_monitor` CLI:**

```bash
# Interactively pick from all session files — searches upward for .alfmonitor
alf_monitor

# Search a specific directory instead
alf_monitor --dir /scratch/user/my_project/.alfmonitor

# Skip selection and open the most recent session
alf_monitor --latest

# Open a specific session file directly
alf_monitor .alfmonitor/session_20260513_102314.json

# Change the auto-refresh interval (default: 30 s)
alf_monitor --refresh 60
```

**Full scripted workflow:**

```python
from py_alf import ALF_source, Simulation, ClusterSubmitter
from py_alf.submission_tui import SubmissionReview

alf_src = ALF_source(...)
sims    = [...]
cs      = ClusterSubmitter("slurm", slurm_mem="8G",
                           partition_rules={"short": 2, "medium": 48, "long": 336})

# Review, submit, and seamlessly hand off to the monitor if the user presses 'm'.
# The monitor launches in a fresh subprocess, which resets the terminal cleanly
# in SSH environments.
SubmissionReview(cs, sims).run_with_monitor()
```

**Jupyter / SSH workflow** — when the TUI must run in an SSH terminal rather than the notebook kernel:

```python
# In the Jupyter notebook cell:
from py_alf.submission_tui import save_for_ssh

save_for_ssh(cs, sims)
# Output:
#   Saved. In the terminal (VS Code Ctrl+`) run:
#   python -c "import pickle; from py_alf.submission_tui import SubmissionReview; ..."

# Then paste the printed command into VS Code's integrated terminal (Ctrl+`).
# The command restores cs and sims from the pickle and calls run_with_monitor().
```

**Later — reload from the session manifest without re-running the script:**

```python
from py_alf.monitor import SimulationMonitor
SimulationMonitor.from_session(".alfmonitor/session_20260513_102314.json").run()
```

**Demos** — runnable examples that mock all SLURM calls (no cluster required):

```bash
python demos/demo_submission_tui.py   # SubmissionReview + monitor handover
python demos/demo_monitor_tui.py      # SimulationMonitor standalone
```

### Detect partition rules

Functionality exists to be able automatically detect the required partition rules on a SLURM cluster, using `detect_partition_rules`.

```python
from py_alf import detect_partition_rules, ClusterSubmitter

rules = detect_partition_rules(exclude=["gpu", "debug"])
cs = ClusterSubmitter("slurm", slurm_mem="8G", partition_rules=rules)
```
