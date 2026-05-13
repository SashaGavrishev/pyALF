# pyALF (fork)

This is a personal fork of the [pyALF](https://github.com/ALF-QMC/pyALF) package.

For documentation, installation instructions, and the full project description, please refer to the upstream repository at **https://github.com/ALF-QMC/pyALF**.

## Key fork branches

| Branch | Description |
|--------|-------------|
| `master` | Kept in sync with pyALF `master` |
| `development` | Personal development |

## Fork additions

### TUI extras — installation

Both TUI components require the `textual` package:

```bash
pip install 'pyALF[tui]'
# or with uv
uv sync --extra tui
```

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
    submit_dir="submitit",
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
    "submitit/session_20260513_102314.json",
    refresh_interval=30.0,
).run()
```

`from_session` reads the job IDs, Hamiltonian metadata, and `ClusterSubmitter` configuration from the manifest — no re-import of the original `Simulation` objects is needed.

The monitor displays a table with one row per simulation. Columns include the Hamiltonian name, any `param_keys` you specify, `n_omp`/`n_mpi`, SLURM partition and memory (when a `ClusterSubmitter` is provided), bin count, master array ID, individual job ID, colour-coded status, elapsed runtime, and estimated time remaining (ETA).

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

After a successful SLURM submission `SubmissionReview` writes a JSON manifest to `submit_dir`:

```
submitit/session_20260513_102314.json
```

The manifest records the submission timestamp, all `ClusterSubmitter` settings, and per-simulation metadata (sim directory, job ID, Hamiltonian name, parallelism, `sim_dict`). It is the link between a submission and a later monitoring session.

**`alf_monitor` CLI:**

```bash
# Interactively pick from all session files in ./submitit (default)
alf_monitor

# Search a different directory
alf_monitor --dir /scratch/user/my_project/submitit

# Skip selection and open the most recent session
alf_monitor --latest

# Open a specific session file directly
alf_monitor submitit/session_20260513_102314.json

# Change the auto-refresh interval (default: 30 s)
alf_monitor --refresh 60
```

**Full scripted workflow:**

```python
from py_alf import ALF_source, Simulation, ClusterSubmitter
from py_alf.submission_tui import SubmissionReview
from py_alf.monitor import SimulationMonitor

alf_src = ALF_source(...)
sims    = [...]
cs      = ClusterSubmitter("slurm", submit_dir="submitit", slurm_mem="8G",
                           partition_rules={"short": 2, "medium": 48, "long": 336})

# 1. Review and submit
app       = SubmissionReview(cs, sims)
submitted = app.run()

if not submitted:
    print("Cancelled.")
else:
    print(f"Submitted {len(submitted)} sim(s). Session: {app.session_path}")

    # 2a. Seamless handover — user pressed 'm' inside the TUI
    if app.open_monitor:
        SimulationMonitor.from_session(
            app.session_path,
            cluster_submitter=app.submitted_cs,
        ).run()

# 2b. Later — reload from the session manifest without re-running the script
# SimulationMonitor.from_session("submitit/session_20260513_102314.json").run()
```

**Demos** — runnable examples that mock all SLURM calls (no cluster required):

```bash
python demos/demo_submission_tui.py   # SubmissionReview + monitor handover
python demos/demo_monitor_tui.py      # SimulationMonitor standalone
```

### Detect partition rules

Functionality exists to be able automatically detect the required partition rules on a SLURM cluster, using `detect_partition_rules`.

```python
/dev/null/run_project.py#L1-8
from py_alf import detect_partition_rules, ClusterSubmitter

rules = detect_partition_rules(exclude=["gpu", "debug"])
cs = ClusterSubmitter("slurm", slurm_mem="8G", partition_rules=rules)
```
