#!/usr/bin/env python3
"""
Visual demo for the SubmissionReview TUI — no ALF installation required.

Run from the repo root:

    python demos/demo_submission_tui.py

What this exercises
-------------------
Sim table
  space   toggle sim on / off
  e       edit parameters and sim_dir in the edit modal
  a       clone a sim and add a new entry (tests _SimProxy submission)

Settings panel
  Executor buttons  slurm / local / debug
  Memory, submit dir, job name, WCKey inputs
  Mail-type toggle  END / FAIL / ALL (mutually exclusive with ALL)
  stderr → stdout toggle
  On-fail stop toggle

Architecture panel
  Partition colormap (green → red by wall-time)
  Wall-time editor: change CPU_MAX live and watch partition update
  "--- (too long)" guard when CPU_MAX exceeds all partitions

Submission flow
  s  submit → confirmation dialog → progress view
       writes real jobid.txt to disk so _run_submission tracking works
       emits a session manifest to submit_dir
  Post-submission: UI locks; footer shows "m open monitor  q exit"
  m  hand off to SimulationMonitor with mocked SLURM statuses
  q  exit (session_path printed to stdout)

Batch selection
  0  short    · serial  · 4 OMP   (CPU_MAX = 1 h  → short partition)
  1  medium   · MPI×2   · 4 OMP   (CPU_MAX = 24 h → medium partition)
  2  long     · MPI×4   · 4 OMP   (CPU_MAX = 72 h → long partition)
  3  extra_long · MPI×8 · 8 OMP   (CPU_MAX = 400 h → extra_long)
  4  unfit partition               (CPU_MAX = 800 h → submit blocked)
  5  local executor                (no cluster, no partition display)
"""

from __future__ import annotations

import atexit
import random
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from py_alf.cluster_submission import ClusterSubmitter
from py_alf.monitor import SimulationMonitor
from py_alf.simulation import Simulation
from py_alf.submission_tui import SubmissionReview

# ---------------------------------------------------------------------------
# Temporary workspace — wiped on exit
# ---------------------------------------------------------------------------

_TMPDIR = Path(tempfile.mkdtemp(prefix="alf_demo_submit_"))
atexit.register(shutil.rmtree, _TMPDIR, ignore_errors=True)

# ---------------------------------------------------------------------------
# Cluster partition table
#
# Partition    Wall-time limit   Notes
# ─────────────────────────────────────────────────────────────────────────
# debug        10 min            cluster-access test; not auto-selected
# short         2 h              most nodes
# medium        2 days
# long         14 days
# extra_long   28 days
# ---------------------------------------------------------------------------

CPU_PARTITIONS: dict[str, float] = {
    "short": 2,
    "medium": 48,
    "long": 336,
    "extra_long": 672,
}


# ---------------------------------------------------------------------------
# Mock simulation factory
# ---------------------------------------------------------------------------


def _sd(name: str) -> str:
    """Return a sim_dir path inside the temp workspace."""
    return str(_TMPDIR / "ALF_data" / name)


def make_mock_sim(
    ham_name: str,
    sim_dir: str,
    sim_dict: dict,
    *,
    n_omp: int = 4,
    n_mpi: int = 1,
    mpi: bool = False,
    machine: str = "GNU",
) -> MagicMock:
    """Return a Simulation-shaped mock; no ALF source or filesystem needed."""
    sim = MagicMock()
    sim.__class__ = Simulation
    sim.ham_name = ham_name
    sim.sim_dir = sim_dir
    sim.sim_dict = sim_dict
    sim.n_omp = n_omp
    sim.n_mpi = n_mpi
    sim.mpi = mpi
    sim.config = f"{machine} HDF5 {'MPI' if mpi else 'NOMPI'}"
    return sim


# ---------------------------------------------------------------------------
# Demo batches
# ---------------------------------------------------------------------------


def _cs(mem: str) -> ClusterSubmitter:
    return ClusterSubmitter(
        "slurm",
        submit_dir=str(_TMPDIR / "submitit"),
        slurm_mem=mem,
        partition_rules=CPU_PARTITIONS,
    )


BATCHES: dict[str, dict] = {
    # ── 0: short partition — serial × 4 OMP ──────────────────────────────
    "short  · serial  · 4 OMP": dict(
        sims=[
            make_mock_sim(
                "Hubbard_Plain_Vanilla",
                _sd(f"Hubbard_PV_Beta={b}_L=4"),
                {"Beta": b, "L1": 4, "L2": 4, "CPU_MAX": 1, "NBin": 40},
                n_omp=4,
            )
            for b in [1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
        ],
        cs=_cs("4G"),
    ),
    # ── 1: medium partition — MPI×2 × 4 OMP ─────────────────────────────
    "medium · MPI×2   · 4 OMP": dict(
        sims=[
            make_mock_sim(
                "Hubbard_Plain_Vanilla",
                _sd(f"Hubbard_PV_MPI2_Beta={b}_L=6"),
                {"Beta": b, "L1": 6, "L2": 6, "CPU_MAX": 24, "NBin": 20},
                n_omp=4,
                n_mpi=2,
                mpi=True,
            )
            for b in [0.5, 1.0, 2.0, 4.0, 8.0]
        ],
        cs=_cs("8G"),
    ),
    # ── 2: long partition — MPI×4 × 4 OMP ───────────────────────────────
    "long   · MPI×4   · 4 OMP": dict(
        sims=[
            make_mock_sim(
                "Hubbard_Plain_Vanilla",
                _sd(f"Hubbard_PV_MPI4_Beta={b}_L=8"),
                {"Beta": b, "L1": 8, "L2": 8, "CPU_MAX": 72, "NBin": 10},
                n_omp=4,
                n_mpi=4,
                mpi=True,
            )
            for b in [0.5, 1.0, 2.0, 4.0]
        ],
        cs=_cs("16G"),
    ),
    # ── 3: extra_long — MPI×8 × 8 OMP, 2-row rank grid ──────────────────
    "extra_long · MPI×8 · 8 OMP": dict(
        sims=[
            make_mock_sim(
                "tV_Model",
                _sd(f"tV_V={v}_L=16"),
                {"V": v, "L1": 16, "L2": 1, "CPU_MAX": 400, "NBin": 5},
                n_omp=8,
                n_mpi=8,
                mpi=True,
                machine="INTEL",
            )
            for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        ],
        cs=_cs("32G"),
    ),
    # ── 4: unfit partition — CPU_MAX exceeds every limit ─────────────────
    #    Arch panel shows "--- (too long)"; submit button stays disabled.
    #    Try raising wall-time via the editor to make it fit.
    "unfit partition  (CPU_MAX > extra_long)": dict(
        sims=[
            make_mock_sim(
                "Hubbard_Plain_Vanilla",
                _sd(f"Hubbard_Unfit_Beta={b}"),
                {"Beta": b, "L1": 4, "L2": 4, "CPU_MAX": 800, "NBin": 5},
                n_omp=4,
            )
            for b in [1.0, 2.0, 4.0]
        ],
        cs=_cs("4G"),
    ),
    # ── 5: local executor — no cluster, no partition display ─────────────
    "local executor (no cluster)": dict(
        sims=[
            make_mock_sim(
                "Kondo",
                _sd(f"Kondo_JK={jk}_L=4"),
                {"JK": jk, "L1": 4, "NBin": 5},
                n_omp=2,
            )
            for jk in [0.5, 1.0, 1.5, 2.0]
        ],
        cs=ClusterSubmitter("local"),
    ),
}


# ---------------------------------------------------------------------------
# Mock submit
# ---------------------------------------------------------------------------


def mock_submit(self: ClusterSubmitter, sims, **kwargs) -> list:
    """
    Stand-in for ClusterSubmitter.submit.

    Writes a real jobid.txt to each sim_dir so that _run_submission's
    before/after snapshot logic correctly marks each sim as submitted,
    and the session manifest contains valid job IDs.
    """
    sims = list(sims)
    master_id = random.randint(10000, 99999)
    jobs = []
    for i, sim in enumerate(sims):
        jid = f"{master_id}_{i}"
        Path(sim.sim_dir).mkdir(parents=True, exist_ok=True)
        (Path(sim.sim_dir) / "jobid.txt").write_text(jid)
        job = MagicMock()
        job.job_id = jid
        jobs.append(job)
    print(
        f"\n[MOCK] executor={self.executor!r}  "
        f"array {master_id}  ({len(sims)} task(s))\n"
        + "\n".join(f"  {master_id}_{i}  {s.sim_dir}" for i, s in enumerate(sims))
    )
    return jobs


# ---------------------------------------------------------------------------
# Monitor handover helper
# ---------------------------------------------------------------------------

_STATUS_CYCLE = ["RUNNING", "RUNNING", "PENDING", "COMPLETED", "FAILED", "CANCELLED"]


def _launch_monitor(sims: list, app: SubmissionReview) -> None:
    """
    Launch SimulationMonitor for the just-submitted sims.

    get_job_id reads from jobid.txt (written by mock_submit, no patch needed).
    SLURM status and bin counts are mocked so no cluster is required.
    """
    # Build a status map keyed by job ID.
    statuses: dict[str, dict] = {}
    bin_map: dict[str, int] = {}
    for i, sim in enumerate(sims):
        jid_file = Path(str(sim.sim_dir)) / "jobid.txt"
        if not jid_file.exists():
            continue
        jid = jid_file.read_text().strip()
        status = _STATUS_CYCLE[i % len(_STATUS_CYCLE)]
        nodelist = f"compute{(i % 4) + 1:02d}" if status == "RUNNING" else None
        statuses[jid] = {
            "status": status,
            "runtime": f"{i}:0{i % 60}:00",
            "nodelist": nodelist,
        }
        bin_map[jid] = {"COMPLETED": 40, "RUNNING": 10 + i * 3}.get(status, 0)

    def mock_bin_count(sim, **_kw):
        jid_file = Path(str(sim.sim_dir)) / "jobid.txt"
        jid = jid_file.read_text().strip() if jid_file.exists() else ""
        return bin_map.get(jid, 0)

    # Create a fake log file per job ID so the log viewer works.
    submit_dir = (
        app.submitted_cs.submit_dir if app.submitted_cs else _TMPDIR / "submitit"
    )
    submit_dir.mkdir(parents=True, exist_ok=True)
    for jid in statuses:
        log = submit_dir / f"{jid}_0_log.out"
        log.write_text(
            f"[ALF] job {jid} started\n"
            f"[ALF] Hamiltonian initialised\n"
            f"[ALF] Warmup  — 1000 sweeps\n"
            f"[ALF] Production — 5000 sweeps\n"
            f"[ALF] Writing observables to data.h5\n"
            f"[ALF] Done.\n"
        )

    with (
        patch("py_alf.monitor._get_slurm_status_bulk", return_value=statuses),
        patch("py_alf.monitor._bin_count", side_effect=mock_bin_count),
        patch("py_alf.monitor.cancel_cluster_job", return_value=True),
        patch.object(ClusterSubmitter, "submit", mock_submit),
    ):
        if app.session_path:
            monitor = SimulationMonitor.from_session(
                app.session_path,
                cluster_submitter=app.submitted_cs,
            )
        else:
            monitor = SimulationMonitor(sims, cluster_submitter=app.submitted_cs)
        monitor.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    names = list(BATCHES)
    print("Available demo batches:\n")
    for i, name in enumerate(names):
        print(f"  {i}  {name}")
    raw = input(f"\nPick batch [0–{len(names) - 1}, default 0]: ").strip()
    idx = int(raw) if raw.isdigit() and int(raw) < len(names) else 0
    chosen = names[idx]
    print(f"\nLaunching: {chosen}")
    print(f"Temp workspace: {_TMPDIR}\n")

    batch = BATCHES[chosen]

    with patch.object(ClusterSubmitter, "submit", mock_submit):
        app = SubmissionReview(batch["cs"], batch["sims"])
        result = app.run()

    print()
    if result is None:
        print("Cancelled — no jobs submitted.")
        return

    print(f"Submitted {len(result)} simulation(s):")
    for s in result:
        print(f"  {s.sim_dir}")

    if app.session_path:
        print(f"\nSession manifest: {app.session_path}")
        print("  (load later with: SimulationMonitor.from_session(path))")
        print(
            "  (or from CLI:     alf_monitor --dir", Path(app.session_path).parent, ")"
        )

    if app.open_monitor:
        print("\nOpening monitor…\n")
        _launch_monitor(result, app)


if __name__ == "__main__":
    main()
