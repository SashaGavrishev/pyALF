#!/usr/bin/env python3
"""
Demo for checkpoint-restart detection in SubmissionReview and SimulationMonitor.

Run from the repo root:

    python demos/demo_checkpoint_restart.py

What this demonstrates
----------------------
Submission TUI
  Three of the six sims already have confin_* files on disk, meaning ALF
  would continue from a checkpoint rather than start fresh.

  When you press  s  to submit, the confirmation dialog shows:

      ⚠  3 checkpoint restart(s) detected.
      ALF will append to existing data.h5.

  Selecting the jobs individually (space) lets you submit only fresh or
  only checkpoint-restart sims and watch the warning count update.

Monitor TUI
  After submission the monitor opens with a pre-built mix of statuses:

    RUNNING  ↺  (green)   — checkpoint-restart job is actively running
    PENDING  ↺  (green)   — checkpoint-restart job is queued
    COMPLETED ↺ (yellow)  — completed restart; confin_* still present,
                             ready for another continuation
    RUNNING               — fresh job (no checkpoint)
    COMPLETED             — fresh job, done, no confin_* files

  Press  r  on any row to see the resubmit confirm dialog — checkpoint
  sims show the ⚠ warning; fresh sims do not.

Controls
--------
  (submission TUI)
  space   toggle sim selected / deselected
  s       submit → confirmation dialog
  q       quit

  (monitor TUI)
  r       resubmit selected sim
  l       view log
  q       quit
"""

from __future__ import annotations

import atexit
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from py_alf.cluster_submission import ClusterSubmitter
from py_alf.monitor import SimulationMonitor
from py_alf.submission_tui import SubmissionReview

# ---------------------------------------------------------------------------
# Temporary workspace — wiped on exit
# ---------------------------------------------------------------------------

_TMPDIR = Path(tempfile.mkdtemp(prefix="alf_demo_ckpt_"))
atexit.register(shutil.rmtree, _TMPDIR, ignore_errors=True)

_SUBMIT_DIR = _TMPDIR / ".alfmonitor"
_SUBMIT_DIR.mkdir()

CPU_PARTITIONS: dict[str, float] = {
    "short": 2,
    "medium": 48,
    "long": 336,
}

# ---------------------------------------------------------------------------
# Sim setup
# ---------------------------------------------------------------------------

# (ham_name, Beta, has_checkpoint)
_SIM_SPECS = [
    ("Hubbard_Plain_Vanilla", 1.0, True),   # checkpoint restart
    ("Hubbard_Plain_Vanilla", 2.0, True),   # checkpoint restart
    ("Hubbard_Plain_Vanilla", 4.0, True),   # checkpoint restart
    ("Hubbard_Plain_Vanilla", 6.0, False),  # fresh run
    ("Hubbard_Plain_Vanilla", 8.0, False),  # fresh run
    ("tV_Model",              1.0, False),  # fresh run
]


def _setup_sims() -> list:
    sims = []
    for ham, beta, has_ckpt in _SIM_SPECS:
        name = f"{ham}_Beta={beta}"
        sim_dir = _TMPDIR / "ALF_data" / name
        sim_dir.mkdir(parents=True, exist_ok=True)

        if has_ckpt:
            (sim_dir / "confin_0").touch()
            (sim_dir / "confin_1").touch()

        sim = SimpleNamespace(
            ham_name=ham,
            sim_dir=str(sim_dir),
            sim_dict={"Beta": beta, "L1": 4, "L2": 4, "CPU_MAX": 24, "NBin": 40},
            n_omp=4,
            n_mpi=1,
            mpi=False,
            mpiexec="mpiexec",
            mpiexec_args=[],
            config="GNU HDF5 NOMPI",
        )
        sims.append(sim)
    return sims


# ---------------------------------------------------------------------------
# Mock submit
# ---------------------------------------------------------------------------

_ARRAY_ID = "55123"
_next_task = [0]


def _mock_submit(self: ClusterSubmitter, sims, **kwargs) -> list:
    sims = list(sims) if not hasattr(sims, "sim_dir") else [sims]
    jobs = []
    for sim in sims:
        i = _next_task[0]
        _next_task[0] += 1
        jid = f"{_ARRAY_ID}_{i}"
        Path(sim.sim_dir).mkdir(parents=True, exist_ok=True)
        (Path(sim.sim_dir) / "jobid.txt").write_text(jid)
        job = MagicMock()
        job.job_id = jid
        jobs.append(job)
        print(f"[MOCK] submitted {jid}  →  {sim.sim_dir}")
    return jobs


# ---------------------------------------------------------------------------
# Monitor handover
# ---------------------------------------------------------------------------

# Assign statuses that showcase all ↺ variants:
#   checkpoint sims (indices 0–2): RUNNING ↺, PENDING ↺, COMPLETED ↺
#   fresh sims      (indices 3–5): RUNNING,   COMPLETED, COMPLETED
_STATUS_CYCLE = ["RUNNING", "PENDING", "COMPLETED", "RUNNING", "COMPLETED", "COMPLETED"]


def _launch_monitor(sims: list, app: SubmissionReview) -> None:
    statuses: dict[str, dict] = {}
    bin_map: dict[str, int] = {}

    for i, sim in enumerate(sims):
        jid_file = Path(str(sim.sim_dir)) / "jobid.txt"
        if not jid_file.exists():
            continue
        jid = jid_file.read_text().strip()
        status = _STATUS_CYCLE[i % len(_STATUS_CYCLE)]
        runtime = f"1:{(i * 7) % 60:02d}:00" if status != "PENDING" else None
        nodelist = f"node{i + 1:02d}" if status == "RUNNING" else None
        statuses[jid] = {"status": status, "runtime": runtime, "nodelist": nodelist}
        bin_map[str(sim.sim_dir)] = {"COMPLETED": 40, "RUNNING": 12 + i * 4}.get(
            status, 0
        )

        # Write a fake log for running / completed jobs.
        log = _SUBMIT_DIR / f"{jid}_0_log.out"
        ckpt = any(Path(sim.sim_dir).glob("confin_*"))
        mode = "checkpoint restart" if ckpt else "fresh start"
        lines = [
            f"[ALF] job {jid}  ham={sim.ham_name}  Beta={sim.sim_dict['Beta']}",
            f"[ALF] Mode: {mode}",
            "[ALF] Warmup  — 1000 sweeps",
            "[ALF] Production — 5000 sweeps",
        ]
        if status == "COMPLETED":
            lines += ["[ALF] Writing observables to data.h5", "[ALF] Done."]
        else:
            lines += ["[ALF] Still running…"]
        log.write_text("\n".join(lines) + "\n")

    def _mock_bin_count(sim, **_kw):
        return bin_map.get(str(sim.sim_dir), 0)

    cs = app.submitted_cs or ClusterSubmitter(
        "slurm",
        submit_dir=str(_SUBMIT_DIR),
        partition_rules=CPU_PARTITIONS,
    )

    with (
        patch("py_alf.monitor._get_slurm_status_bulk", return_value=statuses),
        patch("py_alf.monitor._bin_count", side_effect=_mock_bin_count),
        patch("py_alf.monitor.cancel_cluster_job", return_value=True),
        patch.object(ClusterSubmitter, "submit", _mock_submit),
    ):
        if app.session_path:
            monitor = SimulationMonitor.from_session(
                app.session_path, cluster_submitter=cs, param_keys=["Beta"]
            )
        else:
            monitor = SimulationMonitor(sims, cluster_submitter=cs, param_keys=["Beta"])
        monitor.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print(__doc__)
    print(f"Temp workspace: {_TMPDIR}\n")

    sims = _setup_sims()
    ckpt_count = sum(any(Path(s.sim_dir).glob("confin_*")) for s in sims)
    fresh_count = len(sims) - ckpt_count
    print(f"Prepared {len(sims)} simulations: {ckpt_count} checkpoint, {fresh_count} fresh\n")

    cs = ClusterSubmitter(
        "slurm",
        submit_dir=str(_SUBMIT_DIR),
        slurm_mem="8G",
        partition_rules=CPU_PARTITIONS,
    )

    with patch.object(ClusterSubmitter, "submit", _mock_submit):
        app = SubmissionReview(cs, sims)
        result = app.run()

    print()
    if result is None:
        print("Cancelled — no jobs submitted.")
        return

    print(f"Submitted {len(result)} simulation(s).")

    if app.open_monitor:
        print("Opening monitor…\n")
        _launch_monitor(result, app)


if __name__ == "__main__":
    main()
