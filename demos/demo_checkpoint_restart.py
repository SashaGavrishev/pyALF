#!/usr/bin/env python3
"""
Demo for checkpoint-restart detection in SubmissionReview and SimulationMonitor.

Run from the repo root:

    python demos/demo_checkpoint_restart.py

What this demonstrates
----------------------
Submission TUI — four sim states are pre-built on disk:

  confin_*  (2 sims)   Already have confin_*.h5 files: checkpoint restarts.
                        The confirmation dialog shows a tree of confin files.

  confout_* (2 sims)   Have confout_*.h5 files written by a completed ALF run.
                        _prep_sim_dir auto-renames them to confin_*.h5 on submit.
                        The dialog shows "confout_0.h5  →  confin_0.h5 (auto)".

  conflict  (1 sim)    Has data.h5 but no conf files: interrupted run.
                        Submission is blocked (⚠ mark); the submit button reads
                        "Cannot submit — 1 sim(s) have existing data".
                        Deselect it with  space  to re-enable the submit button.

  fresh     (2 sims)   Empty directories: no special handling.

When you press  s  to submit (after deselecting the blocked sim), the
confirmation dialog shows:

    ⚠  4 checkpoint restart(s) detected.
    ALF will append to existing data.h5.
       confout_* → confin_* will be auto-renamed on submit.

      ↳ Hubbard_Plain_Vanilla_Beta=1.0/
          ├── confin_0.h5
          └── confin_1.h5

      ↳ Hubbard_Plain_Vanilla_Beta=4.0/
          └── confout_0.h5  →  confin_0.h5

  … (etc.)

Monitor TUI
  After submission the monitor shows a mix of statuses.
  Press  i  on a COMPLETED row to view the ALF info file.
  Press  l  on any row to view the log.
  Cancel Job / Cancel Array are disabled for terminal states (COMPLETED etc.).

Controls
--------
  (submission TUI)
  space   toggle sim selected / deselected
  s       submit → confirmation dialog
  m       open monitor (after submission)
  q       quit

  (monitor TUI)
  i       view ALF info file (only active when COMPLETED)
  l       view log
  c       cancel individual job  (mocked)
  a       cancel SLURM array    (mocked)
  r       resubmit selected sim (mocked)
  f       manual refresh
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

# (ham_name, Beta, state)
#   "confin"   — has confin_*.h5 files (prior checkpoint)
#   "confout"  — has confout_*.h5 files (completed run, auto-renamed on submit)
#   "conflict" — has data.h5 only, no conf files (blocked ⚠)
#   None       — fresh directory
_SIM_SPECS = [
    ("Hubbard_Plain_Vanilla", 1.0, "confin"),    # checkpoint restart
    ("Hubbard_Plain_Vanilla", 2.0, "confin"),    # checkpoint restart
    ("Hubbard_Plain_Vanilla", 4.0, "confout"),   # completed → auto-rename on submit
    ("Hubbard_Plain_Vanilla", 6.0, "confout"),   # completed → auto-rename on submit
    ("Hubbard_Plain_Vanilla", 8.0, "conflict"),  # blocked ⚠ — deselect before submit
    ("Hubbard_Plain_Vanilla", 10.0, None),       # fresh
    ("tV_Model", 1.0, None),                     # fresh
]


def _setup_sims() -> list:
    sims = []
    for ham, beta, state in _SIM_SPECS:
        name = f"{ham}_Beta={beta}"
        sim_dir = _TMPDIR / "ALF_data" / name
        sim_dir.mkdir(parents=True, exist_ok=True)

        if state == "confin":
            (sim_dir / "confin_0.h5").touch()
            (sim_dir / "confin_1.h5").touch()
            (sim_dir / "data.h5").touch()
        elif state == "confout":
            (sim_dir / "confout_0.h5").touch()
            (sim_dir / "data.h5").touch()
        elif state == "conflict":
            # data.h5 without any conf files — submission is blocked
            (sim_dir / "data.h5").touch()
        # None → fresh, leave directory empty

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

_STATUS_CYCLE = [
    "RUNNING",    # confin beta=1   ↺ running
    "PENDING",    # confin beta=2   ↺ queued
    "COMPLETED",  # confout beta=4  ↺ completed (confin after rename)
    "RUNNING",    # confout beta=6  running
    "COMPLETED",  # fresh beta=10   completed
    "RUNNING",    # fresh tV        running
]


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

        # Write fake log for RUNNING / COMPLETED jobs.
        log = _SUBMIT_DIR / f"{jid}_0_log.out"
        has_confin = any(Path(sim.sim_dir).glob("confin_*"))
        has_confout = any(Path(sim.sim_dir).glob("confout_*"))
        if has_confout:
            mode = "continuation (confout → confin auto-renamed)"
        elif has_confin:
            mode = "checkpoint restart"
        else:
            mode = "fresh start"
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

        # Write ALF info file for COMPLETED jobs so the  i  keybinding works.
        if status == "COMPLETED":
            info_file = Path(sim.sim_dir) / "info"
            info_file.write_text(
                f"Hamiltonian : {sim.ham_name}\n"
                f"Beta        : {sim.sim_dict['Beta']}\n"
                f"L1 / L2     : {sim.sim_dict['L1']} / {sim.sim_dict['L2']}\n"
                f"OMP threads : {sim.n_omp}\n"
                f"NBin target : {sim.sim_dict.get('NBin', '—')}\n"
                f"Bins done   : 40\n"
                f"Mode        : {mode}\n"
                f"Status      : COMPLETED\n"
            )

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
    counts: dict = {}
    for _, _, state in _SIM_SPECS:
        counts[state] = counts.get(state, 0) + 1
    print(
        f"Prepared {len(sims)} simulations:  "
        f"{counts.get('confin', 0)} confin  "
        f"{counts.get('confout', 0)} confout  "
        f"{counts.get('conflict', 0)} blocked (⚠)  "
        f"{counts.get(None, 0)} fresh"
    )
    print()
    print("  The blocked sim (Beta=8.0) has data.h5 but no checkpoint files.")
    print("  Deselect it with  space  before pressing  s  to submit.\n")

    cs = ClusterSubmitter(
        "slurm",
        submit_dir=str(_SUBMIT_DIR),
        slurm_mem="8G",
        partition_rules=CPU_PARTITIONS,
        job_name="ckpt_demo",
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
