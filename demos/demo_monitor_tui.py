#!/usr/bin/env python3
"""
Visual demo for the SimulationMonitor TUI — no ALF installation required.

Run from the repo root:

    python demos/demo_monitor_tui.py

Two demo modes
--------------
  0  fresh sims — launches the monitor directly with a pre-built mix of
                  SLURM statuses (RUNNING, PENDING, COMPLETED, FAILED,
                  CANCELLED, INACTIVE, CRASHED)
  1  from session — writes a session manifest to disk and loads it via
                    SimulationMonitor.from_session(), exercising the same
                    code path used after a real SubmissionReview run

All SLURM queries, bin counts, and cluster operations are mocked so no
cluster connection is needed.

Controls inside the TUI
-----------------------
  q     quit
  l     view log (shows fake ALF output for jobs that have a log file)
  c     cancel individual job    (mocked — prints confirmation)
  a     cancel SLURM array       (mocked)
  r     resubmit selected sim    (mocked)
  f5    manual refresh
"""

from __future__ import annotations

import atexit
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from py_alf.cluster_submission import ClusterSubmitter
from py_alf.monitor import SimulationMonitor

# ---------------------------------------------------------------------------
# Temporary workspace — wiped on exit
# ---------------------------------------------------------------------------

_TMPDIR = Path(tempfile.mkdtemp(prefix="alf_demo_monitor_"))
atexit.register(shutil.rmtree, _TMPDIR, ignore_errors=True)

_SUBMIT_DIR = _TMPDIR / ".alfmonitor"
_SUBMIT_DIR.mkdir()

# ---------------------------------------------------------------------------
# Mock SLURM data
# ---------------------------------------------------------------------------

ARRAY_ID = "44321"

# job_id → (status, runtime, n_bins)
_SIM_TABLE: list[tuple[str, str, str | None, int]] = [
    # ham_name,       job_id             status       runtime       n_bins
    ("Hubbard_PV", f"{ARRAY_ID}_0", "RUNNING", "2:15:30", 18),
    ("Hubbard_PV", f"{ARRAY_ID}_1", "RUNNING", "2:14:02", 15),
    ("Hubbard_PV", f"{ARRAY_ID}_2", "PENDING", None, 0),
    ("Hubbard_PV", f"{ARRAY_ID}_3", "COMPLETED", "1:58:44", 40),
    ("Hubbard_PV", f"{ARRAY_ID}_4", "COMPLETED", "2:02:17", 40),
    ("Hubbard_PV", f"{ARRAY_ID}_5", "FAILED", "0:03:11", 3),
    ("tV_Model", f"{ARRAY_ID}_6", "CANCELLED", "0:00:05", 0),
    ("tV_Model", None, "INACTIVE", None, 0),  # no job ID
]

_BETA_VALUES = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 0.5, 1.0]
_L_VALUES = [4, 4, 4, 4, 4, 4, 8, 8]

CPU_PARTITIONS: dict[str, float] = {
    "short": 2,
    "medium": 48,
    "long": 336,
    "extra_long": 672,
}


# ---------------------------------------------------------------------------
# Build temp filesystem (sim dirs + jobid.txt + fake logs)
# ---------------------------------------------------------------------------


def _setup_workspace() -> tuple[list, dict, dict]:
    """
    Create sim dirs, write jobid.txt where applicable, write fake log files.

    Returns
    -------
    sims      : list of lightweight sim-like objects
    statuses  : dict for _get_slurm_status_bulk mock
    bin_map   : sim_dir → n_bins for _bin_count mock
    """
    from types import SimpleNamespace

    sims = []
    statuses: dict[str, dict] = {}
    bin_map: dict[str, int] = {}

    for i, (ham, jid, status, runtime, n_bins) in enumerate(_SIM_TABLE):
        beta = _BETA_VALUES[i]
        L = _L_VALUES[i]
        sim_dir = _TMPDIR / "ALF_data" / f"{ham}_Beta={beta}_L={L}"
        sim_dir.mkdir(parents=True, exist_ok=True)

        if jid is not None:
            (sim_dir / "jobid.txt").write_text(jid)
            statuses[jid] = {"status": status, "runtime": runtime}
            if status in ("RUNNING", "COMPLETED"):
                log = _SUBMIT_DIR / f"{jid}_0_log.out"
                log.write_text(_fake_log(jid, ham, beta, L, status))
        else:
            # INACTIVE: leave a RUNNING sentinel file to trigger CRASHED display
            # if you want to test that path, uncomment the next line:
            # (sim_dir / "RUNNING").touch()
            pass

        sim = SimpleNamespace(
            ham_name=ham,
            sim_dir=str(sim_dir),
            sim_dict={"Beta": beta, "L1": L, "L2": L, "CPU_MAX": 24},
            n_omp=4,
            n_mpi=1,
            mpi=False,
            mpiexec="mpiexec",
            mpiexec_args=[],
            config="GNU HDF5 NOMPI",
        )
        sims.append(sim)
        bin_map[str(sim_dir)] = n_bins

    return sims, statuses, bin_map


def _fake_log(jid: str, ham: str, beta: float, L: int, status: str) -> str:
    lines = [
        f"[ALF] job {jid}  ham={ham}  Beta={beta}  L={L}",
        "[ALF] Initialising Hamiltonian...",
        "[ALF] Warmup phase  — 1000 sweeps",
        "[ALF] Production phase — 5000 sweeps",
    ]
    if status == "COMPLETED":
        lines += [
            "[ALF] Writing observables to data.h5",
            "[ALF] Done. All bins collected.",
        ]
    else:
        lines += ["[ALF] Still running..."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Mock patches
# ---------------------------------------------------------------------------


def _mock_bin_count_factory(bin_map: dict[str, int]):
    def _mock(sim, **_kw):
        return bin_map.get(str(sim.sim_dir), 0)

    return _mock


def _mock_resubmit(self: ClusterSubmitter, sims, **kwargs) -> list:
    sims = list(sims) if not hasattr(sims, "sim_dir") else [sims]
    print(f"\n[MOCK] resubmit: {[str(s.sim_dir) for s in sims]}")
    jobs = []
    for _s in sims:
        j = MagicMock()
        j.job_id = "99999_resubmit"
        jobs.append(j)
    return jobs


# ---------------------------------------------------------------------------
# Session manifest for mode 1
# ---------------------------------------------------------------------------


def _write_demo_session(sims: list, statuses: dict) -> Path:
    """Write a session JSON that from_session() can load."""
    entries = []
    for s in sims:
        jid_file = Path(s.sim_dir) / "jobid.txt"
        entries.append(
            {
                "sim_dir": s.sim_dir,
                "job_id": jid_file.read_text().strip() if jid_file.exists() else None,
                "ham_name": s.ham_name,
                "n_omp": s.n_omp,
                "n_mpi": s.n_mpi,
                "mpi": s.mpi,
                "mpiexec": s.mpiexec,
                "mpiexec_args": s.mpiexec_args,
                "sim_dict": s.sim_dict,
            }
        )

    manifest = {
        "version": 1,
        "submitted_at": "2026-05-13T10:00:00",
        "cluster_submitter": {
            "executor": "slurm",
            "submit_dir": str(_SUBMIT_DIR),
            "slurm_mem": "8G",
            "partition_rules": CPU_PARTITIONS,
            "job_name": None,
            "mail_type": "END",
            "wckey": None,
            "stderr_to_stdout": False,
            "slurm_kwargs": {},
        },
        "entries": entries,
    }
    path = _SUBMIT_DIR / "session_demo.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

MODES = {
    "fresh sims (direct SimulationMonitor)": "fresh",
    "from session manifest (SimulationMonitor.from_session)": "session",
}


def main() -> None:
    sims, statuses, bin_map = _setup_workspace()

    names = list(MODES)
    print("Demo modes:\n")
    for i, name in enumerate(names):
        print(f"  {i}  {name}")
    raw = input(f"\nPick mode [0–{len(names) - 1}, default 0]: ").strip()
    idx = int(raw) if raw.isdigit() and int(raw) < len(names) else 0
    mode = list(MODES.values())[idx]

    print(f"\nTemp workspace: {_TMPDIR}\n")

    cs = ClusterSubmitter(
        "slurm",
        submit_dir=str(_SUBMIT_DIR),
        slurm_mem="8G",
        partition_rules=CPU_PARTITIONS,
    )

    with (
        patch("py_alf.monitor._get_slurm_status_bulk", return_value=statuses),
        patch(
            "py_alf.monitor._bin_count", side_effect=_mock_bin_count_factory(bin_map)
        ),
        patch("py_alf.monitor.cancel_cluster_job", return_value=True),
        patch.object(ClusterSubmitter, "submit", _mock_resubmit),
    ):
        if mode == "fresh":
            monitor = SimulationMonitor(
                sims,
                cluster_submitter=cs,
                submit_dir=str(_SUBMIT_DIR),
                param_keys=["Beta", "L1"],
            )
        else:
            session_path = _write_demo_session(sims, statuses)
            print(f"Session manifest: {session_path}\n")
            monitor = SimulationMonitor.from_session(
                session_path,
                cluster_submitter=cs,
                param_keys=["Beta", "L1"],
            )

        monitor.run()


if __name__ == "__main__":
    main()
