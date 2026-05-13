#!/usr/bin/env python3
"""
Visual demo for the SubmissionReview TUI — no ALF installation required.

Run from the repo root:

    python demo_submission_tui.py

Controls inside the TUI:
  space     toggle the highlighted simulation on / off
  s         submit selected simulations (mocked — nothing actually runs)
  q         cancel and exit
  Executor buttons switch the backend; architecture panel updates live.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

from py_alf.cluster_submission import ClusterSubmitter
from py_alf.monitor import SimulationMonitor
from py_alf.simulation import Simulation
from py_alf.submission_tui import SubmissionReview


# ---------------------------------------------------------------------------
# Cluster partition table (excluding the graphic/GPU partition)
#
# Partition    Wall-time    Notes
# ─────────────────────────────────────────────────────────────────────────
# debug        10 min       Cluster-access test only; not auto-selected
#                           because CPU_MAX is always ≥ 1 h.  Submit via
#                           job_properties if needed.
# short        2 h          Most nodes; quick jobs
# medium       2 days       Subset of short nodes
# long         14 days      Subset of medium nodes
# extra_long   28 days      Small partition for extreme runtimes
# ---------------------------------------------------------------------------

CPU_PARTITIONS: dict[str, float] = {
    "short":      2,        # 2 h
    "medium":     48,       # 2 days
    "long":       336,      # 14 days
    "extra_long": 672,      # 28 days
}


# ---------------------------------------------------------------------------
# Mock simulation factory
# ---------------------------------------------------------------------------


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
    sim.__class__ = Simulation          # satisfies isinstance checks
    sim.ham_name = ham_name
    sim.sim_dir = sim_dir
    sim.sim_dict = sim_dict
    sim.n_omp = n_omp
    sim.n_mpi = n_mpi
    sim.mpi = mpi
    mpi_flag = "MPI" if mpi else "NOMPI"
    sim.config = f"{machine} HDF5 {mpi_flag}"
    return sim


# ---------------------------------------------------------------------------
# Demo batches — each exercises a different partition / resource shape.
# All sims within a batch share the same n_omp / n_mpi so the architecture
# panel is consistent (the submit validator enforces this constraint anyway).
# ---------------------------------------------------------------------------

BATCHES: dict[str, dict] = {

    # ── short partition: 1 process × 4 OMP, CPU_MAX=1h (1 ≤ 2h limit) ──
    "short  · serial  · 4 OMP": dict(
        sims=[
            make_mock_sim(
                "Hubbard_Plain_Vanilla",
                f"ALF_data/Hubbard_PV_Beta={b}_L1=4_L2=4",
                {"Beta": b, "L1": 4, "L2": 4, "CPU_MAX": 1, "NBin": 40},
                n_omp=4,
            )
            for b in [1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
        ],
        cs=ClusterSubmitter(
            slurm_mem="4G",
            partition_rules=CPU_PARTITIONS,
        ),
    ),

    # ── medium partition: 2 ranks × 4 OMP, CPU_MAX=24h (2 < 24 ≤ 48h) ──
    "medium · MPI×2   · 4 OMP": dict(
        sims=[
            make_mock_sim(
                "Hubbard_Plain_Vanilla",
                f"ALF_data/Hubbard_PV_MPI2_Beta={b}_L1=6_L2=6",
                {"Beta": b, "L1": 6, "L2": 6, "CPU_MAX": 24, "NBin": 20},
                n_omp=4,
                n_mpi=2,
                mpi=True,
            )
            for b in [0.5, 1.0, 2.0, 4.0, 8.0]
        ],
        cs=ClusterSubmitter(
            slurm_mem="8G",
            partition_rules=CPU_PARTITIONS,
        ),
    ),

    # ── long partition: 4 ranks × 4 OMP, CPU_MAX=72h (48 < 72 ≤ 336h) ──
    "long   · MPI×4   · 4 OMP": dict(
        sims=[
            make_mock_sim(
                "Hubbard_Plain_Vanilla",
                f"ALF_data/Hubbard_PV_MPI4_Beta={b}_L1=8_L2=8",
                {"Beta": b, "L1": 8, "L2": 8, "CPU_MAX": 72, "NBin": 10},
                n_omp=4,
                n_mpi=4,
                mpi=True,
            )
            for b in [0.5, 1.0, 2.0, 4.0]
        ],
        cs=ClusterSubmitter(
            slurm_mem="16G",
            partition_rules=CPU_PARTITIONS,
        ),
    ),

    # ── extra_long: 8 ranks × 8 OMP; full grid exercises 2-row layout ──
    "extra_long · MPI×8 · 8 OMP": dict(
        sims=[
            make_mock_sim(
                "tV_Model",
                f"ALF_data/tV_V={v}_L1=16_L2=1",
                {"V": v, "L1": 16, "L2": 1, "CPU_MAX": 400, "NBin": 5},
                n_omp=8,
                n_mpi=8,
                mpi=True,
                machine="INTEL",
            )
            for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        ],
        cs=ClusterSubmitter(
            slurm_mem="32G",
            partition_rules=CPU_PARTITIONS,
        ),
    ),

    # ── local executor: no SLURM params, architecture shows local mode ──
    "local executor (no cluster)": dict(
        sims=[
            make_mock_sim(
                "Kondo",
                f"ALF_data/Kondo_JK={jk}_L1=4",
                {"JK": jk, "L1": 4, "NBin": 5},
                n_omp=2,
            )
            for jk in [0.5, 1.0, 1.5, 2.0]
        ],
        cs=ClusterSubmitter("local"),
    ),
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def mock_submit(self: ClusterSubmitter, sims, **kwargs) -> list:
    """No-op replacement for ClusterSubmitter.submit — prints what would run."""
    sims = list(sims)
    print(
        f"\n[MOCK] executor={self.executor!r} — "
        f"would submit {len(sims)} sim(s):\n"
        + "\n".join(f"  {s.sim_dir}" for s in sims)
    )
    return [MagicMock(job_id=f"mock_{i}") for i in range(len(sims))]


def main() -> None:
    names = list(BATCHES)
    print("Available demo batches:\n")
    for i, name in enumerate(names):
        print(f"  {i}  {name}")
    raw = input(f"\nPick batch [0–{len(names)-1}, default 0]: ").strip()
    idx = int(raw) if raw.isdigit() and int(raw) < len(names) else 0
    chosen = names[idx]
    print(f"\nLaunching: {chosen}\n")

    batch = BATCHES[chosen]

    with patch.object(ClusterSubmitter, "submit", mock_submit):
        app = SubmissionReview(batch["cs"], batch["sims"])
        result = app.run()

    print()
    if result is None:
        print("Cancelled — no jobs submitted.")
    else:
        print(f"Submitted {len(result)} simulation(s):")
        for s in result:
            print(f"  {s.sim_dir}")
        if app.open_monitor:
            print("\nOpening monitor…")
            SimulationMonitor(
                result,
                cluster_submitter=app.submitted_cs,
            ).run()


if __name__ == "__main__":
    main()
