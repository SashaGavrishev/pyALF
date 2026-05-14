"""Tests for SubmissionReview helpers — save_for_ssh."""

from __future__ import annotations

import pickle
from pathlib import Path

from py_alf.submission_tui import save_for_ssh

# ---------------------------------------------------------------------------
# Helpers — simple picklable stand-ins (MagicMock with overridden __class__
# cannot be pickled, so we use plain objects instead)
# ---------------------------------------------------------------------------


class _DummyCS:
    executor = "slurm"


class _DummySim:
    def __init__(self, sim_dir: Path) -> None:
        self.sim_dir = str(sim_dir)
        self.ham_name = "Hubbard"
        self.n_omp = 4
        self.n_mpi = 1
        self.mpi = False
        self.sim_dict = {"U": 4.0}


# ---------------------------------------------------------------------------
# save_for_ssh
# ---------------------------------------------------------------------------


def test_save_for_ssh_creates_pickle(tmp_path):
    pkl = tmp_path / "state.pkl"
    save_for_ssh(_DummyCS(), [_DummySim(tmp_path / "sim0")], path=pkl)
    assert pkl.exists()


def test_save_for_ssh_pickle_contains_cs_and_sims(tmp_path):
    pkl = tmp_path / "state.pkl"
    cs = _DummyCS()
    sims = [_DummySim(tmp_path / f"sim{i}") for i in range(2)]
    save_for_ssh(cs, sims, path=pkl)
    data = pickle.loads(pkl.read_bytes())
    assert "cs" in data
    assert "sims" in data
    assert len(data["sims"]) == 2


def test_save_for_ssh_returns_command_string(tmp_path):
    pkl = tmp_path / "state.pkl"
    cmd = save_for_ssh(_DummyCS(), [_DummySim(tmp_path / "sim0")], path=pkl)
    assert isinstance(cmd, str)


def test_save_for_ssh_command_references_pickle_path(tmp_path):
    pkl = tmp_path / "state.pkl"
    cmd = save_for_ssh(_DummyCS(), [_DummySim(tmp_path / "sim0")], path=pkl)
    assert str(pkl) in cmd


def test_save_for_ssh_command_references_submission_review(tmp_path):
    pkl = tmp_path / "state.pkl"
    cmd = save_for_ssh(_DummyCS(), [_DummySim(tmp_path / "sim0")], path=pkl)
    assert "SubmissionReview" in cmd


def test_save_for_ssh_command_uses_run_with_monitor(tmp_path):
    """Generated command calls run_with_monitor so the monitor handover works."""
    pkl = tmp_path / "state.pkl"
    cmd = save_for_ssh(_DummyCS(), [_DummySim(tmp_path / "sim0")], path=pkl)
    assert "run_with_monitor" in cmd


def test_save_for_ssh_default_path_is_tmp(tmp_path, capsys):
    """When no path is given, the pickle lands under /tmp."""
    save_for_ssh(_DummyCS(), [_DummySim(tmp_path / "sim0")])
    out = capsys.readouterr().out
    assert "/tmp/" in out


def test_save_for_ssh_prints_confirmation(tmp_path, capsys):
    pkl = tmp_path / "state.pkl"
    save_for_ssh(_DummyCS(), [_DummySim(tmp_path / "sim0")], path=pkl)
    out = capsys.readouterr().out
    assert str(pkl) in out
    assert "SubmissionReview" in out
