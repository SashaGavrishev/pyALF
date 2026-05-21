"""Tests for SubmissionReview helpers — save_for_ssh and !mail command."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock

from py_alf.cluster_submission import ClusterSubmitter
from py_alf.simulation import Simulation
from py_alf.submission_tui import SubmissionReview, save_for_ssh

# ---------------------------------------------------------------------------
# Helpers for SubmissionReview tests
# ---------------------------------------------------------------------------


def _make_sim(sim_dir: Path) -> MagicMock:
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim = MagicMock()
    sim.__class__ = Simulation
    sim.ham_name = "Hubbard"
    sim.sim_dir = str(sim_dir)
    sim.sim_dict = {"Beta": 4.0, "CPU_MAX": 24}
    sim.n_omp = 4
    sim.n_mpi = 1
    sim.mpi = False
    sim.mpiexec = "mpiexec"
    sim.mpiexec_args = []
    sim.config = "GNU HDF5 NOMPI"
    return sim


def _app(tmp_path: Path, mail_type: str | None = None) -> SubmissionReview:
    cs = ClusterSubmitter(
        "slurm",
        submit_dir=str(tmp_path / ".alfmonitor"),
        slurm_mem="4G",
        partition_rules={"short": 2, "medium": 48},
        mail_type=mail_type,
    )
    return SubmissionReview(cs, [_make_sim(tmp_path / "sim0")])

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


# ---------------------------------------------------------------------------
# !mail command
# ---------------------------------------------------------------------------


async def test_mail_cmd_sets_end(tmp_path):
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._exec_command("!mail end")
        assert app._mail_flags == {"END"}


async def test_mail_cmd_sets_fail(tmp_path):
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._exec_command("!mail fail")
        assert app._mail_flags == {"FAIL"}


async def test_mail_cmd_sets_all(tmp_path):
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._exec_command("!mail all")
        assert app._mail_flags == {"ALL"}


async def test_mail_cmd_sets_end_and_fail(tmp_path):
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._exec_command("!mail end fail")
        assert app._mail_flags == {"END", "FAIL"}


async def test_mail_cmd_none_clears_flags(tmp_path):
    app = _app(tmp_path, mail_type="END FAIL")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._exec_command("!mail none")
        assert app._mail_flags == set()


async def test_mail_cmd_clear_flag_clears_flags(tmp_path):
    app = _app(tmp_path, mail_type="END")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._exec_command("!mail --clear")
        assert app._mail_flags == set()


async def test_mail_cmd_no_arg_clears_flags(tmp_path):
    app = _app(tmp_path, mail_type="FAIL")
    async with app.run_test() as pilot:
        await pilot.pause()
        app._exec_command("!mail")
        assert app._mail_flags == set()


async def test_mail_cmd_alias_mailtype(tmp_path):
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._exec_command("!mailtype end")
        assert app._mail_flags == {"END"}


async def test_mail_cmd_alias_mail_type(tmp_path):
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._exec_command("!mail_type fail")
        assert app._mail_flags == {"FAIL"}


async def test_mail_cmd_unknown_flag_notifies(tmp_path):
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        notified = []
        app.notify = lambda msg, **kw: notified.append(msg)
        app._exec_command("!mail begin")
        assert any("begin" in m.lower() or "unknown" in m.lower() or "valid" in m.lower() for m in notified)


async def test_mail_cmd_all_exclusive_with_end_notifies(tmp_path):
    app = _app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        notified = []
        app.notify = lambda msg, **kw: notified.append(msg)
        app._exec_command("!mail all end")
        assert app._mail_flags == set()  # unchanged (command rejected)
        assert any("exclusive" in m.lower() or "all" in m.lower() for m in notified)


async def test_mail_cmd_syncs_buttons(tmp_path):
    from rich.text import Text
    from textual.widgets import Button

    from py_alf.submission_tui import _btn_label

    def _plain(tag: str, active: bool) -> str:
        return Text.from_markup(_btn_label(tag, active)).plain

    app = _app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._exec_command("!mail end")
        await pilot.pause()
        end_btn = app.query_one("#mail-end", Button)
        fail_btn = app.query_one("#mail-fail", Button)
        all_btn = app.query_one("#mail-all", Button)
        assert str(end_btn.label) == _plain("END", True)
        assert str(fail_btn.label) == _plain("FAIL", False)
        assert str(all_btn.label) == _plain("ALL", False)
