"""Tests for SimulationMonitor TUI (py_alf.monitor)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from py_alf.cluster_submission import ClusterSubmitter
from py_alf.monitor import ConfirmScreen, LogViewerScreen, SimulationMonitor, _styled
from py_alf.simulation import Simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_sim(sim_dir: Path, ham="Hubbard", n_omp=4, n_mpi=1, mpi=False):
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim = MagicMock()
    sim.__class__ = Simulation
    sim.sim_dir = str(sim_dir)
    sim.ham_name = ham
    sim.n_omp = n_omp
    sim.n_mpi = n_mpi
    sim.mpi = mpi
    sim.sim_dict = {"U": 4.0, "beta": 10.0}
    return sim


def _make_mock_cs(submit_dir=None):
    cs = MagicMock()
    cs.__class__ = ClusterSubmitter
    cs.slurm_partition = "short"
    cs.slurm_mem = "4G"
    cs.submit_dir = Path(submit_dir) if submit_dir else Path("submitit")
    return cs


def _monitor(sims, **kwargs):
    """Create a SimulationMonitor with a very long refresh interval."""
    kwargs.setdefault("refresh_interval", 9999.0)
    return SimulationMonitor(sims, **kwargs)


class _HostApp:
    """Minimal App factory used to host isolated modal screens in tests."""

    @staticmethod
    def make():
        from textual.app import App, ComposeResult
        from textual.widgets import Label

        class _App(App):
            CSS = ""

            def compose(self) -> ComposeResult:
                yield Label("host")

        return _App()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def no_slurm():
    """Patch every SLURM-touching helper so tests run without a real cluster."""
    with (
        patch("py_alf.monitor.get_job_id", return_value=None),
        patch("py_alf.monitor._get_slurm_status_bulk", return_value={}),
        patch("py_alf.monitor._bin_count", return_value=0),
    ):
        yield


# ---------------------------------------------------------------------------
# Pure unit tests (synchronous)
# ---------------------------------------------------------------------------


def test_styled_running():
    t = _styled("RUNNING")
    assert str(t) == "RUNNING"
    assert "green" in t.style


def test_styled_failed():
    t = _styled("FAILED")
    assert "red" in t.style


def test_styled_pending():
    t = _styled("PENDING")
    assert "yellow" in t.style


def test_styled_unknown_status():
    t = _styled("NOVELSTATUS")
    assert t.style == ""


def test_init_defaults(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    app = _monitor([sim])
    assert app._sims == [sim]
    assert app._cs is None
    assert app._submit_dir is None
    assert app._param_keys == []
    assert app._param_headers == []


def test_init_submit_dir_falls_back_to_submitter(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    cs = _make_mock_cs(tmp_path / "logs")
    app = _monitor([sim], cluster_submitter=cs)
    assert app._submit_dir == cs.submit_dir


def test_init_explicit_submit_dir_overrides_submitter(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    cs = _make_mock_cs(tmp_path / "logs")
    explicit = tmp_path / "custom_logs"
    app = _monitor([sim], cluster_submitter=cs, submit_dir=explicit)
    assert app._submit_dir == explicit


def test_init_param_headers_default_to_keys(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    app = _monitor([sim], param_keys=["U", "beta"])
    assert app._param_headers == ["U", "beta"]


def test_init_explicit_param_headers(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    app = _monitor([sim], param_keys=["U", "beta"], param_headers=["U", "β"])
    assert app._param_headers == ["U", "β"]


# ---------------------------------------------------------------------------
# Textual (async) tests
# ---------------------------------------------------------------------------


async def test_confirm_screen_yes():
    results = []
    app = _HostApp.make()
    async with app.run_test() as pilot:
        app.push_screen(ConfirmScreen("Continue?"), results.append)
        await pilot.pause()
        await pilot.click("#confirm-yes")
        await pilot.pause()
    assert results == [True]


async def test_confirm_screen_no():
    results = []
    app = _HostApp.make()
    async with app.run_test() as pilot:
        app.push_screen(ConfirmScreen("Continue?"), results.append)
        await pilot.pause()
        await pilot.click("#confirm-no")
        await pilot.pause()
    assert results == [False]


async def test_confirm_screen_escape():
    results = []
    app = _HostApp.make()
    async with app.run_test() as pilot:
        app.push_screen(ConfirmScreen("Continue?"), results.append)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert results == [False]


async def test_log_viewer_shows_content():
    app = _HostApp.make()
    async with app.run_test() as pilot:
        app.push_screen(LogViewerScreen("My Log", "line one\nline two"))
        await pilot.pause()
        # app.screen is the topmost (modal) screen after push_screen
        body = app.screen.query_one("#log-body")
        assert "line one" in str(body.content)


async def test_log_viewer_close_returns_to_host():
    app = _HostApp.make()
    async with app.run_test() as pilot:
        app.push_screen(LogViewerScreen("Log", "content"))
        await pilot.pause()
        await pilot.click("#log-close")
        await pilot.pause()
        # After dismissal the LogViewerScreen is no longer the active screen
        assert not isinstance(app.screen, LogViewerScreen)


async def test_log_viewer_escape_returns_to_host():
    app = _HostApp.make()
    async with app.run_test() as pilot:
        app.push_screen(LogViewerScreen("Log", "content"))
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, LogViewerScreen)


async def test_monitor_table_has_one_row_per_sim(tmp_path, no_slurm):
    sims = [_make_mock_sim(tmp_path / f"sim{i}") for i in range(3)]
    app = _monitor(sims)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        table = app.query_one("DataTable")
        assert table.row_count == 3


async def test_monitor_status_column_present(tmp_path, no_slurm):
    sim = _make_mock_sim(tmp_path / "sim0")
    app = _monitor([sim])
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        table = app.query_one("DataTable")
        col_labels = [str(col.label) for col in table.columns.values()]
        assert "Status" in col_labels
        assert "Elapsed" in col_labels


async def test_monitor_shows_slurm_params_when_submitter_provided(tmp_path, no_slurm):
    sim = _make_mock_sim(tmp_path / "sim0")
    cs = _make_mock_cs(tmp_path / "logs")
    app = _monitor([sim], cluster_submitter=cs)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        table = app.query_one("DataTable")
        col_labels = [str(col.label) for col in table.columns.values()]
        assert "partition" in col_labels
        assert "mem" in col_labels


async def test_action_view_logs_warns_when_no_jobid(tmp_path, no_slurm):
    sim = _make_mock_sim(tmp_path / "sim0")
    app = _monitor([sim])
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        with patch.object(app, "notify") as mock_notify:
            await pilot.press("l")
            await app.workers.wait_for_complete()
            await pilot.pause()
    assert any("No job ID" in str(call.args[0]) for call in mock_notify.call_args_list)


async def test_action_view_logs_errors_when_log_not_found(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="99"),
        patch("py_alf.monitor._get_slurm_status_bulk", return_value={"99": {"status": "COMPLETED", "runtime": None}}),
        patch("py_alf.monitor._bin_count", return_value=0),
        patch("py_alf.monitor._find_job_log", return_value=None),
    ):
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            with patch.object(app, "notify") as mock_notify:
                await pilot.press("l")
                await app.workers.wait_for_complete()
                await pilot.pause()
    assert any("Cannot locate" in str(call.args[0]) for call in mock_notify.call_args_list)


async def test_action_view_logs_opens_log_screen(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    log_file = tmp_path / "test.log"
    log_file.write_text("first line\nsecond line\n")
    with (
        patch("py_alf.monitor.get_job_id", return_value="55"),
        patch("py_alf.monitor._get_slurm_status_bulk", return_value={"55": {"status": "COMPLETED", "runtime": None}}),
        patch("py_alf.monitor._bin_count", return_value=8),
        patch("py_alf.monitor._find_job_log", return_value=log_file),
    ):
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("l")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, LogViewerScreen)


async def test_action_cancel_job_warns_when_no_jobid(tmp_path, no_slurm):
    sim = _make_mock_sim(tmp_path / "sim0")
    app = _monitor([sim])
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        with patch.object(app, "notify") as mock_notify:
            await pilot.press("c")
            await pilot.pause()
    assert any("No job ID" in str(call.args[0]) for call in mock_notify.call_args_list)


async def test_action_cancel_job_confirmed(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="77"),
        patch("py_alf.monitor._get_slurm_status_bulk", return_value={"77": {"status": "RUNNING", "runtime": "00:01:00"}}),
        patch("py_alf.monitor._bin_count", return_value=0),
        patch("py_alf.monitor.cancel_cluster_job", return_value=True) as mock_cancel,
    ):
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            await pilot.click("#confirm-yes")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
    mock_cancel.assert_called_once_with(sim)


async def test_action_cancel_job_declined(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="77"),
        patch("py_alf.monitor._get_slurm_status_bulk", return_value={"77": {"status": "RUNNING", "runtime": "00:01:00"}}),
        patch("py_alf.monitor._bin_count", return_value=0),
        patch("py_alf.monitor.cancel_cluster_job", return_value=True) as mock_cancel,
    ):
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            await pilot.click("#confirm-no")
            await pilot.pause()
    mock_cancel.assert_not_called()


async def test_action_cancel_array_uses_base_id(tmp_path):
    """scancel is called with the base array ID, not the task-specific ID."""
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="88_2"),
        patch("py_alf.monitor._get_slurm_status_bulk", return_value={"88_2": {"status": "RUNNING", "runtime": "00:02:00"}}),
        patch("py_alf.monitor._bin_count", return_value=0),
        patch("py_alf.monitor.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("a")
            await pilot.pause()
            await pilot.click("#confirm-yes")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
    mock_run.assert_called_once_with(
        ["scancel", "88"], capture_output=True, text=True
    )


async def test_action_cancel_array_declined(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="88_2"),
        patch("py_alf.monitor._get_slurm_status_bulk", return_value={"88_2": {"status": "RUNNING", "runtime": None}}),
        patch("py_alf.monitor._bin_count", return_value=0),
        patch("py_alf.monitor.subprocess.run") as mock_run,
    ):
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("a")
            await pilot.pause()
            await pilot.click("#confirm-no")
            await pilot.pause()
    mock_run.assert_not_called()


async def test_action_resubmit_errors_without_submitter(tmp_path, no_slurm):
    sim = _make_mock_sim(tmp_path / "sim0")
    app = _monitor([sim])  # no cluster_submitter
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        with patch.object(app, "notify") as mock_notify:
            await pilot.press("r")
            await pilot.pause()
    assert any(
        "ClusterSubmitter" in str(call.args[0])
        for call in mock_notify.call_args_list
    )


async def test_action_resubmit_confirmed(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    cs = _make_mock_cs(tmp_path / "logs")
    with (
        patch("py_alf.monitor.get_job_id", return_value=None),
        patch("py_alf.monitor._get_slurm_status_bulk", return_value={}),
        patch("py_alf.monitor._bin_count", return_value=3),
    ):
        app = _monitor([sim], cluster_submitter=cs)
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            await pilot.click("#confirm-yes")
            await pilot.pause()
    cs.submit.assert_called_once_with(sim)


async def test_action_resubmit_declined(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    cs = _make_mock_cs(tmp_path / "logs")
    with (
        patch("py_alf.monitor.get_job_id", return_value=None),
        patch("py_alf.monitor._get_slurm_status_bulk", return_value={}),
        patch("py_alf.monitor._bin_count", return_value=0),
    ):
        app = _monitor([sim], cluster_submitter=cs)
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            await pilot.click("#confirm-no")
            await pilot.pause()
    cs.submit.assert_not_called()


async def test_f5_retriggers_refresh(tmp_path, no_slurm):
    sim = _make_mock_sim(tmp_path / "sim0")
    app = _monitor([sim])
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("f5")
        await app.workers.wait_for_complete()
        await pilot.pause()
        # Table should still have one row after the refresh
        assert app.query_one("DataTable").row_count == 1