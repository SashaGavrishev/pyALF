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
    sim.job_id = None  # MagicMock auto-creates truthy attributes; pin to None so the
    # monitor falls back to get_job_id() as it would for real Simulation objects.
    return sim


def _make_mock_cs(submit_dir=None):
    cs = MagicMock()
    cs.__class__ = ClusterSubmitter
    cs.executor = "slurm"
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
        patch("py_alf.monitor._get_jobs_resources_bulk", return_value={}),
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
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            return_value={"99": {"status": "COMPLETED", "runtime": None}},
        ),
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
    assert any(
        "Cannot locate" in str(call.args[0]) for call in mock_notify.call_args_list
    )


async def test_action_view_logs_opens_log_screen(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    log_file = tmp_path / "test.log"
    log_file.write_text("first line\nsecond line\n")
    with (
        patch("py_alf.monitor.get_job_id", return_value="55"),
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            return_value={"55": {"status": "COMPLETED", "runtime": None}},
        ),
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
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            return_value={"77": {"status": "RUNNING", "runtime": "00:01:00"}},
        ),
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
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            return_value={"77": {"status": "RUNNING", "runtime": "00:01:00"}},
        ),
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
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            return_value={"88_2": {"status": "RUNNING", "runtime": "00:02:00"}},
        ),
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
    mock_run.assert_called_once_with(["scancel", "88"], capture_output=True, text=True)


async def test_action_cancel_array_declined(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="88_2"),
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            return_value={"88_2": {"status": "RUNNING", "runtime": None}},
        ),
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
        "ClusterSubmitter" in str(call.args[0]) for call in mock_notify.call_args_list
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
    cs.submit.assert_called_once_with(sim, confirm_checkpoint=False)


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


# ---------------------------------------------------------------------------
# _sanitise_nodelist
# ---------------------------------------------------------------------------


def test_sanitise_nodelist_returns_real_node():
    from py_alf.cluster_submission import _sanitise_nodelist

    assert _sanitise_nodelist("compute01") == "compute01"
    assert _sanitise_nodelist("node[001-004]") == "node[001-004]"


def test_sanitise_nodelist_rejects_pending_reason():
    from py_alf.cluster_submission import _sanitise_nodelist

    assert _sanitise_nodelist("(Priority)") is None
    assert _sanitise_nodelist("(Resources)") is None
    assert _sanitise_nodelist("(None)") is None


def test_sanitise_nodelist_rejects_sacct_none_literal():
    from py_alf.cluster_submission import _sanitise_nodelist

    assert _sanitise_nodelist("None") is None
    assert _sanitise_nodelist("N/A") is None
    assert _sanitise_nodelist("none") is None


def test_sanitise_nodelist_rejects_empty_and_none():
    from py_alf.cluster_submission import _sanitise_nodelist

    assert _sanitise_nodelist("") is None
    assert _sanitise_nodelist(None) is None


# ---------------------------------------------------------------------------
# Node column — table structure
# ---------------------------------------------------------------------------


async def test_monitor_table_has_node_column(tmp_path, no_slurm):
    """The monitor table always includes a 'Node' column."""
    sim = _make_mock_sim(tmp_path / "sim0")
    app = _monitor([sim])
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        col_labels = [
            str(col.label) for col in app.query_one("DataTable").columns.values()
        ]
    assert "Node" in col_labels


async def test_node_column_position_between_status_and_elapsed(tmp_path, no_slurm):
    """'Node' appears immediately after 'Status' and before 'Elapsed'."""
    sim = _make_mock_sim(tmp_path / "sim0")
    app = _monitor([sim])
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        labels = [str(col.label) for col in app.query_one("DataTable").columns.values()]
    status_idx = labels.index("Status")
    node_idx = labels.index("Node")
    elapsed_idx = labels.index("Elapsed")
    assert status_idx < node_idx < elapsed_idx


# ---------------------------------------------------------------------------
# Node column — per-status display values
# ---------------------------------------------------------------------------


async def test_node_shown_for_running_job(tmp_path):
    """A RUNNING job with a nodelist shows the node name in the Node column."""
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="42_0"),
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            return_value={
                "42_0": {
                    "status": "RUNNING",
                    "runtime": "01:00:00",
                    "nodelist": "compute03",
                }
            },
        ),
        patch("py_alf.monitor._bin_count", return_value=5),
    ):
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
    assert app._row_data[0]["node"] == "compute03"


async def test_node_dash_for_pending_job(tmp_path):
    """A PENDING job (no nodelist from SLURM) shows '-' in the Node column."""
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="7"),
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            return_value={
                "7": {"status": "PENDING", "runtime": None, "nodelist": None}
            },
        ),
        patch("py_alf.monitor._bin_count", return_value=0),
    ):
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
    assert app._row_data[0]["node"] == "-"


async def test_node_dash_for_inactive_sim(tmp_path, no_slurm):
    """A sim with no jobid.txt (INACTIVE) always shows '-' in the Node column."""
    sim = _make_mock_sim(tmp_path / "sim0")
    # no_slurm fixture: get_job_id returns None → INACTIVE path
    app = _monitor([sim])
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert app._row_data[0]["node"] == "-"


async def test_node_shown_for_completed_job_from_sacct(tmp_path):
    """sacct history also carries nodelist; completed jobs display the node."""
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="99"),
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            return_value={
                "99": {
                    "status": "COMPLETED",
                    "runtime": "02:30:00",
                    "nodelist": "hpc-node07",
                }
            },
        ),
        patch("py_alf.monitor._bin_count", return_value=40),
    ):
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
    assert app._row_data[0]["node"] == "hpc-node07"


async def test_node_dash_when_nodelist_key_absent(tmp_path):
    """Old mock dicts without 'nodelist' key are handled gracefully."""
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="55"),
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            # Legacy dict without 'nodelist' key
            return_value={"55": {"status": "RUNNING", "runtime": "00:30:00"}},
        ),
        patch("py_alf.monitor._bin_count", return_value=2),
        patch("py_alf.monitor._get_jobs_resources_bulk", return_value={}),
    ):
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
    assert app._row_data[0]["node"] == "-"


# ---------------------------------------------------------------------------
# _bins_cell
# ---------------------------------------------------------------------------


def test_bins_cell_no_target():
    from py_alf.monitor import _bins_cell

    assert _bins_cell(42, None) == "42"
    assert _bins_cell(0, None) == "0"


def test_bins_cell_zero_target_treated_as_no_target():
    from py_alf.monitor import _bins_cell

    assert _bins_cell(5, 0) == "5"


def test_bins_cell_partial_progress_returns_rich_text():
    from rich.text import Text

    from py_alf.monitor import _bins_cell

    result = _bins_cell(50, 100)
    assert isinstance(result, Text)
    assert "50/100" in result.plain
    assert "░" in result.plain  # bar is not fully filled


def test_bins_cell_complete_progress_no_empty_blocks():
    from rich.text import Text

    from py_alf.monitor import _bins_cell

    result = _bins_cell(100, 100)
    assert isinstance(result, Text)
    assert "100/100" in result.plain
    assert "░" not in result.plain  # bar is fully filled


def test_bins_cell_complete_uses_green_style():
    from py_alf.monitor import _bins_cell

    result = _bins_cell(100, 100)
    assert any("green" in str(span.style) for span in result._spans)


def test_bins_cell_partial_uses_yellow_style():
    from py_alf.monitor import _bins_cell

    result = _bins_cell(50, 100)
    assert any("yellow" in str(span.style) for span in result._spans)


def test_bins_cell_over_target_capped_at_full_bar():
    from py_alf.monitor import _bins_cell

    result = _bins_cell(200, 100)
    assert "░" not in result.plain


# ---------------------------------------------------------------------------
# _SessionEntry.run
# ---------------------------------------------------------------------------


def test_session_entry_has_run_method():
    from py_alf.monitor import _SessionEntry

    e = _SessionEntry("dir", "Ham", 4, 1, False, {})
    assert callable(e.run)


def test_session_entry_run_skips_only_prep():
    from py_alf.monitor import _SessionEntry

    e = _SessionEntry("/tmp/dir", "Ham", 4, 1, False, {})
    with patch("subprocess.run") as mock_run:
        e.run(only_prep=True)
    mock_run.assert_not_called()


def test_session_entry_run_calls_alf_binary(tmp_path):
    import contextlib

    from py_alf.monitor import _SessionEntry

    e = _SessionEntry(str(tmp_path), "Ham", 4, 1, False, {})

    @contextlib.contextmanager
    def _fake_cd(path):
        yield

    with (
        patch("py_alf.simulation.cd", new=_fake_cd),
        patch("subprocess.run") as mock_run,
    ):
        e.run()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd == [str(tmp_path / "ALF.out")]
    assert mock_run.call_args[1]["env"]["OMP_NUM_THREADS"] == "4"


def test_session_entry_run_mpi_wraps_with_mpiexec(tmp_path):
    import contextlib

    from py_alf.monitor import _SessionEntry

    e = _SessionEntry(str(tmp_path), "Ham", 4, 2, True, {})

    @contextlib.contextmanager
    def _fake_cd(path):
        yield

    with (
        patch("py_alf.simulation.cd", new=_fake_cd),
        patch("subprocess.run") as mock_run,
    ):
        e.run()

    cmd = mock_run.call_args[0][0]
    assert cmd == ["mpiexec", "-n", "2", str(tmp_path / "ALF.out")]


# ---------------------------------------------------------------------------
# Peak Mem / CPU Eff columns
# ---------------------------------------------------------------------------


async def test_monitor_table_has_peak_mem_and_cpu_eff_columns(tmp_path, no_slurm):
    sim = _make_mock_sim(tmp_path / "sim0")
    app = _monitor([sim])
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        col_labels = [
            str(col.label) for col in app.query_one("DataTable").columns.values()
        ]
    assert "Peak Mem" in col_labels
    assert "CPU Eff" in col_labels


async def test_monitor_peak_resources_populated_for_completed_job(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="42"),
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            return_value={
                "42": {"status": "COMPLETED", "runtime": "02:00:00", "nodelist": None}
            },
        ),
        patch("py_alf.monitor._bin_count", return_value=100),
        patch(
            "py_alf.monitor._get_jobs_resources_bulk",
            return_value={"42": {"max_rss": "8M", "cpu_eff": "94%"}},
        ),
    ):
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
    assert app._row_data[0]["peak_mem"] == "8M"
    assert app._row_data[0]["cpu_eff"] == "94%"


async def test_monitor_peak_resources_dash_for_running_job(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    with (
        patch("py_alf.monitor.get_job_id", return_value="43"),
        patch(
            "py_alf.monitor._get_slurm_status_bulk",
            return_value={
                "43": {"status": "RUNNING", "runtime": "01:00:00", "nodelist": "node01"}
            },
        ),
        patch("py_alf.monitor._bin_count", return_value=50),
        patch("py_alf.monitor._get_jobs_resources_bulk", return_value={}),
    ):
        app = _monitor([sim])
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
    assert app._row_data[0]["peak_mem"] == "-"
    assert app._row_data[0]["cpu_eff"] == "-"


async def test_monitor_nbin_target_from_sim_dict(tmp_path, no_slurm):
    """When sim_dict contains 'NBin', the row carries a numeric nbin_target."""
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.sim_dict = {"U": 4.0, "NBin": 200}
    app = _monitor([sim])
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert app._row_data[0]["nbin_target"] == 200
