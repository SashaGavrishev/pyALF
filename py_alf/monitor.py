"""
TUI for monitoring ALF simulations on a SLURM cluster.

Usage::

    from py_alf.monitor import SimulationMonitor
    monitor = SimulationMonitor(sims, cluster_submitter=cs)
    monitor.run()

Requires the ``textual`` package (install with ``pip install 'pyALF[tui]'``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from rich.text import Text
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, ScrollableContainer
    from textual.screen import ModalScreen
    from textual.widgets import Button, DataTable, Footer, Header, Label, Static
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The simulation monitor requires the 'textual' package. "
        "Install it with: pip install 'pyALF[tui]'"
    ) from exc

from .cluster_submission import (
    ClusterSubmitter,
    _bin_count,
    _find_job_log,
    _get_slurm_status_bulk,
    cancel_cluster_job,
    get_job_id,
)
from .simulation import Simulation

_STATUS_COLORS: Dict[str, str] = {
    "RUNNING": "bold green",
    "PENDING": "bold yellow",
    "FAILED": "bold red",
    "CANCELLED": "bold red",
    "TIMEOUT": "bold red",
    "CRASHED": "bold red",
    "COMPLETED": "blue",
    "INACTIVE": "dim",
    "UNKNOWN": "dim",
    "ERROR": "bold red",
}


def _styled(status: str) -> Text:
    return Text(status, style=_STATUS_COLORS.get(status, ""))


# ---------------------------------------------------------------------------
# Modal screens
# ---------------------------------------------------------------------------


class LogViewerScreen(ModalScreen):
    """Full-window modal for viewing a job's log file."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    def __init__(self, title: str, content: str) -> None:
        super().__init__()
        self._log_title = title
        self._content = content

    def compose(self) -> ComposeResult:
        with Container(id="log-dialog"):
            yield Label(self._log_title, id="log-title")
            with ScrollableContainer(id="log-scroll"):
                yield Static(self._content, id="log-body")
            yield Button("Close  [esc]", variant="primary", id="log-close")

    @on(Button.Pressed, "#log-close")
    def close(self) -> None:
        self.dismiss()


class ConfirmScreen(ModalScreen[bool]):
    """Confirmation dialog that resolves to True when the user clicks Yes."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm-dialog"):
            yield Label(self._message, id="confirm-msg")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", variant="error", id="confirm-yes")
                yield Button("No", variant="default", id="confirm-no")

    @on(Button.Pressed, "#confirm-yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def _no(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class SimulationMonitor(App):
    """
    Interactive TUI for monitoring ALF simulations on a SLURM cluster.

    Parameters
    ----------
    sims : list of Simulation
        Simulations to monitor.
    cluster_submitter : ClusterSubmitter, optional
        Required for the resubmit action. If omitted, resubmission is disabled.
    submit_dir : str or Path, optional
        Directory where submitit writes logs. Used for log discovery via the
        ``{jobid}_0_log.out`` naming convention.  Falls back to
        ``cluster_submitter.submit_dir`` when not set.
    refresh_interval : float
        Seconds between automatic SLURM status polls (default: 30).
    param_keys : list of str, optional
        Keys from ``sim.sim_dict`` to display as extra parameter columns.
    param_headers : list of str, optional
        Column header labels for ``param_keys``. Defaults to the key names.
    """

    TITLE = "pyALF Simulation Monitor"

    CSS = """
    Screen {
        background: $background;
    }
    DataTable {
        height: 1fr;
    }
    /* --- Log viewer modal --- */
    LogViewerScreen {
        align: center middle;
    }
    #log-dialog {
        background: $panel;
        border: solid $primary;
        padding: 1 2;
        width: 92%;
        height: 92%;
    }
    #log-title {
        text-align: center;
        text-style: bold;
        background: $primary-darken-1;
        color: $text;
        padding: 0 1;
        margin-bottom: 1;
    }
    #log-scroll {
        height: 1fr;
        border: solid $primary-darken-3;
    }
    #log-body {
        padding: 0 1;
    }
    #log-close {
        margin-top: 1;
        width: 100%;
    }
    /* --- Confirmation modal --- */
    ConfirmScreen {
        align: center middle;
    }
    #confirm-dialog {
        background: $panel;
        border: solid $warning;
        padding: 2 4;
        width: 64;
        height: auto;
    }
    #confirm-msg {
        text-align: center;
        margin-bottom: 2;
    }
    #confirm-buttons {
        align: center middle;
        height: 3;
    }
    #confirm-buttons Button {
        margin: 0 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("l", "view_logs", "Logs", show=True),
        Binding("c", "cancel_job", "Cancel Job", show=True),
        Binding("a", "cancel_array", "Cancel Array", show=True),
        Binding("r", "resubmit", "Resubmit", show=True),
        Binding("f5", "refresh_data", "Refresh", show=True),
    ]

    def __init__(
        self,
        sims: List[Simulation],
        cluster_submitter: Optional[ClusterSubmitter] = None,
        submit_dir: Optional[Union[str, Path]] = None,
        refresh_interval: float = 30.0,
        param_keys: Optional[List[str]] = None,
        param_headers: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._sims = list(sims)
        self._cs = cluster_submitter
        if submit_dir is not None:
            self._submit_dir: Optional[Path] = Path(submit_dir)
        elif cluster_submitter is not None:
            self._submit_dir = cluster_submitter.submit_dir
        else:
            self._submit_dir = None
        self._refresh_interval = refresh_interval
        self._param_keys: List[str] = list(param_keys or [])
        self._param_headers: List[str] = (
            list(param_headers) if param_headers else list(self._param_keys)
        )
        self._row_data: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="sim-table", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#sim-table", DataTable)
        table.add_column("#", key="idx")
        table.add_column("Hamiltonian", key="ham")
        for key, hdr in zip(self._param_keys, self._param_headers):
            table.add_column(hdr, key=key)
        table.add_column("n_omp", key="n_omp")
        table.add_column("n_mpi", key="n_mpi")
        if self._cs is not None:
            table.add_column("partition", key="partition")
            table.add_column("mem", key="mem")
        table.add_column("N_bins", key="n_bins")
        table.add_column("JobID", key="jobid")
        table.add_column("Status", key="status")
        table.add_column("Elapsed", key="elapsed")

        self._trigger_refresh()
        self.set_interval(self._refresh_interval, self._trigger_refresh)

    # ------------------------------------------------------------------
    # Data fetching (background thread)
    # ------------------------------------------------------------------

    def _trigger_refresh(self) -> None:
        self.sub_title = "Refreshing…"
        self._fetch_and_update()

    @work(thread=True)
    def _fetch_and_update(self) -> None:
        jobid_map: Dict[str, str] = {}
        for sim in self._sims:
            jid = get_job_id(sim)
            if jid:
                jobid_map[sim.sim_dir] = jid

        statuses = (
            _get_slurm_status_bulk(list(set(jobid_map.values())))
            if jobid_map
            else {}
        )

        rows: List[Dict[str, Any]] = []
        for idx, sim in enumerate(self._sims):
            jobid = jobid_map.get(sim.sim_dir)
            if jobid:
                se = statuses.get(jobid, {"status": "UNKNOWN", "runtime": None})
                status = se.get("status", "UNKNOWN")
                runtime = se.get("runtime")
            else:
                running_file = Path(sim.sim_dir) / "RUNNING"
                status = "CRASHED" if running_file.exists() else "INACTIVE"
                runtime = None

            n_bins = _bin_count(sim, refresh=(status == "RUNNING"))

            sim_dict = sim.sim_dict
            if isinstance(sim_dict, list):
                sim_dict = sim_dict[0] if sim_dict else {}

            row: Dict[str, Any] = {
                "idx": idx,
                "ham": sim.ham_name,
                "n_omp": sim.n_omp,
                "n_mpi": sim.n_mpi if getattr(sim, "mpi", False) else 1,
                "n_bins": n_bins,
                "jobid": jobid or "-",
                "status": status,
                "elapsed": runtime or "-",
            }
            for key in self._param_keys:
                row[key] = str(sim_dict.get(key, "-"))
            if self._cs is not None:
                row["partition"] = self._cs.slurm_partition
                row["mem"] = self._cs.slurm_mem
            rows.append(row)

        self.call_from_thread(self._apply_rows, rows)

    def _apply_rows(self, rows: List[Dict[str, Any]]) -> None:
        self._row_data = rows
        table = self.query_one("#sim-table", DataTable)
        saved_cursor = table.cursor_row

        table.clear()
        for row in rows:
            values: List[Any] = [row["idx"], row["ham"]]
            for key in self._param_keys:
                values.append(row.get(key, "-"))
            values.extend([row["n_omp"], row["n_mpi"]])
            if self._cs is not None:
                values.extend([row["partition"], row["mem"]])
            values.extend([row["n_bins"], row["jobid"]])
            values.append(_styled(row["status"]))
            values.append(row["elapsed"])
            table.add_row(*values, key=str(row["idx"]))

        if rows:
            table.move_cursor(row=min(saved_cursor, len(rows) - 1))

        self.sub_title = (
            f"{len(rows)} simulation(s) | auto-refresh every {self._refresh_interval:.0f}s"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _selected_sim(self) -> Optional[Simulation]:
        table = self.query_one("#sim-table", DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._sims):
            return self._sims[row]
        return None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_view_logs(self) -> None:
        sim = self._selected_sim()
        if sim is None:
            self.notify("No simulation selected.", severity="warning")
            return
        self._load_and_show_log(sim)

    @work(thread=True)
    def _load_and_show_log(self, sim: Simulation) -> None:
        jobid = get_job_id(sim)
        if not jobid:
            self.call_from_thread(
                self.notify, "No job ID for this simulation.", severity="warning"
            )
            return
        log_path = _find_job_log(
            jobid, root_dir=[sim.sim_dir, "."], submit_dir=self._submit_dir
        )
        if log_path is None:
            self.call_from_thread(
                self.notify,
                f"Cannot locate log for job {jobid}.",
                severity="error",
            )
            return
        try:
            lines = log_path.read_text(errors="replace").splitlines()
            if len(lines) > 2000:
                header = f"[... {len(lines)} lines total; showing last 2000 ...]\n"
                content = header + "\n".join(lines[-2000:])
            else:
                content = "\n".join(lines)
            title = f"Log: job {jobid}  |  {Path(sim.sim_dir).name}"
            self.call_from_thread(self._show_log_screen, title, content)
        except Exception as exc:
            self.call_from_thread(
                self.notify, f"Error reading log: {exc}", severity="error"
            )

    def _show_log_screen(self, title: str, content: str) -> None:
        self.push_screen(LogViewerScreen(title, content))

    def action_cancel_job(self) -> None:
        sim = self._selected_sim()
        if sim is None:
            self.notify("No simulation selected.", severity="warning")
            return
        jobid = get_job_id(sim)
        if not jobid:
            self.notify("No job ID for this simulation.", severity="warning")
            return

        def _on_confirm(confirmed: Optional[bool]) -> None:
            if not confirmed:
                return
            ok = cancel_cluster_job(sim)
            if ok:
                self.notify(f"Cancelled job {jobid}.")
            else:
                self.notify(f"Failed to cancel job {jobid}.", severity="error")
            self._trigger_refresh()

        self.push_screen(ConfirmScreen(f"Cancel job {jobid}?"), _on_confirm)

    def action_cancel_array(self) -> None:
        sim = self._selected_sim()
        if sim is None:
            self.notify("No simulation selected.", severity="warning")
            return
        jobid = get_job_id(sim)
        if not jobid:
            self.notify("No job ID for this simulation.", severity="warning")
            return
        base_id = jobid.split("_")[0]

        def _on_confirm(confirmed: Optional[bool]) -> None:
            if not confirmed:
                return
            try:
                result = subprocess.run(
                    ["scancel", base_id], capture_output=True, text=True
                )
                if result.returncode == 0:
                    self.notify(f"Cancelled array {base_id}.")
                else:
                    self.notify(
                        f"scancel failed: {result.stderr.strip()}", severity="error"
                    )
            except Exception as exc:
                self.notify(f"Error: {exc}", severity="error")
            self._trigger_refresh()

        self.push_screen(
            ConfirmScreen(
                f"Cancel entire SLURM array {base_id}?\n"
                "(All tasks sharing this array ID will be cancelled.)"
            ),
            _on_confirm,
        )

    def action_resubmit(self) -> None:
        if self._cs is None:
            self.notify(
                "No ClusterSubmitter provided — resubmission is unavailable.",
                severity="error",
            )
            return
        sim = self._selected_sim()
        if sim is None:
            self.notify("No simulation selected.", severity="warning")
            return
        sim_name = Path(sim.sim_dir).name

        def _on_confirm(confirmed: Optional[bool]) -> None:
            if not confirmed:
                return
            try:
                self._cs.submit(sim)
                self.notify(f"Resubmitted {sim_name}.")
            except Exception as exc:
                self.notify(f"Resubmission failed: {exc}", severity="error")
            self._trigger_refresh()

        self.push_screen(
            ConfirmScreen(f"Force resubmit {sim_name}?"), _on_confirm
        )

    def action_refresh_data(self) -> None:
        self._trigger_refresh()
