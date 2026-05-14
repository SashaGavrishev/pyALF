"""
TUI for monitoring ALF simulations on a SLURM cluster.

Usage::

    from py_alf.monitor import SimulationMonitor
    monitor = SimulationMonitor(sims, cluster_submitter=cs)
    monitor.run()
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, DataTable, Footer, Label, Static

from .cluster_submission import (
    _TERMINAL_STATES,
    ClusterSubmitter,
    _bin_count,
    _exec_alf_binary,
    _find_job_log,
    _get_jobs_resources_bulk,
    _get_slurm_status_bulk,
    _is_submitit_timeout,
    cancel_cluster_job,
    get_job_id,
)
from .simulation import Simulation

_STATUS_COLORS: dict[str, str] = {
    "RUNNING": "bold green",
    "PENDING": "bold yellow",
    "COMPLETING": "bold yellow",
    "FAILED": "bold red",
    "CANCELLED": "bold red",
    "TIMEOUT": "bold red",
    "OUT_OF_MEMORY": "bold red",
    "NODE_FAIL": "bold red",
    "PREEMPTED": "bold yellow",
    "CRASHED": "bold red",
    "COMPLETED": "blue",
    "INACTIVE": "dim",
    "UNKNOWN": "dim",
    "ERROR": "bold red",
}


def _styled(status: str) -> Text:
    return Text(status, style=_STATUS_COLORS.get(status, ""))


def _parse_elapsed_hours(runtime: str) -> float | None:
    """Parse a SLURM runtime string (``[D-]HH:MM:SS``) to fractional hours."""
    if not runtime:
        return None
    try:
        days = 0
        t = runtime
        if "-" in runtime:
            d, t = runtime.split("-", 1)
            days = int(d)
        parts = t.split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, s = 0, int(parts[0]), int(parts[1])
        else:
            return None
        return days * 24 + h + m / 60 + s / 3600
    except (ValueError, IndexError):
        return None


def _bins_cell(n_bins: int, nbin_target: int | None) -> Any:
    """Return a Rich Text progress bar for the N_bins column."""
    if nbin_target is None or nbin_target <= 0:
        return str(n_bins)
    bar_width = 8
    filled = min(bar_width, round(bar_width * n_bins / nbin_target))
    bar = "█" * filled + "░" * (bar_width - filled)
    # Pad n_bins to the same digit width as nbin_target so bars stay aligned.
    w = len(str(nbin_target))
    t = Text(f"{n_bins:{w}d}/{nbin_target} ")
    t.append(bar, style="green" if n_bins >= nbin_target else "yellow")
    return t


def _eta_cell(cpu_max_h: float, elapsed_h: float | None) -> Any:
    bar_width = 8
    if elapsed_h is None:
        return Text("-")
    fraction = min(1.0, elapsed_h / cpu_max_h) if cpu_max_h > 0 else 1.0
    filled = round(bar_width * fraction)
    bar = "█" * filled + "░" * (bar_width - filled)
    remaining_h = cpu_max_h - elapsed_h
    if remaining_h <= 0:
        t = Text("overtime ")
        t.append(bar, style="red")
    else:
        total_m = int(remaining_h * 60)
        h, m = divmod(total_m, 60)
        eta_str = f"{h}h{m:02d}m" if h else f"{m}m"
        t = Text(f"{eta_str} ")
        t.append(bar, style="green" if fraction < 0.9 else "yellow")
    return t


_MONO_THEME = Theme(
    name="monitor-mono",
    primary="ansi_default",
    secondary="ansi_default",
    warning="ansi_yellow",
    error="ansi_red",
    success="ansi_default",
    accent="ansi_default",
    foreground="ansi_default",
    background="ansi_default",
    surface="ansi_default",
    panel="ansi_default",
    boost="ansi_default",
    dark=True,
    ansi=True,
    variables={
        "ansi-background": "ansi_default",
        "ansi-foreground": "ansi_default",
        "border-blurred": "ansi_default",
        "block-cursor-foreground": "ansi_default",
        "block-cursor-background": "ansi_default",
        "input-cursor-background": "ansi_default",
        "input-cursor-foreground": "ansi_default",
        "input-cursor-text-style": "reverse",
        "input-selection-background": "ansi_default",
        "input-selection-foreground": "ansi_default",
        "screen-selection-background": "ansi_default",
        "screen-selection-foreground": "ansi_default",
    },
)


# ---------------------------------------------------------------------------
# Session utilities (usable from Jupyter without launching the TUI)
# ---------------------------------------------------------------------------


def list_sessions(submit_dir: str | Path) -> list[Path]:
    """Return session manifest paths inside *submit_dir*, newest first.

    Parameters
    ----------
    submit_dir : str or Path
        Directory that contains ``session_YYYYMMDD_HHMMSS.json`` files
        (the ``.alfmonitor`` or custom submit directory passed to
        ``ClusterSubmitter``).

    Returns
    -------
    list of Path
        Matching session files sorted by modification time, most recent first.
    """
    return sorted(
        Path(submit_dir).glob("session_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_session_sims(session_path: str | Path) -> list[_SessionEntry]:
    """Load simulation objects from a session manifest written by SubmissionReview.

    Reconstructed objects expose the same attributes used by analysis
    workflows: ``sim_dir``, ``ham_name``, ``sim_dict``, ``n_omp``,
    ``n_mpi``, ``mpi``, and ``job_id``.

    Parameters
    ----------
    session_path : str or Path
        Path to a ``session_YYYYMMDD_HHMMSS.json`` file.

    Returns
    -------
    list of _SessionEntry

    Example
    -------
    In a Jupyter notebook::

        from py_alf.monitor import list_sessions, load_session_sims

        sessions = list_sessions("/scratch/user/.alfmonitor")
        print(sessions)            # newest first
        sims = load_session_sims(sessions[0])
        # use sim.sim_dir with py_alf.analysis() etc.
    """
    data = json.loads(Path(session_path).read_text())
    return [_SessionEntry(**entry) for entry in data["entries"]]


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
                yield Static(self._content, id="log-body", markup=False)
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

    def on_mount(self) -> None:
        self.query_one("#confirm-no", Button).focus()

    @on(Button.Pressed, "#confirm-yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def _no(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# Session manifest reconstruction
# ---------------------------------------------------------------------------


class _SessionEntry:
    """Minimal sim-like object reconstructed from a session manifest JSON."""

    __slots__ = (
        "sim_dir",
        "ham_name",
        "n_omp",
        "n_mpi",
        "mpi",
        "sim_dict",
        "config",
        "alf_dir",
        "job_id",
        "mpiexec",
        "mpiexec_args",
    )

    def __init__(
        self,
        sim_dir,
        ham_name,
        n_omp,
        n_mpi,
        mpi,
        sim_dict,
        job_id=None,
        mpiexec="mpiexec",
        mpiexec_args=None,
        config="",
        alf_dir=".",
        **_extra,
    ):
        self.sim_dir = sim_dir
        self.ham_name = ham_name
        self.n_omp = n_omp
        self.n_mpi = n_mpi
        self.mpi = mpi
        self.sim_dict = sim_dict
        self.config = config
        self.alf_dir = alf_dir
        self.job_id = job_id
        self.mpiexec = mpiexec
        self.mpiexec_args = mpiexec_args or []

    def run(
        self,
        copy_bin: bool = False,
        only_prep: bool = False,
        bin_in_sim_dir: bool = False,
    ) -> None:
        """Run the simulation from the already-prepared sim_dir."""
        if only_prep:
            return
        _exec_alf_binary(
            self.sim_dir,
            self.n_omp,
            self.n_mpi,
            self.mpi,
            self.mpiexec,
            self.mpiexec_args,
            config=self.config,
            alf_dir=self.alf_dir,
        )


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

    TITLE = "pyALF · Simulation Monitor"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen {
        background: transparent;
        color: ansi_default;
        scrollbar-size: 0 0;
    }

    ScrollableContainer { scrollbar-size: 0 0; }
    DataTable { scrollbar-size: 0 0; }

    #title-bar {
        height: 1;
        background: ansi_bright_black;
        color: ansi_default;
        text-style: bold;
        padding: 0 1;
    }

    #status-bar {
        height: 1;
        color: ansi_default;
        padding: 0 1;
    }

    Horizontal, Container, ScrollableContainer, Static, Label { background: transparent; }
    DataTable { height: 1fr; background: transparent; }
    DataTable > .datatable--header { color: ansi_default; background: transparent; }
    DataTable > .datatable--cursor { background: ansi_bright_black; color: ansi_default; text-style: bold; }
    DataTable > .datatable--hover { background: ansi_bright_black; color: ansi_default; }
    DataTable > .datatable--even-row { background: transparent; }
    DataTable > .datatable--odd-row { background: transparent; }

    Footer { background: transparent; color: ansi_default; }
    FooterKey .footer-key--key { background: ansi_bright_black; color: ansi_default; padding: 0 1; }
    FooterKey .footer-key--description { padding-left: 1; padding-right: 2; }

    Button { border: blank; color: ansi_default; background: transparent; }
    Button.-primary { background: ansi_bright_black; color: ansi_default; }
    Button.-error { background: ansi_red; color: ansi_default; }
    Button.-error.-active { background: ansi_red; color: ansi_default; text-style: bold; }
    Button.-error:hover { background: ansi_red; color: ansi_default; text-style: bold; }
    Button.-active { background: ansi_bright_black; color: ansi_default; text-style: bold; }
    Button.-primary.-active { background: ansi_bright_black; color: ansi_default; text-style: bold; }
    Button:hover { color: ansi_default; background: transparent; }
    Button.-primary:hover { color: ansi_default; background: ansi_bright_black; text-style: bold; }

    /* --- Log viewer modal --- */
    LogViewerScreen { align: center middle; }
    #log-dialog {
        background: ansi_default;
        border: solid ansi_default;
        padding: 1 2;
        width: 92%;
        height: 92%;
    }
    #log-title {
        text-style: bold;
        background: ansi_bright_black;
        color: ansi_default;
        padding: 0 1;
        margin-bottom: 1;
    }
    #log-scroll {
        height: 1fr;
        border: solid ansi_default;
        scrollbar-size: 0 0;
    }
    #log-body { padding: 0 1; }
    #log-close { margin-top: 1; width: 100%; }

    /* --- Confirmation modal --- */
    ConfirmScreen { align: center middle; }
    #confirm-dialog {
        background: ansi_default;
        border: solid ansi_default;
        padding: 2 4;
        width: 64;
        height: auto;
    }
    #confirm-msg {
        text-align: center;
        margin-bottom: 2;
        color: ansi_default;
    }
    #confirm-buttons { align: center middle; height: 3; }
    #confirm-buttons Button { margin: 0 2; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("l", "view_logs", "Logs", show=True),
        Binding("c", "cancel_job", "Cancel Job", show=True),
        Binding("a", "cancel_array", "Cancel Array", show=True),
        Binding("r", "resubmit", "Resubmit", show=True),
        Binding("f", "refresh_data", "Refresh", show=True),
        Binding("left", "pan_left", show=False),
        Binding("right", "pan_right", show=False),
    ]

    def __init__(
        self,
        sims: list[Simulation],
        cluster_submitter: ClusterSubmitter | None = None,
        submit_dir: str | Path | None = None,
        refresh_interval: float = 30.0,
        param_keys: list[str] | None = None,
        param_headers: list[str] | None = None,
        submitted_at: str | None = None,
    ) -> None:
        super().__init__()
        self._sims = list(sims)
        self._cs = cluster_submitter
        if submit_dir is not None:
            self._submit_dir: Path | None = Path(submit_dir)
        elif cluster_submitter is not None:
            self._submit_dir = cluster_submitter.submit_dir
        else:
            self._submit_dir = None
        self._refresh_interval = refresh_interval
        self._param_keys: list[str] = list(param_keys or [])
        self._param_headers: list[str] = (
            list(param_headers) if param_headers else list(self._param_keys)
        )
        self._submitted_at: str | None = submitted_at
        self._row_data: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Session manifest
    # ------------------------------------------------------------------

    @classmethod
    def from_session(
        cls,
        path: str | Path,
        cluster_submitter: ClusterSubmitter | None = None,
        refresh_interval: float = 30.0,
        param_keys: list[str] | None = None,
        param_headers: list[str] | None = None,
    ) -> SimulationMonitor:
        """Reconstruct a monitor from a session manifest written by SubmissionReview.

        Parameters
        ----------
        path:
            Path to a ``session_YYYYMMDD_HHMMSS.json`` file produced by
            ``SubmissionReview`` after a successful SLURM submission.
        cluster_submitter:
            Optional ``ClusterSubmitter`` for partition display and resubmission.
            If omitted, the ``submit_dir`` stored in the manifest is used for
            log discovery.
        """
        data = json.loads(Path(path).read_text())
        sims = [_SessionEntry(**entry) for entry in data["entries"]]
        if cluster_submitter is None and "cluster_submitter" in data:
            cs_data = dict(data["cluster_submitter"])
            slurm_kwargs = cs_data.pop("slurm_kwargs", {})
            with contextlib.suppress(Exception):
                cluster_submitter = ClusterSubmitter(**cs_data, **slurm_kwargs)
        return cls(
            sims,
            cluster_submitter=cluster_submitter,
            submit_dir=data.get("cluster_submitter", {}).get("submit_dir"),
            refresh_interval=refresh_interval,
            param_keys=param_keys,
            param_headers=param_headers,
            submitted_at=data.get("submitted_at"),
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(f"  {self.TITLE}", id="title-bar")
        yield Static("", id="status-bar")
        yield DataTable(id="sim-table", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(_MONO_THEME)
        self.theme = "monitor-mono"
        self._any_mpi: bool = any(getattr(s, "mpi", False) for s in self._sims)
        table = self.query_one("#sim-table", DataTable)
        table.add_column("#", key="idx")
        table.add_column("Hamiltonian", key="ham")
        for key, hdr in zip(self._param_keys, self._param_headers):
            table.add_column(hdr, key=key)
        table.add_column("n_omp", key="n_omp")
        if self._any_mpi:
            table.add_column("n_mpi", key="n_mpi")
        if self._cs is not None and self._cs.executor == "slurm":
            table.add_column("partition", key="partition")
            table.add_column("mem", key="mem")
        table.add_column("N_bins", key="n_bins")
        table.add_column("Array", key="array_id")
        table.add_column("JobID", key="jobid")
        table.add_column("Status", key="status")
        table.add_column("Node", key="node")
        table.add_column("Elapsed", key="elapsed")
        table.add_column("ETA", key="eta")
        table.add_column("Peak Mem", key="peak_mem")
        table.add_column("CPU Eff", key="cpu_eff")

        self._trigger_refresh()
        self.set_interval(self._refresh_interval, self._trigger_refresh)
        self.call_after_refresh(self._remove_ansi_scrollbar_class)

    def watch_theme(self, _theme: str) -> None:
        self.call_after_refresh(self._remove_ansi_scrollbar_class)

    def _remove_ansi_scrollbar_class(self) -> None:
        for widget in self.query(".-ansi-scrollbar"):
            widget.remove_class("-ansi-scrollbar")

    # ------------------------------------------------------------------
    # Data fetching (background thread)
    # ------------------------------------------------------------------

    def _trigger_refresh(self) -> None:
        self.query_one("#status-bar", Static).update("[dim]Refreshing…[/dim]")
        self._fetch_and_update()

    @work(thread=True)
    def _fetch_and_update(self) -> None:
        # Build a per-index job ID list.  Prefer the ID stored in the session
        # manifest (sim.job_id on _SessionEntry objects) so that a later
        # re-submission that overwrites jobid.txt does not corrupt this session's
        # view.  Fall back to get_job_id() for live Simulation objects.
        jobid_list: list[str | None] = [
            getattr(sim, "job_id", None) or get_job_id(sim) for sim in self._sims
        ]

        all_jids = list({jid for jid in jobid_list if jid})
        statuses = _get_slurm_status_bulk(all_jids) if all_jids else {}

        terminal_jids = [
            jid
            for jid in all_jids
            if statuses.get(jid, {}).get("status") in _TERMINAL_STATES
        ]
        resources = _get_jobs_resources_bulk(terminal_jids) if terminal_jids else {}

        # Compute partition once from the first sim's CPU_MAX — matches the
        # behaviour of ClusterSubmitter.submit(), which uses filtered_sims[0]
        # for the whole array's SLURM parameters.
        _shared_partition: str | None = None
        _shared_mem: str | None = None
        if self._cs is not None and self._cs.executor == "slurm" and self._sims:
            _first_sd = self._sims[0].sim_dict
            if isinstance(_first_sd, list):
                _first_sd = _first_sd[0] if _first_sd else {}
            _timeout_h_0 = max(1, int(_first_sd.get("CPU_MAX", 24)))
            try:
                _shared_partition = self._cs._select_partition(_timeout_h_0)
            except ValueError:
                _shared_partition = "???"
            _shared_mem = self._cs.slurm_mem

        rows: list[dict[str, Any]] = []
        for idx, sim in enumerate(self._sims):
            jobid = jobid_list[idx]
            if jobid:
                se = statuses.get(jobid, {"status": "UNKNOWN", "runtime": None})
                status = se.get("status", "UNKNOWN")
                runtime = se.get("runtime")
                nodelist = se.get("nodelist")
                if (
                    status in {"FAILED", "COMPLETED"}
                    and self._submit_dir is not None
                    and _is_submitit_timeout(jobid, self._submit_dir, status=status)
                ):
                    status = "TIMEOUT"
            else:
                running_file = Path(sim.sim_dir) / "RUNNING"
                status = "CRASHED" if running_file.exists() else "INACTIVE"
                runtime = None
                nodelist = None

            n_bins = _bin_count(sim, refresh=(status in {"RUNNING"} | _TERMINAL_STATES))
            has_checkpoint = any(Path(sim.sim_dir).glob("confin_*"))

            sim_dict = sim.sim_dict
            if isinstance(sim_dict, list):
                sim_dict = sim_dict[0] if sim_dict else {}

            array_id = jobid.split("_")[0] if jobid and "_" in jobid else jobid or "-"

            if status == "RUNNING" and runtime:
                _cpu_max_raw = (
                    sim_dict.get("CPU_MAX") if isinstance(sim_dict, dict) else None
                )
                _cpu_max = float(_cpu_max_raw) if _cpu_max_raw is not None else None
                _elapsed_h = _parse_elapsed_hours(runtime)
            else:
                _cpu_max = None
                _elapsed_h = None

            nbin_target = sim_dict.get("NBin") or sim_dict.get("Nbin")

            res: dict = {}
            if jobid and status in _TERMINAL_STATES:
                res = resources.get(jobid, {})
                res_file = Path(sim.sim_dir) / "peak_resources.json"
                if res.get("max_rss") or res.get("cpu_eff"):
                    with contextlib.suppress(OSError):
                        res_file.write_text(json.dumps(res))
                elif res_file.exists():
                    with contextlib.suppress(Exception):
                        res = json.loads(res_file.read_text())

            row: dict[str, Any] = {
                "idx": idx,
                "ham": sim.ham_name,
                "n_omp": sim.n_omp,
                "n_mpi": sim.n_mpi if getattr(sim, "mpi", False) else 1,
                "n_bins": n_bins,
                "nbin_target": int(nbin_target) if nbin_target is not None else None,
                "array_id": array_id,
                "jobid": jobid or "-",
                "status": status,
                "has_checkpoint": has_checkpoint,
                "node": nodelist or "-",
                "elapsed": runtime or "-",
                "cpu_max": _cpu_max,
                "elapsed_h": _elapsed_h,
                "peak_mem": res.get("max_rss") or "-",
                "cpu_eff": res.get("cpu_eff") or "-",
            }
            for key in self._param_keys:
                row[key] = str(sim_dict.get(key, "-"))
            if self._cs is not None and self._cs.executor == "slurm":
                row["partition"] = _shared_partition
                row["mem"] = _shared_mem
            rows.append(row)

        self.call_from_thread(self._apply_rows, rows)

    def _apply_rows(self, rows: list[dict[str, Any]]) -> None:
        self._row_data = rows
        table = self.query_one("#sim-table", DataTable)
        saved_cursor = table.cursor_row

        table.clear()
        array_ids: list[str] = list(
            dict.fromkeys(
                r["array_id"]
                for r in rows
                if r.get("array_id", "-") not in ("-", "") and "_" in r.get("jobid", "")
            )
        )

        for row in rows:
            values: list[Any] = [row["idx"], row["ham"]]
            for key in self._param_keys:
                values.append(row.get(key, "-"))
            values.append(row["n_omp"])
            if self._any_mpi:
                values.append(row["n_mpi"])
            if self._cs is not None and self._cs.executor == "slurm":
                values.extend([row["partition"], row["mem"]])
            _cpu_max = row.get("cpu_max")
            if _cpu_max is not None and _cpu_max > 0:
                _bins_val = _bins_cell(row["n_bins"], None)
                _eta_val = _eta_cell(_cpu_max, row.get("elapsed_h"))
            else:
                _nbin_target = (
                    None if row.get("has_checkpoint") else row.get("nbin_target")
                )
                _bins_val = _bins_cell(row["n_bins"], _nbin_target)
                _eta_val = Text("-")
            values.extend(
                [
                    _bins_val,
                    row["array_id"],
                    row["jobid"],
                ]
            )
            status_cell = _styled(row["status"])
            if row.get("has_checkpoint"):
                if row["status"] in ("RUNNING", "PENDING"):
                    status_cell.append(" ↺", style="bold green")
                else:
                    status_cell.append(" ↺", style="yellow")
            values.append(status_cell)
            values.append(row["node"])
            values.extend(
                [
                    row["elapsed"],
                    _eta_val,
                    row.get("peak_mem", "-"),
                    row.get("cpu_eff", "-"),
                ]
            )
            table.add_row(*values, key=str(row["idx"]))

        if rows:
            table.move_cursor(row=min(saved_cursor, len(rows) - 1))

        if array_ids:
            label = "Array" if len(array_ids) == 1 else "Arrays"
            array_suffix = f"  ·  {label} {', '.join(array_ids)}"
        else:
            array_suffix = ""
        self.query_one("#title-bar", Static).update(f"  {self.TITLE}{array_suffix}")

        at_str = f"  ·  submitted {self._submitted_at}" if self._submitted_at else ""
        self.query_one("#status-bar", Static).update(
            f"[dim]{len(rows)} simulation(s)  ·  auto-refresh every {self._refresh_interval:.0f}s{at_str}[/dim]"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _selected_sim(self) -> Simulation | None:
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

        def _on_confirm(confirmed: bool | None) -> None:
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

        def _on_confirm(confirmed: bool | None) -> None:
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

        has_checkpoint = any(Path(sim.sim_dir).glob("confin_*"))
        msg = f"Force resubmit {sim_name}?"
        if has_checkpoint:
            msg += "\n\n⚠  Checkpoint restart detected.\nALF will append to existing data.h5."

        def _on_confirm(confirmed: bool | None) -> None:
            if not confirmed:
                return
            try:
                self._cs.submit(sim, confirm_checkpoint=False)
                self.notify(f"Resubmitted {sim_name}.")
            except Exception as exc:
                self.notify(f"Resubmission failed: {exc}", severity="error")
            self._trigger_refresh()

        self.push_screen(ConfirmScreen(msg), _on_confirm)

    def action_pan_left(self) -> None:
        self.query_one("#sim-table", DataTable).scroll_left(animate=False)

    def action_pan_right(self) -> None:
        self.query_one("#sim-table", DataTable).scroll_right(animate=False)

    def action_refresh_data(self) -> None:
        self._trigger_refresh()
