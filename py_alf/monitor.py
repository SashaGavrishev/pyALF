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
import re as _re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, DataTable, Label, Static

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

_ANIM_FRAMES = ("· ", " ·")

# Upper bound on concurrent filesystem probes.  These threads spend their time
# blocked on a networked filesystem, so the useful width is set by I/O latency
# rather than by core count.  Below _MIN_FANOUT items the pool costs more to
# start than the I/O it would overlap.
_MAX_IO_WORKERS = 16
_MIN_FANOUT = 3


def _map_io(fn, items: list) -> list:
    """Apply *fn* to *items*, concurrently when there is enough work to justify it.

    The probes are independent and block on filesystem latency, so on a cluster
    filesystem the fan-out dominates: at ~5 ms per operation this turns a 32-row
    refresh from ~200 ms into ~15 ms.  On a local disk the pool is pure overhead,
    but well under a millisecond — far below the refresh interval either way.
    """
    if len(items) < _MIN_FANOUT:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(len(items), _MAX_IO_WORKERS)) as pool:
        return list(pool.map(fn, items))


_STATUS_COLORS: dict[str, str] = {
    "RUNNING": "",
    "PENDING": "dim",
    "COMPLETING": "dim",
    "FAILED": "red",
    "CANCELLED": "dim",
    "TIMEOUT": "red",
    "OUT_OF_MEMORY": "red",
    "NODE_FAIL": "red",
    "PREEMPTED": "dim",
    "CRASHED": "red",
    "COMPLETED": "",
    "INACTIVE": "dim",
    "UNKNOWN": "dim",
    "ERROR": "red",
}


# States whose bin count can still change and so must be re-checked on refresh.
# Terminal states are included so the last bins written before exit are picked
# up; _bin_count then freezes the result.
_REFRESHING_STATES: frozenset[str] = frozenset({"RUNNING"}) | _TERMINAL_STATES


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


def _bins_cell(n_bins: int, nbin_target: int | None, max_bins_w: int = 1) -> Any:
    """Return a Rich Text progress bar for the N_bins column."""
    if nbin_target is None or nbin_target <= 0:
        return str(n_bins)
    bar_width = 8
    filled = min(bar_width, round(bar_width * n_bins / nbin_target))
    bar = "[" + "/" * filled + "-" * (bar_width - filled) + "]"
    # Pad n_bins to the widest value across all rows so bars start at the same column.
    w = max(max_bins_w, len(str(n_bins)))
    t = Text(f"{n_bins:{w}d}/{nbin_target} ")
    t.append(bar, style="" if n_bins >= nbin_target else "dim")
    return t


def _eta_cell(cpu_max_h: float, elapsed_h: float | None) -> Any:
    bar_width = 8
    if elapsed_h is None:
        return Text("-")
    fraction = min(1.0, elapsed_h / cpu_max_h) if cpu_max_h > 0 else 1.0
    filled = round(bar_width * fraction)
    bar = "[" + "/" * filled + "-" * (bar_width - filled) + "]"
    remaining_h = cpu_max_h - elapsed_h
    if remaining_h <= 0:
        t = Text("overtime ")
        t.append(bar, style="red")
    else:
        total_m = int(remaining_h * 60)
        h, m = divmod(total_m, 60)
        eta_str = f"{h}h{m:02d}m" if h else f"{m}m"
        t = Text(f"{eta_str} ")
        t.append(bar, style="dim")
    return t


_MONO_THEME = Theme(
    name="monitor-mono",
    primary="ansi_default",
    secondary="ansi_default",
    warning="ansi_default",
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
# Shared TUI widgets
# ---------------------------------------------------------------------------

_SQUARE_LEAD_RE = _re.compile(r"^\[dim\]□ |^■ ")
_DIM_CLOSE_RE = _re.compile(r"\[/dim\]$")


def _strip_square_markup(label: str) -> str:
    return _DIM_CLOSE_RE.sub("", _SQUARE_LEAD_RE.sub("", label))


class _ToggleButton(Button):
    """Button that shows '>' while hovered or focused instead of inverting."""

    _hovered: bool = False
    _focused: bool = False
    _base_label: str = ""
    _setting_hover: bool = False

    def __init__(self, label: str = "", **kwargs) -> None:
        super().__init__(label, **kwargs)
        self._base_label = label
        if _strip_square_markup(label) == label:
            self._base_label = f"■ {label}"
            self._setting_hover = True
            self.label = f"■ {label}"  # type: ignore[assignment]
            self._setting_hover = False

    def validate_label(self, label: object) -> object:
        if isinstance(label, str) and not self._setting_hover:
            self._base_label = label
        return super().validate_label(label)  # type: ignore[arg-type]

    def _sync_label(self) -> None:
        want_active = (self._hovered or self._focused) and not self.disabled
        stripped = _strip_square_markup(self._base_label)
        target = "> " + stripped if want_active else self._base_label
        self._setting_hover = True
        self.label = target  # type: ignore[assignment]
        self._setting_hover = False

    def _apply_hover(self) -> None:
        if self.disabled or self._hovered:
            return
        self._hovered = True
        self._sync_label()

    def _remove_hover(self) -> None:
        if not self._hovered:
            return
        self._hovered = False
        self._sync_label()

    def on_focus(self) -> None:
        if self._focused:
            return
        self._focused = True
        self._sync_label()

    def on_blur(self) -> None:
        if not self._focused:
            return
        self._focused = False
        self._sync_label()

    def on_enter(self) -> None:
        self._apply_hover()

    def on_mouse_enter(self) -> None:
        self._apply_hover()

    def on_leave(self) -> None:
        self._remove_hover()

    def on_mouse_leave(self) -> None:
        self._remove_hover()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.call_after_refresh(self._clear_after_click)

    def _clear_after_click(self) -> None:
        self._hovered = False
        self._focused = False
        self._sync_label()


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
            yield _ToggleButton("Close  [esc]", variant="primary", id="log-close")

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
                yield _ToggleButton("Yes", variant="error", id="confirm-yes")
                yield _ToggleButton("No", id="confirm-no")

    def on_mount(self) -> None:
        self.query_one("#confirm-no", _ToggleButton).focus()

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
# PARALLEL_PARAMS expansion
# ---------------------------------------------------------------------------


def _is_parallel_params(sim) -> bool:
    """True if *sim* is a PARALLEL_PARAMS job (one SLURM job, Temp_i/ per rank).

    Detected from the build config rather than ``sim_dict`` because a session
    manifest stores only the first parameter dict, so the list is not available
    after reattachment — but the ``PARALLEL_PARAMS`` config tag is preserved.
    """
    return "PARALLEL_PARAMS" in (getattr(sim, "config", "") or "")


def _probe_unit(task: tuple[Any, str, Path, bool, bool]) -> tuple[int, bool]:
    """Read the on-disk progress of one row: bin count and checkpoint presence.

    Split out of ``_fetch_and_update`` so the per-row filesystem work — a stat
    or h5py open of ``data.h5`` plus a ``confin_*`` scan — can be fanned out
    across a thread pool.  Called by module-global name so tests that patch
    ``py_alf.monitor._bin_count`` still take effect.
    """
    sim, status, eff_dir, refresh, force = task
    n_bins = _bin_count(
        sim,
        refresh=refresh,
        data_dir=str(eff_dir),
        final=(status in _TERMINAL_STATES),
        force=force,
    )
    has_checkpoint = any(eff_dir.glob("confin_*"))
    return n_bins, has_checkpoint


def _parallel_param_temp_dirs(sim_dir: str | Path) -> list[tuple[int, Path]]:
    """Return ``(realisation_index, Temp_i_path)`` pairs under *sim_dir*, sorted.

    ALF prepares these directories at submission time (``run(only_prep=True)``),
    so they exist even while the job is still PENDING.
    """
    found: list[tuple[int, Path]] = []
    for p in Path(sim_dir).glob("Temp_*"):
        m = _re.fullmatch(r"Temp_(\d+)", p.name)
        if m and p.is_dir():
            found.append((int(m.group(1)), p))
    return sorted(found, key=lambda t: t[0])


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

    TITLE = "pyALF │ Simulation Monitor"
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
        height: 2;
        color: ansi_default;
        padding: 0 1;
        border-bottom: solid ansi_default;
    }

    #status-bar {
        height: 1;
        color: ansi_default;
        padding: 0 1;
    }

    Horizontal, Container, ScrollableContainer, Static, Label { background: transparent; }
    DataTable { height: 1fr; background: transparent; }
    DataTable > .datatable--header { color: ansi_default; background: transparent; text-style: none; }
    DataTable > .datatable--cursor { background: transparent; text-style: none; }
    DataTable > .datatable--hover { background: transparent; }
    DataTable > .datatable--even-row { background: transparent; }
    DataTable > .datatable--odd-row { background: transparent; }

    #footer-keys { height: 1; content-align: left middle; padding: 0 1; }

    Button { border: none; height: 1; width: auto; color: ansi_default; background: transparent; text-style: none; content-align: left middle; text-align: left; }
    Button.-error { color: ansi_red; }
    Button.-error.-active { color: ansi_red; }
    Button.-active { color: ansi_default; }

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
    #log-close { margin-top: 1; }

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
    #confirm-buttons { align: center middle; height: 1; }
    #confirm-buttons Button { margin: 0 2; min-width: 9; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("l", "view_logs", "Logs", show=True),
        Binding("i", "view_info", "Info", show=True),
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
        self._dt_cursor_row: int = 0
        self._anim_tick: int = 0

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
        yield DataTable(
            id="sim-table",
            zebra_stripes=True,
            cursor_type="row",
            cursor_foreground_priority="renderable",
        )
        yield Static("", id="footer-keys", markup=False)

    def on_mount(self) -> None:
        self.register_theme(_MONO_THEME)
        self.theme = "monitor-mono"
        self._update_footer_keys()
        self._any_mpi: bool = any(getattr(s, "mpi", False) for s in self._sims)
        table = self.query_one("#sim-table", DataTable)
        table.add_column("", key="cur", width=3)
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
        self.set_interval(2.0, self._step_animation)
        self.call_after_refresh(self._remove_ansi_scrollbar_class)

    def watch_theme(self, _theme: str) -> None:
        self.call_after_refresh(self._remove_ansi_scrollbar_class)

    def _remove_ansi_scrollbar_class(self) -> None:
        for widget in self.query(".-ansi-scrollbar"):
            widget.remove_class("-ansi-scrollbar")

    # ------------------------------------------------------------------
    # Data fetching (background thread)
    # ------------------------------------------------------------------

    def _trigger_refresh(self, force: bool = False) -> None:
        self.query_one("#status-bar", Static).update("[dim]Refreshing…[/dim]")
        self._fetch_and_update(force)

    @work(thread=True)
    def _fetch_and_update(self, force: bool = False) -> None:
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

        # The Temp_* scans are independent per sim and each costs a readdir on a
        # networked filesystem, so run them concurrently before the row build.
        pp_refs = [i for i, s in enumerate(self._sims) if _is_parallel_params(s)]
        temp_dirs_by_sim: dict[int, list[tuple[int, Path]]] = dict(
            zip(
                pp_refs,
                _map_io(
                    lambda i: _parallel_param_temp_dirs(self._sims[i].sim_dir), pp_refs
                ),
            )
        )

        rows: list[dict[str, Any]] = []
        probe_tasks: list[tuple[Any, str, Path, bool, bool]] = []
        display_idx = 0
        for sim_ref, sim in enumerate(self._sims):
            jobid = jobid_list[sim_ref]
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

            sim_dict = sim.sim_dict
            if isinstance(sim_dict, list):
                sim_dict = sim_dict[0] if sim_dict else {}

            # A PARALLEL_PARAMS job is a *single* SLURM job whose ranks each write
            # an independent realisation into Temp_i/.  Expand it into one row per
            # realisation so per-config bin progress is visible.  These rows share
            # the one job id — this is NOT a SLURM array (see array_id below).
            if _is_parallel_params(sim):
                temp_dirs = temp_dirs_by_sim[sim_ref]
                units: list[tuple[int | None, Path]] = temp_dirs or [
                    (None, Path(sim.sim_dir))
                ]
            else:
                units = [(None, Path(sim.sim_dir))]

            _cpu_max_raw = (
                sim_dict.get("CPU_MAX") if isinstance(sim_dict, dict) else None
            )
            _cpu_max = float(_cpu_max_raw) if _cpu_max_raw is not None else None
            _elapsed_h = (
                _parse_elapsed_hours(runtime)
                if status == "RUNNING" and runtime
                else None
            )

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

            for realisation, eff_dir in units:
                is_pp_row = realisation is not None
                probe_tasks.append(
                    (sim, status, eff_dir, status in _REFRESHING_STATES, force)
                )

                # array_id drives the "Cancel Array" action and the array title
                # note; only true array tasks (jobid like 1234_5) qualify.  A PP
                # expansion is one job, so array_id stays "-" and the column shows
                # a distinct "PP[i]" tag instead.
                if jobid and "_" in jobid:
                    array_id = jobid.split("_")[0]
                elif is_pp_row:
                    array_id = "-"
                else:
                    array_id = jobid or "-"
                array_display = f"PP[{realisation}]" if is_pp_row else array_id

                if is_pp_row:
                    n_mpi = int(sim_dict.get("mpi_per_parameter_set", 1) or 1)
                else:
                    n_mpi = sim.n_mpi if getattr(sim, "mpi", False) else 1

                row: dict[str, Any] = {
                    "idx": display_idx,
                    "_sim_ref": sim_ref,
                    "realisation": realisation,
                    "effective_dir": str(eff_dir),
                    "is_pp": is_pp_row,
                    "ham": sim.ham_name,
                    "n_omp": sim.n_omp,
                    "n_mpi": n_mpi,
                    # Filled in by the probe pass below.
                    "n_bins": 0,
                    "nbin_target": int(nbin_target) if nbin_target is not None else None,
                    "array_id": array_id,
                    "array_display": array_display,
                    "jobid": jobid or "-",
                    "status": status,
                    "has_checkpoint": False,
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
                display_idx += 1

        # Each row's progress needs a stat (or h5py open) of data.h5 plus a
        # confin_* scan.  Note that h5py serialises actual reads on its global
        # lock, so the fan-out buys time on the stat and readdir calls; the
        # stat gate in _bin_count is what keeps the h5py opens rare.
        for row, (n_bins, has_checkpoint) in zip(rows, _map_io(_probe_unit, probe_tasks)):
            row["n_bins"] = n_bins
            row["has_checkpoint"] = has_checkpoint

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

        restore_cursor = min(saved_cursor, len(rows) - 1) if rows else 0

        # Pre-compute the width of the widest n_bins value across all rows so
        # the fraction text has a uniform prefix and bars start at the same column.
        max_bins_w = max((len(str(r["n_bins"])) for r in rows), default=1)

        for i, row in enumerate(rows):
            if i == restore_cursor:
                cur_indicator = ">"
            elif row["status"] == "RUNNING":
                cur_indicator = _ANIM_FRAMES[self._anim_tick % 2]
            else:
                cur_indicator = " "
            values: list[Any] = [cur_indicator, row["idx"], row["ham"]]
            for key in self._param_keys:
                values.append(row.get(key, "-"))
            values.append(row["n_omp"])
            if self._any_mpi:
                values.append(row["n_mpi"])
            if self._cs is not None and self._cs.executor == "slurm":
                values.extend([row["partition"], row["mem"]])
            _cpu_max = row.get("cpu_max")
            _nbin_target = None if row.get("has_checkpoint") else row.get("nbin_target")
            # CPU_MAX mode: wall-time budget set, no NBin target → show ETA bar.
            # NBin mode: bin target set (or no cpu_max) → show bins progress bar.
            cpu_max_mode = _cpu_max is not None and _cpu_max > 0 and not _nbin_target
            _bins_val = _bins_cell(row["n_bins"], None if cpu_max_mode else _nbin_target, max_bins_w)
            if cpu_max_mode:
                assert _cpu_max is not None  # guaranteed by cpu_max_mode
                if row["status"] == "COMPLETED":
                    _eta_val = Text("[" + "/" * 8 + "]")
                else:
                    _eta_val = _eta_cell(_cpu_max, row.get("elapsed_h"))
            else:
                _eta_val = Text("-")
            values.extend(
                [
                    _bins_val,
                    row.get("array_display", row["array_id"]),
                    row["jobid"],
                ]
            )
            status_cell = _styled(row["status"])
            if row.get("has_checkpoint"):
                if row["status"] in ("RUNNING", "PENDING"):
                    status_cell.append(" [R]")
                else:
                    status_cell.append(" [R]", style="dim")
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
            self._dt_cursor_row = restore_cursor
            table.move_cursor(row=restore_cursor)

        if array_ids:
            label = "Array" if len(array_ids) == 1 else "Arrays"
            array_suffix = f"  ·  {label} {', '.join(array_ids)}"
        else:
            array_suffix = ""
        # PARALLEL_PARAMS jobs are expanded into one row per Temp_i realisation
        # that all share a single job id — flag this so it is not mistaken for a
        # SLURM array (whose tasks have distinct {base}_{i} ids).
        n_pp_jobs = len({r["jobid"] for r in rows if r.get("is_pp")})
        if n_pp_jobs:
            pp_suffix = (
                f"  ·  PARALLEL_PARAMS expansion"
                f"{f' ×{n_pp_jobs}' if n_pp_jobs > 1 else ''} (rows per realisation, not a SLURM array)"
            )
        else:
            pp_suffix = ""
        job_name_suffix = (
            f"  ·  job: {self._cs.job_name}" if self._cs and self._cs.job_name else ""
        )
        self.query_one("#title-bar", Static).update(
            f"  {self.TITLE}{job_name_suffix}{array_suffix}{pp_suffix}"
        )

        n_jobs = len(self._sims)
        count_str = (
            f"{len(rows)} row(s) · {n_jobs} job(s)"
            if len(rows) != n_jobs
            else f"{len(rows)} simulation(s)"
        )
        at_str = f"  ·  submitted {self._submitted_at}" if self._submitted_at else ""
        self.query_one("#status-bar", Static).update(
            f"[dim]{count_str}  ·  auto-refresh every {self._refresh_interval:.0f}s{at_str}[/dim]"
        )
        self._update_footer_keys()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        try:
            table = self.query_one("#sim-table", DataTable)
        except Exception:
            return None  # DOM not ready yet
        row = table.cursor_row
        status = (
            self._row_data[row]["status"] if 0 <= row < len(self._row_data) else None
        )
        if action == "view_logs":
            jobid = (
                self._row_data[row].get("jobid", "-")
                if 0 <= row < len(self._row_data)
                else "-"
            )
            return True if jobid != "-" else None
        if action == "view_info":
            return True if status == "COMPLETED" else None
        if action == "cancel_job":
            return None if status in _TERMINAL_STATES else True
        if action == "cancel_array":
            if not (0 <= row < len(self._row_data)):
                return None
            selected_array_id = self._row_data[row].get("array_id", "-")
            if selected_array_id == "-":
                return None
            if "_" in self._row_data[row].get("jobid", ""):
                # Array job: enable if any task in this array is still active.
                any_active = any(
                    r.get("array_id") == selected_array_id
                    and r.get("status") not in _TERMINAL_STATES
                    for r in self._row_data
                )
                return True if any_active else None
            # Single job: same logic as cancel_job.
            return None if status in _TERMINAL_STATES else True
        return True

    @on(DataTable.RowHighlighted, "#sim-table")
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#sim-table", DataTable)
        old_row = self._dt_cursor_row
        self._dt_cursor_row = event.cursor_row
        with contextlib.suppress(Exception):
            old_status = (
                self._row_data[old_row]["status"]
                if old_row < len(self._row_data)
                else ""
            )
            old_indicator = (
                _ANIM_FRAMES[self._anim_tick % 2] if old_status == "RUNNING" else " "
            )
            table.update_cell(str(old_row), "cur", old_indicator)
        with contextlib.suppress(Exception):
            table.update_cell(str(self._dt_cursor_row), "cur", ">")
        self._update_footer_keys()

    def _update_footer_keys(self) -> None:
        _KEYS: list[tuple[str, str, str | None]] = [
            ("q", "Quit", None),
            ("l", "Logs", "view_logs"),
            ("i", "Info", "view_info"),
            ("c", "Cancel Job", "cancel_job"),
            ("a", "Cancel Array", "cancel_array"),
            ("r", "Resubmit", None),
            ("f", "Refresh", None),
        ]
        t = Text()
        for idx, (key, label, action) in enumerate(_KEYS):
            enabled = self.check_action(action, ()) if action else True
            style = "" if enabled else "dim"
            if idx:
                t.append("    ")
            t.append(f"[{key}] {label}", style=style)
        with contextlib.suppress(Exception):
            self.query_one("#footer-keys", Static).update(t)

    def _step_animation(self) -> None:
        self._anim_tick += 1
        frame = _ANIM_FRAMES[self._anim_tick % 2]
        table = self.query_one("#sim-table", DataTable)
        for i, row in enumerate(self._row_data):
            if row["status"] == "RUNNING" and i != self._dt_cursor_row:
                with contextlib.suppress(Exception):
                    table.update_cell(str(i), "cur", frame)

    def _selected_row(self) -> dict[str, Any] | None:
        table = self.query_one("#sim-table", DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._row_data):
            return self._row_data[row]
        return None

    def _selected_sim(self) -> Simulation | None:
        # Rows and sims are no longer 1:1 (a PARALLEL_PARAMS sim expands into
        # one row per realisation), so resolve the underlying sim via the row's
        # _sim_ref.  Job-level actions (logs, cancel, resubmit) act on this sim.
        rowd = self._selected_row()
        if rowd is None:
            return None
        return self._sims[rowd["_sim_ref"]]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_view_logs(self) -> None:
        sim = self._selected_sim()
        if sim is None:
            self.notify("No simulation selected.", severity="warning")
            return
        jobid = get_job_id(sim)
        if not jobid:
            self.notify("No job ID for this simulation.", severity="warning")
            return
        self._load_and_show_log(sim, jobid)

    @work(thread=True)
    def _load_and_show_log(self, sim: Simulation, jobid: str) -> None:
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

    def action_view_info(self) -> None:
        sim = self._selected_sim()
        rowd = self._selected_row()
        if sim is None or rowd is None:
            self.notify("No simulation selected.", severity="warning")
            return
        # For a PARALLEL_PARAMS row, info lives in the realisation's Temp_i/.
        info_dir = Path(rowd.get("effective_dir") or sim.sim_dir)
        info_path = info_dir / "info"
        if rowd.get("realisation") is not None:
            where = f"{Path(sim.sim_dir).name}/{info_dir.name}"
        else:
            where = Path(sim.sim_dir).name
        if not info_path.exists():
            self.notify(f"No info file found in {where}.", severity="warning")
            return
        try:
            content = info_path.read_text(errors="replace")
        except Exception as exc:
            self.notify(f"Error reading info file: {exc}", severity="error")
            return
        self.push_screen(LogViewerScreen(f"Info: {where}", content))

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
            msg += "\n\n[!]  Checkpoint restart detected.\nALF will append to existing data.h5."

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
        # An explicit refresh bypasses the (mtime, size) short-circuit, so a
        # stale filesystem attribute cache cannot pin the bin count.
        self._trigger_refresh(force=True)
