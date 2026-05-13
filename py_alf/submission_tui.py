"""
TUI for reviewing and confirming ALF job submission.

Usage::

    from py_alf.submission_tui import SubmissionReview
    app = SubmissionReview(cs, sims)
    submitted = app.run()   # blocks; returns submitted sim list, or None if cancelled
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Group as RichGroup
from rich.panel import Panel as RichPanel
from rich.table import Table as RichTable
from rich.text import Text as RichText
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, DataTable, Footer, Input, Label, Static

from .cluster_submission import (
    ClusterSubmitter,
    PartitionSpec,
    _format_hours,
    _parse_mem_gb,
)
from .simulation import Simulation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim_dict_of(sim: Simulation) -> dict:
    d = sim.sim_dict
    return d[0] if isinstance(d, list) else d


def _all_param_keys(sims: list[Simulation]) -> list[str]:
    """Union of all sim_dict keys across all sims, preserving first-seen order."""
    seen: dict[str, None] = {}
    for sim in sims:
        for k in _sim_dict_of(sim):
            seen[k] = None
    return list(seen)


def _wall_time_str(cpu_max_h: float) -> str:
    total_s = int(cpu_max_h * 3600)
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _completion_str(cpu_max_h: float) -> str:
    return (datetime.now() + timedelta(hours=cpu_max_h)).strftime("%Y-%m-%d %H:%M")


def _coerce(raw: str, original: object) -> object:
    """Parse raw string into the same type as original."""
    if isinstance(original, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(original, int):
        try:
            return int(raw)
        except ValueError:
            return original
    if isinstance(original, float):
        try:
            return float(raw)
        except ValueError:
            return original
    return raw


def _parse_wall_time(s: str) -> float | None:
    """Parse HH:MM:SS to fractional hours, or None if the string is invalid."""
    parts = s.strip().split(":")
    if len(parts) == 3:
        try:
            h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
            total = h + m / 60 + sec / 3600
            return total if total > 0 else None
        except ValueError:
            pass
    return None


_PARTITION_GRADIENT = [
    "bright_green",
    "green",
    "yellow",
    "red",
    "bright_red",
]


def _partition_color(partition: str, rules: dict[str, PartitionSpec]) -> str:
    """Return an ANSI color for *partition* along a green→red gradient."""
    if partition not in rules:
        return "ansi_white"
    sorted_names = sorted(rules, key=lambda n: rules[n].get("max_hours", 0))
    idx = sorted_names.index(partition)
    n = len(sorted_names)
    if n == 1:
        return _PARTITION_GRADIENT[0]
    grad_idx = round(idx / (n - 1) * (len(_PARTITION_GRADIENT) - 1))
    return _PARTITION_GRADIENT[grad_idx]


def _btn_label(text: str, active: bool) -> str:
    """Rich-markup label: bold ● for active, dim ○ for inactive."""
    return f"[bold]● {text}[/bold]" if active else f"[dim]○ {text}[/dim]"


_SUBMIT_STATUS_MAP: dict[str, tuple[str, str]] = {
    "queued": ("·  queued", "dim"),
    "preparing": ("…  preparing...", ""),
    "submitted": ("✓  submitted", "green"),
    "skipped": ("↷  skipped", "dim"),
    "failed": ("✗  failed", "red"),
    "cancelled": ("⊘  cancelled", "dim"),
}


def _status_label(status: str, detail: str = "") -> RichText:
    text, style = _SUBMIT_STATUS_MAP[status]
    t = RichText(text, style=style)
    if detail:
        t.append(f"  {detail}", style="dim")
    return t


def _write_session_manifest(
    submitted: list[Simulation],
    job_ids: list[str],
    cs: ClusterSubmitter,
    executor: str,
) -> Path | None:
    """Write a JSON record of the submitted sims to submit_dir. Returns the path, or None on error."""
    entries = [
        {
            "sim_dir": str(sim.sim_dir),
            "job_id": jid,
            "ham_name": sim.ham_name,
            "n_omp": sim.n_omp,
            "n_mpi": getattr(sim, "n_mpi", 1),
            "mpi": getattr(sim, "mpi", False),
            "sim_dict": dict(_sim_dict_of(sim)),
        }
        for sim, jid in zip(submitted, job_ids)
    ]
    cs_record: dict = {
        "executor": executor,
        "submit_dir": str(cs.submit_dir),
        "slurm_mem": cs.slurm_mem,
        "partition_rules": cs.partition_rules,
        "job_name": cs.job_name,
        "mail_type": cs.mail_type,
        "wckey": cs.wckey,
        "stderr_to_stdout": cs.stderr_to_stdout,
        "slurm_kwargs": cs.slurm_kwargs,
    }
    manifest = {
        "version": 1,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "cluster_submitter": cs_record,
        "entries": entries,
    }
    out_path = (
        cs.submit_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    try:
        cs.submit_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2, default=str))
        return out_path
    except Exception:
        return None


class _SimProxy:
    """Lightweight editable sim entry for manually added array jobs.

    Cloned from an existing simulation so it shares the same hamiltonian,
    config, and parallelism — the user edits parameters and sim_dir in the
    modal before the job is appended to the submission list.
    """

    def __init__(self, template: Simulation) -> None:
        self._template: Simulation = template
        self.ham_name: str = template.ham_name
        self.sim_dir: str = str(template.sim_dir)
        self.sim_dict: dict = copy.deepcopy(_sim_dict_of(template))
        self.n_omp: int = template.n_omp
        self.n_mpi: int = getattr(template, "n_mpi", 1)
        self.mpi: bool = getattr(template, "mpi", False)
        self.config: str = getattr(template, "config", "")

    def run(
        self,
        copy_bin: bool = False,
        only_prep: bool = False,
        bin_in_sim_dir: bool = False,
    ) -> None:
        """Delegate directory prep to the template Simulation with this proxy's sim_dir/sim_dict."""
        old_dir = self._template.sim_dir
        old_dict = self._template.sim_dict
        self._template.sim_dir = self.sim_dir
        self._template.sim_dict = self.sim_dict
        try:
            self._template.run(
                copy_bin=copy_bin, only_prep=only_prep, bin_in_sim_dir=bin_in_sim_dir
            )
        finally:
            self._template.sim_dir = old_dir
            self._template.sim_dict = old_dict


def _arch_renderable(
    sim: Simulation,
    executor: str,
    cs: ClusterSubmitter,
    mem_display: str | None = None,
) -> object:
    """Return a Rich renderable describing the job's resource layout."""
    if executor in ("local", "debug"):
        body = RichText()
        body.append(
            {
                "local": "Runs as local subprocesses — no cluster queue.",
                "debug": "Runs inline (synchronous) — debug mode.",
            }[executor]
            + "\n\n",
            style="dim",
        )
        body.append(f"{sim.n_omp}", style="bold")
        body.append(" OMP thread(s) per process")
        return RichPanel(body, title=f"[bold]{executor}[/bold]", border_style="dim")

    # --- SLURM ---
    n_ranks = sim.n_mpi if sim.mpi else 1
    n_omp = sim.n_omp
    sim_dict = _sim_dict_of(sim)
    timeout_h = max(0.0, float(sim_dict.get("CPU_MAX", 24)))

    try:
        partition = cs._select_partition(timeout_h)
        p_spec = cs.partition_rules[partition]
        p_note = f"CPU_MAX={_format_hours(timeout_h)}  ≤  {_format_hours(p_spec['max_hours'])} limit"
    except ValueError:
        partition = "---"
        p_spec = {}
        p_note = "too long"

    # Full rank grid — all workers, 4 columns, wrapping rows
    COLS = 4
    n_rows = math.ceil(n_ranks / COLS)
    bar = "█" * min(n_omp, 8) + ("…" if n_omp > 8 else "")

    grid = RichTable.grid(padding=(0, 1))
    for _ in range(COLS):
        grid.add_column(justify="center", min_width=8)

    for row_i in range(n_rows):
        start = row_i * COLS
        end = min(start + COLS, n_ranks)
        n_in_row = end - start
        pad = COLS - n_in_row

        labels = [RichText(f"r{i}", style="bold") for i in range(start, end)]
        bars = [RichText(bar, style="bold")] * n_in_row
        omps = [RichText(f"{n_omp} OMP")] * n_in_row

        grid.add_row(*(labels + [RichText("")] * pad))
        grid.add_row(*(bars + [RichText("")] * pad))
        grid.add_row(*(omps + [RichText("")] * pad))
        if row_i < n_rows - 1:
            grid.add_row(*([RichText("")] * COLS))

    mem_str = mem_display if mem_display else (cs.slurm_mem or "—")

    total_cpus = n_ranks * n_omp
    max_cpus: int | None = p_spec.get("max_cpus")  # type: ignore[assignment]
    max_mem_gb: float | None = p_spec.get("max_mem_gb")  # type: ignore[assignment]

    # Compute violations (only when limits are known and partition was found)
    cpu_over = max_cpus is not None and total_cpus > max_cpus
    mem_over = False
    if max_mem_gb is not None and mem_str:
        with contextlib.suppress(ValueError):
            mem_over = _parse_mem_gb(mem_str) > max_mem_gb

    info = RichText()
    info.append("\n")
    info.append("Partition: ", style="bold")
    if partition == "---":
        info.append("--- (too long)", style="bold bright_red")
        info.append("\n")
    else:
        part_color = (
            _partition_color(partition, cs.partition_rules or {})
            if cs.partition_rules
            else "white"
        )
        info.append(f"{partition}", style=part_color)
        info.append(f"   {p_note}\n", style="dim")
    info.append("Timeout:   ", style="bold")
    info.append(f"{_format_hours(timeout_h)}\n")
    info.append("Memory:    ", style="bold")
    if max_mem_gb is not None:
        limit_str = f"{max_mem_gb:g} GB"
        info.append(
            f"{mem_str} / {limit_str}",
            style="bold red" if mem_over else "",
        )
        if mem_over:
            info.append("  ✗", style="bold red")
    else:
        info.append(f"{mem_str}")
    info.append("\n")
    info.append("CPUs:      ", style="bold")
    if max_cpus is not None:
        info.append(
            f"{total_cpus} / {max_cpus}",
            style="bold red" if cpu_over else "",
        )
        if cpu_over:
            info.append("  ✗", style="bold red")
    else:
        info.append(f"{total_cpus}")

    title = f"[bold]Node[/bold]  ·  {n_ranks} rank(s) × {n_omp} OMP"
    return RichPanel(RichGroup(grid, info), title=title, border_style="dim")


# ---------------------------------------------------------------------------
# Edit modal
# ---------------------------------------------------------------------------


class EditSimScreen(ModalScreen):
    """Modal for editing a single simulation's parameters."""

    BINDINGS = [Binding("escape", "dismiss", "Close", show=True)]

    CSS = """
    EditSimScreen { align: center middle; }
    #edit-dialog {
        width: 62;
        max-height: 80%;
        background: ansi_default;
        border: solid ansi_default;
        padding: 1 2;
    }
    #edit-title { text-style: bold; margin-bottom: 1; color: ansi_default; }
    #edit-dialog Label { margin-top: 1; height: 1; color: ansi_default; }
    #edit-dialog Input { margin-bottom: 0; background: transparent; color: ansi_yellow; border: solid ansi_default; }
    #edit-dialog Input:focus { border: solid ansi_default; }
    #edit-buttons { height: 3; margin-top: 1; align: center middle; }
    #edit-buttons Button { margin: 0 1; }
    """

    def __init__(self, sim: Simulation, all_keys: list[str]) -> None:
        super().__init__()
        self._sim = sim
        self._sd = _sim_dict_of(sim)
        self._all_keys = all_keys

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="edit-dialog"):
            yield Label(Path(self._sim.sim_dir).name, id="edit-title")
            yield Label("sim_dir")
            yield Input(
                str(self._sim.sim_dir),
                id="field-sim_dir",
                placeholder="output directory",
            )
            for key in self._all_keys:
                yield Label(key)
                yield Input(
                    str(self._sd.get(key, "")), id=f"field-{key}", placeholder=key
                )
            yield Label("n_omp")
            yield Input(
                str(self._sim.n_omp), id="field-n_omp", placeholder="OMP threads"
            )
            yield Label("n_mpi")
            yield Input(str(self._sim.n_mpi), id="field-n_mpi", placeholder="MPI ranks")
            with Horizontal(id="edit-buttons"):
                yield Button("Save", id="btn-save", variant="primary")
                yield Button("Save to All", id="btn-save-all", variant="warning")
                yield Button("Cancel  [Esc]", id="btn-cancel", variant="default")

    def _apply_form(self) -> None:
        """Write current form values back to the sim object."""
        with contextlib.suppress(AttributeError):
            self._sim.sim_dir = self.query_one("#field-sim_dir", Input).value
        for key in self._all_keys:
            try:
                widget = self.query_one(f"#field-{key}", Input)
                self._sd[key] = _coerce(widget.value, self._sd.get(key, ""))
            except Exception:
                pass
        with contextlib.suppress(ValueError, AttributeError):
            self._sim.n_omp = int(self.query_one("#field-n_omp", Input).value)
        try:
            n_mpi = int(self.query_one("#field-n_mpi", Input).value)
            self._sim.n_mpi = n_mpi
            self._sim.mpi = n_mpi > 1
        except (ValueError, AttributeError):
            pass

    @on(Button.Pressed, "#btn-save")
    def _save(self) -> None:
        self._apply_form()
        self.dismiss("save")

    @on(Button.Pressed, "#btn-save-all")
    def _save_all(self) -> None:
        self._apply_form()
        self.dismiss("all")

    @on(Button.Pressed, "#btn-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Monochrome ANSI theme — replaces Textual's default blue primary with white
# ---------------------------------------------------------------------------

_MONO_THEME = Theme(
    name="submission-mono",
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
# Confirmation dialog
# ---------------------------------------------------------------------------


class ConfirmSubmitScreen(ModalScreen):
    """Yes / No confirmation before submitting jobs."""

    BINDINGS = [Binding("escape", "dismiss", show=False)]

    CSS = """
    ConfirmSubmitScreen { align: center middle; }
    #confirm-dialog {
        width: 44;
        height: auto;
        background: ansi_default;
        border: solid ansi_default;
        padding: 1 2;
    }
    #confirm-msg { color: ansi_default; margin-bottom: 1; }
    #confirm-buttons { height: 1; align: left middle; }
    #confirm-buttons Button { height: 1; border: none; margin: 0 2 0 0; }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(self._message, id="confirm-msg")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", id="btn-yes", variant="primary")
                yield Button("No", id="btn-no")

    @on(Button.Pressed, "#btn-yes")
    def _on_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#btn-no")
    def _on_no(self) -> None:
        self.dismiss(False)

    def action_dismiss(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class SubmissionReview(App):
    """
    Interactive TUI for reviewing and confirming ALF job submission.

    Presents a pre-submission checklist: toggle which simulations to include,
    inspect the SLURM resource layout, review model parameters and output
    directories, and choose the backend executor before committing.

    Parameters
    ----------
    cs : ClusterSubmitter
        Configured submitter whose settings are displayed and used on submit.
    sims : list of Simulation
        Candidate simulations. All are pre-selected; the user may deselect any.
    param_keys : list of str, optional
        Keys from ``sim.sim_dict`` to display as table columns. Defaults to
        auto-discovering all keys present across all simulations.
    param_headers : list of str, optional
        Column labels for *param_keys* (defaults to the key names).

    Returns
    -------
    list[Simulation] | None
        ``app.run()`` returns the list of submitted simulations, or ``None``
        if the user cancelled without submitting.
    """

    TITLE = "pyALF · Submission Review"
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

    #body { height: 1fr; }
    #left-panel { height: 1fr; width: 100%; padding: 0 1; }
    #right-panel { height: 1fr; width: 100%; border-top: solid ansi_default; }
    #settings-col { width: 1fr; padding: 0 1; }
    #arch-col { width: 1fr; border-left: solid ansi_default; padding: 0 1; }

    Horizontal, Vertical, ScrollableContainer, Static, Label { background: transparent; }
    DataTable { height: 1fr; background: transparent; }
    DataTable > .datatable--header { color: ansi_default; background: transparent; }
    DataTable > .datatable--cursor { background: ansi_bright_black; color: ansi_default; text-style: bold; }
    DataTable > .datatable--hover { background: ansi_bright_black; color: ansi_default; }
    DataTable > .datatable--even-row { background: transparent; }
    DataTable > .datatable--odd-row { background: transparent; }

    #progress-table { display: none; }

    #job-bar { height: 1; color: ansi_default; }

    .hd { height: 1; margin-top: 1; color: ansi_default; text-style: bold; }

    .sr { height: 3; }
    .sl { width: 11; height: 3; content-align: left middle; color: ansi_default; }

    Input { background: transparent; color: ansi_default; border: solid ansi_default; }
    Input:focus { border: solid ansi_default; }

    #settings-col .sr { height: 1; }
    #settings-col .sl { height: 1; content-align: left middle; }
    #settings-col Input { height: 1; border: none; padding: 0 1; background: transparent; color: ansi_yellow; }
    #settings-col Input:focus { border: none; background: transparent; color: ansi_yellow; }
    #mail-type-row Button { min-width: 9; height: 1; border: none; margin: 0 1 0 0; }
    #stderr-toggle-row Button { min-width: 10; height: 1; border: none; margin: 0 1 0 0; }
    #stop-on-fail-btn { min-width: 9; height: 1; border: none; }

    #executor-row { height: 3; }
    #executor-row Button { min-width: 10; height: 3; margin: 0 1 0 0; }

    #arch-col .sr { height: 1; }
    #arch-col .sl { height: 1; content-align: left middle; }
    #arch-col Input { height: 1; border: none; padding: 0 1; background: transparent; color: ansi_blue; }
    #arch-col Input:focus { border: none; background: transparent; color: ansi_blue; }

    Button { border: blank; color: ansi_default; background: transparent; }
    Button.-primary { background: ansi_bright_black; color: ansi_default; }
    Button.-success { background: ansi_bright_black; color: ansi_default; }
    Button.-active { background: ansi_bright_black; color: ansi_default; text-style: bold; }
    Button.-primary.-active { background: ansi_bright_black; color: ansi_default; text-style: bold; }
    Button:hover { color: ansi_default; background: transparent; }
    Button.-primary:hover { color: ansi_default; background: ansi_bright_black; text-style: bold; }

    #footer-bar { height: 1; }
    #footer-bar Footer { width: 1fr; background: transparent; color: ansi_default; }
    #footer-done { display: none; width: 1fr; height: 1; content-align: left middle; padding: 0 1; }
    FooterKey .footer-key--key { background: ansi_bright_black; color: ansi_default; padding: 0 1; }
    FooterKey .footer-key--description { padding-left: 1; padding-right: 2; }
    #btn-submit { height: 1; border: blank; min-width: 0; padding: 0 2; background: ansi_bright_black; color: ansi_default; text-style: bold; }
    #btn-submit:hover { background: ansi_bright_black; color: ansi_default; text-style: bold; }
    #btn-submit.-active { background: ansi_bright_black; color: ansi_default; text-style: bold; }
    """

    BINDINGS = [
        Binding("s", "submit", "Submit", show=True),
        Binding("space", "toggle_sim", "Toggle", show=True),
        Binding("e", "edit_sim", "Edit", show=True),
        Binding("a", "add_sim", "Add", show=True),
        Binding("m", "open_monitor", "Monitor", show=False),
        Binding("q", "quit_cancel", "Cancel", show=True),
    ]

    def __init__(
        self,
        cs: ClusterSubmitter,
        sims: list[Simulation],
        param_keys: list[str] | None = None,
        param_headers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._cs = cs
        self._sims = list(sims)
        self._selected = [True] * len(sims)
        self._param_keys: list[str] = (
            list(param_keys) if param_keys else _all_param_keys(sims)
        )
        self._param_headers: list[str] = (
            list(param_headers) if param_headers else list(self._param_keys)
        )
        self._executor_state: str = cs.executor
        self._can_use_slurm: bool = cs.executor == "slurm"
        self._focused_idx: int = 0
        self._mail_flags: set[str] = set((cs.mail_type or "").upper().split())
        self._stderr_to_stdout: bool = cs.stderr_to_stdout
        self._is_submitting: bool = False
        self._submitted_result: list[Simulation] | None = None
        self._stop_on_fail: bool = False
        self.open_monitor: bool = False
        self.submitted_cs: ClusterSubmitter | None = None
        self.session_path: Path | None = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(f"  {self.TITLE}", id="title-bar")
        with Vertical(id="body"):
            # ── left: table ──
            with Vertical(id="left-panel"):
                yield DataTable(id="sim-table", zebra_stripes=True, cursor_type="row")
                yield DataTable(
                    id="progress-table", cursor_type="none", zebra_stripes=True
                )
                yield Static("", id="job-bar")
            # ── right: settings + arch + schedule ──
            with Horizontal(id="right-panel"):
                with ScrollableContainer(id="settings-col"):
                    yield Label("Executor", classes="hd")
                    with Horizontal(id="executor-row"):
                        for ex in ("slurm", "local", "debug"):
                            disabled = ex == "slurm" and not self._can_use_slurm
                            yield Button(
                                _btn_label(ex, ex == self._executor_state),
                                id=f"exec-{ex}",
                                disabled=disabled,
                            )
                    yield Label("Settings", classes="hd")
                    with Horizontal(classes="sr"):
                        yield Label("Memory", classes="sl")
                        yield Input(
                            self._cs.slurm_mem or "", id="mem-input", placeholder="4G"
                        )
                    with Horizontal(classes="sr"):
                        yield Label("Submit dir", classes="sl")
                        yield Input(
                            str(self._cs.submit_dir),
                            id="dir-input",
                            placeholder="submitit",
                        )
                    with Horizontal(classes="sr"):
                        yield Label("Job name", classes="sl")
                        yield Input(
                            self._cs.job_name or "",
                            id="job-name-input",
                            placeholder="(hamiltonian)",
                        )
                    with Horizontal(classes="sr"):
                        yield Label("Mail type", classes="sl")
                        with Horizontal(id="mail-type-row"):
                            for tag in ("END", "FAIL", "ALL"):
                                yield Button(
                                    _btn_label(tag, tag in self._mail_flags),
                                    id=f"mail-{tag.lower()}",
                                )
                    with Horizontal(classes="sr"):
                        yield Label("WCKey", classes="sl")
                        yield Input(
                            self._cs.wckey or "",
                            id="wckey-input",
                            placeholder="project key",
                        )
                    with Horizontal(classes="sr", id="stderr-row"):
                        yield Label("stderr→out", classes="sl")
                        with Horizontal(id="stderr-toggle-row"):
                            for val, label in (("true", "true"), ("false", "false")):
                                active = (val == "true") == self._cs.stderr_to_stdout
                                yield Button(
                                    _btn_label(label, active),
                                    id=f"stderr-{val}",
                                )
                    with Horizontal(classes="sr"):
                        yield Label("On fail", classes="sl")
                        yield Button(
                            _btn_label("stop", self._stop_on_fail),
                            id="stop-on-fail-btn",
                        )
                with ScrollableContainer(id="arch-col"):
                    yield Label("Architecture", classes="hd")
                    yield Static("", id="arch-panel")
                    yield Label("Schedule", classes="hd")
                    with Horizontal(classes="sr"):
                        yield Label("Wall time", classes="sl")
                        yield Input("", id="walltime-input", placeholder="HH:MM:SS")
                    yield Static("", id="schedule-panel")
        # ── footer row: key hints left, submit button right ──
        with Horizontal(id="footer-bar"):
            yield Footer()
            yield Static("", id="footer-done")
            yield Button("", id="btn-submit")

    def on_mount(self) -> None:
        self.register_theme(_MONO_THEME)
        self.theme = "submission-mono"
        self._setup_table()
        self._refresh_all()
        self.call_after_refresh(self._remove_ansi_scrollbar_class)

    def watch_theme(self, _theme: str) -> None:
        self.call_after_refresh(self._remove_ansi_scrollbar_class)

    def _remove_ansi_scrollbar_class(self) -> None:
        for widget in self.query(".-ansi-scrollbar"):
            widget.remove_class("-ansi-scrollbar")

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _setup_table(self) -> None:
        table = self.query_one("#sim-table", DataTable)
        table.add_column("", key="sel", width=3)
        table.add_column("#", key="idx", width=4)
        table.add_column("Hamiltonian", key="ham")
        for key, hdr in zip(self._param_keys, self._param_headers):
            table.add_column(hdr, key=key)
        table.add_column("OMP", key="omp", width=5)
        table.add_column("MPI", key="mpi_r", width=5)
        self._repopulate_table()

    def _repopulate_table(self) -> None:
        table = self.query_one("#sim-table", DataTable)
        saved_row = table.cursor_row
        table.clear()
        for i, sim in enumerate(self._sims):
            sd = _sim_dict_of(sim)
            mark = (
                RichText("✓", style="bold")
                if self._selected[i]
                else RichText("·", style="dim")
            )
            row = [mark, str(i), sim.ham_name]
            for key in self._param_keys:
                row.append(str(sd.get(key, "—")))
            row += [str(sim.n_omp), str(sim.n_mpi if sim.mpi else 1)]
            table.add_row(*row, key=str(i))
        if self._sims:
            table.move_cursor(row=min(saved_row, len(self._sims) - 1))

    # ------------------------------------------------------------------
    # Panel refresh
    # ------------------------------------------------------------------

    def _refresh_all(self) -> None:
        if not self._sims:
            return
        self._refresh_job_bar()
        self._refresh_right_panel()
        self._refresh_submit_button()

    def _refresh_job_bar(self) -> None:
        n_sel = sum(self._selected)
        n_total = len(self._sims)
        job_type = "array job" if n_sel > 1 else "single job"
        self.query_one("#job-bar", Static).update(
            f"[dim]{n_sel}/{n_total} selected  ·  {job_type}[/dim]"
        )

    def _refresh_right_panel(self) -> None:
        if not self._sims or self._focused_idx >= len(self._sims):
            return
        sim = self._sims[self._focused_idx]
        sd = _sim_dict_of(sim)
        timeout_h = max(0.0, float(sd.get("CPU_MAX", 24)))
        mem = self.query_one("#mem-input", Input).value or (self._cs.slurm_mem or "")

        self.query_one("#arch-panel", Static).update(
            _arch_renderable(sim, self._executor_state, self._cs, mem_display=mem)
        )

        wt_input = self.query_one("#walltime-input", Input)
        if not wt_input.has_focus:
            wt_input.value = _wall_time_str(timeout_h)

        self.query_one("#schedule-panel", Static).update(
            f"[bold]Completes ~:[/bold] [ansi_blue]{_completion_str(timeout_h)}[/ansi_blue]\n"
            f"[bold]Submit dir:[/bold]  [dim]{self.query_one('#dir-input', Input).value or str(self._cs.submit_dir)}[/dim]"
        )

    def _has_unfit_partition(self) -> bool:
        """True when any selected sim exceeds a partition wall-time or node resource limit."""
        if self._executor_state != "slurm" or not self._cs.partition_rules:
            return False
        mem_str = self.query_one("#mem-input", Input).value or (
            self._cs.slurm_mem or ""
        )
        for sim, sel in zip(self._sims, self._selected):
            if not sel:
                continue
            timeout_h = max(0.0, float(_sim_dict_of(sim).get("CPU_MAX", 24)))
            try:
                partition = self._cs._select_partition(timeout_h)
            except ValueError:
                return True
            try:
                self._cs._check_node_fit(sim, partition, slurm_mem=mem_str or None)
            except ValueError:
                return True
        return False

    def _refresh_submit_button(self) -> None:
        n_sel = sum(self._selected)
        btn = self.query_one("#btn-submit", Button)
        if self._submitted_result is not None:
            btn.label = f"Done — {len(self._submitted_result)} submitted  [q]"
            btn.disabled = True
        elif self._is_submitting:
            btn.label = "Submitting...  [s]"
            btn.disabled = True
        elif n_sel == 0:
            btn.label = f"Submit  ({n_sel} / {len(self._sims)})  [s]"
            btn.disabled = True
        elif self._has_unfit_partition():
            btn.label = "Cannot submit — partition unfit  [s]"
            btn.disabled = True
        else:
            btn.label = f"Submit  ({n_sel} / {len(self._sims)})  [s]"
            btn.disabled = False

    def _set_executor(self, executor: str) -> None:
        self._executor_state = executor
        for ex in ("slurm", "local", "debug"):
            self.query_one(f"#exec-{ex}", Button).label = _btn_label(ex, ex == executor)
        self._refresh_right_panel()
        self._refresh_submit_button()

    # ------------------------------------------------------------------
    # Submit / cancel
    # ------------------------------------------------------------------

    def _do_submit(self) -> None:
        selected_sims = [s for s, sel in zip(self._sims, self._selected) if sel]
        if not selected_sims:
            self.notify("No simulations selected.", severity="warning")
            return
        try:
            cs = self._effective_cs()
        except Exception as exc:
            self.notify(f"Configuration error: {exc}", severity="error")
            return
        self.submitted_cs = cs
        self._is_submitting = True
        self._refresh_submit_button()
        self._show_progress_view(selected_sims)
        self._run_submission(cs, selected_sims)

    def _show_progress_view(self, sims: list[Simulation]) -> None:
        self.query_one("#sim-table").display = False
        ptable = self.query_one("#progress-table", DataTable)
        ptable.display = True
        ptable.clear(columns=True)
        ptable.add_column("#", key="idx", width=4)
        ptable.add_column("Simulation", key="sim")
        ptable.add_column("Status", key="status")
        for i, sim in enumerate(sims):
            ptable.add_row(
                str(i), Path(str(sim.sim_dir)).name, _status_label("queued"), key=str(i)
            )
        self.query_one("#job-bar", Static).update(
            f"[bold]Submitting {len(sims)} job(s)...[/bold]"
        )

    def _set_progress(self, idx: int, status: str, detail: str = "") -> None:
        self.query_one("#progress-table", DataTable).update_cell(
            str(idx), "status", _status_label(status, detail), update_width=True
        )

    def _on_all_submitted(
        self, submitted: list[Simulation], n_total: int, job_ids: list[str]
    ) -> None:
        self._is_submitting = False
        self._submitted_result = submitted
        n_ok = len(submitted)
        n_fail = n_total - n_ok
        parts = [f"[green]{n_ok}/{n_total} submitted[/green]"]
        if n_fail:
            parts.append(f"[red]{n_fail} failed/cancelled[/red]")
        array_parents = {jid.split("_")[0] for jid in job_ids if "_" in jid}
        if len(array_parents) == 1:
            parts.append(f"[dim]array: {array_parents.pop()}[/dim]")
        if (
            submitted
            and self._executor_state == "slurm"
            and self.submitted_cs is not None
        ):
            self.session_path = _write_session_manifest(
                submitted, job_ids, self.submitted_cs, self._executor_state
            )
            if self.session_path:
                parts.append(f"[dim]session: {self.session_path}[/dim]")
        self.query_one("#job-bar", Static).update("  ".join(parts))
        self._lock_ui()

    def _lock_ui(self) -> None:
        for w in self.query("Input"):
            w.disabled = True
        for w in self.query("Button"):
            w.disabled = True
        self.query_one("Footer").display = False
        self.query_one("#btn-submit").display = False
        label = RichText()
        if self._submitted_result and self._executor_state == "slurm":
            label.append(" m ", style="on bright_black")
            label.append("  open monitor")
            label.append("    ")
        label.append(" q ", style="on bright_black")
        label.append("  exit")
        done = self.query_one("#footer-done", Static)
        done.display = True
        done.update(label)

    @work(thread=True)
    def _run_submission(self, cs: ClusterSubmitter, sims: list[Simulation]) -> None:
        # Snapshot existing job IDs so we can identify which sims are newly submitted.
        old_jids: dict[str, str | None] = {}
        for sim in sims:
            jid_file = Path(str(sim.sim_dir)) / "jobid.txt"
            old_jids[str(sim.sim_dir)] = (
                jid_file.read_text().strip() if jid_file.exists() else None
            )

        for i in range(len(sims)):
            self.call_from_thread(self._set_progress, i, "preparing")

        try:
            jobs = cs.submit(sims)
        except Exception as exc:
            for i in range(len(sims)):
                self.call_from_thread(self._set_progress, i, "failed", str(exc))
            self.call_from_thread(self._on_all_submitted, [], len(sims), [])
            return

        new_jid_set = {j.job_id for j in jobs}
        submitted: list[Simulation] = []
        job_ids: list[str] = []

        for i, sim in enumerate(sims):
            jid_file = Path(str(sim.sim_dir)) / "jobid.txt"
            if jid_file.exists():
                new_jid = jid_file.read_text().strip()
                if new_jid != old_jids.get(str(sim.sim_dir)) and new_jid in new_jid_set:
                    self.call_from_thread(self._set_progress, i, "submitted", new_jid)
                    submitted.append(sim)
                    job_ids.append(new_jid)
                    continue
            self.call_from_thread(self._set_progress, i, "skipped")

        self.call_from_thread(self._on_all_submitted, submitted, len(sims), job_ids)

    def _effective_cs(self) -> ClusterSubmitter:
        mem = self.query_one("#mem-input", Input).value.strip() or (
            self._cs.slurm_mem or ""
        )
        submit_dir = self.query_one("#dir-input", Input).value.strip() or str(
            self._cs.submit_dir
        )
        job_name = self.query_one("#job-name-input", Input).value.strip() or None
        mail_type = (
            " ".join(sorted(self._mail_flags, key=["END", "FAIL", "ALL"].index)) or None
        )
        wckey = self.query_one("#wckey-input", Input).value.strip() or None
        stderr_to_stdout = self._stderr_to_stdout

        if self._executor_state == "slurm" and self._cs.executor == "slurm":
            self._cs.slurm_mem = mem
            self._cs.submit_dir = Path(submit_dir)
            self._cs.job_name = job_name
            self._cs.mail_type = mail_type
            self._cs.wckey = wckey
            self._cs.stderr_to_stdout = stderr_to_stdout
            return self._cs
        return ClusterSubmitter(
            self._executor_state,
            submit_dir=submit_dir,
            job_name=job_name,
            stderr_to_stdout=stderr_to_stdout,
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_submit(self) -> None:
        if self._is_submitting or self._submitted_result is not None:
            return
        n_sel = sum(self._selected)
        if n_sel == 0:
            self.notify("No simulations selected.", severity="warning")
            return
        if self._has_unfit_partition():
            self.notify(
                "Cannot submit — a selected job exceeds all configured partitions.",
                severity="error",
            )
            return
        msg = f"Submit {n_sel} simulation(s)?"

        def _confirmed(result: bool | None) -> None:
            if result:
                self._do_submit()

        self.push_screen(ConfirmSubmitScreen(msg), _confirmed)

    def action_toggle_sim(self) -> None:
        table = self.query_one("#sim-table", DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._sims):
            self._selected[row] = not self._selected[row]
            self._repopulate_table()
            self._refresh_job_bar()
            self._refresh_submit_button()

    def action_edit_sim(self) -> None:
        table = self.query_one("#sim-table", DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._sims):
            sim = self._sims[row]

            def on_done(result: object) -> None:
                if result is None:
                    return
                self._repopulate_table()
                self._refresh_right_panel()
                if result == "all":
                    n_omp, n_mpi, mpi = sim.n_omp, sim.n_mpi, sim.mpi
                    for s in self._sims:
                        if s is not sim:
                            with contextlib.suppress(AttributeError):
                                s.n_omp = n_omp
                            with contextlib.suppress(AttributeError):
                                s.n_mpi = n_mpi
                                s.mpi = mpi
                    self._repopulate_table()
                    self.notify(
                        f"Applied n_omp={n_omp}, n_mpi={n_mpi} to all {len(self._sims)} sims."
                    )

            self.push_screen(
                EditSimScreen(sim, _all_param_keys([sim])),
                callback=on_done,
            )

    def action_add_sim(self) -> None:
        table = self.query_one("#sim-table", DataTable)
        row = table.cursor_row
        template = self._sims[row] if 0 <= row < len(self._sims) else self._sims[0]
        proxy = _SimProxy(template)
        all_keys = _all_param_keys([proxy])

        def on_done(result: object) -> None:
            if result is None:
                return
            self._sims.append(proxy)
            self._selected.append(True)
            self._repopulate_table()
            self._refresh_job_bar()
            self._refresh_submit_button()
            self.query_one("#sim-table", DataTable).move_cursor(row=len(self._sims) - 1)

        self.push_screen(EditSimScreen(proxy, all_keys), callback=on_done)

    def action_quit_cancel(self) -> None:
        self.exit(self._submitted_result)

    def action_open_monitor(self) -> None:
        if self._submitted_result is None or self._executor_state != "slurm":
            return
        self.open_monitor = True
        self.exit(self._submitted_result)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    @on(DataTable.RowHighlighted, "#sim-table")
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None:
            with contextlib.suppress(ValueError, TypeError):
                self._focused_idx = int(str(event.row_key.value))
            self._refresh_right_panel()

    @on(Input.Changed, "#mem-input")
    def _on_mem_changed(self) -> None:
        self._refresh_right_panel()

    @on(Input.Changed, "#dir-input")
    def _on_dir_changed(self) -> None:
        self._refresh_right_panel()

    @on(Input.Changed, "#walltime-input")
    def _on_walltime_changed(self) -> None:
        val = self.query_one("#walltime-input", Input).value
        hours = _parse_wall_time(val)
        if hours is not None and self._focused_idx < len(self._sims):
            _sim_dict_of(self._sims[self._focused_idx])["CPU_MAX"] = hours
            self._refresh_right_panel()
            self._refresh_submit_button()

    @on(Button.Pressed, "#mail-end")
    def _on_mail_end(self) -> None:
        self._toggle_mail("END")

    @on(Button.Pressed, "#mail-fail")
    def _on_mail_fail(self) -> None:
        self._toggle_mail("FAIL")

    @on(Button.Pressed, "#mail-all")
    def _on_mail_all(self) -> None:
        self._toggle_mail("ALL")

    def _toggle_mail(self, tag: str) -> None:
        if tag in self._mail_flags:
            self._mail_flags.discard(tag)
        else:
            if tag == "ALL":
                # ALL is exclusive — deselect END and FAIL
                for other in ("END", "FAIL"):
                    self._mail_flags.discard(other)
                    self.query_one(f"#mail-{other.lower()}", Button).label = _btn_label(
                        other, False
                    )
            else:
                # Selecting END or FAIL implicitly removes ALL
                self._mail_flags.discard("ALL")
                self.query_one("#mail-all", Button).label = _btn_label("ALL", False)
            self._mail_flags.add(tag)
        self.query_one(f"#mail-{tag.lower()}", Button).label = _btn_label(
            tag, tag in self._mail_flags
        )

    @on(Button.Pressed, "#stderr-true")
    def _on_stderr_true(self) -> None:
        self._set_stderr(True)

    @on(Button.Pressed, "#stderr-false")
    def _on_stderr_false(self) -> None:
        self._set_stderr(False)

    def _set_stderr(self, value: bool) -> None:
        self._stderr_to_stdout = value
        self.query_one("#stderr-true", Button).label = _btn_label("true", value)
        self.query_one("#stderr-false", Button).label = _btn_label("false", not value)

    @on(Button.Pressed, "#stop-on-fail-btn")
    def _on_stop_on_fail(self) -> None:
        self._stop_on_fail = not self._stop_on_fail
        self.query_one("#stop-on-fail-btn", Button).label = _btn_label(
            "stop", self._stop_on_fail
        )

    @on(Button.Pressed, "#exec-slurm")
    def _on_exec_slurm(self) -> None:
        self._set_executor("slurm")

    @on(Button.Pressed, "#exec-local")
    def _on_exec_local(self) -> None:
        self._set_executor("local")

    @on(Button.Pressed, "#exec-debug")
    def _on_exec_debug(self) -> None:
        self._set_executor("debug")

    @on(Button.Pressed, "#btn-submit")
    def _on_submit_pressed(self) -> None:
        self.action_submit()
