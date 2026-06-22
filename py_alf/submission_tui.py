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
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import rich.box as RichBox
from rich.panel import Panel as RichPanel
from rich.text import Text as RichText
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, DataTable, Input, Label, Static

from .cluster_submission import (
    ClusterSubmitter,
    _format_hours,
    _hours_to_hms,
    _parse_mem_gb,
    _parse_slurm_time_hours,
    _slurm_time_to_minutes,
)
from .cluster_submission import (
    write_session_manifest as _write_session_manifest,
)
from .simulation import Simulation

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def _sim_dict_of(sim: Simulation) -> dict:
    d = sim.sim_dict
    return d[0] if isinstance(d, list) else d


_EXCLUDED_PARAM_KEYS_LOWER = frozenset({"model", "cpu_max"})


def _all_param_keys(sims: list[Simulation]) -> list[str]:
    """Union of all sim_dict keys across all sims, preserving first-seen order.

    Keys in _EXCLUDED_PARAM_KEYS_LOWER are omitted: Model is implementation
    detail noise; CPU_MAX is surfaced globally via the wall-time input.
    """
    seen: dict[str, None] = {}
    for sim in sims:
        for k in _sim_dict_of(sim):
            if k.lower() not in _EXCLUDED_PARAM_KEYS_LOWER:
                seen[k] = None
    return list(seen)


def _completion_str(cpu_max_h: float) -> str:
    return (datetime.now() + timedelta(hours=cpu_max_h)).strftime("%Y-%m-%d %H:%M")


def _data_conflict(sim: Simulation) -> bool:
    """True if sim_dir has a partial/interrupted run that would cause a hang.

    Directories with ``confin_*`` files are legitimate checkpoint restarts.
    Directories with ``confout_*`` files are completed runs: ``_prep_sim_dir``
    calls ``out_to_in()`` automatically on submit, renaming them to ``confin_*``.
    Both cases are handled transparently and do not block submission.

    The only genuinely problematic case is ``data.h5`` existing without any
    conf files, which indicates an interrupted run with no usable checkpoint.
    """
    sim_path = Path(str(sim.sim_dir))
    if not sim_path.exists():
        return False
    if any(sim_path.glob("confin_*")) or any(sim_path.glob("confout_*")):
        return False  # checkpoint or completed run — handled by _prep_sim_dir
    return (sim_path / "data.h5").exists()


def _sim_mark(selected: bool, conflict: bool) -> RichText:
    """Return the selection-column cell for a sim table row."""
    if conflict and selected:
        return RichText(" ! ", style="red")
    if conflict:
        return RichText(" ! ", style="dim")
    return RichText("S") if selected else RichText("-", style="dim")


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


def _relative_or_abs(p: Path) -> str:
    """Return p relative to cwd, falling back to absolute if outside cwd."""
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def _tail_path(path: str, keep: int = 2) -> str:
    """Return '.../last/N/parts' when the path is longer than *keep* components."""
    if not path:
        return path
    parts = Path(path).parts
    return ".../" + "/".join(parts[-keep:])


def _common_sim_data_dir(sims: list) -> str:
    """
    Return the deepest common ancestor of all sim_dirs, relative to cwd if
    possible.
    """
    paths = [Path(str(s.sim_dir)).resolve() for s in sims]
    if not paths:
        return ""
    if len(paths) == 1:
        return _relative_or_abs(paths[0])
    common = Path(os.path.commonpath([str(p) for p in paths]))
    return _relative_or_abs(common)


def _parse_wall_time(s: str) -> float | None:
    """Parse HH:MM:SS to fractional hours, or None if the string is invalid."""
    parts = s.strip().split(":")
    if len(parts) == 3:
        try:
            h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
            if not (h >= 0 and 0 <= m <= 59 and 0 <= sec < 60):
                return None
            total = h + m / 60 + sec / 3600
            return total
        except ValueError:
            pass
    return None


def _btn_label(text: str, active: bool) -> str:
    """Rich-markup label: ■ for active, dim □ for inactive."""
    return f"■ {text}" if active else f"[dim]□ {text}[/dim]"


_FOOTER_VIEW = r"\[s] Submit    \[q] Cancel"
_FOOTER_EDIT = r"\[space] Toggle    \[e] Edit    \[a] Add    \[r] Remove    \[q] Done"

# Command palette: canonical name → input widget id (None = special handling).
_CMD_ALIASES: dict[str, str] = {
    "exec": "executor",
    "executor": "executor",
    "omp": "omp",
    "mpi": "mpi",
    "mem": "mem",
    "memory": "mem",
    "walltime": "walltime",
    "time": "walltime",
    "slurm_time": "walltime",
    "nbin": "nbin",
    "partition": "partition",
    "part": "partition",
    "jobname": "jobname",
    "job_name": "jobname",
    "name": "jobname",
    "wckey": "wckey",
    "datadir": "datadir",
    "dir": "datadir",
    "end": "end",
    "endby": "end",
    "stderr": "stderr",
    "mail": "mail",
    "mailtype": "mail",
    "mail_type": "mail",
}

_CMD_INPUT_ID: dict[str, str] = {
    "omp": "#omp-input",
    "mpi": "#mpi-input",
    "mem": "#mem-input",
    "walltime": "#walltime-input",
    "nbin": "#nbin-input",
    "partition": "#partition-input",
    "jobname": "#job-name-input",
    "wckey": "#wckey-input",
    "datadir": "#data-dir-display",
}

_SUBMIT_STATUS_MAP: dict[str, tuple[str, str]] = {
    "QUEUED": ("QUEUED", "dim"),
    "PREPARING": ("PREPARING", ""),
    "SUBMITTED": ("SUBMITTED", ""),
    "SKIPPED": ("SKIPPED", "dim"),
    "FAILED": ("FAILED", "red"),
    "CANCELLED": ("CANCELLED", "dim"),
}


def _status_label(status: str, detail: str = "") -> RichText:
    text, style = _SUBMIT_STATUS_MAP[status]
    t = RichText(text, style=style)
    if detail:
        t.append(f" >>> {detail}", style="dim")
    return t


def save_for_ssh(
    cs: ClusterSubmitter,
    sims: list,
    path: str | Path | None = None,
) -> str:
    """Pickle submission state and print the command to launch the TUI over SSH.

    Call this from a Jupyter notebook cell after building ``cs`` and ``sims``.
    It saves the objects to a temporary file and prints a one-liner to paste
    into the VS Code integrated terminal (or any SSH shell) to open the
    submission review followed by the monitor.

    Parameters
    ----------
    cs : ClusterSubmitter
        Configured submitter.
    sims : list of Simulation
        Simulation objects to submit.
    path : str or Path, optional
        Pickle destination. Defaults to ``/tmp/alf_tui_state.pkl``.

    Returns
    -------
    str
        The shell command that was printed.
    """
    import pickle

    path = Path("/tmp/alf_tui_state.pkl") if path is None else Path(path)

    with open(path, "wb") as f:
        pickle.dump({"sims": sims, "cs": cs}, f)

    cmd = (
        f'python -c "'
        f"import pickle; from py_alf.submission_tui import SubmissionReview; "
        f"d=pickle.load(open('{path}','rb')); "
        f"SubmissionReview(d['cs'],d['sims']).run_with_monitor()\""
    )
    print(f"{cmd}")
    return cmd


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
        """
        Delegate directory prep to the template Simulation with this proxy's
        sim_dir/sim_dict.
        """
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
    partition_override: str | None = None,
    machine_config: str = "",
    n_omp_display: int | None = None,
    n_mpi_display: int | None = None,
    job_name: str | None = None,
    wckey: str | None = None,
) -> object:
    """Return a Rich renderable describing the job's resource layout."""
    name_part = job_name if job_name else f"auto  <<<  {sim.ham_name}"
    job_label = f"[{wckey}]  {name_part}" if wckey else name_part

    # Outline a "summary" panel, that changes depending on slurm / local / debug.

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
        body.append(f"{sim.n_omp}")
        body.append(" OMP thread(s) per process")
        if machine_config:
            body.append("\n\nALF configuration: ")
            body.append(machine_config, style="dim")
        return RichPanel(body, title=executor, border_style="dim", box=RichBox.SQUARE)

    # If omp or mpi are not provided, display "?".
    omp_label = str(n_omp_display) if n_omp_display is not None else "?"
    mpi_label = str(n_mpi_display) if n_mpi_display is not None else "?"

    sim_dict = _sim_dict_of(sim)

    # If CPU_MAX is not provided assume NBin and hence set to 0.
    _cpu_max = float(sim_dict.get("CPU_MAX", 0))

    # Account for the possibility that slurm_time could be int or str.
    fixed_slurm_time: str | int | None = (cs.slurm_kwargs or {}).get("slurm_time")
    if isinstance(fixed_slurm_time, int):
        fixed_slurm_h: float | None = fixed_slurm_time / 60
    elif fixed_slurm_time:
        fixed_slurm_h = _parse_slurm_time_hours(fixed_slurm_time)
    else:
        fixed_slurm_h = None

    # If the slurm_time is None it means we are possibly in NBin mode.
    nbin_needs_time: bool = fixed_slurm_h is None and _cpu_max == 0

    if nbin_needs_time:
        partition = "---"
        p_spec = {}
        p_note = ""
    else:
        try:
            if partition_override:  # allow manual override of auto partition
                partition = partition_override
                p_spec = (cs.partition_rules or {}).get(partition, {})
                p_note = "manual override"
            elif fixed_slurm_h is not None:  # auto select partition
                partition = cs._select_partition(fixed_slurm_h)
                p_spec = cs.partition_rules[partition]
                p_note = f"slurm_time={_format_hours(fixed_slurm_h)}\
                     ≤  {_format_hours(p_spec['max_hours'])} limit"
            else:
                partition = cs._select_partition(_cpu_max)
                p_spec = cs.partition_rules[partition]
                p_note = ""
        except ValueError:
            partition = "---"
            p_spec = {}
            p_note = "too long"

    # SLURM wall time: use user-supplied slurm_time if present, otherwise
    # compute as CPU_MAX + 10% buffer capped at the partition limit.

    if fixed_slurm_h is not None:
        slurm_h = fixed_slurm_h
    else:
        slurm_h_raw = _cpu_max * 1.1
        slurm_h = slurm_h_raw
        if partition != "---" and p_spec:
            slurm_h = min(slurm_h_raw, float(p_spec.get("max_hours", slurm_h_raw)))

    info = RichText()

    info.append(job_label, style="dim")
    info.append("\n\n")

    # OMP/MPI parallelism line with total CPU count (only when values are known)
    if sim.mpi:
        info.append(f"{mpi_label} MPI rank(s)")
        info.append("  ×  ", style="dim")
        info.append(f"{omp_label} OMP thread(s)")
        if n_omp_display is not None and n_mpi_display is not None:
            info.append(f"  =  {n_omp_display * n_mpi_display} CPUs", style="dim")
    else:
        info.append(f"{omp_label} OMP thread(s)")
        if n_omp_display is not None:
            info.append(f"  =  {n_omp_display} CPUs", style="dim")
    info.append("\n\n")

    info.append("SLURM partition:".ljust(20))
    if nbin_needs_time:
        pass
    elif partition == "---":
        info.append("[!] too long", style="red")
    else:
        info.append(f"{partition}", style="white")
        auto_tag = "" if partition_override else " [auto]"
        info.append(auto_tag, style="dim")
        if p_note:
            info.append(f"\n└─ {p_note}", style="dim")
        if fixed_slurm_h is None and slurm_h > 0:
            info.append(
                f"\n└─ slurm_time={_format_hours(slurm_h)}  (CPU_MAX + 10%)",
                style="dim",
            )
    if machine_config:
        info.append("\n\n")
        info.append("ALF Machine Config:".ljust(20))
        info.append(machine_config, style="dim")

    return RichPanel(info, title="SLURM", border_style="dim", box=RichBox.SQUARE)


# --------------------------------------------------------------------------------------
# Edit modal
# --------------------------------------------------------------------------------------


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
    #edit-title { margin-bottom: 1; color: ansi_default; }
    #edit-dialog Label { margin-top: 1; height: 1; color: ansi_default; }
    #edit-dialog Input { margin-bottom: 0; background: transparent; color: ansi_blue; border: solid ansi_default; }
    #edit-dialog Input:focus { border: solid ansi_blue; }
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
            with Horizontal(id="edit-buttons"):
                yield _ToggleButton("Save", id="btn-save", variant="primary")
                yield _ToggleButton("Save to All", id="btn-save-all", variant="warning")
                yield _ToggleButton("Cancel  [Esc]", id="btn-cancel", variant="default")

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


# --------------------------------------------------------------------------------------
# Simple monochrome ANSI theme
# --------------------------------------------------------------------------------------

_MONO_THEME = Theme(
    name="submission-mono",
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

# --------------------------------------------------------------------------------------
# Confirmation dialog
# --------------------------------------------------------------------------------------


class ConfirmSubmitScreen(ModalScreen):
    """Simple [Yes / No] confirmation before submitting jobs."""

    BINDINGS = [Binding("escape", "dismiss", show=False)]

    CSS = """
    ConfirmSubmitScreen { align: center middle; }
    #confirm-dialog {
        width: 62;
        height: auto;
        background: ansi_default;
        border: solid ansi_default;
        padding: 1 2;
    }
    #confirm-msg { color: ansi_default; margin-bottom: 1; }
    #confirm-buttons { height: 1; margin-top: 1; align: center middle; }
    #confirm-buttons Button { min-width: 9; width: auto; border: none; margin: 0 2; text-align: left; content-align: left middle; }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def on_mount(self) -> None:
        self.query_one("#btn-no").focus()

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(self._message, id="confirm-msg", markup=False)
            with Horizontal(id="confirm-buttons"):
                yield _ToggleButton("Yes", id="btn-yes", variant="primary")
                yield _ToggleButton("No", id="btn-no")

    @on(Button.Pressed, "#btn-yes")
    def _on_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#btn-no")
    def _on_no(self) -> None:
        self.dismiss(False)

    def action_dismiss(self) -> None:
        self.dismiss(False)


# --------------------------------------------------------------------------------------
# Helper widgets
# --------------------------------------------------------------------------------------

# Matches "■ " or "[dim]□ " at the start and optional "[/dim]" at the end.
_SQUARE_LEAD_RE = re.compile(r"^\[dim\]□ |^■ ")
_DIM_CLOSE_RE = re.compile(r"\[/dim\]$")


def _strip_square_markup(label: str) -> str:
    "Remove square boxes from start and end of string in toggles."
    return _DIM_CLOSE_RE.sub("", _SQUARE_LEAD_RE.sub("", label))


class _ToggleButton(Button):
    """Button that replaces its square indicator with '>' while hovered or focused."""

    _hovered: bool = False
    _focused: bool = False
    _base_label: str = ""
    _setting_hover: bool = False  # True only during internal label writes

    def __init__(self, label: str = "", **kwargs) -> None:
        super().__init__(label, **kwargs)
        self._base_label = label
        # Plain-label buttons (no ■/□ prefix) get a ■ prefix so the label text
        # never shifts between inactive (■) and active (>) states.
        if _strip_square_markup(label) == label:
            self._base_label = f"■ {label}"
            self._setting_hover = True
            self.label = f"■ {label}"  # type: ignore[assignment]
            self._setting_hover = False

    def validate_label(self, label: object) -> object:
        # Only update _base_label for external writes (not our own indicator changes).
        # Ignore non-string values: Textual internally re-sets label as a Content object
        # on layout refreshes, and str(Content) would strip markup from _base_label.
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

    # Cover both Textual naming conventions across versions.
    def on_enter(self) -> None:
        self._apply_hover()

    def on_mouse_enter(self) -> None:
        self._apply_hover()

    def on_leave(self) -> None:
        self._remove_hover()

    def on_mouse_leave(self) -> None:
        self._remove_hover()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # After a click, Textual gives keyboard focus to this button AFTER the
        # action handler runs.  on_focus then calls _sync_label → "> label",
        # overwriting the externally-set base label.  Schedule a reset to run
        # after the full event queue drains so the correct label wins.
        self.call_after_refresh(self._clear_after_click)

    def _clear_after_click(self) -> None:
        self._hovered = False
        self._focused = False
        self._sync_label()


class _DataDirInput(Input):
    """DATA_DIR input: shows a tailed path when unfocused, full path when focused."""

    _full_value: str = ""
    # Set to the tailed string just before writing it to .value.  Checked by
    # _on_data_dir_changed to distinguish display-only writes from real edits.
    # Cannot use a bool flag because Input.Changed is dispatched asynchronously
    # through the message queue — the flag would already be cleared by the time
    # the handler runs.
    _tail_display: str = ""

    def on_mount(self) -> None:
        self._full_value = self.value
        self._apply_tail()

    def _apply_tail(self) -> None:
        tailed = _tail_path(self._full_value)
        self._tail_display = tailed
        self.value = tailed
        self.cursor_position = 0  # keep "..." visible at left edge

    def on_focus(self) -> None:
        self._tail_display = ""  # clear sentinel so handler treats next change as real
        self.value = self._full_value

    def on_blur(self) -> None:
        self._full_value = self.value
        self._apply_tail()


class _CmdInput(Input):
    """Command-palette input — Escape hides the panel and returns focus to the sim table."""

    def on_key(self, event: object) -> None:
        from textual.events import Key

        if isinstance(event, Key) and event.key == "escape":
            with contextlib.suppress(Exception):
                self.app._hide_cmd()  # type: ignore[attr-defined]
            event.stop()  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------------------


class SubmissionReview(App):
    """
    Interactive TUI for reviewing and confirming ALF job submission as an array or
    single job.

    Presents a pre-submission checklist:
     - choose the backend executor (slurm / local / debug)
     - choose the end mode for the single / array job (CPU_MAX OR Nbin)
     - construct and edit job arrays
     - inspect and modify SLURM resources to commit to single / array job.
     - inspect and modify the simulation directory
     - identifies data conflicts if launching simulation in existing directories
     - identifies checkpoint restart simulations

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

    # app title which goes in the header
    TITLE = "pyALF │ Submission Review"

    # do not use textual's built in command pallete.
    ENABLE_COMMAND_PALETTE = False

    # the app CSS
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

    #body { height: 1fr; }
    #top-panel { height: 1fr; width: 100%; padding: 0 1; }
    #bottom-panel { height: 1fr; width: 100%; border-top: solid ansi_default; }
    #settings-col { width: 1fr; padding: 0 1; }
    #arch-col { width: 1fr; border-left: solid ansi_default; padding: 0 1; }

    Horizontal, Vertical, ScrollableContainer, Static, Label { background: transparent; }
    DataTable { height: 1fr; background: transparent; }
    DataTable > .datatable--header { color: ansi_default; background: transparent; text-style: none; }
    DataTable > .datatable--cursor { background: transparent; text-style: none; }
    DataTable > .datatable--hover { background: transparent; }
    DataTable > .datatable--even-row { background: transparent; }
    DataTable > .datatable--odd-row { background: transparent; }

    #progress-table { display: none; }

    #job-bar { height: 1; color: ansi_default; }
    #sim-warning { display: none; height: 2; text-align: right; padding: 0 1; }

    .hd { height: 1; margin-top: 1; color: ansi_default; }

    .sr { height: 3; }
    .sl { width: 13; height: 3; content-align: left middle; color: ansi_default; }

    Input { background: transparent; color: ansi_default; border: solid ansi_default; }
    Input:focus { border: solid ansi_default; }

    #settings-col .sr { height: 1; }
    #settings-col .sl { height: 1; width: 18; content-align: left middle; }
    #settings-col Input { height: 1; width: 1fr; border: none; padding: 0 1; background: transparent; color: ansi_blue; }
    #settings-col Input:focus { border: none; background: transparent; color: ansi_blue; }
    #settings-col Input:disabled { color: ansi_default; text-style: dim; }
    #settings-col Input .input--placeholder { color: ansi_default; text-style: dim; }
    #settings-col Input.req .input--placeholder { color: ansi_red; text-style: none; }
    #settings-col Input:disabled .input--placeholder { color: ansi_default; text-style: dim; }
    #mail-type-row Button { min-width: 10; width: auto; height: 1; border: none; margin: 0 1 0 0; text-align: left; content-align: left middle; }
    #stderr-toggle-row Button { min-width: 10; width: auto; height: 1; border: none; margin: 0 1 0 0; text-align: left; content-align: left middle; }
    #executor-col { width: 14; padding: 0 1; border-right: solid ansi_default; }
    #executor-col .hd { margin-bottom: 1; }
    #executor-divider { height: 1; width: 100%; border-top: solid ansi_default; margin-top: 1; }
    #end-by-hd { margin-top: 1; }
    #executor-col Button { min-width: 10; height: 1; border: none; width: 100%; text-align: left; content-align: left middle; }
    #data-dir-display { margin-bottom: 1; }
    #settings-col .hd { margin-bottom: 1; }
    #arch-col .hd { margin-bottom: 1; }
    .fv { height: 1; width: auto; content-align: left middle; padding: 0 1; }
    #walltime-hint { width: 14; }
    #cpu-max-hint { width: 14; }
    #walltime-input.needs-time { color: ansi_red; }
    #walltime-input.needs-time:focus { color: ansi_red; }

    #slurm-settings { display: none; }
    #slurm-divider { height: 1; width: 100%; border-top: solid ansi_default; margin-top: 1; }
    #cpu-max-row { display: none; }
    #mpi-row { display: none; }

    #nbin-row { display: none; }

    Button { border: blank; color: ansi_default; background: transparent; text-style: none; }
    Button.-primary { color: ansi_default; }
    Button.-success { color: ansi_default; }
    Button.-active { color: ansi_default; }
    Button.-primary.-active { color: ansi_default; }
    Button:hover { color: ansi_default; background: transparent; }
    Button.-primary:hover { color: ansi_default; }
    Button:disabled { color: ansi_default; text-style: dim; }

    #cmd-prefix { display: none; height: 1; width: auto; content-align: left middle; padding: 0 0 0 1; }
    #cmd-input { display: none; height: 1; border: none; padding: 0; background: transparent; color: ansi_default; width: 1fr; }
    #cmd-input:focus { border: none; background: transparent; }
    #cmd-input .input--placeholder { color: ansi_default; text-style: dim; }
    #cmd-input:disabled { color: ansi_default; text-style: dim; }

    #footer-bar { height: 1; margin-top: 1; }
    #footer-keys { width: 1fr; height: 1; content-align: left middle; padding: 0 1; }
    #footer-done { display: none; width: 1fr; height: 1; content-align: left middle; padding: 0 1; }
    #btn-submit { display: none; width: 0; }
    """

    BINDINGS = [
        Binding("s", "submit", "Submit", show=True),
        Binding("space", "toggle_sim", "Toggle", show=False),
        Binding("e", "edit_sim", "Edit", show=False),
        Binding("a", "add_sim", "Add", show=False),
        Binding("r", "remove_sim", "Remove", show=False),
        Binding(":", "focus_cmd", "Command", show=False),
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
        self.open_monitor: bool = False
        self.submitted_cs: ClusterSubmitter | None = None
        self.session_path: Path | None = None
        # End criterion: read initial state from first sim's dict
        _first_sd = _sim_dict_of(self._sims[0]) if self._sims else {}
        _nbin_val = _first_sd.get("NBin") or _first_sd.get("Nbin")
        self._end_mode: str = "nbin" if _nbin_val else "cpu_max"
        self._nbin_initial: str = str(_nbin_val) if _nbin_val else "40"
        # Track whether slurm_time was provided by the caller so we can clean it
        # up if the user adds it via the walltime input during NBin mode and then
        # switches back to CPU_MAX mode.
        self._slurm_time_was_preset: bool = "slurm_time" in (cs.slurm_kwargs or {})
        # Snapshots for !reset — captured once at construction.
        self._initial_end_mode: str = self._end_mode
        self._initial_sim_dicts: list[dict] = [
            copy.deepcopy(_sim_dict_of(s)) for s in self._sims
        ]
        self._initial_param_keys: list[str] = list(self._param_keys)
        self._initial_param_headers: list[str] = list(self._param_headers)
        self._initial_omp: str = str(self._sims[0].n_omp) if self._sims else "1"
        self._initial_mpi: str = str(self._sims[0].n_mpi) if self._sims else "1"
        self._initial_data_dir: str = _common_sim_data_dir(self._sims)
        # Manual overrides for partition and OMP (arch-col inputs)
        self._manual_partition: str | None = None
        self._manual_omp: int | None = None
        self._manual_nbin: bool = False
        self._edit_mode: bool = False
        self._dt_cursor_row: int = -1
        self._dt_hover_row: int = -1
        # Per-sim path suffixes relative to the common data dir prefix (absolute).
        # Used to propagate DATA_DIR edits to individual sim.sim_dir values.
        _sim_dirs_abs = [str(Path(str(s.sim_dir)).resolve()) for s in self._sims]
        _prefix_abs = (
            str(Path(os.path.commonpath(_sim_dirs_abs))) if _sim_dirs_abs else ""
        )
        self._data_dir_suffixes: list[str] = [
            d[len(_prefix_abs) :] for d in _sim_dirs_abs
        ]

    # Layout ---------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(f"  {self.TITLE}", id="title-bar")
        with Vertical(id="body"):
            # top: table ---------------------------------------------------------------
            with Vertical(id="top-panel"):
                yield DataTable(
                    id="sim-table",
                    zebra_stripes=True,
                    cursor_type="row",
                    cursor_foreground_priority="renderable",
                )
                yield DataTable(
                    id="progress-table", cursor_type="none", zebra_stripes=True
                )
                yield Static("", id="sim-warning")
                yield Static("", id="job-bar")
            # bottom: executor / end | settings | summary ------------------------------
            with Horizontal(id="bottom-panel"):
                with Vertical(id="executor-col"):
                    # bottom left panel: executor --------------------------------------
                    yield Label("Executor", classes="hd")
                    for ex in ("slurm", "local", "debug"):
                        disabled = ex == "slurm" and not self._can_use_slurm
                        yield _ToggleButton(
                            _btn_label(ex, ex == self._executor_state),
                            id=f"exec-{ex}",
                            disabled=disabled,
                        )
                    yield Static("", id="executor-divider")  # divider
                    # bottom left panel: end by ----------------------------------------
                    yield Label("End by", classes="hd", id="end-by-hd")
                    yield _ToggleButton(
                        _btn_label("CPU_MAX", self._end_mode == "cpu_max"),
                        id="end-by-cpu-max",
                    )
                    yield _ToggleButton(
                        _btn_label("NBin", self._end_mode == "nbin"),
                        id="end-by-nbin",
                    )
                # bottom middle: settings ----------------------------------------------
                with ScrollableContainer(id="settings-col"):
                    # bottom middle: settings ------------------------------------------
                    yield Label("Settings", classes="hd")
                    with Horizontal(classes="sr"):
                        yield Label("DATA_DIR", classes="sl")
                        yield _DataDirInput(
                            _common_sim_data_dir(self._sims),
                            id="data-dir-display",
                            placeholder="[enter]",
                            classes="req",
                        )
                    with Horizontal(classes="sr"):
                        yield Label("OMP", classes="sl")
                        yield Input(
                            str(self._sims[0].n_omp) if self._sims else "1",
                            id="omp-input",
                            placeholder="[enter]",
                            classes="req",
                            restrict=r"\d*",
                        )
                        yield Static("", id="omp-violation", classes="fv")
                    with Horizontal(classes="sr", id="mpi-row"):
                        yield Label("MPI ranks", classes="sl")
                        yield Input(
                            str(self._sims[0].n_mpi) if self._sims else "1",
                            id="mpi-input",
                            placeholder="[enter]",
                            classes="req",
                            restrict=r"\d*",
                        )
                    with Horizontal(classes="sr"):
                        yield Label("Memory", classes="sl")
                        yield Input(
                            self._cs.slurm_mem or "",
                            id="mem-input",
                            placeholder="[enter]",
                            classes="req",
                        )
                        yield Static("", id="mem-violation", classes="fv")
                    with Horizontal(classes="sr", id="cpu-max-row"):
                        yield Label("CPU_MAX", classes="sl")
                        yield Input(
                            "",
                            id="cpu-max-input",
                            placeholder="[enter]",
                            classes="req",
                            restrict=r"\d{0,4}(:\d{0,2}(:\d{0,2})?)?",
                        )
                        yield Static("", id="cpu-max-hint", classes="fv")
                    with Horizontal(classes="sr", id="nbin-row"):
                        yield Label("NBin", classes="sl")
                        yield Input(
                            self._nbin_initial,
                            id="nbin-input",
                            placeholder="[enter]",
                            classes="req",
                            restrict=r"\d*",
                        )
                    # bottom middle: SLURM settings ------------------------------------
                    with Vertical(id="slurm-settings"):
                        yield Static("", id="slurm-divider")
                        yield Label("SLURM", classes="hd")
                        with Horizontal(classes="sr"):
                            yield Label("Job name", classes="sl")
                            yield Input(
                                self._cs.job_name or "",
                                id="job-name-input",
                                placeholder="[auto]",
                            )
                        with Horizontal(classes="sr"):
                            yield Label("WCKey", classes="sl")
                            yield Input(
                                self._cs.wckey or "",
                                id="wckey-input",
                                placeholder="[enter]",
                            )
                        with Horizontal(classes="sr"):
                            yield Label("Mail type", classes="sl")
                            with Horizontal(id="mail-type-row"):
                                for tag in ("END", "FAIL", "ALL"):
                                    yield _ToggleButton(
                                        _btn_label(tag, tag in self._mail_flags),
                                        id=f"mail-{tag.lower()}",
                                    )
                        with Horizontal(classes="sr", id="stderr-row"):
                            yield Label("STDERR → STDOUT", classes="sl")
                            with Horizontal(id="stderr-toggle-row"):
                                for val, label in (
                                    ("true", "true"),
                                    ("false", "false"),
                                ):
                                    active = (
                                        val == "true"
                                    ) == self._cs.stderr_to_stdout
                                    yield _ToggleButton(
                                        _btn_label(label, active),
                                        id=f"stderr-{val}",
                                    )
                        with Horizontal(classes="sr", id="walltime-row"):
                            yield Label("SLURM time", classes="sl")
                            yield Input(
                                "",
                                id="walltime-input",
                                placeholder="[enter]",
                                classes="req",
                                restrict=r"\d{0,4}(:\d{0,2}(:\d{0,2})?)?",
                            )
                            yield Static("", id="walltime-hint", classes="fv")
                        with Horizontal(classes="sr", id="partition-row"):
                            yield Label("SLURM partition", classes="sl")
                            yield Input("", id="partition-input", placeholder="[auto]")
                with ScrollableContainer(id="arch-col"):
                    yield Label("Summary", classes="hd")
                    yield Static("", id="arch-panel")
                    yield Static("", id="schedule-panel")

        # ── footer row: key hints / vim command line / submit button ──
        with Horizontal(id="footer-bar"):
            yield Static("", id="footer-keys")
            yield Static("", id="footer-done")
            yield Static(":", id="cmd-prefix")
            yield _CmdInput("", id="cmd-input", placeholder="")
            yield Button("", id="btn-submit")

    def on_mount(self) -> None:
        self.register_theme(_MONO_THEME)
        self.theme = "submission-mono"
        _cpu_max = self._end_mode == "cpu_max"
        self.query_one("#nbin-row").display = self._end_mode == "nbin"
        self.query_one("#cpu-max-row").display = _cpu_max
        self.query_one("#mpi-row").display = any(
            getattr(s, "mpi", False) for s in self._sims
        )
        _slurm = self._executor_state == "slurm"
        self.query_one("#slurm-settings").display = _slurm
        if _slurm:
            self.query_one("#walltime-input", Input).disabled = _cpu_max
        # In NBin mode, ensure CPU_MAX=0 in all sim_dicts so ALF does not use a
        # time-based stopping criterion alongside the bin count.
        if self._end_mode == "nbin":
            for sim in self._sims:
                _sim_dict_of(sim)["CPU_MAX"] = 0
        self._setup_table()
        self._refresh_all()

    # table detailing the job array ----------------------------------------------------

    def _setup_table(self) -> None:
        table = self.query_one("#sim-table", DataTable)
        table.add_column("", key="cur", width=3)
        table.add_column("", key="sel", width=3)
        table.add_column("#", key="idx", width=4)
        table.add_column("Hamiltonian", key="ham")
        for key, hdr in zip(self._param_keys, self._param_headers):
            table.add_column(hdr, key=key)
        self._repopulate_table()

    def _repopulate_table(self) -> None:
        table = self.query_one("#sim-table", DataTable)
        saved_row = table.cursor_row
        table.clear()
        for i, sim in enumerate(self._sims):
            sd = _sim_dict_of(sim)
            mark = _sim_mark(self._selected[i], _data_conflict(sim))
            row = ["   ", mark, str(i), sim.ham_name]
            for key in self._param_keys:
                row.append(str(sd.get(key, "—")))
            table.add_row(*row, key=str(i))
        if self._sims:
            table.move_cursor(row=min(saved_row, len(self._sims) - 1))

    def _rebuild_table_with_columns(self) -> None:
        """Full table rebuild including columns — use when _param_keys has changed."""
        table = self.query_one("#sim-table", DataTable)
        saved_row = table.cursor_row
        table.clear(columns=True)
        table.add_column("", key="cur", width=3)
        table.add_column("", key="sel", width=3)
        table.add_column("#", key="idx", width=4)
        table.add_column("Hamiltonian", key="ham")
        for key, hdr in zip(self._param_keys, self._param_headers):
            table.add_column(hdr, key=key)
        for i, sim in enumerate(self._sims):
            sd = _sim_dict_of(sim)
            mark = _sim_mark(self._selected[i], _data_conflict(sim))
            row = ["   ", mark, str(i), sim.ham_name]
            for key in self._param_keys:
                row.append(str(sd.get(key, "—")))
            table.add_row(*row, key=str(i))
        if self._sims:
            table.move_cursor(row=min(saved_row, len(self._sims) - 1))

    # refreshing panels ----------------------------------------------------------------

    def _refresh_all(self) -> None:
        if not self._sims:
            return
        self._refresh_job_bar()
        self._refresh_right_panel()
        self._refresh_submit_button()

    def _refresh_job_bar(self) -> None:
        n_sel = sum(self._selected)
        n_total = len(self._sims)
        if n_sel == 0:
            text = f"{n_sel}/{n_total} selected"
        elif n_sel > 1:
            text = f"{n_sel}/{n_total} selected  ·  array job"
        else:
            text = f"{n_sel}/{n_total} selected  ·  single job"
        self.query_one("#job-bar", Static).update(f"[dim]{text}[/dim]")

    def _refresh_right_panel(self) -> None:
        if not self._sims or self._focused_idx >= len(self._sims):
            return
        sim = self._sims[self._focused_idx]
        sd = _sim_dict_of(sim)
        timeout_h = max(0.0, float(sd.get("CPU_MAX", 24)))
        mem = self.query_one("#mem-input", Input).value or (self._cs.slurm_mem or "")

        omp_input = self.query_one("#omp-input", Input)
        try:
            _omp_disp: int | None = int(omp_input.value)
            if _omp_disp < 1:
                _omp_disp = None
        except ValueError:
            _omp_disp = None
        _mpi_disp: int | None = None
        if any(getattr(s, "mpi", False) for s in self._sims):
            try:
                _mpi_disp = int(self.query_one("#mpi-input", Input).value)
                if _mpi_disp < 1:
                    _mpi_disp = None
            except ValueError:
                pass
        _job_name = self.query_one("#job-name-input", Input).value.strip() or None
        _wckey = self.query_one("#wckey-input", Input).value.strip() or None
        self.query_one("#arch-panel", Static).update(
            _arch_renderable(
                sim,
                self._executor_state,
                self._cs,
                mem_display=mem,
                partition_override=self._manual_partition,
                machine_config=getattr(sim, "config", "") or "—",
                n_omp_display=_omp_disp,
                n_mpi_display=_mpi_disp,
                job_name=_job_name,
                wckey=_wckey,
            )
        )

        # Inline violation indicators next to OMP and Memory inputs.
        p_spec: dict = {}
        if self._executor_state == "slurm" and self._cs.partition_rules:
            fixed_slurm_time = (self._cs.slurm_kwargs or {}).get("slurm_time")
            if isinstance(fixed_slurm_time, int):
                wall_h: float = fixed_slurm_time / 60.0
            elif fixed_slurm_time:
                wall_h = _parse_slurm_time_hours(fixed_slurm_time) or 0.0
            else:
                wall_h = max(0.0, float(_sim_dict_of(self._sims[0]).get("CPU_MAX", 24)))
            with contextlib.suppress(Exception):
                part = self._manual_partition or self._cs._select_partition(wall_h)
                p_spec = self._cs.partition_rules.get(part, {})

        n_ranks = sim.n_mpi if sim.mpi else 1
        try:
            n_omp_val = int(omp_input.value)
        except (ValueError, TypeError):
            n_omp_val = sim.n_omp
        total_cpus = n_ranks * n_omp_val
        max_cpus: int | None = p_spec.get("max_cpus")
        max_mem_gb: float | None = p_spec.get("max_mem_gb")

        omp_stat = self.query_one("#omp-violation", Static)
        if max_cpus is not None:
            if total_cpus > max_cpus:
                omp_stat.update(f"[ansi_red]{total_cpus} / {max_cpus} ✗[/ansi_red]")
            else:
                omp_stat.update(f"[dim]{total_cpus} / {max_cpus}[/dim]")
        else:
            omp_stat.update("")

        mem_stat = self.query_one("#mem-violation", Static)
        if max_mem_gb is not None:
            mem_over = False
            with contextlib.suppress(ValueError):
                if mem:
                    mem_over = _parse_mem_gb(mem) > max_mem_gb
            limit_str = f"{max_mem_gb:g} GB"
            mem_stat.update(
                f"[ansi_red]/ {limit_str} ✗[/ansi_red]"
                if mem_over
                else f"[dim]/ {limit_str}[/dim]"
            )
        else:
            mem_stat.update("")

        wt_input = self.query_one("#walltime-input", Input)
        fixed_slurm_time = (self._cs.slurm_kwargs or {}).get("slurm_time")
        nbin_needs_time = self._end_mode == "nbin" and not fixed_slurm_time
        wt_input.set_class(nbin_needs_time, "needs-time")
        if self._end_mode == "cpu_max":
            wt_input.placeholder = "auto  <<<  CPU_MAX + 10%"
            if not wt_input.has_focus:
                wt_input.value = ""
                array_timeout_h = max(
                    0.0, float(_sim_dict_of(self._sims[0]).get("CPU_MAX", 0))
                )
                if array_timeout_h > 0:
                    cpu_max_input = self.query_one("#cpu-max-input", Input)
                    if not cpu_max_input.has_focus:
                        cpu_max_input.value = _hours_to_hms(array_timeout_h)
        else:
            wt_input.placeholder = "[enter]"
            if not wt_input.has_focus and fixed_slurm_time:
                wt_input.value = (
                    _hours_to_hms(fixed_slurm_time / 60)
                    if isinstance(fixed_slurm_time, int)
                    else fixed_slurm_time
                )

        nbin_target = sd.get("NBin") or sd.get("Nbin")
        if nbin_target:
            if timeout_h > 0:
                completes_line = (
                    f"CPU_MAX end ~: {_completion_str(timeout_h)}"
                    f"  [dim](NBin={nbin_target} may finish earlier)[/dim]"
                )
            else:
                completes_line = ""  # pure NBin mode — End by row already shows this
        else:
            completes_line = f"Completes ~: {_completion_str(timeout_h)}"
        self.query_one("#schedule-panel", Static).update(completes_line)

        sim_warning = self.query_one("#sim-warning", Static)
        if self._edit_mode and _data_conflict(sim):
            sim_warning.display = True
            selected = self._selected[self._focused_idx]
            if selected:
                sim_warning.update(
                    "[ansi_red][!] data.h5 exists — no checkpoint files — submission blocked[/ansi_red]\n"
                    "space deselect"
                    "  ·  delete data.h5 for a fresh run"
                    "  ·  add confin_* to restart from a checkpoint"
                )
            else:
                sim_warning.update(
                    "[!] data conflict — sim deselected, not submitted\n"
                    "resolve before reselecting:"
                    "  delete data.h5 for a fresh run"
                    "  ·  add confin_* to restart from a checkpoint"
                )
        else:
            sim_warning.display = False
            sim_warning.update("")

    def _needs_slurm_time(self) -> bool:
        """True when any selected sim has CPU_MAX=0 and no slurm_time is configured."""
        if self._executor_state != "slurm":
            return False
        if (self._cs.slurm_kwargs or {}).get("slurm_time") is not None:
            return False
        return any(
            sel and float(_sim_dict_of(sim).get("CPU_MAX", 1)) <= 0
            for sim, sel in zip(self._sims, self._selected)
        )

    def _needs_data_dir(self) -> bool:
        inp = self.query_one("#data-dir-display", _DataDirInput)
        return not (inp._full_value or inp.value).strip()

    def _needs_omp(self) -> bool:
        val = self.query_one("#omp-input", Input).value.strip()
        try:
            return int(val) < 1
        except ValueError:
            return True

    def _needs_mem(self) -> bool:
        return not self.query_one("#mem-input", Input).value.strip()

    def _needs_nbin(self) -> bool:
        if self._end_mode != "nbin":
            return False
        val = self.query_one("#nbin-input", Input).value.strip()
        try:
            return int(val) < 1
        except ValueError:
            return True

    def _needs_cpu_max(self) -> bool:
        if self._end_mode != "cpu_max":
            return False
        val = self.query_one("#cpu-max-input", Input).value
        return _parse_wall_time(val) is None

    def _needs_mpi(self) -> bool:
        if not any(getattr(s, "mpi", False) for s in self._sims):
            return False
        val = self.query_one("#mpi-input", Input).value.strip()
        try:
            return int(val) < 1
        except ValueError:
            return True

    def _missing_required_fields(self) -> list[str]:
        missing = []
        if self._needs_data_dir():
            missing.append("DATA_DIR")
        if self._needs_omp():
            missing.append("OMP")
        if self._needs_mem():
            missing.append("Memory")
        if self._needs_mpi():
            missing.append("MPI ranks")
        if self._needs_nbin():
            missing.append("NBin")
        if self._needs_cpu_max():
            missing.append("CPU_MAX")
        if self._needs_slurm_time():
            missing.append("SLURM time")
        return missing

    def _has_unfit_partition(self) -> bool:
        """True when any selected sim exceeds a partition wall-time or node resource limit."""
        if self._executor_state != "slurm" or not self._cs.partition_rules:
            return False
        mem_str = self.query_one("#mem-input", Input).value or (
            self._cs.slurm_mem or ""
        )
        _raw_st = (self._cs.slurm_kwargs or {}).get("slurm_time")
        try:
            _slurm_time_h: float | None = (
                _slurm_time_to_minutes(_raw_st) / 60 if _raw_st is not None else None
            )
        except ValueError:
            _slurm_time_h = None
        for sim, sel in zip(self._sims, self._selected):
            if not sel:
                continue
            _c = float(_sim_dict_of(sim).get("CPU_MAX", 0))
            timeout_h = (
                _slurm_time_h if _slurm_time_h is not None else (_c if _c > 0 else 24.0)
            )
            if self._manual_partition:
                partition = self._manual_partition
                # Manual partition may not be in partition_rules — skip limit check
                # (limits are unknown; ClusterSubmitter will enforce at submit time).
                if partition not in self._cs.partition_rules:
                    continue
            else:
                try:
                    partition = self._cs._select_partition(timeout_h)
                except ValueError:
                    return True
            try:
                self._cs._check_node_fit(sim, partition, slurm_mem=mem_str or None)
            except ValueError:
                return True
        return False

    def _count_selected_data_conflicts(self) -> int:
        """Number of selected sims whose sim_dir has data but no checkpoint files."""
        return sum(
            1
            for sim, sel in zip(self._sims, self._selected)
            if sel and _data_conflict(sim)
        )

    def _refresh_submit_button(self) -> None:
        n_sel = sum(self._selected)
        btn = self.query_one("#btn-submit", Button)
        if (
            self._submitted_result is not None
            or self._is_submitting
            or n_sel == 0
            or self._edit_mode
            or self._count_selected_data_conflicts() > 0
            or self._missing_required_fields()
            or self._has_unfit_partition()
        ):
            btn.disabled = True
        else:
            btn.disabled = False
        footer_keys = self.query_one("#footer-keys", Static)
        schedule = self.query_one("#schedule-panel", Static)
        if self._submitted_result is not None or self._is_submitting:
            footer_keys.update("")
            schedule.update("")
        elif self._edit_mode:
            footer_keys.update(f"---JOB EDIT---  │  {_FOOTER_EDIT}")
            schedule.update("")
        elif btn.disabled:
            footer_keys.update(
                f"[ansi_red][!] missing information[/ansi_red]  │  {_FOOTER_VIEW}"
            )
            schedule.update("")
        else:
            footer_keys.update(f">>> ready to submit  │  {_FOOTER_VIEW}")
            completion = ""
            if self._sims:
                _sd0 = _sim_dict_of(self._sims[0])
                _cpu_h = max(0.0, float(_sd0.get("CPU_MAX", 0)))
                _fixed = (self._cs.slurm_kwargs or {}).get("slurm_time")
                if _fixed:
                    _cpu_h = (
                        (_fixed / 60)
                        if isinstance(_fixed, int)
                        else (_parse_slurm_time_hours(_fixed) or 0)
                    )
                if _cpu_h > 0:
                    completion = f"Completes ~ {_completion_str(_cpu_h)}"
            schedule.update(completion)

    def _set_executor(self, executor: str) -> None:
        self._executor_state = executor
        for ex in ("slurm", "local", "debug"):
            self.query_one(f"#exec-{ex}", Button).label = _btn_label(ex, ex == executor)
        _slurm = executor == "slurm"
        _cpu_max = self._end_mode == "cpu_max"
        self.query_one("#slurm-settings").display = _slurm
        self.query_one("#cpu-max-row").display = _cpu_max
        if _slurm:
            self.query_one("#walltime-input", Input).disabled = _cpu_max
        self._refresh_right_panel()
        self._refresh_submit_button()

    # submit / cancel jobs -------------------------------------------------------------

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
        self._lock_ui()
        self._show_progress_view(selected_sims)
        job_properties: dict = {}
        if self._manual_partition:
            job_properties["slurm_partition"] = self._manual_partition
        self._run_submission(cs, selected_sims, job_properties or None)

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
                str(i), Path(str(sim.sim_dir)).name, _status_label("QUEUED"), key=str(i)
            )
        self.query_one("#job-bar", Static).update(f"Submitting {len(sims)} job(s)...")

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
        parts = [f"{n_ok}/{n_total} submitted"]
        if n_fail:
            parts.append(f"[ansi_red]{n_fail} failed/cancelled[/ansi_red]")
        array_parents = {jid.split("_")[0] for jid in job_ids if "_" in jid}
        if len(array_parents) == 1:
            parts.append(f"[dim]array: {array_parents.pop()}[/dim]")
        if self.submitted_cs is not None:
            _jn = self.submitted_cs.job_name
            if _jn:
                parts.append(f"[dim]job: {_jn}[/dim]")
            elif submitted:
                parts.append(f"[dim]job: {submitted[0].ham_name} (auto)[/dim]")
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
        self.query_one("#footer-keys", Static).display = False
        label = RichText()
        if self._submitted_result and self._executor_state == "slurm":
            label.append("[m]")
            label.append(" open monitor ")
        label.append("[q]")
        label.append(" exit ")
        done = self.query_one("#footer-done", Static)
        done.display = True
        done.update(label)

    @work(thread=True)
    def _run_submission(
        self,
        cs: ClusterSubmitter,
        sims: list[Simulation],
        job_properties: dict | None = None,
    ) -> None:
        # Snapshot existing job IDs so we can identify which sims are newly submitted.
        old_jids: dict[str, str | None] = {}
        for sim in sims:
            jid_file = Path(str(sim.sim_dir)) / "jobid.txt"
            old_jids[str(sim.sim_dir)] = (
                jid_file.read_text().strip() if jid_file.exists() else None
            )

        for i in range(len(sims)):
            self.call_from_thread(self._set_progress, i, "PREPARING")

        try:
            jobs = cs.submit(
                sims,
                job_properties=job_properties,
                confirm_checkpoint=False,
                write_session=False,
            )
        except Exception as exc:
            for i in range(len(sims)):
                self.call_from_thread(self._set_progress, i, "FAILED", str(exc))
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
                    self.call_from_thread(self._set_progress, i, "SUBMITTED", new_jid)
                    submitted.append(sim)
                    job_ids.append(new_jid)
                    continue
            self.call_from_thread(self._set_progress, i, "SKIPPED")

        self.call_from_thread(self._on_all_submitted, submitted, len(sims), job_ids)

    def _effective_cs(self) -> ClusterSubmitter:
        mem = self.query_one("#mem-input", Input).value.strip() or (
            self._cs.slurm_mem or ""
        )
        submit_dir = str(self._cs.submit_dir)
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

    # actions --------------------------------------------------------------------------

    def action_submit(self) -> None:
        if self._is_submitting or self._submitted_result is not None:
            return
        n_sel = sum(self._selected)
        if n_sel == 0:
            self.notify("No simulations selected.", severity="warning")
            return
        if self._count_selected_data_conflicts() > 0:
            self.notify(
                "[!] Cannot submit — deselect sims with existing data or add"
                " checkpoint files first.",
                severity="error",
            )
            return
        if missing := self._missing_required_fields():
            self.notify(
                f"[1] Cannot submit — fill in: {', '.join(missing)}",
                severity="error",
            )
            return
        if self._has_unfit_partition():
            self.notify(
                "Cannot submit — a selected job exceeds all configured partitions.",
                severity="error",
            )
            return
        checkpoint_sims = [
            s
            for s, sel in zip(self._sims, self._selected)
            if sel
            and (
                any(Path(str(s.sim_dir)).glob("confin_*"))
                or any(Path(str(s.sim_dir)).glob("confout_*"))
            )
        ]
        msg = f"Submit {n_sel} simulation(s)?"
        if checkpoint_sims:
            has_confout = any(
                any(Path(str(s.sim_dir)).glob("confout_*")) for s in checkpoint_sims
            )
            msg += f"\n\n[!] {len(checkpoint_sims)} checkpoint restart(s) detected.\n"
            msg += "    ALF will append new measurement bins to the existing\n"
            msg += "    data.h5 rather than starting a fresh run.\n"
            if has_confout:
                msg += "\n    confout_* files will be auto-renamed to confin_*\n"
                msg += "    by ALF on submit — no manual action needed.\n"
            msg += "\n    To start fresh instead: delete the conf* files from\n"
            msg += "    the sim directory before submitting.\n"
            for sim in checkpoint_sims:
                sim_path = Path(str(sim.sim_dir))
                name = sim_path.name
                if len(name) > 44:
                    name = name[:41] + "…"
                msg += f"\n  ↳ {name}/\n"
                confin_files = sorted(f.name for f in sim_path.glob("confin_*"))
                confout_files = sorted(f.name for f in sim_path.glob("confout_*"))
                entries = list(confin_files)
                for fname in confout_files:
                    confin_name = fname.replace("confout_", "confin_", 1)
                    entries.append(f"{fname}  →  {confin_name}")
                shown = entries[:6]
                total = len(entries)
                for i, entry in enumerate(shown):
                    connector = "└──" if (i == len(shown) - 1 and total <= 6) else "├──"
                    msg += f"    {connector} {entry}\n"
                if total > 6:
                    msg += f"    └── … ({total - 6} more)\n"

        def _confirmed(result: bool | None) -> None:
            if result:
                self._do_submit()

        self.push_screen(ConfirmSubmitScreen(msg), _confirmed)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        locked = self._is_submitting or self._submitted_result is not None
        if locked and action in (
            "toggle_sim",
            "edit_sim",
            "add_sim",
            "remove_sim",
            "submit",
            "focus_cmd",
        ):
            return False
        if self._edit_mode and action == "submit":
            return False
        return self._edit_mode or action not in (
            "toggle_sim",
            "edit_sim",
            "add_sim",
            "remove_sim",
        )

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
            edit_keys = _all_param_keys([sim])

            def on_done(result: object) -> None:
                if result is None:
                    return
                if result == "all":
                    sd_src = _sim_dict_of(sim)
                    for s in self._sims:
                        if s is not sim:
                            sd_dst = _sim_dict_of(s)
                            for k in edit_keys:
                                if k in sd_src:
                                    sd_dst[k] = sd_src[k]
                    self.notify(f"Applied parameters to all {len(self._sims)} sims.")
                self._repopulate_table()
                self._refresh_right_panel()
                self._refresh_end_mode_display()

            self.push_screen(
                EditSimScreen(sim, edit_keys),
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

    def action_remove_sim(self) -> None:
        table = self.query_one("#sim-table", DataTable)
        row = table.cursor_row
        if not (0 <= row < len(self._sims)):
            return
        if len(self._sims) == 1:
            self.notify("Cannot remove the last simulation.", severity="warning")
            return
        del self._sims[row]
        del self._selected[row]
        self._dt_cursor_row = min(row, len(self._sims) - 1)
        self._focused_idx = self._dt_cursor_row
        self._repopulate_table()
        self._refresh_job_bar()
        self._refresh_submit_button()

    def _show_cmd(self) -> None:
        self.query_one("#footer-keys", Static).display = False
        self.query_one("#footer-done", Static).display = False
        self.query_one("#cmd-prefix", Static).display = True
        cmd = self.query_one("#cmd-input", Input)
        cmd.display = True
        cmd.focus()

    def _hide_cmd(self) -> None:
        self.query_one("#cmd-prefix", Static).display = False
        cmd = self.query_one("#cmd-input", Input)
        cmd.display = False
        cmd.clear()
        with contextlib.suppress(Exception):
            self.query_one("#sim-table", DataTable).focus()
        if self._submitted_result is not None or self._is_submitting:
            self.query_one("#footer-done", Static).display = True
        else:
            self.query_one("#footer-keys", Static).display = True

    def action_focus_cmd(self) -> None:
        self._show_cmd()

    def _set_edit_mode(self, enabled: bool) -> None:
        self._edit_mode = enabled
        self._repopulate_table()
        self._refresh_submit_button()
        bottom = self.query_one("#bottom-panel")
        for btn in bottom.query(_ToggleButton):
            btn._hovered = False
            btn._focused = False
            btn._sync_label()
        bottom.display = not enabled
        for w in bottom.query("Input"):
            w.disabled = enabled
        table = self.query_one("#sim-table", DataTable)
        if enabled:
            with contextlib.suppress(Exception):
                table.update_cell(str(self._dt_cursor_row), "cur", ">")
        table.focus()

    def _exec_command(self, raw: str) -> None:
        raw = raw.strip()
        if not raw.startswith("!"):
            self.notify("Commands must start with '!'", severity="warning")
            return
        parts = raw[1:].split(None, 1)
        if not parts:
            return
        cmd_raw = parts[0].lower()
        val = parts[1].strip().lower() if len(parts) > 1 else ""
        cmd = _CMD_ALIASES.get(cmd_raw, cmd_raw)

        if cmd_raw == "reset":
            self._cmd_reset()
            return
        if cmd_raw == "edit" and val.lower() == "jobs":
            self._set_edit_mode(True)
            return

        if cmd in _CMD_INPUT_ID:
            inp = self.query_one(_CMD_INPUT_ID[cmd], Input)
            inp.value = "" if val == "--clear" else val
            return

        if cmd == "executor":
            if val == "--clear":
                self._set_executor(self._cs.executor)
            elif val in ("slurm", "local", "debug"):
                self._set_executor(val)
            else:
                self.notify(
                    "executor must be slurm, local, or debug", severity="warning"
                )
            return

        if cmd == "end":
            if val in ("cpu_max", "nbin"):
                self._set_end_mode(val)
            elif val == "--clear":
                self._set_end_mode(self._end_mode)
            else:
                self.notify("end must be cpu_max or nbin", severity="warning")
            return

        if cmd == "stderr":
            if val.lower() in ("true", "1", "yes"):
                self._set_stderr(True)
            elif val.lower() in ("false", "0", "no", "--clear"):
                self._set_stderr(False)
            else:
                self.notify("stderr must be true or false", severity="warning")
            return

        if cmd == "mail":
            _VALID_FLAGS = {"END", "FAIL", "ALL"}
            if not val or val in ("--clear", "none"):
                self._set_mail_flags(set())
            else:
                tokens = {t.upper() for t in val.split()}
                unknown = tokens - _VALID_FLAGS
                if unknown:
                    self.notify(
                        f"!mail: unknown flag(s): {', '.join(sorted(unknown))}. "
                        "Valid: END, FAIL, ALL, none",
                        severity="warning",
                    )
                    return
                if "ALL" in tokens and len(tokens) > 1:
                    self.notify(
                        "!mail: ALL is exclusive with END and FAIL", severity="warning"
                    )
                    return
                self._set_mail_flags(tokens)
            return

        self.notify(f"Unknown command: !{cmd_raw}", severity="warning")

    def _cmd_reset(self) -> None:
        if self._edit_mode:
            self._set_edit_mode(False)

        # Restore sim dicts to their original state (before any end-mode mutations).
        for sim, initial_sd in zip(self._sims, self._initial_sim_dicts):
            sd = _sim_dict_of(sim)
            sd.clear()
            sd.update(copy.deepcopy(initial_sd))

        # Restore selection and column layout.
        self._selected = [True] * len(self._sims)
        self._param_keys = list(self._initial_param_keys)
        self._param_headers = list(self._initial_param_headers)

        # Reset manual override flags.
        self._manual_partition = None
        self._manual_omp = None
        self._manual_nbin = False

        # Reset mail flags.
        self._mail_flags = set((self._cs.mail_type or "").upper().split())
        for tag in ("END", "FAIL", "ALL"):
            self.query_one(f"#mail-{tag.lower()}", Button).label = _btn_label(
                tag, tag in self._mail_flags
            )

        # Reset all inputs.
        self.query_one("#omp-input", Input).value = self._initial_omp
        self.query_one("#mpi-input", Input).value = self._initial_mpi
        self.query_one("#nbin-input", Input).value = self._nbin_initial
        self.query_one("#cpu-max-input", Input).value = ""
        self.query_one("#walltime-input", Input).value = ""
        self.query_one("#partition-input", Input).value = ""
        self.query_one("#mem-input", Input).value = self._cs.slurm_mem or ""
        self.query_one("#job-name-input", Input).value = self._cs.job_name or ""
        self.query_one("#wckey-input", Input).value = self._cs.wckey or ""
        _dd = self.query_one("#data-dir-display", _DataDirInput)
        _dd._full_value = self._initial_data_dir
        _dd._apply_tail()

        # Restore executor, stderr, and end mode (reads from restored sim dicts).
        self._set_executor(self._cs.executor)
        self._set_stderr(self._cs.stderr_to_stdout)
        self._set_end_mode(self._initial_end_mode)

        # Rebuild the table with the correct columns and repopulate.
        self._rebuild_table_with_columns()
        self._refresh_all()
        self.notify("Settings reset.")

    def run_with_monitor(self, refresh_interval: float = 30.0) -> None:
        """Run the submission TUI, then open the monitor automatically if requested.

        Replaces the ``app.run()`` / ``if app.open_monitor`` pattern that the
        caller would otherwise have to write manually.

        The monitor is launched in a fresh subprocess so that the terminal is
        fully reset between the two Textual apps — this fixes the blank-screen
        issue that occurs when running sequentially in SSH environments.
        """
        import subprocess
        import sys

        self.run()
        if self.open_monitor and self.session_path:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from py_alf.monitor import SimulationMonitor; "
                        f"SimulationMonitor.from_session({str(self.session_path)!r},"
                        f" refresh_interval={refresh_interval}).run()"
                    ),
                ],
                check=False,
            )

    def action_quit_cancel(self) -> None:
        if self._edit_mode:
            self._set_edit_mode(False)
        else:
            self.exit(self._submitted_result)

    def action_open_monitor(self) -> None:
        if self._submitted_result is None or self._executor_state != "slurm":
            return
        self.open_monitor = True
        self.exit(self._submitted_result)

    # events ---------------------------------------------------------------------------

    def on_key(self, event: object) -> None:
        from textual.events import Key

        if (
            isinstance(event, Key)
            and event.key == "escape"
            and isinstance(self.focused, Input)
        ):
            with contextlib.suppress(Exception):
                self.query_one("#sim-table", DataTable).focus()
            event.stop()  # type: ignore[attr-defined]

    def _refresh_end_mode_display(self) -> None:
        """
        Sync toggle buttons and NBin/CPU_MAX rows from the focused sim — no sim_dict
        writes.
        """
        if not self._sims or self._focused_idx >= len(self._sims):
            return
        sim = self._sims[self._focused_idx]
        sd = _sim_dict_of(sim)
        # Sync OMP from the newly focused sim (only if user hasn't overridden it).
        omp_input = self.query_one("#omp-input", Input)
        if not omp_input.has_focus and self._manual_omp is None:
            omp_input.value = str(sim.n_omp)
        nbin_val = sd.get("NBin") or sd.get("Nbin")
        mode = "nbin" if nbin_val else "cpu_max"
        self._end_mode = mode
        self.query_one("#end-by-cpu-max", Button).label = _btn_label(
            "CPU_MAX", mode == "cpu_max"
        )
        self.query_one("#end-by-nbin", Button).label = _btn_label(
            "NBin", mode == "nbin"
        )
        _slurm = self._executor_state == "slurm"
        self.query_one("#nbin-row").display = mode == "nbin"
        self.query_one("#cpu-max-row").display = mode == "cpu_max"
        if _slurm:
            self.query_one("#walltime-input", Input).disabled = mode == "cpu_max"
        if mode == "cpu_max":
            cpu_max_h = max(0.0, float(sd.get("CPU_MAX", 0)))
            cpu_max_input = self.query_one("#cpu-max-input", Input)
            if cpu_max_h > 0 and not cpu_max_input.has_focus:
                cpu_max_input.value = _hours_to_hms(cpu_max_h)
        if nbin_val and not self._manual_nbin:
            nbin_input = self.query_one("#nbin-input", Input)
            if not nbin_input.has_focus:
                nbin_input.value = str(nbin_val)

    @on(DataTable.RowHighlighted, "#sim-table")
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self._edit_mode:
            table = self.query_one("#sim-table", DataTable)
            with contextlib.suppress(Exception):
                if self._dt_cursor_row != self._dt_hover_row:
                    table.update_cell(str(self._dt_cursor_row), "cur", " ")
                self._dt_cursor_row = event.cursor_row
                table.update_cell(str(self._dt_cursor_row), "cur", ">")
        else:
            self._dt_cursor_row = event.cursor_row
        if event.row_key is not None:
            with contextlib.suppress(ValueError, TypeError):
                self._focused_idx = int(str(event.row_key.value))
            self._refresh_right_panel()
            self._refresh_end_mode_display()
            self._refresh_submit_button()

    def _update_dt_hover(self, new_hover: int) -> None:
        if new_hover == self._dt_hover_row:
            return
        if self._edit_mode:
            table = self.query_one("#sim-table", DataTable)
            with contextlib.suppress(Exception):
                if self._dt_hover_row != self._dt_cursor_row:
                    table.update_cell(str(self._dt_hover_row), "cur", " ")
                self._dt_hover_row = new_hover
                if self._dt_hover_row != self._dt_cursor_row:
                    table.update_cell(str(self._dt_hover_row), "cur", ">")
        else:
            self._dt_hover_row = new_hover

    @on(Input.Changed, "#data-dir-display")
    def _on_data_dir_changed(self) -> None:
        inp = self.query_one("#data-dir-display", _DataDirInput)
        if inp.value == inp._tail_display:
            return  # display-only tailing write — not a real edit
        inp._full_value = inp.value  # keep in sync while typing
        new_base = inp.value.strip()
        if not new_base or not self._sims:
            return
        for sim, suffix in zip(self._sims, self._data_dir_suffixes):
            sim.sim_dir = new_base + suffix if suffix else new_base
        self._repopulate_table()

    @on(Input.Changed, "#mem-input")
    def _on_mem_changed(self) -> None:
        self._refresh_right_panel()
        self._refresh_submit_button()

    @on(Input.Changed, "#walltime-input")
    def _on_walltime_changed(self) -> None:
        val = self.query_one("#walltime-input", Input).value
        hours = _parse_wall_time(val)
        self.query_one("#walltime-hint", Static).update(
            "[HH:MM:SS]" if (val and hours is None) else ""
        )
        if hours is not None and self._sims:
            if self._end_mode == "nbin":
                # In NBin mode, the walltime sets SLURM --time directly.
                # CPU_MAX stays at 0 so ALF stops only on the bin count.
                if self._cs.slurm_kwargs is None:
                    self._cs.slurm_kwargs = {}
                self._cs.slurm_kwargs["slurm_time"] = int(hours * 60)
            elif "slurm_time" in (self._cs.slurm_kwargs or {}):
                self._cs.slurm_kwargs["slurm_time"] = int(hours * 60)
        elif (
            self._end_mode == "nbin"
            and val.strip() == ""
            and self._sims
            and self._cs.slurm_kwargs
        ):
            # User cleared the wall-time field in NBin mode — remove slurm_time
            # so the arch panel reverts to the "set SLURM time" prompts.
            self._cs.slurm_kwargs.pop("slurm_time", None)
        self._refresh_right_panel()
        self._refresh_submit_button()

    @on(Input.Changed, "#omp-input")
    def _on_omp_changed(self) -> None:
        try:
            n_omp = int(self.query_one("#omp-input", Input).value)
        except ValueError:
            self._refresh_right_panel()
            self._refresh_submit_button()
            return
        if n_omp < 1:
            self._refresh_right_panel()
            self._refresh_submit_button()
            return
        self._manual_omp = n_omp
        for sim in self._sims:
            with contextlib.suppress(AttributeError):
                sim.n_omp = n_omp
        self._repopulate_table()
        self._refresh_right_panel()
        self._refresh_submit_button()

    @on(Input.Changed, "#mpi-input")
    def _on_mpi_changed(self) -> None:
        try:
            n_mpi = int(self.query_one("#mpi-input", Input).value)
        except ValueError:
            self._refresh_right_panel()
            self._refresh_submit_button()
            return
        if n_mpi < 1:
            self._refresh_right_panel()
            self._refresh_submit_button()
            return
        for sim in self._sims:
            with contextlib.suppress(AttributeError):
                sim.n_mpi = n_mpi
        self._repopulate_table()
        self._refresh_right_panel()
        self._refresh_submit_button()

    @on(Input.Changed, "#job-name-input")
    @on(Input.Changed, "#wckey-input")
    def _on_job_meta_changed(self) -> None:
        self._refresh_right_panel()

    @on(Input.Changed, "#partition-input")
    def _on_partition_changed(self) -> None:
        val = self.query_one("#partition-input", Input).value.strip()
        self._manual_partition = val or None
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

    def _set_mail_flags(self, flags: set[str]) -> None:
        """Set mail notification flags directly and sync all mail buttons."""
        self._mail_flags = flags
        for tag in ("END", "FAIL", "ALL"):
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

    @on(Button.Pressed, "#end-by-cpu-max")
    def _on_end_by_cpu_max(self) -> None:
        self._set_end_mode("cpu_max")

    @on(Button.Pressed, "#end-by-nbin")
    def _on_end_by_nbin(self) -> None:
        self._set_end_mode("nbin")

    def _set_end_mode(self, mode: str) -> None:
        self._end_mode = mode
        self.query_one("#end-by-cpu-max", Button).label = _btn_label(
            "CPU_MAX", mode == "cpu_max"
        )
        self.query_one("#end-by-nbin", Button).label = _btn_label(
            "NBin", mode == "nbin"
        )
        _slurm = self._executor_state == "slurm"
        self.query_one("#nbin-row").display = mode == "nbin"
        self.query_one("#cpu-max-row").display = mode == "cpu_max"
        if _slurm:
            self.query_one("#walltime-input", Input).disabled = mode == "cpu_max"
        if mode == "cpu_max":
            # Restore CPU_MAX from the current sim (NBin mode may have zeroed it).
            cpu_max_h = (
                max(0.0, float(_sim_dict_of(self._sims[0]).get("CPU_MAX", 0)))
                if self._sims
                else 0.0
            )
            for sim in self._sims:
                sd = _sim_dict_of(sim)
                sd.pop("NBin", None)
                sd.pop("Nbin", None)
                if cpu_max_h > 0:
                    sd["CPU_MAX"] = cpu_max_h
            cpu_max_input = self.query_one("#cpu-max-input", Input)
            if cpu_max_h > 0 and not cpu_max_input.has_focus:
                cpu_max_input.value = _hours_to_hms(cpu_max_h)
            # In CPU_MAX mode slurm_time is auto-computed — always clear it.
            if self._cs.slurm_kwargs:
                self._cs.slurm_kwargs.pop("slurm_time", None)
            self._repopulate_table()
        else:
            # Switching to NBin mode — clear the walltime input so it shows [enter].
            wt = self.query_one("#walltime-input", Input)
            wt.value = ""
            self.query_one("#walltime-hint", Static).update("")
            try:
                nbin_int = int(self.query_one("#nbin-input", Input).value)
            except ValueError:
                nbin_int = 40
            for sim in self._sims:
                sd = _sim_dict_of(sim)
                sd.pop("Nbin", None)
                sd["NBin"] = nbin_int
                # CPU_MAX=0 tells ALF to stop only when NBin bins are reached.
                sd["CPU_MAX"] = 0
            if "NBin" not in self._param_keys:
                self._param_keys.append("NBin")
                self._param_headers.append("NBin")
                self._rebuild_table_with_columns()
            else:
                self._repopulate_table()
        self._refresh_right_panel()
        self._refresh_submit_button()

    @on(Input.Changed, "#nbin-input")
    def _on_nbin_changed(self) -> None:
        if self._end_mode != "nbin":
            return
        try:
            nbin_int = int(self.query_one("#nbin-input", Input).value)
        except ValueError:
            self._refresh_submit_button()
            return
        self._manual_nbin = True
        for sim in self._sims:
            sd = _sim_dict_of(sim)
            sd.pop("Nbin", None)
            sd["NBin"] = nbin_int
        self._repopulate_table()
        self._refresh_right_panel()
        self._refresh_submit_button()

    @on(Input.Changed, "#cpu-max-input")
    def _on_cpu_max_changed(self) -> None:
        if self._end_mode != "cpu_max":
            return
        val = self.query_one("#cpu-max-input", Input).value
        hours = _parse_wall_time(val)
        self.query_one("#cpu-max-hint", Static).update(
            "[HH:MM:SS]" if (val and hours is None) else ""
        )
        if hours is not None and self._sims:
            for sim in self._sims:
                _sim_dict_of(sim)["CPU_MAX"] = hours
            self._repopulate_table()
        self._refresh_right_panel()
        self._refresh_submit_button()

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

    @on(Input.Submitted, "#cmd-input")
    def _on_cmd_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        self._hide_cmd()
        if raw:
            self._exec_command(raw)
