#!/usr/bin/env python3
"""CLI for launching the ALF simulation monitor from a session manifest."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from rich.console import Console

from py_alf.monitor import SimulationMonitor


def _get_arg_parser():
    parser = ArgumentParser(
        description=(
            "Launch the ALF simulation monitor. "
            "Scans a submit directory for session manifests written by "
            "SubmissionReview and lets you pick one to open, or pass a "
            "specific session file directly."
        ),
    )
    parser.add_argument(
        "session",
        nargs="?",
        metavar="SESSION_JSON",
        help=(
            "Path to a specific session JSON file. "
            "If omitted, all session files in --dir are listed for selection."
        ),
    )
    parser.add_argument(
        "--dir",
        default=".alfmonitor",
        metavar="DIR",
        help="Directory to search for session JSON files (default: .alfmonitor).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Skip selection and open the most recent session automatically.",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Auto-refresh interval in seconds (default: 30).",
    )
    return parser


def _find_sessions(directory: Path) -> list[Path]:
    """Return session JSON files sorted newest first."""
    return sorted(directory.glob("session_*.json"), key=lambda p: p.name, reverse=True)


def _find_alfmonitor_dir() -> Path | None:
    """Search upward from CWD for a .alfmonitor directory."""
    current = Path.cwd()
    while True:
        candidate = current / ".alfmonitor"
        if candidate.is_dir():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _param_display(values: list) -> str:
    """Compact display for a list of param values across sims."""
    unique = list(dict.fromkeys(values))
    if len(unique) == 1:
        return str(unique[0])
    try:
        nums = [float(v) for v in unique]
        lo, hi = min(nums), max(nums)
        fmt = lambda x: str(int(x)) if x == int(x) else str(x)  # noqa: E731
        return f"{fmt(lo)}–{fmt(hi)}"
    except (ValueError, TypeError):
        pass
    if len(unique) <= 4:
        return ", ".join(str(v) for v in unique)
    return f"{unique[0]}, …, {unique[-1]} ({len(unique)} vals)"


_EXECUTOR_STYLE = {"slurm": "green", "local": "dim", "debug": "yellow"}


def _print_session(i: int, path: Path, console) -> None:
    """Print one session entry with rich formatting."""
    from rich.text import Text

    # Index chip + filename
    chip = Text(f" {i} ", style="bold on bright_black")
    chip.append(f"  {path.name}", style="bold default")
    console.print(chip)

    try:
        data = json.loads(path.read_text())
    except Exception:
        console.print("    [red](unreadable)[/red]")
        return

    entries = data.get("entries", [])
    cs = data.get("cluster_submitter", {})
    executor = cs.get("executor", "?")

    # Line 1: timestamp · count · executor · mem
    t1 = Text("    ")
    t1.append(data.get("submitted_at", "?"), style="dim")
    t1.append("  ·  ")
    t1.append(str(len(entries)), style="bold")
    t1.append(" sim(s)  ·  ")
    t1.append(executor, style=_EXECUTOR_STYLE.get(executor, ""))
    if cs.get("slurm_mem"):
        t1.append("  ·  ")
        t1.append(f"mem={cs['slurm_mem']}", style="yellow")
    console.print(t1)

    if not entries:
        return

    first = entries[0]

    # Line 2: hamiltonian(s) · resources
    hams = list(dict.fromkeys(e.get("ham_name", "?") for e in entries))
    n_omp = first.get("n_omp", "?")
    n_mpi = first.get("n_mpi", 1)
    mpi = first.get("mpi", False)

    t2 = Text("    ")
    t2.append(", ".join(hams), style="bold cyan")
    t2.append("  ·  ")
    t2.append(str(n_omp), style="bold")
    t2.append(" OMP")
    if mpi and n_mpi > 1:
        t2.append("  ×  ")
        t2.append(str(n_mpi), style="bold")
        t2.append(" MPI")
    console.print(t2)

    # Line 3: sim_dict parameters
    all_keys = list(dict.fromkeys(k for e in entries for k in e.get("sim_dict", {})))
    if all_keys:
        t3 = Text("    ")
        for j, key in enumerate(all_keys):
            if j:
                t3.append("  ·  ", style="dim")
            t3.append(f"{key}: ", style="dim")
            vals = [e["sim_dict"][key] for e in entries if key in e.get("sim_dict", {})]
            t3.append(_param_display(vals))
        console.print(t3)


def _main():
    parser = _get_arg_parser()
    args = parser.parse_args()

    console = Console()

    # Direct path supplied — skip discovery.
    if args.session:
        path = Path(args.session)
        if not path.exists():
            console.print(f"[red]Error:[/red] {path} not found.", file=sys.stderr)
            sys.exit(1)
        SimulationMonitor.from_session(path, refresh_interval=args.refresh).run()
        return

    # Discovery mode.
    search_dir = Path(args.dir)
    if not search_dir.is_dir() and args.dir == ".alfmonitor":
        # Auto-discover .alfmonitor by searching upward from CWD.
        found = _find_alfmonitor_dir()
        if found is not None:
            search_dir = found
    if not search_dir.is_dir():
        console.print(
            f"[red]Error:[/red] directory [bold]{search_dir}[/bold] does not exist."
        )
        sys.exit(1)

    sessions = _find_sessions(search_dir)
    if not sessions:
        console.print(
            f"[dim]No session files found in [bold]{search_dir}[/bold].[/dim]"
        )
        sys.exit(1)

    if args.latest or len(sessions) == 1:
        chosen = sessions[0]
    else:
        console.print(f"\nSession files in [bold]{search_dir}[/bold]:\n")
        for i, p in enumerate(sessions):
            _print_session(i, p, console)
            console.print()
        raw = input(f"Pick session [0–{len(sessions) - 1}, default 0]: ").strip()
        if raw == "":
            idx = 0
        elif raw.isdigit() and int(raw) < len(sessions):
            idx = int(raw)
        else:
            console.print(
                f"[red]Error:[/red] {raw!r} is not a valid session index (0–{len(sessions) - 1})."
            )
            sys.exit(1)
        chosen = sessions[idx]

    console.print(f"\n[dim]Opening:[/dim] [bold]{chosen.name}[/bold]\n")
    SimulationMonitor.from_session(chosen, refresh_interval=args.refresh).run()


if __name__ == "__main__":
    _main()
