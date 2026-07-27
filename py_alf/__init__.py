"""pyALF, a Python package for the Algorithms for Lattice Fermions (ALF)."""
# pylint: disable=inconsistent-return-statements
# pylint: disable=import-outside-toplevel

# Classes
from .alf_source import ALF_source
from .campaign import Campaign, Chain, ChainStatus, Ledger, SegmentPlan, SegmentPolicy
from .cluster_submission import ClusterSubmitter, PartitionSpec, detect_partition_rules
from .lattice import Lattice
from .simulation import Simulation

__all__ = [
    "ALF_source",
    "Simulation",
    "Lattice",
    "Campaign",
    "Chain",
    "ChainStatus",
    "ClusterSubmitter",
    "Ledger",
    "PartitionSpec",
    "SegmentPlan",
    "SegmentPolicy",
    "detect_partition_rules",
    "SubmissionReview",
    "save_for_ssh",
]

_LAZY: dict[str, tuple[str, str]] = {
    "SubmissionReview": (".submission_tui", "SubmissionReview"),
    "save_for_ssh": (".submission_tui", "save_for_ssh"),
    "list_sessions": (".monitor", "list_sessions"),
    "load_session_sims": (".monitor", "load_session_sims"),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        from importlib import import_module  # noqa: PLC0415

        module = import_module(module_path, package=__package__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def check_warmup(*args, gui="tk", **kwargs):
    """Plot bins to determine n_skip.

    Calls either :func:`py_alf.check_warmup_tk` or
    :func:`py_alf.check_warmup_ipy`.

    Parameters
    ----------
    *args
    gui : {"tk", "ipy"}
    **kwargs

    """
    if gui == "tk":
        from .check_warmup_tk import check_warmup_tk

        check_warmup_tk(*args, **kwargs)
    elif gui == "ipy":
        from .check_warmup_ipy import check_warmup_ipy

        return check_warmup_ipy(*args, **kwargs)
    else:
        raise TypeError(f"Illegal value gui={gui}")


def check_rebin(*args, gui="tk", **kwargs):
    """Plot error vs n_rebin in a Jupyter Widget.

    Calls either :func:`py_alf.check_rebin_tk` or
    :func:`py_alf.check_rebin_ipy`.

    Parameters
    ----------
    *args
    gui : {"tk", "ipy"}
    **kwargs

    """
    if gui == "tk":
        from .check_rebin_tk import check_rebin_tk

        check_rebin_tk(*args, **kwargs)
    elif gui == "ipy":
        from .check_rebin_ipy import check_rebin_ipy

        return check_rebin_ipy(*args, **kwargs)
    else:
        raise TypeError(f"Illegal value gui={gui}")
