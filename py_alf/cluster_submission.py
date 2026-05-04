"""

cluster_submission
==================

Provides interfaces for running ALF simulations on a cluster.

"""

__author__ = "Johannes Hofmann"
__copyright__ = "Copyright 2020-2025, The ALF Project"
__license__ = "GPL"

import logging
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import submitit
from colorama import Fore
from tabulate import tabulate
from tqdm import tqdm

from .simulation import Simulation

logger = logging.getLogger(__name__)


def _run_alf(sim: Simulation) -> None:
    """
    Execute an ALF simulation on a cluster node.

    Called by submitit on the remote worker. Assumes the binary has already been copied
    into sim.sim_dir by the pre-submission preparation step.
    """
    sim.run(bin_in_sim_dir=True)


class ClusterSubmitter:
    """
    Handles job submission to a SLURM cluster using submitit.

    Parameters
    ----------
    submit_dir : str or Path
        Directory where submitit writes job logs and state.
    slurm_partition : str
        SLURM partition to submit to.
    slurm_mem : str
        Memory request per node (e.g. ``'2G'``, ``'8G'``). Required.
    **slurm_kwargs
        Additional keyword arguments passed to
        ``executor.update_parameters()``. Any key prefixed with ``slurm_``
        is forwarded as a raw ``#SBATCH`` directive.
    """

    def __init__(
        self,
        submit_dir="submitit",
        slurm_partition="short",
        *,
        slurm_mem: str,
        **slurm_kwargs,
    ):
        self.submit_dir = Path(submit_dir)
        self.slurm_partition = slurm_partition
        self.slurm_mem = slurm_mem
        self.slurm_kwargs = slurm_kwargs

    def submit(
        self,
        sims: Union[Simulation, Iterable[Simulation]],
        job_properties: Optional[Dict[str, Any]] = None,
        submit_dir: Optional[Union[str, Path]] = None,
    ) -> List[submitit.Job]:
        """
        Submit one or more Simulation instances to the SLURM cluster.

        Prepares simulation directories, filters out already-running or broken
        jobs, then submits via submitit. Job IDs are written to ``jobid.txt``
        inside each simulation directory so that the status-checking helpers
        in this module continue to work.

        Parameters
        ----------
        sims : Simulation or iterable of Simulation
            Simulation(s) to submit.
        job_properties : dict, optional
            Override default SLURM parameters. Keys must match
            ``executor.update_parameters()`` keyword arguments.
        submit_dir : str or Path, optional
            Directory for submitit logs and state for this submission.
            Overrides the instance-level ``submit_dir`` set at construction.

        Returns
        -------
        list of submitit.Job
            One Job object per submitted simulation.
        """

        if isinstance(sims, Iterable) and not isinstance(
            sims, (str, bytes, Simulation)
        ):
            sim_list = list(sims)
        else:
            sim_list = [sims]

        for s in sim_list:
            if not isinstance(s, Simulation):
                raise TypeError(f"Expected Simulation, got {type(s)}")

        filtered_sims = []

        for s in sim_list:
            jobid_file = Path(s.sim_dir) / "jobid.txt"
            running_file = Path(s.sim_dir) / "RUNNING"

            jobid: Optional[str] = (
                jobid_file.read_text().strip() if jobid_file.exists() else None
            )

            if jobid is not None:
                status_entry = _get_slurm_status_sacct(jobid)
                if status_entry.get("status") in ("PENDING", "RUNNING"):
                    logger.info(
                        f"Skipping {s.sim_dir}: job {jobid} is \
                           {status_entry.get('status')}"
                    )
                    continue

            if running_file.exists():
                if jobid is not None:
                    status_entry = _get_slurm_status_sacct(jobid)
                    if status_entry.get("status") == "RUNNING":
                        logger.info(f"Skipping {s.sim_dir}: job {jobid} is RUNNING")
                        continue
                logger.warning(f"Leftover RUNNING file detected in {s.sim_dir}.")
                logger.warning("This indicates an error in the previous run.")
                choice = (
                    input("Remove RUNNING file to enable resubmission? [y/N]: ")
                    .strip()
                    .lower()
                )
                if choice in ("yes", "y"):
                    running_file.unlink()
                    logger.info("File removed.")
                else:
                    logger.info(f"Skipping {s.sim_dir}.")
                    continue

            filtered_sims.append(s)

        if not filtered_sims:
            logger.info("No inactive simulations to submit.")
            return []

        sim = filtered_sims[0]

        # Guard: all sims in an array job must share the same resource shape,
        # since submitit applies one set of SLURM parameters to every task.
        if len(filtered_sims) > 1:
            for s in filtered_sims[1:]:
                if s.n_omp != sim.n_omp or s.n_mpi != sim.n_mpi or s.mpi != sim.mpi:
                    raise ValueError(
                        "All simulations in an array job must have the same n_omp, n_mpi, "
                        f"and mpi settings (derived from filtered_sims[0]: n_omp={sim.n_omp}, "
                        f"n_mpi={sim.n_mpi}, mpi={sim.mpi}), but {s.sim_dir} has "
                        f"n_omp={s.n_omp}, n_mpi={s.n_mpi}, mpi={s.mpi}."
                    )

        timeout_hours = max(1, int(sim.sim_dict.get("CPU_MAX", 24)))

        # Build executor parameters from defaults, instance-level kwargs,
        # then per-call overrides.
        params: Dict[str, Any] = {
            "name": sim.ham_name,
            "timeout_min": timeout_hours * 60,
            "nodes": 1,
            "cpus_per_task": sim.n_omp,
            "tasks_per_node": sim.n_mpi if sim.mpi else 1,
            "slurm_mem": self.slurm_mem,
            "slurm_partition": self.slurm_partition,
        }
        params.update(self.slurm_kwargs)
        if job_properties:
            params.update(job_properties)

        # Prepare simulation directories and copy binary.
        for s in filtered_sims:
            s.run(only_prep=True, copy_bin=True)

        effective_submit_dir = (
            Path(submit_dir) if submit_dir is not None else self.submit_dir
        )
        effective_submit_dir.mkdir(parents=True, exist_ok=True)

        executor = submitit.AutoExecutor(
            folder=str(effective_submit_dir), cluster="slurm"
        )
        executor.update_parameters(**params)

        if len(filtered_sims) == 1:
            jobs = [executor.submit(_run_alf, filtered_sims[0])]
        else:
            jobs = executor.map_array(_run_alf, filtered_sims)

        # Write job IDs for compatibility with get_status / get_status_all.
        for s, job in zip(filtered_sims, jobs):
            Path(s.sim_dir, "jobid.txt").write_text(job.job_id)

        logger.info(f"Submitted {len(jobs)} job(s): {[j.job_id for j in jobs]}")
        return jobs

    def resubmission(
        self,
        sims_to_resubmit: Iterable[Simulation],
        job_properties: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
        print_first: bool = True,
        confirm: bool = True,
        counting_obs: str = "Ener_scal",
    ) -> None:
        """
        Resubmit simulations that have too few bins.

        Parameters
        ----------
        sims_to_resubmit : iterable of Simulation
            Simulations to resubmit.
        job_properties : dict, optional
            SLURM parameter overrides forwarded to :meth:`submit`.
        params : dict, optional
            Mapping of display-label → sim_dict key used when printing.
        print_first : bool
            Print a summary of simulations before resubmitting.
        confirm : bool
            Ask for confirmation before submitting.
        counting_obs : str
            Observable used to count bins.
        """
        sims_to_resubmit = list(sims_to_resubmit)
        if not sims_to_resubmit:
            logger.info("No simulations to resubmit.")
            return

        if print_first:
            print(f"{len(sims_to_resubmit)} simulations will be resubmitted.")
            for sim in sims_to_resubmit:
                num_bins = sim.bin_count(counting_obs=counting_obs, refresh=True)
                status = sim.get_cluster_job_status()
                label = (
                    "".join(
                        f"{k}={sim.sim_dict[v]}, " if v in sim.sim_dict else ""
                        for k, v in params.items()
                    )
                    if params
                    else sim.sim_dir
                )
                print(f"Sim (Nbins={num_bins}) {label} with status {status}")

        if confirm:
            choice = input("Proceed with resubmission? [y/N]: ").strip().lower()
            if choice not in ("yes", "y"):
                logger.info("Resubmission cancelled.")
                return

        self.submit(sims=sims_to_resubmit, job_properties=job_properties)


# --- Status functions ---
def get_status(sim: Simulation, colored: bool = True) -> str:
    """
    Returns colorized SLURM job status for a simulation.
    Args:
        sim: Simulation instance.
        colored: Colorize output if True.
    Returns:
        Colorized status string.
    """
    jobid_file = Path(sim.sim_dir) / "jobid.txt"
    running_file = Path(sim.sim_dir) / "RUNNING"
    if not jobid_file.exists():
        status = "CRASHED" if running_file.exists() else "NO_JOBID"
    else:
        jobid = jobid_file.read_text().strip()
        entry = _get_slurm_status_sacct(jobid)
        status = entry.get("status", "UNKNOWN")
    if colored:
        status = _colorize_status(status)
    return status


def get_job_id(sim: Simulation) -> Optional[str]:
    """
    Returns colorized SLURM job status for a simulation.
    Args:
        sim: Simulation instance.
        colored: Colorize output if True.
    Returns:
        Colorized status string.
    """
    jobid_file = Path(sim.sim_dir) / "jobid.txt"
    if not jobid_file.exists():
        return None
    else:
        return jobid_file.read_text().strip()


def _get_slurm_status_sacct(jobid: str) -> Dict[str, Optional[str]]:
    """
    Query SLURM sacct for job status and elapsed time.
    Returns dict: {'status': <status_str>, 'runtime': <elapsed_or_None>}
    """
    try:
        result = subprocess.run(
            ["sacct", "-j", jobid, "--format=State,Elapsed", "--noheader", "--array"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = result.stdout.strip().splitlines()
        logger.debug(lines)
        for line in lines:
            parts = line.split()
            if len(parts) >= 1:
                state = parts[0]
                runtime = parts[1] if len(parts) > 1 else None
                return {"status": state, "runtime": runtime}
        return {"status": "UNKNOWN", "runtime": None}
    except Exception as e:
        logger.error(f"sacct error for job {jobid}: {e}")
        return {"status": "ERROR", "runtime": None}


def _get_slurm_status_bulk_sacct(
    jobids: List[str],
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Query SLURM sacct for multiple job IDs (including array tasks) in one call.
    Returns dict: jobid[_index] -> {'status': <str>, 'runtime': <str|None>}
    """
    status_map: Dict[str, Dict[str, Optional[str]]] = {
        jid: {"status": "UNKNOWN", "runtime": None} for jid in jobids
    }
    if not jobids:
        return status_map

    try:
        result = subprocess.run(
            ["sacct", "--format=JobID,State,Elapsed", "--noheader", "--array"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                jobid = parts[0]
                state = parts[1]
                runtime = parts[2] if len(parts) > 2 else None
                status_map[jobid] = {"status": state, "runtime": runtime}
    except Exception as e:
        logger.error(f"sacct bulk error: {e}")
        for jid in jobids:
            status_map[jid] = {"status": "ERROR", "runtime": None}
    return status_map


def _get_slurm_status_bulk(jobids: List[str]) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Query SLURM for multiple job IDs (including array tasks) in one call.
    Args:
        jobids: List of job IDs.
    Returns:
        Dict mapping jobid[_index] to {'status':..., 'runtime':...}.
    """
    if not jobids:
        return {}

    status_map: Dict[str, Dict[str, Optional[str]]] = {
        jid: {"status": "FINISHED_OR_NOT_FOUND", "runtime": None} for jid in jobids
    }
    found_in_squeue = set()

    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%A %i %T %M", "--array", "-j", ",".join(jobids)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        squeue_failed = False
    except subprocess.TimeoutExpired:
        logger.warning(
            "squeue command timed out. Falling back to sacct for job status."
        )
        squeue_failed = True
    except Exception as e:
        logger.error(f"Error running squeue: {e}")
        squeue_failed = True

    if not squeue_failed:
        for line in result.stdout.strip().splitlines():
            try:
                parts = line.split(maxsplit=3)
                if len(parts) != 4:
                    logger.warning(f"Unexpected squeue output line: '{line}'")
                    continue
                jid, idx, state, runtime = parts
                full_id = jid if idx == "N/A" else idx
                status_map[full_id] = {"status": state, "runtime": runtime}
                found_in_squeue.add(full_id)
            except Exception as e:
                logger.error(f"Error parsing squeue output line '{line}': {e}")
                continue
        # For jobs not found in squeue output, fallback to sacct bulk
        missing_jobids = [jid for jid in jobids if jid not in found_in_squeue]
        if missing_jobids:
            sacct_statuses = _get_slurm_status_bulk_sacct(missing_jobids)
            for jid in missing_jobids:
                status_map[jid] = sacct_statuses.get(
                    jid, {"status": "UNKNOWN", "runtime": None}
                )
    else:
        # squeue failed, use sacct bulk for all jobids
        sacct_statuses = _get_slurm_status_bulk_sacct(jobids)
        for jid in jobids:
            status_map[jid] = sacct_statuses.get(
                jid, {"status": "UNKNOWN", "runtime": None}
            )

    return status_map


def get_status_all(
    sims: Iterable[Simulation],
    header: Optional[List[str]] = None,
    keys: Optional[List[str]] = None,
    filter_out: Optional[List[str]] = None,
    crash_tags: Optional[List[str]] = None,
    showid: bool = True,
    counting_obs: str = "Ener_scal",
    refresh_cache: bool = False,
    min_bins: int = 4,
    **tabargs,
) -> tuple[Optional[List[Simulation]], Optional[List[Simulation]]]:
    """

    Prints a table of statuses for all simulations (bulk SLURM query).

    Args:
        sims: Iterable of Simulation instances.
        header: List of column headers.
        keys: List of keys to extract from sim.sim_dict.
        filter_out: List of statuses to filter out from display.
        crash_tags: List of tags indicating crashed simulations.
        showid: Whether to show job ID column.
        counting_obs: Observable name for bin counting.
        refresh_cache: Whether to refresh bin count cache.
        min_bins: Minimum number of bins required; simulations with fewer are returned.
        tabargs: Additional arguments for tabulate.

    Returns:
        Tuple of (sims_with_too_few_bins, crashed_sims)
    """
    sims = list(sims)  # Accept any iterable
    if header is None:
        header = ["dir"]
    if keys is None:
        keys = ["sim_dir"]
    if filter_out is None:
        filter_out = ["INACTIVE"]
    if crash_tags:
        crash_tags = [tag.upper() for tag in crash_tags]
        if "CRASHED" not in crash_tags:
            crash_tags.append("CRASHED")
        if "FAILED" not in crash_tags:
            crash_tags.append("FAILED")
    else:
        crash_tags = ["CRASHED", "FAILED"]
    jobid_map: Dict[str, str] = {}
    for sim in sims:
        jobid_file = Path(sim.sim_dir) / "jobid.txt"
        if jobid_file.exists():
            jobid_map[sim.sim_dir] = jobid_file.read_text().strip()

    statuses = _get_slurm_status_bulk(list(jobid_map.values()))

    summary: Dict[str, int] = {}
    header = header + ["N_bin", "JobID", "status", "time"]
    if showid:
        header = ["SimID"] + header
    entries: List[List[Any]] = []

    sims_with_too_few_bins: List[Simulation] = []
    crashed_sims: List[Simulation] = []
    for idx, sim in enumerate(sims):
        runtime: Optional[str] = None
        jobid = jobid_map.get(sim.sim_dir)
        if jobid is None:
            running_file = Path(sim.sim_dir) / "RUNNING"
            status = "CRASHED" if running_file.exists() else "INACTIVE"
        else:
            status_entry = statuses.get(jobid, {"status": "UNKNOWN", "runtime": None})
            status = status_entry.get("status", "UNKNOWN")
            runtime = status_entry.get("runtime", None)

        num_bins = _bin_count(
            sim, counting_obs, refresh=(status == "RUNNING") or refresh_cache
        )
        row = [sim.sim_dict.get(key, None) for key in keys]
        if showid:
            row = [idx] + row
        row.append(num_bins)
        row.append(jobid)
        row.append(_colorize_status(status))
        row.append(
            _pad_runtime(runtime)
            if status == "RUNNING" and runtime is not None
            else None
        )
        if filter_out and status not in filter_out:
            entries.append(row)
        if status in crash_tags:
            crashed_sims.append(sim)
        if num_bins < min_bins and status not in ("RUNNING", "PENDING"):
            # double-check bin count to avoid cache issues
            num_bins = _bin_count(sim, counting_obs, refresh=True)
            if num_bins < min_bins:
                sims_with_too_few_bins.append(sim)

        summary[status] = summary.get(status, 0) + 1

    print(
        tabulate(
            entries,
            headers=header,
            tablefmt=tabargs.pop("tablefmt", "fancy_grid"),
            stralign=tabargs.pop("stralign", "right"),
            **tabargs,
        )
    )

    print("\nSummary:")
    if "RUNNING" in summary:
        key = "RUNNING"
        val = summary.pop(key)
        _print_summary_entry(key, val, len(sims), filter_out)
    if "PENDING" in summary:
        key = "PENDING"
        val = summary.pop(key)
        _print_summary_entry(key, val, len(sims), filter_out)
    for key, val in summary.items():
        if filter_out and key in filter_out:
            print(_colorize_status(key), f":\t{val}/{len(sims)}\t(filtered out)")
        else:
            print(_colorize_status(key), f":\t{val}/{len(sims)}")

    if sims_with_too_few_bins:
        print(
            f"{len(sims_with_too_few_bins)} simulations with fewer than {min_bins} bins in '{counting_obs}'."
        )
    return (
        sims_with_too_few_bins if sims_with_too_few_bins else None,
        crashed_sims if crashed_sims else None,
    )


def find_sims_by_status(
    sims: Iterable[Simulation], filter: List[str]
) -> Optional[List[Simulation]]:
    """
    Prints a table of statuses for all simulations (bulk SLURM query).
    Args:
        sims: Iterable of Simulation instances.
        filter: List of statuses to return.
    """
    sims = list(sims)  # Accept any iterable
    jobid_map: Dict[str, str] = {}
    for sim in sims:
        jobid_file = Path(sim.sim_dir) / "jobid.txt"
        if jobid_file.exists():
            jobid_map[sim.sim_dir] = jobid_file.read_text().strip()

    statuses = _get_slurm_status_bulk(list(jobid_map.values()))

    sims_with_status: List[Simulation] = []
    for sim in sims:
        jobid = jobid_map.get(sim.sim_dir)
        if jobid is None:
            running_file = Path(sim.sim_dir) / "RUNNING"
            status = "CRASHED" if running_file.exists() else "INACTIVE"
        else:
            status_entry = statuses.get(jobid, {"status": "UNKNOWN", "runtime": None})
            status = status_entry.get("status", "UNKNOWN")
        if status in filter:
            sims_with_status.append(sim)
    if not sims_with_status:
        logger.info(f"No simulations found with status in {filter}.")
    return sims_with_status if sims_with_status else None


def _print_summary_entry(
    key: str, val: int, total: int, filter_out: Optional[List[str]] = None
) -> None:
    if filter_out and key in filter_out:
        print(_colorize_status(key), f":\t{val}/{total}\t(not shown)")
    else:
        print(_colorize_status(key), f":\t{val}/{total}")


def _get_slurm_status(jobid_element: str) -> str:
    """
    Query SLURM for a single job or array element.
    Args:
        jobid_element: Job ID string.
    Returns:
        Status string.
    """
    result = subprocess.run(
        ["squeue", "-j", jobid_element, "-h", "-o", "%T"],
        capture_output=True,
        text=True,
    )
    status = result.stdout.strip()
    return status if status else "FINISHED_OR_NOT_FOUND"


_bin_cache: Dict[Any, int] = {}


def _bin_count(
    sim: Simulation, counting_obs: str = "Ener_scal", refresh: bool = False
) -> int:
    """
    Counts bins for a given observable in simulation data, with caching.
    Args:
        sim: Simulation instance.
        counting_obs: Observable name.
        refresh: Whether to refresh cache.
    Returns:
        Number of bins.
    """
    import h5py

    filename = os.path.join(sim.sim_dir, "data.h5")
    key = (filename, counting_obs)

    if (key in _bin_cache) and (not refresh):
        return _bin_cache[key]

    N_bins = 0
    try:
        with h5py.File(filename, "r") as f:
            if counting_obs in f:
                N_bins = f[counting_obs + "/obser"].shape[0]
    except FileNotFoundError:
        N_bins = 0
    except (OSError, KeyError) as e:
        logger.error(f"Error reading {filename}: {e}")
        N_bins = 0

    _bin_cache[key] = N_bins
    return N_bins


def _colorize_status(status: str) -> str:
    """
    Returns a colorized status string for terminal output.
    Args:
        status: Status string.
    Returns:
        Colorized status string.
    """
    if status == "RUNNING":
        status = Fore.GREEN + status + Fore.RESET
    elif status == "PENDING":
        status = Fore.YELLOW + status + Fore.RESET
    elif status in ("CRASHED", "FAILED"):
        status = Fore.RED + status + Fore.RESET
    elif status == "FINISHED_OR_NOT_FOUND":
        status = Fore.BLUE + status + Fore.RESET
    return status


def _pad_runtime(runtime: Optional[str], width: int = 10) -> str:
    """
    Pads runtime string for table formatting.
    Args:
        runtime: Runtime string.
        width: Desired width.
    Returns:
        Padded runtime string.
    """
    return runtime.rjust(width) if runtime is not None else "".rjust(width)


def print_logfile(
    sim: Simulation,
    logfile: Optional[str] = None,
    tail: Optional[int] = None,
    head: Optional[int] = None,
    return_content: bool = False,
    show_progress: bool = False,
    submit_dir: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """
    Prints the logfile of a simulation to the terminal, with options for tail/head
    and progress.

    Args:
        sim: Simulation instance.
        logfile: Path to logfile. If None, tries to auto-detect.
        tail: If set, print only the last N lines.
        head: If set, print only the first N lines.
        return_content: If True, return log content as string.
        show_progress: If True, show a progress bar for large files.
        submit_dir: submitit log directory (the ``submit_dir`` passed to
            ``ClusterSubmitter``). When provided, logs are searched there
            using submitit's ``{jobid}_0_log.out`` naming convention in
            addition to the legacy ``job-*.log`` glob.
    Returns:
        Log content as string if return_content is True, else None.
    """
    log_file = None
    if logfile:
        log_file = Path(logfile)
        if not log_file.exists():
            logger.error(f"Log file {log_file} does not exist.")
            return None
    else:
        log_file = Path(sim.sim_dir) / "latest_cluster_run.log"
        if not log_file.exists():
            logger.info("Searching by job ID...")
            jobid = get_job_id(sim)
            if jobid:
                status = get_status(sim, colored=False)
                print(f"Found job ID {jobid} with status {status}.")
                if status == "PENDING":
                    logger.info(
                        f"Job {jobid} is pending. Logfile cannot be located by job ID."
                    )
                    return None
            else:
                logger.info("No job ID found. Logfile cannot be located by job ID.")
                return None
            logfile_path = _find_job_log(
                jobid, root_dir=[sim.sim_dir, "."], submit_dir=submit_dir
            )
            if logfile_path is None:
                logger.error("Cannot locate logfile.")
                return None
            log_file = logfile_path

    try:
        with log_file.open("r") as f:
            lines = f.readlines()
            total_lines = len(lines)
            if head is not None:
                lines = lines[:head]
            elif tail is not None:
                lines = lines[-tail:]
            if show_progress and total_lines > 1000:
                for line in tqdm(lines, desc="Reading logfile"):
                    print(line, end="")
            else:
                print("".join(lines))
            if return_content:
                return "".join(lines)
    except FileNotFoundError:
        logger.warning(f"Log file {log_file} does not exist.")
    except Exception as e:
        logger.error(f"Error reading {log_file}: {e}")
    return None


def _find_job_log(
    jobid: str,
    root_dir: List[str] = None,
    submit_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    if root_dir is None:
        root_dir = ["."]
    if jobid is None:
        logger.info("No job ID provided for logfile search.")
        return None

    # submitit names logs as {jobid}_{task_id}_log.out; task_id is always 0.
    if submit_dir is not None:
        submitit_log = Path(submit_dir) / f"{jobid}_0_log.out"
        if submitit_log.exists():
            return submitit_log

    # Legacy SLURM-template naming: job-{jobid}.log (underscores → dashes).
    all_matches = []
    for dir in root_dir:
        pattern = f"job-{jobid.replace('_', '-')}.log"
        all_matches.extend(Path(dir).rglob(pattern))
    if not all_matches:
        logger.error(f"Could not find logfile for job {jobid} in {root_dir}")
        return None
    if len(all_matches) != 1:
        logger.warning(f"Multiple logfiles found for job {jobid} in {root_dir}")
    return all_matches[0]


# --- Attach status method to Simulation class ---
Simulation.get_cluster_job_status = get_status
Simulation.get_cluster_job_id = get_job_id
Simulation.print_cluster_logfile = print_logfile
Simulation.bin_count = _bin_count


def simulation_submit_to_cluster(
    self: Simulation,
    cluster_submitter: ClusterSubmitter,
    job_properties: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Submits this simulation as a single job to the cluster using the provided ClusterSubmitter.
    Args:
        self: Simulation instance.
        cluster_submitter: ClusterSubmitter instance.
        job_properties: Optional dictionary of SLURM job properties.
    """
    if not isinstance(cluster_submitter, ClusterSubmitter):
        raise TypeError("cluster_submitter must be a ClusterSubmitter instance")
    cluster_submitter.submit(self, job_properties=job_properties)


Simulation.submit_to_cluster = simulation_submit_to_cluster


def cancel_cluster_job(sim: Simulation) -> bool:
    """
    Cancels the SLURM job associated with this simulation.
    Returns True if cancellation was attempted, False otherwise.
    """
    jobid = get_job_id(sim)
    if jobid is None:
        logger.warning(f"No job ID found for simulation in {sim.sim_dir}.")
        return False
    try:
        result = subprocess.run(["scancel", jobid], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Cancelled job {jobid} for simulation in {sim.sim_dir}.")
            return True
        else:
            logger.error(f"Failed to cancel job {jobid}: {result.stderr.strip()}")
            return False
    except Exception as e:
        logger.error(f"Error cancelling job {jobid}: {e}")
        return False


def cancel_cluster_jobs(sims: Iterable[Simulation]) -> None:
    """
    Cancels SLURM jobs for a list or iterable of simulations.
    """
    for sim in sims:
        cancel_cluster_job(sim)


# Attach to Simulation class
Simulation.cancel_cluster_job = cancel_cluster_job


def remove_RUNNING_file(sim: Simulation) -> bool:
    """
    Removes the RUNNING file from the simulation directory if it exists.
    Returns True if the file was removed, False otherwise.
    """
    running_file = Path(sim.sim_dir) / "RUNNING"
    if running_file.exists():
        try:
            running_file.unlink()
            logger.info(f"Removed RUNNING file from {sim.sim_dir}.")
            return True
        except Exception as e:
            logger.error(f"Error removing RUNNING file from {sim.sim_dir}: {e}")
            return False
    else:
        logger.info(f"No RUNNING file found in {sim.sim_dir}.")
        return False


Simulation.remove_RUNNING_file = remove_RUNNING_file
