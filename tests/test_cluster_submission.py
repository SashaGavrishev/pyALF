"""Tests for ClusterSubmitter in py_alf.cluster_submission."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from py_alf.cluster_submission import ClusterSubmitter, _find_job_log, _run_alf
from py_alf.simulation import Simulation


def test_init_defaults():
    cs = ClusterSubmitter(slurm_mem='2G')
    assert cs.submit_dir == Path('submitit')
    assert cs.slurm_partition == 'short'
    assert cs.slurm_mem == '2G'
    assert cs.slurm_kwargs == {}


def test_init_custom():
    cs = ClusterSubmitter(submit_dir='/tmp/logs', slurm_partition='gpu', slurm_mem='8G', slurm_extra='foo')
    assert cs.submit_dir == Path('/tmp/logs')
    assert cs.slurm_partition == 'gpu'
    assert cs.slurm_mem == '8G'
    assert cs.slurm_kwargs == {'slurm_extra': 'foo'}


def test_init_requires_slurm_mem():
    with pytest.raises(TypeError, match="slurm_mem"):
        ClusterSubmitter()


def test_submit_type_error():
    cs = ClusterSubmitter(slurm_mem='2G')
    with pytest.raises(TypeError, match="Expected Simulation"):
        cs.submit("not_a_simulation")


def test_submit_empty_list():
    cs = ClusterSubmitter(slurm_mem='2G')
    result = cs.submit([])
    assert result == []


def test_submit_single_sim(tmp_path):
    """Single simulation is submitted as a one-element job list."""
    sim = _make_mock_sim(tmp_path / "sim0")

    mock_job = MagicMock()
    mock_job.job_id = "42"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(submit_dir=tmp_path / "logs", slurm_mem="2G")
        jobs = cs.submit(sim)

    assert jobs == [mock_job]
    assert (tmp_path / "sim0" / "jobid.txt").read_text() == "42"
    mock_executor.return_value.submit.assert_called_once_with(_run_alf, sim)


def test_submit_multiple_sims_uses_map_array(tmp_path):
    """Multiple simulations are submitted via map_array."""
    sims = [_make_mock_sim(tmp_path / f"sim{i}") for i in range(3)]

    mock_jobs = [MagicMock(job_id=f"99_{i}") for i in range(3)]

    with _patch_submitit(mock_jobs, multi=True) as mock_executor:
        cs = ClusterSubmitter(submit_dir=tmp_path / "logs", slurm_mem="2G")
        jobs = cs.submit(sims)

    assert jobs == mock_jobs
    for i, sim in enumerate(sims):
        assert (tmp_path / f"sim{i}" / "jobid.txt").read_text() == f"99_{i}"
    mock_executor.return_value.map_array.assert_called_once_with(_run_alf, sims)


def test_submit_heterogeneous_resources_raises(tmp_path):
    """Array submission raises ValueError when sims have different resource shapes."""
    sim_a = _make_mock_sim(tmp_path / "sim0")
    sim_b = _make_mock_sim(tmp_path / "sim1")
    sim_b.n_omp = 8  # differs from sim_a's n_omp=4

    with pytest.raises(ValueError, match="n_omp"):
        ClusterSubmitter(submit_dir=tmp_path / "logs", slurm_mem="2G").submit([sim_a, sim_b])


def test_submit_skips_running_job(tmp_path):
    """A simulation whose jobid.txt reports RUNNING is skipped."""
    sim = _make_mock_sim(tmp_path / "sim0")
    (tmp_path / "sim0" / "jobid.txt").write_text("7")

    with patch(
        "py_alf.cluster_submission._get_slurm_status_sacct",
        return_value={"status": "RUNNING"},
    ):
        cs = ClusterSubmitter(submit_dir=tmp_path / "logs", slurm_mem="2G")
        jobs = cs.submit(sim)

    assert jobs == []


def test_submit_skips_pending_job(tmp_path):
    """A simulation whose jobid.txt reports PENDING is skipped."""
    sim = _make_mock_sim(tmp_path / "sim0")
    (tmp_path / "sim0" / "jobid.txt").write_text("7")

    with patch(
        "py_alf.cluster_submission._get_slurm_status_sacct",
        return_value={"status": "PENDING"},
    ):
        cs = ClusterSubmitter(submit_dir=tmp_path / "logs", slurm_mem="2G")
        jobs = cs.submit(sim)

    assert jobs == []


def test_submit_executor_parameters(tmp_path):
    """Executor is configured with correct default SLURM parameters."""
    sim = _make_mock_sim(tmp_path / "sim0")

    mock_job = MagicMock()
    mock_job.job_id = "1"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(submit_dir=tmp_path / "logs", slurm_partition='long', slurm_mem='4G')
        cs.submit(sim, job_properties={'timeout_min': 120})

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert call_kwargs['slurm_partition'] == 'long'
    assert call_kwargs['slurm_mem'] == '4G'
    assert call_kwargs['timeout_min'] == 120
    assert call_kwargs['cpus_per_task'] == sim.n_omp
    assert call_kwargs['tasks_per_node'] == 1  # non-MPI sim


def test_find_job_log_submitit(tmp_path):
    """_find_job_log returns the submitit-named log when submit_dir is provided."""
    submit_dir = tmp_path / "logs"
    submit_dir.mkdir()
    log_file = submit_dir / "42_0_0_log.out"
    log_file.write_text("output")

    result = _find_job_log("42_0", submit_dir=submit_dir)
    assert result == log_file


def test_find_job_log_falls_back_to_legacy(tmp_path):
    """_find_job_log falls back to the job-*.log glob when no submitit log exists."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    legacy_log = sim_dir / "job-42.log"
    legacy_log.write_text("output")

    result = _find_job_log("42", root_dir=[str(sim_dir)])
    assert result == legacy_log


# --- helpers ---

def _make_mock_sim(sim_dir: Path):
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim = MagicMock()
    sim.__class__ = Simulation  # makes isinstance(sim, Simulation) return True
    sim.sim_dir = str(sim_dir)
    sim.ham_name = "Hubbard"
    sim.n_omp = 4
    sim.n_mpi = 1
    sim.mpi = False
    sim.sim_dict = {"CPU_MAX": 2}
    return sim


def _patch_submitit(job_or_jobs, multi=False):
    mock_executor = MagicMock()
    if multi:
        mock_executor.return_value.map_array.return_value = job_or_jobs
    else:
        mock_executor.return_value.submit.return_value = job_or_jobs
    return patch("py_alf.cluster_submission.submitit.AutoExecutor", mock_executor)
