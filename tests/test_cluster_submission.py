"""Tests for ClusterSubmitter in py_alf.cluster_submission."""

from pathlib import Path

import pytest

from py_alf.cluster_submission import ClusterSubmitter

TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "Jobscripts" / "Generic.slurm.j2"

MINIMAL_CONTEXT = {
    "nodes": 1,
    "tasks": 1,
    "threads": 4,
    "name": "test_job",
    "mem": "2G",
    "queue": "short",
    "time": 2,
    "alf_src": "/path/to/alf",
    "config": "GNU",
    "sim_dir": "/path/to/sim",
    "array": False,
    "array_max": None,
    "slurm_id_file": "jobid.txt",
}


def test_init_valid_path():
    cs = ClusterSubmitter(TEMPLATE_PATH)
    assert cs.template is not None


def test_init_accepts_string_path():
    cs = ClusterSubmitter(str(TEMPLATE_PATH))
    assert cs.template is not None


def test_init_invalid_path():
    from jinja2 import TemplateNotFound

    with pytest.raises(TemplateNotFound):
        ClusterSubmitter("/nonexistent/path/to/template.slurm.j2")


def test_render_single_job():
    cs = ClusterSubmitter(TEMPLATE_PATH)
    result = cs.render(MINIMAL_CONTEXT)
    assert "#SBATCH --job-name=test_job" in result
    assert "#SBATCH --mem=2G" in result
    assert "#SBATCH --partition=short" in result
    assert "#SBATCH --time=2:00:00" in result
    assert 'simdir="/path/to/sim"' in result
    assert "#SBATCH --array" not in result


def test_render_array_job():
    context = {**MINIMAL_CONTEXT, "array": True, "array_max": 4}
    cs = ClusterSubmitter(TEMPLATE_PATH)
    result = cs.render(context)
    assert "#SBATCH --array=0-4" in result
    assert "directories.txt" in result
    assert 'simdir="/path/to/sim"' not in result


def test_render_slurm_id_file():
    cs = ClusterSubmitter(TEMPLATE_PATH)
    result = cs.render(MINIMAL_CONTEXT)
    assert "jobid.txt" in result


def test_render_missing_variable():
    from jinja2 import UndefinedError

    cs = ClusterSubmitter(TEMPLATE_PATH)
    incomplete = {k: v for k, v in MINIMAL_CONTEXT.items() if k != "name"}
    with pytest.raises(UndefinedError):
        cs.render(incomplete)
