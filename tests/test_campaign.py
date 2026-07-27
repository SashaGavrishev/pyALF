"""Tests for the campaign layer in py_alf.campaign.

These cover the decisions a campaign makes without running ALF: how long a
segment may ask for, what a chain is called, what the ledger remembers, and how
a chain's progress is judged. Anything needing a real ALF binary lives in the
consuming project's integration tests.
"""

import json
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import submitit

from py_alf.campaign import (
    Ledger,
    SegmentPolicy,
    chain_id,
    chain_point,
    run_segment,
)
from py_alf.campaign.campaign import Campaign, ChainStatus
from py_alf.campaign.worker import _claim_running, _clear_own_running
from py_alf.simulation import Simulation

# A three-tier cluster of the shape these policies exist to cope with.
RULES = {
    "short": {"max_hours": 2},
    "medium": {"max_hours": 48},
    "long": {"max_hours": 336},
}


# --- SegmentPolicy ----------------------------------------------------------


def test_cpu_max_never_exceeds_the_capped_partition():
    """A budget request must stay inside `medium`, however much work is left."""
    policy = SegmentPolicy()
    cap = policy.max_hours(RULES)
    assert cap == pytest.approx(48 * 0.95)
    for remaining, hours_per_bin in [(1, 0.5), (100, 0.5), (100_000, 10.0)]:
        assert policy.cpu_max(remaining, hours_per_bin, RULES) <= cap


def test_cpu_max_shrinks_to_the_work_that_is_left():
    """A nearly-finished chain asks for little, so it can land in a fast queue."""
    policy = SegmentPolicy(min_hours=0.0)
    small = policy.cpu_max(1, 0.5, RULES)
    large = policy.cpu_max(100, 0.5, RULES)
    assert small < large
    assert small == pytest.approx(1 * 0.5 * policy.safety)


def test_cpu_max_respects_the_floor():
    """Below the floor the queue wait dwarfs the run, so do not ask for less."""
    policy = SegmentPolicy(min_hours=0.25)
    assert policy.cpu_max(1, 1e-9, RULES) == pytest.approx(0.25)


def test_max_hours_unbounded_without_partition_rules():
    """Local and debug executors define no partitions, so nothing caps them."""
    assert SegmentPolicy().max_hours({}) == float("inf")


def test_segments_needed_grows_with_the_work():
    policy = SegmentPolicy()
    assert policy.segments_needed(1, 0.01, RULES) == 1
    assert policy.segments_needed(100, 2.0, RULES) > 1
    # However bad the estimate, never queue more attempts than allowed.
    assert policy.segments_needed(10**9, 10.0, RULES) <= policy.max_segments


def test_unknown_max_partition_is_rejected():
    with pytest.raises(SystemExit):
        SegmentPolicy(max_partition="nonexistent").max_hours(RULES)


# --- chain_id ---------------------------------------------------------------


def _sim(sim_dir, ham_name="Hubbard"):
    """Stand-in for a Simulation: chain_id reads only these two attributes.

    A real Simulation would need an ALF source tree to validate parameters
    against, which these unit tests deliberately do not require.
    """
    stub = MagicMock()
    stub.__class__ = Simulation
    stub.ham_name = ham_name
    stub.sim_dir = str(sim_dir)
    return stub


def test_chain_id_is_deterministic_and_unique(tmp_path):
    a = _sim(tmp_path / "Hubbard_L1=4")
    b = _sim(tmp_path / "Hubbard_L1=6")
    assert chain_id(a, 1) == chain_id(a, 1)
    assert chain_id(a, 1) != chain_id(b, 1), "parameters must distinguish chains"
    assert chain_id(a, 1) != chain_id(a, 2), "mc_seed must distinguish chains"
    assert chain_id(a, 1) != chain_id(_sim(tmp_path / "Hubbard_L1=4", "Other"), 1)


def test_chain_id_survives_relocating_the_data(tmp_path):
    """The id names the chain, not where its output happens to live.

    It hashes the directory's *name* -- already a canonical rendering of every
    Hamiltonian parameter -- so moving the data root renumbers nothing.
    """
    here = _sim(tmp_path / "here" / "Hubbard_L1=4")
    there = _sim(tmp_path / "there" / "Hubbard_L1=4")
    assert chain_id(here, 1) == chain_id(there, 1)


# --- Ledger -----------------------------------------------------------------


def _ledger(tmp_path, **kwargs):
    return Ledger.new(
        tmp_path / "c.json",
        name="c",
        target_bins=100,
        policy={},
        **kwargs,
    )


def test_ledger_round_trips(tmp_path):
    led = _ledger(tmp_path)
    led.data["chains"]["abc"] = {"sim_dir": "/d", "point": {"disorder_seed": 5}}
    led.save()
    assert Ledger.load(tmp_path / "c.json").chains["abc"]["point"]["disorder_seed"] == 5


def test_ledger_save_is_atomic(tmp_path):
    """A crashed driver must never leave a half-written index behind."""
    led = _ledger(tmp_path)
    led.save()
    assert not list(tmp_path.glob("*.tmp"))
    json.loads((tmp_path / "c.json").read_text())


def test_loading_a_missing_ledger_is_a_clean_error(tmp_path):
    with pytest.raises(SystemExit):
        Ledger.load(tmp_path / "absent.json")


def test_chain_point_folds_in_legacy_top_level_keys():
    """Ledgers written before the seed moved into `point` must still read."""
    legacy = {"sim_dir": "/d", "disorder_seed": 42, "point": {"N": 8}}
    assert chain_point(legacy) == {"N": 8, "disorder_seed": 42}
    # A point that already carries the key wins over the legacy copy.
    both = {"disorder_seed": 42, "point": {"disorder_seed": 7}}
    assert chain_point(both)["disorder_seed"] == 7


def test_by_point_key_indexes_and_tolerates_absent_keys(tmp_path):
    """A model with no disorder simply has no such key -- that is not an error."""
    led = _ledger(tmp_path)
    led.data["chains"] = {
        "a": {"point": {"disorder_seed": 1}},
        "b": {"point": {"disorder_seed": 2}},
        "c": {"point": {"beta": 5.0}},
    }
    assert led.by_point_key("disorder_seed") == {1: "a", 2: "b"}
    assert led.by_point_key("nothing_uses_this") == {}
    assert led.chain_ids_by_point_key("disorder_seed") == {1: ["a"], 2: ["b"]}


def test_chain_ids_by_point_key_keeps_every_chain_sharing_a_value(tmp_path):
    """A grid reuses seeds across points, so one value maps to several chains."""
    led = _ledger(tmp_path)
    led.data["chains"] = {
        "a": {"point": {"disorder_seed": 1, "N": 8}},
        "b": {"point": {"disorder_seed": 1, "N": 10}},
    }
    assert sorted(led.chain_ids_by_point_key("disorder_seed")[1]) == ["a", "b"]


def test_absorb_segment_records_merges_worker_output(tmp_path):
    """Only the node knows what a segment actually did; fold that back in."""
    sim_dir = tmp_path / "sim"
    (sim_dir / "segments").mkdir(parents=True)
    (sim_dir / "segments" / "000-7_0-x.json").write_text(
        json.dumps({"job_id": "7_0", "bins_before": 0, "bins_after": 40})
    )
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = {
        "sim_dir": str(sim_dir),
        "point": {},
        "segments": [{"index": 0, "job_id": "7_0"}],
    }
    assert led.absorb_segment_records() == 1
    assert led.chains["a"]["segments"][0]["bins_after"] == 40


def test_absorb_segment_records_ignores_unreadable_files(tmp_path):
    """A record truncated by a hard kill must not break the whole scan."""
    sim_dir = tmp_path / "sim"
    (sim_dir / "segments").mkdir(parents=True)
    (sim_dir / "segments" / "000-7_0-x.json").write_text("{ truncated")
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = {"sim_dir": str(sim_dir), "point": {}, "segments": []}
    led.absorb_segment_records()  # must not raise


# --- worker: checkpoint and the RUNNING mutex -------------------------------


def test_run_segment_is_checkpointable(tmp_path):
    """Without this attribute submitit fails a timed-out job instead of requeuing."""
    checkpoint = getattr(run_segment, "checkpoint", None) or getattr(
        run_segment, "__submitit_checkpoint__", None
    )
    assert checkpoint is not None
    resumed = checkpoint(_sim(tmp_path))
    assert isinstance(resumed, submitit.helpers.DelayedSubmission)
    assert resumed.function is run_segment


def test_clear_own_running_removes_only_this_job_s_file(tmp_path):
    """A requeued attempt reuses the job id, so that file is provably ours."""
    _claim_running(tmp_path, "42_1")
    (tmp_path / "RUNNING").write_text("ALF is running")
    assert _clear_own_running(tmp_path, "42_1") is True
    assert not (tmp_path / "RUNNING").exists()


def test_clear_own_running_leaves_a_foreign_file_alone(tmp_path):
    """RUNNING is a mutex: another job's ALF must be allowed to abort us."""
    _claim_running(tmp_path, "999_0")
    (tmp_path / "RUNNING").write_text("ALF is running")
    assert _clear_own_running(tmp_path, "42_1") is False
    assert (tmp_path / "RUNNING").exists()


def test_clear_own_running_leaves_an_unowned_file_alone(tmp_path):
    """No owner marker means we cannot prove it is ours, so do not touch it."""
    (tmp_path / "RUNNING").write_text("ALF is running")
    assert _clear_own_running(tmp_path, "42_1") is False
    assert (tmp_path / "RUNNING").exists()


def test_clear_own_running_is_a_noop_without_a_running_file(tmp_path):
    assert _clear_own_running(tmp_path, "42_1") is False


# --- Campaign.status: progress is judged by bins, not by SLURM --------------


def _campaign(tmp_path, chains=()):
    # status() reads submit_dir to look for submitit's timeout markers; the log
    # is absent here, which is the "cannot tell yet" path.
    submitter = MagicMock()
    submitter.submit_dir = tmp_path / "submit"
    return Campaign(
        name="c",
        chains=list(chains),
        target_bins=100,
        submitter=submitter,
        ledger_path=tmp_path / "c.json",
        partition_rules=RULES,
    )


def _status_with(tmp_path, bins, segments, slurm_state=None):
    """Run Campaign.status against a ledger with one chain in a chosen state."""
    led = _ledger(tmp_path)
    led.data["chains"]["a"] = {
        "sim_dir": str(tmp_path / "sim"),
        "point": {},
        "segments": segments,
    }
    led.save()
    states = {}
    if slurm_state is not None and segments:
        states = {segments[-1]["job_id"]: {"status": slurm_state}}

    camp = _campaign(tmp_path)
    with (
        patch("py_alf.campaign.campaign._get_slurm_status_bulk", return_value=states),
        patch("py_alf.campaign.campaign._bins_in_dir", return_value=bins),
    ):
        return camp.status(Ledger.load(tmp_path / "c.json"))[0]


def test_status_done_when_the_target_is_reached(tmp_path):
    assert (
        _status_with(tmp_path, 100, [{"index": 0, "job_id": "1_0"}]).verdict == "done"
    )


def test_status_unstarted_when_nothing_was_submitted(tmp_path):
    assert _status_with(tmp_path, 0, []).verdict == "unstarted"


def test_status_resumable_when_short_with_bins_on_disk(tmp_path):
    """A stopped chain holding bins is a restart, whatever SLURM called it."""
    got = _status_with(tmp_path, 40, [{"index": 0, "job_id": "1_0"}], "FAILED")
    assert got.verdict == "resumable"


def test_status_suspect_when_short_with_no_bins(tmp_path):
    """Zero bins after a run is a crash signature; requeuing it would loop.

    Its SLURM state is the same FAILED a wall-clock stop produces, which is
    exactly why the bin count and not the state decides.
    """
    got = _status_with(tmp_path, 0, [{"index": 0, "job_id": "1_0"}], "FAILED")
    assert got.verdict == "suspect"


def test_status_active_is_left_alone(tmp_path):
    got = _status_with(tmp_path, 10, [{"index": 0, "job_id": "1_0"}], "RUNNING")
    assert got.verdict == "active"
    assert got.active_job == "1_0"


def test_chain_status_complete_tracks_the_target():
    base = ChainStatus(
        chain_id="a",
        sim_dir="/d",
        point={},
        bins=99,
        target_bins=100,
        segments=1,
        active_job=None,
        last_state=None,
        timed_out=False,
        verdict="resumable",
    )
    assert not base.complete
    assert replace(base, bins=100).complete
