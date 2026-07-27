"""How long each checkpoint-restart segment is allowed to run.

A campaign reaches its bin target through a chain of *segments*, each a separate
SLURM allocation. This module decides one number per segment: ``CPU_MAX``, ALF's
own soft wall time in hours.

That number is what keeps the run out of the ``long`` queue. ALF times every bin
and breaks the bin loop once fewer than ~1.5 bins still fit in ``CPU_MAX``
(``ALF/Prog/control_mod.F90:make_truncation``), having already flushed both
``data.h5`` and ``confout_0`` that bin -- so a segment that ends on its budget
ends cleanly, and the next one resumes from the configuration it left behind.

Capping ``CPU_MAX`` below the ``medium`` partition's limit therefore turns an
arbitrarily long run into a sequence of jobs that each fit the 48 h queue.
``ClusterSubmitter._select_partition`` picks the *smallest* partition that fits,
so a chain with little work left asks for a small budget and lands in ``short``
-- the queueing win is free, not something the caller has to arrange.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentPolicy:
    """Wall-time budget rules shared by every chain in a campaign.

    ``max_partition`` is the largest queue a segment may target; its limit comes
    from the environment's ``partition_rules``. ``margin`` keeps ``CPU_MAX``
    below that limit so SLURM's own ``--time`` (which the submitter sets to
    ``CPU_MAX * 1.1``, capped at the partition limit) still leaves ALF room to
    truncate and write before the hard kill.

    ``safety`` inflates the estimated remaining time, since under-asking costs a
    whole extra segment (queue wait included) while over-asking costs only
    priority. ``max_segments`` bounds the retry budget so a mis-estimate cannot
    keep requeueing forever.
    """

    max_partition: str = "medium"
    margin: float = 0.95
    safety: float = 1.3
    max_segments: int = 6
    # Floor on a segment's budget: ALF needs a few bins before make_truncation
    # has a bin duration to extrapolate from, and below this the queue wait
    # dwarfs the run. Lower it only for tests.
    min_hours: float = 0.25

    def max_hours(self, partition_rules: dict) -> float:
        """Largest ``CPU_MAX`` any segment may ask for, in hours.

        Unbounded where there is no queue to satisfy (the ``local`` and
        ``debug`` executors define no partitions): a segment there simply runs
        until the chain is done, which is one segment.
        """
        if not partition_rules:
            return float("inf")
        try:
            limit = float(partition_rules[self.max_partition]["max_hours"])
        except (KeyError, TypeError) as exc:
            raise SystemExit(
                f"SegmentPolicy.max_partition={self.max_partition!r} is not in this "
                f"environment's partition_rules ({sorted(partition_rules)})"
            ) from exc
        return limit * self.margin

    def cpu_max(
        self, remaining_bins: int, hours_per_bin: float, partition_rules: dict
    ) -> float:
        """``CPU_MAX`` (hours) for a segment with ``remaining_bins`` still to do.

        Sized to *finish* the chain when that fits the partition cap, and pinned
        at the cap otherwise -- in which case the chain simply needs another
        segment, which is the whole point of the mechanism.
        """
        want = remaining_bins * hours_per_bin * self.safety
        return min(max(want, self.min_hours), self.max_hours(partition_rules))

    def segments_needed(
        self, total_bins: int, hours_per_bin: float, partition_rules: dict
    ) -> int:
        """How many segments the *initial* estimate says the target will take.

        Used only to size the requeue budget; the worker re-measures on every
        attempt, and :meth:`~py_alf.campaign.campaign.Campaign.reconcile`
        covers any shortfall, so this being wrong is recoverable either way.
        """
        cap = self.max_hours(partition_rules)
        total_hours = total_bins * hours_per_bin * self.safety
        return max(1, min(self.max_segments, math.ceil(total_hours / cap)))
