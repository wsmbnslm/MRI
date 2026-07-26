"""Detect how many CPUs a process can actually use.

os.cpu_count() and PyTorch's own default thread count both typically
report the *host* machine's logical CPU count, not a container's
cgroup-enforced limit. In Kubernetes this bites two ways depending on
how the cluster enforces pod CPU limits:

- CFS quota (the default, from a pod's `resources.limits.cpu`): exposed
  via cgroup v2 `cpu.max` or cgroup v1 `cpu.cfs_quota_us`/`cpu.cfs_period_us`.
- cpuset pinning (the "static" CPU Manager policy): restricts which CPUs
  the process may even run on, visible via os.sched_getaffinity, not the
  quota files.

A pod can be limited by either mechanism (or neither), so this checks
both and returns the smallest constraint actually found, falling back to
os.cpu_count() if neither applies.
"""

import os


def _cgroup_v2_quota() -> float | None:
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        return None


def _cgroup_v1_quota() -> float | None:
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        return None if quota <= 0 else quota / period
    except (OSError, ValueError):
        return None


def _affinity_count() -> int | None:
    if not hasattr(os, "sched_getaffinity"):
        return None  # not available on macOS/Windows
    return len(os.sched_getaffinity(0))


def available_cpu_count() -> int:
    """Best-effort count of CPUs actually usable by this process."""
    candidates = [
        _cgroup_v2_quota(),
        _cgroup_v1_quota(),
        _affinity_count(),
    ]
    known = [c for c in candidates if c is not None]
    total = min(known) if known else (os.cpu_count() or 1)
    return max(1, int(total))