# =============================================================================
# results_io.py
# Shared CSV export helpers for the BA1, GWO, WOA and PSO wind-farm scripts.
# =============================================================================

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Sequence

import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def export_results(
    algorithm: str,
    best_objective_values: Sequence[float],
    best_positions: Sequence,
    run_times: Sequence[float],
    all_opt_costs: Sequence[Sequence[float]],
    all_search_points: Sequence[Sequence],
    all_search_costs: Sequence[Sequence[float]],
    all_archives: Sequence,
    out_dir: str = "results",
) -> dict:
    """
    Write the four CSVs for one batch of independent runs.

    Parameters mirror the variable names already used in the four algorithm
    scripts so the call site is just `export_results("BA1", ...)`.

    Returns a dict of the four file paths written, for logging.
    """
    _ensure_dir(out_dir)
    ts = _timestamp()
    n_runs = len(best_objective_values)

    summary_path     = os.path.join(out_dir, f"{algorithm}_summary_{ts}.csv")
    convergence_path = os.path.join(out_dir, f"{algorithm}_convergence_{ts}.csv")
    search_path      = os.path.join(out_dir, f"{algorithm}_search_points_{ts}.csv")
    archive_path     = os.path.join(out_dir, f"{algorithm}_archive_{ts}.csv")

    # ---------- 1. SUMMARY (one row per run) -------------------------------
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["algorithm", "run", "lcoe", "runtime_s",
             "evaluations", "best_lon", "best_lat"]
        )
        for i in range(n_runs):
            pos = best_positions[i]
            lon = float(pos[0]) if pos is not None else ""
            lat = float(pos[1]) if pos is not None else ""
            w.writerow([
                algorithm,
                i + 1,
                float(best_objective_values[i]),
                float(run_times[i]),
                len(all_opt_costs[i]),
                lon,
                lat,
            ])

    # ---------- 2. CONVERGENCE (wide: rows = eval_index, cols = runs) -----
    max_len = max((len(oc) for oc in all_opt_costs), default=0)
    padded = np.full((n_runs, max_len), np.nan, dtype=float)
    for i, oc in enumerate(all_opt_costs):
        if len(oc) > 0:
            padded[i, : len(oc)] = np.asarray(oc, dtype=float)

    with open(convergence_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["eval_index"] + [f"run_{i + 1}" for i in range(n_runs)]
        w.writerow(header)
        for k in range(max_len):
            row = [k + 1] + [
                ("" if np.isnan(padded[i, k]) else float(padded[i, k]))
                for i in range(n_runs)
            ]
            w.writerow(row)

    # ---------- 3. SEARCH POINTS (long: every evaluation) ------------------
    with open(search_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "run", "eval_index", "lon", "lat", "lcoe"])
        for i in range(n_runs):
            pts   = all_search_points[i]
            costs = all_search_costs[i]
            n_pts = min(len(pts), len(costs))
            for k in range(n_pts):
                p = pts[k]
                # Accept either np.array([lon, lat]) or (lon, lat) tuples
                lon = float(p[0])
                lat = float(p[1])
                c   = costs[k]
                if c is None or (isinstance(c, float) and np.isnan(c)):
                    continue
                w.writerow([algorithm, i + 1, k + 1, lon, lat, float(c)])

    # ---------- 4. ARCHIVE (long: secondary optima per run) ----------------
    with open(archive_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "run", "lon", "lat", "lcoe"])
        for i in range(n_runs):
            arch = all_archives[i] if i < len(all_archives) else None
            if arch is None:
                continue
            for entry in arch:
                # Archive entries can be either (pos, cost) tuples or
                # arrays/lists of [lon, lat, cost] / [lon, lat, ..., cost].
                lon = lat = cost = None
                try:
                    if isinstance(entry, (tuple, list)) and len(entry) == 2 \
                       and hasattr(entry[0], "__len__"):
                        # (pos_array, cost)
                        pos, cost = entry
                        lon, lat = float(pos[0]), float(pos[1])
                        cost = float(cost)
                    else:
                        # flat sequence: [lon, lat, ..., cost]
                        seq = list(entry)
                        if len(seq) >= 3:
                            lon, lat = float(seq[0]), float(seq[1])
                            cost = float(seq[-1])
                except (TypeError, ValueError, IndexError):
                    continue
                if lon is None or cost is None:
                    continue
                if isinstance(cost, float) and np.isnan(cost):
                    continue
                w.writerow([algorithm, i + 1, lon, lat, cost])

    paths = {
        "summary":     summary_path,
        "convergence": convergence_path,
        "search":      search_path,
        "archive":     archive_path,
    }

    print("\n[results_io] CSVs written to ./{}/:".format(out_dir))
    for k, v in paths.items():
        print(f"  {k:<12s} {v}")

    return paths
