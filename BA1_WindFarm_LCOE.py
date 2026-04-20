# =============================================================================
# BA1_WindFarm_LCOE_wind.py
# Single-parameter Bees Algorithm (BA1) for offshore wind farm siting.
# Minimises LCOE over the North Sea by optimising (longitude, latitude).
#
#
# Original BA1: J. Cressall using - https://doi.org/10.3390/biomimetics9100634
# LCOE adaptation: Arthur Denison-Smith 
# Bugs & improve code: NH
#
# Fixes applied:
#   Fix 1 — incremental_kmeans_1d: replaced raw-distortion minimisation with
#            a penalised criterion so K is not always max_K.
#   Fix 2 — incremental_kmeans_1d: np.random.choice now samples from unique
#            values with replace=True to avoid ValueError on low-diversity data.
#   Fix 3 — Main loop: patch size now scales against the *original* patch_size
#            rather than compounding multiplicatively, preventing collapse.
#   Fix 4 — archive_update: restructured into three explicit distance zones so
#            the D_min < D_aj <= D_max medium zone is actually reachable.
#   Fix 5 — plot_north_sea: objective costs are passed in instead of being
#            recomputed, eliminating redundant evaluations.
#   Fix 6 — Module-level data loading: bounds are now computed lazily inside
#            BA1_implementation (and in __main__), not at import time.
#   Fix 7 — exploitation: mutable NumPy arrays removed as default arguments;
#            None sentinels used instead.
# =============================================================================

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time

from lcoe_model import compute_lcoe, lcoe_breakdown
from spatial_data_local import load_depth_raster, PORTS
from wind_energy import load_wind_mean_speed, coverage_bounds
from results_io import export_results

# =============================================================================
# SEARCH SPACE — bounding box for the North Sea (will be clipped to data coverage)
# =============================================================================
USER_LON_MIN, USER_LON_MAX = -3.0,  3.0
USER_LAT_MIN, USER_LAT_MAX = 51.0, 61.0

# How often (in main-loop iterations) to re-cluster and refresh patches.
# Set to 0 to disable re-clustering (original behaviour).
RE_CLUSTER_EVERY = 0 #disabled, found with recluster perform worse


# =============================================================================
# FIX 6 — Bounds computation is now a callable, not executed at import time
# =============================================================================
def _compute_search_bounds():
    """
    Clip the user bounding box to the overlap of:
      - GEBCO depth coverage
      - Wind NetCDF coverage
    This avoids wasting objective calls on locations that will return inf.
    """
    _, depth_lons, depth_lats = load_depth_raster()
    load_wind_mean_speed()
    w = coverage_bounds()

    lon_min = max(USER_LON_MIN, float(np.min(depth_lons)), float(w["lon_min"]))
    lon_max = min(USER_LON_MAX, float(np.max(depth_lons)), float(w["lon_max"]))
    lat_min = max(USER_LAT_MIN, float(np.min(depth_lats)), float(w["lat_min"]))
    lat_max = min(USER_LAT_MAX, float(np.max(depth_lats)), float(w["lat_max"]))

    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(
            "No overlap between user bounds, depth coverage and wind coverage."
        )

    return lon_min, lon_max, lat_min, lat_max


# These are populated in __main__ and inside BA1_implementation to avoid
# loading NetCDF files at import time (Fix 6).
_LON_MIN = _LON_MAX = _LAT_MIN = _LAT_MAX = None
_lowerBound = _upperBound = _bounds = None


def _ensure_bounds():
    """Lazily initialise module-level bounds the first time they are needed."""
    global _LON_MIN, _LON_MAX, _LAT_MIN, _LAT_MAX
    global _lowerBound, _upperBound, _bounds
    if _lowerBound is None:
        _LON_MIN, _LON_MAX, _LAT_MIN, _LAT_MAX = _compute_search_bounds()
        _lowerBound = np.array([_LON_MIN, _LAT_MIN], dtype=float)
        _upperBound = np.array([_LON_MAX, _LAT_MAX], dtype=float)
        _bounds = [[_LON_MIN, _LON_MAX], [_LAT_MIN, _LAT_MAX]]


def objective(x: np.ndarray) -> float:
    """Thin wrapper so BA1 calls a single function. x = [lon, lat]."""
    return float(compute_lcoe(float(x[0]), float(x[1])))


# =============================================================================
# BA1 CORE FUNCTIONS
# =============================================================================

def Generate_nghk(middle: float, rows: int, cols: int) -> np.ndarray:
    """Generate a (rows × cols) neighbourhood assignment matrix."""
    def d_triangular(mn: float, mid: float, mx: float) -> float:
        m = np.random.randint(1, 11)
        a = (mid - mn) / 10.0
        c = (mx - mid) / 10.0
        return float(np.random.uniform(mid - m * a, mid + m * c))

    M = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = d_triangular(0.0, middle, 1.0)
    return M


# FIX 7 — None sentinels instead of mutable NumPy arrays as default arguments
def exploitation(
    x: np.ndarray,
    PatchSize: np.ndarray,
    nghk: float,
    ub: np.ndarray | None = None,
    lb: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perturb position x within its neighbourhood to generate a worker bee.
    """
    _ensure_bounds()
    if ub is None:
        ub = _upperBound
    if lb is None:
        lb = _lowerBound

    r = nghk * PatchSize
    k = np.random.randint(0, x.size)
    y = np.copy(x)
    if np.ndim(PatchSize) > 0:
        y[k] += np.random.uniform(-r[k], r[k])
    else:
        y[k] += np.random.uniform(-float(r), float(r))
    return np.clip(y, lb, ub)


# FIX 1 & FIX 2 — Corrected k-means clustering
def incremental_kmeans_1d(x: np.ndarray, max_K: int = 5, n_init: int = 5, n_iter: int = 50) -> dict:
    """
    Fit k-means for K = 1 … max_K and select the best K using a penalised criterion:

        score(K) = distortion / n + alpha * K

    where alpha is scaled to the data variance so the penalty is meaningful
    regardless of the magnitude of x.  This prevents K always equalling max_K
    (Fix 1).

    Centroids are sampled from the *unique* values of x with replacement to
    avoid a ValueError when there are fewer distinct values than K (Fix 2).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    max_K = max(1, min(max_K if max_K else n - 1, n - 1))

    # Penalty weight — one tenth of the data variance; avoids dimensionality issue
    alpha = float(np.var(x)) * 0.1 if n > 1 else 0.0

    unique_x = np.unique(x)

    all_results = []
    for K in range(1, max_K + 1):
        best_run = None
        for _ in range(n_init):
            # FIX 2: sample from unique values with replacement
            centroids = np.random.choice(unique_x, size=K, replace=True).astype(float)
            for _ in range(n_iter):
                labels = np.argmin(np.abs(x[:, None] - centroids[None, :]), axis=1)
                new_c = np.array(
                    [
                        x[labels == k].mean() if np.any(labels == k) else centroids[k]
                        for k in range(K)
                    ]
                )
                if np.allclose(new_c, centroids):
                    break
                centroids = new_c
            labels = np.argmin(np.abs(x[:, None] - centroids[None, :]), axis=1)
            distortion = sum(
                float(np.sum((x[labels == k] - centroids[k]) ** 2))
                for k in range(K)
                if np.any(labels == k)
            )

            # FIX 1: penalised score — lower is better, but extra clusters cost alpha each
            penalised_score = (distortion / n) + alpha * K

            if best_run is None or penalised_score < best_run["penalised_score"]:
                best_run = {
                    "K": K,
                    "centroids": centroids.copy(),
                    "labels": labels.copy(),
                    "sum_distortion": distortion,
                    "penalised_score": penalised_score,
                }
        all_results.append(best_run)

    # FIX 1: choose K by the penalised score, not raw distortion
    chosen = min(all_results, key=lambda r: r["penalised_score"])
    return {
        "chosen_K": chosen["K"],
        "labels_for_chosen_K": chosen["labels"],
        "centroids_for_chosen_K": chosen["centroids"],
        "all_results": all_results,
    }


# FIX 4 — Restructured into three explicit distance zones
def archive_update(
    archive: list,
    xj: np.ndarray,
    fj: float,
    bounds: list,
    max_archive: int = 5,
    D_min: float = 0.12,
    D_max: float = 0.25,
    best_so_far_value: float | None = None,
) -> tuple:
    """
    Update the archive with candidate (xj, fj).

    Three explicit distance zones (Fix 4):
      Zone A  D_aj > D_max          — clearly new region
      Zone B  D_min < D_aj <= D_max — medium proximity
      Zone C  D_aj <= D_min         — very close to an existing archive entry
    """
    xj = np.asarray(xj, dtype=float)
    fj = float(fj)
    bnds = np.asarray(bounds, dtype=float)
    lo, hi = bnds[:, 0], bnds[:, 1]
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)
    xj_s = (xj - lo) / span

    def dist_scaled(xi: np.ndarray) -> float:
        d = xj_s - (np.asarray(xi, dtype=float) - lo) / span
        return float(np.sqrt(np.dot(d, d)))

    if not archive:
        archive.append({"x": xj.copy(), "f": fj})
        return archive, "added_first", {}

    dists = [dist_scaled(item["x"]) for item in archive]
    a = int(np.argmin(dists))
    D_aj = float(dists[a])
    b = int(np.argmax([item["f"] for item in archive]))

    # ------------------------------------------------------------------ Zone A
    if D_aj > D_max:
        # Clearly a new region of the search space
        if len(archive) < max_archive:
            archive.append({"x": xj.copy(), "f": fj})
        elif fj < archive[b]["f"]:
            archive[b] = {"x": xj.copy(), "f": fj}
        return archive, "new_region", {"a": a, "b": b, "D_aj": D_aj}

    # ------------------------------------------------------------------ Zone B
    if D_aj > D_min:
        # Medium proximity — only replace if this is the best known overall
        if best_so_far_value is not None and fj <= float(best_so_far_value):
            archive[a] = {"x": xj.copy(), "f": fj}
            return archive, "replaced_best_medium", {"a": a, "b": b, "D_aj": D_aj}
        # Otherwise keep the existing archive entry but don't discard entirely
        return archive, "medium_proximity_kept", {"a": a, "b": b, "D_aj": D_aj}

    # ------------------------------------------------------------------ Zone C
    # Very close to an existing entry — only update if strictly better
    if best_so_far_value is not None and fj <= float(best_so_far_value):
        archive[a] = {"x": xj.copy(), "f": fj}
        return archive, "replaced_best", {"a": a, "b": b, "D_aj": D_aj}
    if fj < archive[a]["f"]:
        archive[a] = {"x": xj.copy(), "f": fj}
        return archive, "replaced_better", {"a": a, "b": b, "D_aj": D_aj}

    return archive, "discarded", {"a": a, "b": b, "D_aj": D_aj}


# =============================================================================
# HELPER — build patches from a bee population
# =============================================================================
def _build_patches(bees: list, patch_size: np.ndarray, counter: int) -> tuple[list, np.ndarray, int]:
    """
    Cluster bees by distance from the best, then create one patch per cluster.
    Returns (patches, ssize, chosen_K).
    """
    best_pos = bees[0]["position"]
    for b in bees:
        b["distance"] = float(np.linalg.norm(best_pos - b["position"]))

    clustering = incremental_kmeans_1d(np.array([b["distance"] for b in bees]))
    chosen_K = clustering["chosen_K"]
    for i, b in enumerate(bees):
        b["cluster"] = int(clustering["labels_for_chosen_K"][i] + 1)

    cluster_sizes = [
        sum(1 for b in bees if b["cluster"] == c) for c in range(1, chosen_K + 1)
    ]

    patches: list = []
    for c in range(1, chosen_K + 1):
        cb = sorted(
            [b for b in bees if b["cluster"] == c], key=lambda b: b["cost"]
        )
        if cb:
            patches.append(
                {
                    "position": cb[0]["position"].copy(),
                    "cost": float(cb[0]["cost"]),
                    "size": patch_size.copy(),
                    "stagnated": 0,
                    "counter": counter,
                    "recruited": int(cluster_sizes[c - 1]),
                }
            )

    ssize = np.linspace(0.0, 1.0, num=len(patches)) if patches else np.array([])
    return patches, ssize, chosen_K


# =============================================================================
# BA1 MAIN LOOP
# =============================================================================
def BA1_implementation(population_size: int, max_evals: int):
    """
    Single-parameter Bees Algorithm (BA1) with consistent evaluation budgeting.

    Returns
    -------
    iters           : int   — number of main-loop iterations completed
    opt_cost        : list  — best-so-far LCOE after each evaluation
    counter         : int   — total objective evaluations performed
    best_position   : ndarray | None
    search_points   : ndarray  — all evaluated positions
    search_costs    : ndarray  — corresponding LCOE values (Fix 5)
    archive         : list
    """
    _ensure_bounds()

    archive: list = []
    dim = 2
    counter = 0
    search_points: list = []
    search_costs_list: list = []   # FIX 5: track costs at evaluation time
    opt_cost: list = []
    patch_size = _upperBound - _lowerBound   # original full-range patch size

    best_sol = {"cost": float("inf"), "position": None}

    def eval_and_track(pos: np.ndarray) -> float:
        nonlocal counter, best_sol
        cost = float(objective(pos))
        counter += 1
        search_points.append(pos.copy())
        search_costs_list.append(cost)      # FIX 5
        if cost < best_sol["cost"]:
            best_sol = {"cost": cost, "position": pos.copy()}
        opt_cost.append(best_sol["cost"])
        return cost

    # --- Scout initialisation ---
    bees: list = []
    for _ in range(population_size):
        pos = np.random.uniform(_lowerBound, _upperBound, size=dim)
        cost = eval_and_track(pos)
        bees.append(
            {
                "position": pos,
                "cost": cost,
                "size": patch_size.copy(),
                "stagnated": 0,
                "cluster": 0,
                "counter": counter,
                "distance": None,
            }
        )

    bees.sort(key=lambda b: b["cost"])

    archive, _, _ = archive_update(
        archive,
        bees[0]["position"],
        bees[0]["cost"],
        _bounds,
        best_so_far_value=best_sol["cost"],
    )

    patches, ssize, chosen_K = _build_patches(bees, patch_size, counter)

    # --- Main loop ---
    iters = 0
    while counter < max_evals:
        iters += 1

        # FIX 8 — Periodic re-clustering so the patch structure stays current #note I made this 0 as it not produce better result
        if RE_CLUSTER_EVERY > 0 and iters > 1 and (iters % RE_CLUSTER_EVERY) == 0:
            # Rebuild a fresh scout population from the current patch centres
            scout_bees = [
                {
                    "position": p["position"].copy(),
                    "cost": p["cost"],
                    "size": patch_size.copy(),
                    "stagnated": 0,
                    "cluster": 0,
                    "counter": counter,
                    "distance": None,
                }
                for p in patches
            ]
            if len(scout_bees) > 1:
                scout_bees.sort(key=lambda b: b["cost"])
                patches, ssize, chosen_K = _build_patches(scout_bees, patch_size, counter)

        for i, patch in enumerate(patches):
            if counter >= max_evals:
                break
            if patch["recruited"] <= 0:
                continue

            assignment = Generate_nghk(ssize[i], 1, patch["recruited"]).ravel()
            best_worker: dict = {"cost": float("inf")}

            for j in range(patch["recruited"]):
                if counter >= max_evals:
                    break
                worker_pos = exploitation(
                    patch["position"], patch["size"], float(assignment[j])
                )
                worker_cost = eval_and_track(worker_pos)

                if worker_cost < best_worker["cost"]:
                    best_worker = {
                        "position": worker_pos,
                        "cost": worker_cost,
                        "size": patch["size"].copy(),
                    }

            if best_worker["cost"] < patch["cost"]:
                patch.update(
                    {
                        "position": best_worker["position"].copy(),
                        "cost": best_worker["cost"],
                        "stagnated": 0,
                        "counter": counter,
                    }
                )
            else:
                patch["stagnated"] += 1
                # FIX 3 — Scale against the *original* patch_size, not iteratively
                progress = counter / float(max_evals)
                shrink_factor = 1.0 - (3.0 * progress / 4.0)
                patch["size"] = patch_size * shrink_factor

                stagnation_limit = int(round(population_size / max(1, chosen_K)))
                if patch["stagnated"] >= stagnation_limit and counter < max_evals:
                    patch["position"] = np.random.uniform(
                        _lowerBound, _upperBound, size=dim
                    )
                    patch["cost"] = eval_and_track(patch["position"])
                    patch["size"] = patch_size.copy()
                    patch["stagnated"] = 0
                    patch["counter"] = counter

            archive, _, _ = archive_update(
                archive,
                patch["position"],
                patch["cost"],
                _bounds,
                best_so_far_value=best_sol["cost"],
            )

        patches.sort(key=lambda p: p["cost"])

    return (
        iters,
        opt_cost,
        counter,
        best_sol["position"],
        np.array(search_points),
        np.array(search_costs_list),   # FIX 5
        archive,
    )


# =============================================================================
# PLOTTING
# =============================================================================
# FIX 5 — Accept pre-computed costs instead of re-evaluating
def plot_north_sea(
    search_points: np.ndarray,
    search_costs: np.ndarray,
    best_pos: np.ndarray | None,
    archive: list,
) -> None:
    """Plot evaluated sites, best location and archive on a North Sea map."""
    _ensure_bounds()
    pts = np.asarray(search_points, dtype=float)
    costs = np.asarray(search_costs, dtype=float)  # FIX 5: no re-evaluation
    valid = pts[~np.isinf(costs)]

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": proj})

    # UK / coastline context
    ax.add_feature(cfeature.LAND, facecolor="lightgrey", edgecolor="black", linewidth=0.6)
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4)

    if len(valid):
        ax.scatter(
            valid[:, 0], valid[:, 1], s=5, alpha=0.25,
            label="Evaluated sites", transform=proj,
        )

    if best_pos is not None:
        bp = np.asarray(best_pos).ravel()
        ax.scatter(
            bp[0], bp[1], s=250, marker="X", color="gold",
            edgecolors="black", zorder=5,
            label=f"Best site ({bp[0]:.2f}°E, {bp[1]:.2f}°N)",
            transform=proj,
        )

    if archive:
        A = np.array([item["x"] for item in archive])
        ax.scatter(
            A[:, 0], A[:, 1], s=80, marker="o",
            edgecolors="black", facecolors="none",
            zorder=4, label="Archived optima",
            transform=proj,
        )

    px, py = zip(*list(PORTS.values()))
    ax.scatter(px, py, s=100, marker="^", color="red", zorder=6,
               label="Ports", transform=proj)

    # Alternate label offsets to reduce overlap with 12+ ports
    port_items = list(PORTS.items())
    for i, (name, (lon, lat)) in enumerate(port_items):
        if i % 2 == 0:
            xytext = (6, 4)
            ha = "left"
        else:
            xytext = (-6, -8)
            ha = "right"
        ax.annotate(
            name, (lon, lat), textcoords="offset points",
            xytext=xytext, fontsize=7, ha=ha,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
        )

    ax.set_xlim(_LON_MIN, _LON_MAX)
    ax.set_ylim(_LAT_MIN, _LAT_MAX)
    ax.set_xlabel("Longitude (°E)", fontsize=13)
    ax.set_ylabel("Latitude (°N)", fontsize=13)
    ax.set_title(
        "BA1 — Offshore Wind Farm LCOE Optimisation (North Sea, incl. wind)",
        fontsize=14,
    )
    ax.gridlines(draw_labels=True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_convergence(cost_history: list) -> None:
    """Plot the best-so-far LCOE convergence curve."""
    y = np.asarray(cost_history, dtype=float)
    if y.size == 0:
        print("[plot_convergence] No history to plot.")
        return
    x = np.arange(1, len(y) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel("Objective function evaluations")
    plt.ylabel("Best LCOE (£/MWh)")
    plt.title("BA1 Convergence — LCOE Minimisation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # FIX 6 — Data is loaded here, not at import time
    load_depth_raster()
    load_wind_mean_speed()
    _ensure_bounds()

    max_evals = 5000
    independentRuns = 10 # ≥10 recommended for meaningful summary statistics
    populationSize = 30

    best_objective_values: list = []
    best_positions: list = []
    run_times: list = []

    all_opt_costs: list = []
    all_search_points: list = []
    all_search_costs: list = []
    all_archives: list = []

    print(
        f"[bounds] Using search box: "
        f"lon {_LON_MIN:.2f}..{_LON_MAX:.2f}, "
        f"lat {_LAT_MIN:.2f}..{_LAT_MAX:.2f}"
    )

    for run in range(independentRuns):
        t0 = time.perf_counter()
        (
            iters,
            opt_cost,
            counter,
            best_pos,
            search_points,
            search_costs,   # FIX 5
            archive,
        ) = BA1_implementation(population_size=populationSize, max_evals=max_evals)
        t1 = time.perf_counter()

        run_times.append(t1 - t0)
        best_cost = opt_cost[-1] if opt_cost else float("inf")
        best_objective_values.append(best_cost)
        best_positions.append(best_pos)

        all_opt_costs.append(opt_cost)
        all_search_points.append(search_points)
        all_search_costs.append(search_costs)
        all_archives.append(archive)

        if best_pos is None:
            print(
                f"Run {run + 1:3d} | LCOE: {best_cost:.2f} £/MWh | "
                f"Pos: (no feasible site) | Time: {t1 - t0:.1f}s"
            )
        else:
            print(
                f"Run {run + 1:3d} | LCOE: {best_cost:.2f} £/MWh | "
                f"Pos: ({best_pos[0]:.3f}°E, {best_pos[1]:.3f}°N) | "
                f"Time: {t1 - t0:.1f}s"
            )

    # =================================================================
    # SUMMARY STATISTICS OVER MULTIPLE RUNS
    # =================================================================
    obj_vals = np.array(best_objective_values)
    times    = np.array(run_times)

    best_run_idx  = int(np.argmin(obj_vals))
    worst_run_idx = int(np.argmax(obj_vals))
    best_loc      = best_positions[best_run_idx]
    worst_loc     = best_positions[worst_run_idx]

    # Count how many evaluations each run actually used
    evals_per_run = np.array([len(oc) for oc in all_opt_costs])

    print("\n" + "=" * 65)
    print(f"  SUMMARY STATISTICS  —  {independentRuns} independent runs")
    print(f"  (population = {populationSize}, max evals = {max_evals})")
    print("=" * 65)

    print(f"\n{'':>4s}{'Metric':<22s}{'LCOE (£/MWh)':>14s}{'Time (s)':>12s}")
    print(f"{'':>4s}{'-' * 48}")
    print(f"{'':>4s}{'Mean':<22s}{np.mean(obj_vals):>14.4f}{np.mean(times):>12.2f}")
    print(f"{'':>4s}{'Median':<22s}{np.median(obj_vals):>14.4f}{np.median(times):>12.2f}")
    print(f"{'':>4s}{'Std deviation':<22s}{np.std(obj_vals, ddof=1) if len(obj_vals) > 1 else 0.0:>14.4f}{np.std(times, ddof=1) if len(times) > 1 else 0.0:>12.2f}")
    print(f"{'':>4s}{'Best (min)':<22s}{np.min(obj_vals):>14.4f}{np.min(times):>12.2f}")
    print(f"{'':>4s}{'Worst (max)':<22s}{np.max(obj_vals):>14.4f}{np.max(times):>12.2f}")
    print(f"{'':>4s}{'Range':<22s}{np.ptp(obj_vals):>14.4f}{np.ptp(times):>12.2f}")

    if len(obj_vals) >= 4:
        q1, q3 = np.percentile(obj_vals, [25, 75])
        iqr = q3 - q1
        print(f"{'':>4s}{'25th percentile':<22s}{q1:>14.4f}")
        print(f"{'':>4s}{'75th percentile':<22s}{q3:>14.4f}")
        print(f"{'':>4s}{'IQR':<22s}{iqr:>14.4f}")


    # Best and worst run locations
    print(f"\n  Best run  (#{best_run_idx + 1}): ", end="")
    if best_loc is not None:
        print(f"LCOE = {obj_vals[best_run_idx]:.4f} £/MWh  "
              f"at ({best_loc[0]:.3f}°E, {best_loc[1]:.3f}°N)")
    else:
        print("(no feasible site)")

    print(f"  Worst run (#{worst_run_idx + 1}): ", end="")
    if worst_loc is not None:
        print(f"LCOE = {obj_vals[worst_run_idx]:.4f} £/MWh  "
              f"at ({worst_loc[0]:.3f}°E, {worst_loc[1]:.3f}°N)")
    else:
        print("(no feasible site)")

    # LCOE breakdown for best overall location
    if best_loc is not None:
        print("\n  Cost/resource breakdown for best location:")
        for k, v in lcoe_breakdown(best_loc[0], best_loc[1]).items():
            print(
                f"    {k:<30s}: {v:.6g}"
                if isinstance(v, float)
                else f"    {k:<30s}: {v}"
            )

    print("=" * 65)

    # =================================================================
    # CSV EXPORT — write results for downstream stats/plot analysis
    # =================================================================
    export_results(
        algorithm="BA1",
        best_objective_values=best_objective_values,
        best_positions=best_positions,
        run_times=run_times,
        all_opt_costs=all_opt_costs,
        all_search_points=all_search_points,
        all_search_costs=all_search_costs,
        all_archives=all_archives,
    )

    # =================================================================
    # PLOTS
    # =================================================================
    if best_loc is not None:
        # 1. Convergence curve for the best run
        plot_convergence(all_opt_costs[best_run_idx])

        # 2. North Sea map for the best run
        plot_north_sea(
            all_search_points[best_run_idx],
            all_search_costs[best_run_idx],
            best_loc,
            all_archives[best_run_idx],
        )

        # 3. Box plot of LCOE across runs (only useful with ≥2 runs)
        if independentRuns >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            axes[0].boxplot(obj_vals, vert=True, patch_artist=True,
                            boxprops=dict(facecolor="lightblue"))
            axes[0].set_ylabel("Best LCOE (£/MWh)")
            axes[0].set_title("LCOE Distribution Across Runs")
            axes[0].grid(True, alpha=0.3)

            axes[1].boxplot(times, vert=True, patch_artist=True,
                            boxprops=dict(facecolor="lightyellow"))
            axes[1].set_ylabel("Run time (s)")
            axes[1].set_title("Run Time Distribution")
            axes[1].grid(True, alpha=0.3)

            fig.suptitle(
                f"BA1 Multi-Run Summary ({independentRuns} runs, "
                f"{max_evals} evals each)",
                fontsize=13,
            )
            plt.tight_layout()
            plt.show()

        # 4. Overlay convergence curves from all runs
        if independentRuns >= 2:
            plt.figure(figsize=(8, 5))
            for i, oc in enumerate(all_opt_costs):
                label = f"Run {i + 1}" if independentRuns <= 10 else None
                alpha = 0.35 if i != best_run_idx else 1.0
                lw    = 1.0  if i != best_run_idx else 2.0
                plt.plot(oc, alpha=alpha, linewidth=lw, label=label)
            # Mean convergence
            max_len = max(len(oc) for oc in all_opt_costs)
            padded = np.full((independentRuns, max_len), np.nan)
            for i, oc in enumerate(all_opt_costs):
                padded[i, : len(oc)] = oc
            mean_curve = np.nanmean(padded, axis=0)
            plt.plot(mean_curve, color="black", linewidth=2.5,
                     linestyle="--", label="Mean")
            plt.xlabel("Objective function evaluations")
            plt.ylabel("Best LCOE (£/MWh)")
            plt.title("BA1 Convergence — All Runs Overlaid")
            plt.legend(fontsize=8, loc="upper right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        
