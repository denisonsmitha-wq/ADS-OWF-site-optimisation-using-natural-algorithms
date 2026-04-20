# =============================================================================
# PSO_WindFarm_LCOE_wind_stats.py
# Particle Swarm Optimisation (PSO) for offshore wind farm siting.
# Minimises LCOE over the North Sea by optimising (longitude, latitude).
#
# Drop-in replacement for BA1_WindFarm_LCOE_wind_stats.py — uses the same
# LCOE model, spatial data, wind resource, plotting and summary statistics.
#
# References:
#   Kennedy & Eberhart (1995) "Particle Swarm Optimization"
#   Shi & Eberhart (1998) linearly decreasing inertia weight
#
# Author: generated for Arthur Denison-Smith's BA1 optimisation project
#
# Fixes applied:
#   Fix 1 — gbest_pos initialised to a random in-bounds position rather than
#            None.  If the entire initial swarm is infeasible (all inf), the
#            velocity update  social = C2 * r2 * (gbest_pos - positions[i])
#            would crash with TypeError: unsupported operand type
#            'NoneType' - 'ndarray'.  gbest_cost stays inf, so the first
#            finite evaluation still correctly displaces this placeholder.

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
# SEARCH SPACE — same bounding box as BA1
# =============================================================================
USER_LON_MIN, USER_LON_MAX = -3.0,  3.0
USER_LAT_MIN, USER_LAT_MAX = 51.0, 61.0

# =============================================================================
# PSO HYPERPARAMETERS
# =============================================================================
W_START = 0.9       # Initial inertia weight (exploration)
W_END   = 0.4       # Final inertia weight   (exploitation)
C1      = 2.0       # Cognitive coefficient   (personal best attraction)
C2      = 2.0       # Social coefficient      (global best attraction)
V_CLAMP = 0.2       # Max velocity as fraction of search range per dimension

# =============================================================================
# BOUNDS (lazily initialised, same pattern as BA1)
# =============================================================================
_LON_MIN = _LON_MAX = _LAT_MIN = _LAT_MAX = None
_lowerBound = _upperBound = _bounds = None


def _compute_search_bounds():
    """Clip user box to overlap of GEBCO depth and wind NetCDF coverage."""
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


def _ensure_bounds():
    global _LON_MIN, _LON_MAX, _LAT_MIN, _LAT_MAX
    global _lowerBound, _upperBound, _bounds
    if _lowerBound is None:
        _LON_MIN, _LON_MAX, _LAT_MIN, _LAT_MAX = _compute_search_bounds()
        _lowerBound = np.array([_LON_MIN, _LAT_MIN], dtype=float)
        _upperBound = np.array([_LON_MAX, _LAT_MAX], dtype=float)
        _bounds = [[_LON_MIN, _LON_MAX], [_LAT_MIN, _LAT_MAX]]


def objective(x: np.ndarray) -> float:
    """Thin wrapper — x = [lon, lat]."""
    return float(compute_lcoe(float(x[0]), float(x[1])))


# =============================================================================
# PSO IMPLEMENTATION
# =============================================================================

def PSO_implementation(population_size: int, max_evals: int):
    """
    Standard Particle Swarm Optimisation with:
      - Linearly decreasing inertia weight (w: 0.9 → 0.4)
      - Cognitive (c1) and social (c2) acceleration coefficients
      - Velocity clamping to prevent particles flying out of bounds
      - Absorbing boundary handling (positions clamped, velocities zeroed at wall)

    Parameters
    ----------
    population_size : int
        Number of particles in the swarm.
    max_evals : int
        Maximum number of objective function evaluations.

    Returns
    -------
    iters           : int       — number of PSO iterations completed
    opt_cost        : list      — best-so-far LCOE after each evaluation
    counter         : int       — total objective evaluations performed
    best_position   : ndarray   — global best position found
    search_points   : ndarray   — all evaluated positions (N × 2)
    search_costs    : ndarray   — corresponding LCOE values (N,)
    archive         : list      — list of dicts with top-K distinct optima
    """
    _ensure_bounds()

    dim = 2
    counter = 0
    search_points: list = []
    search_costs_list: list = []
    opt_cost: list = []

    # Global best
    # Fix 1: initialise gbest_pos to a random in-bounds position rather than

    gbest_pos  = np.random.uniform(_lowerBound, _upperBound)   # Fix 1
    gbest_cost = float("inf")

    # Velocity limits per dimension
    span = _upperBound - _lowerBound
    v_max = V_CLAMP * span
    v_min = -v_max

    def eval_and_track(pos: np.ndarray) -> float:
        nonlocal counter, gbest_pos, gbest_cost
        cost = float(objective(pos))
        counter += 1
        search_points.append(pos.copy())
        search_costs_list.append(cost)
        if cost < gbest_cost:
            gbest_cost = cost
            gbest_pos = pos.copy()
        opt_cost.append(gbest_cost)
        return cost

    # ------------------------------------------------------------------
    # Initialise swarm
    # ------------------------------------------------------------------
    positions  = np.random.uniform(_lowerBound, _upperBound, size=(population_size, dim))
    velocities = np.random.uniform(v_min, v_max, size=(population_size, dim))

    # Personal bests
    pbest_pos  = np.copy(positions)
    pbest_cost = np.full(population_size, float("inf"))

    # Evaluate initial positions
    for i in range(population_size):
        cost = eval_and_track(positions[i])
        pbest_cost[i] = cost

    # ------------------------------------------------------------------
    # Main PSO loop
    # ------------------------------------------------------------------
    iters = 0
    while counter < max_evals:
        iters += 1

        # Linear inertia weight schedule
        progress = min(counter / float(max_evals), 1.0)
        w = W_START - (W_START - W_END) * progress

        for i in range(population_size):
            if counter >= max_evals:
                break

            r1 = np.random.uniform(0.0, 1.0, size=dim)
            r2 = np.random.uniform(0.0, 1.0, size=dim)

            # Velocity update
            cognitive = C1 * r1 * (pbest_pos[i] - positions[i])
            social    = C2 * r2 * (gbest_pos   - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social

            # Clamp velocity
            velocities[i] = np.clip(velocities[i], v_min, v_max)

            # Position update
            positions[i] = positions[i] + velocities[i]

            # Absorbing boundary: clamp position, zero velocity at wall
            for d in range(dim):
                if positions[i, d] < _lowerBound[d]:
                    positions[i, d] = _lowerBound[d]
                    velocities[i, d] = 0.0
                elif positions[i, d] > _upperBound[d]:
                    positions[i, d] = _upperBound[d]
                    velocities[i, d] = 0.0

            # Evaluate
            cost = eval_and_track(positions[i])

            # Update personal best
            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest_pos[i] = positions[i].copy()

    # ------------------------------------------------------------------
    # Build a simple archive of the top-K distinct optima from personal bests
    # ------------------------------------------------------------------
    archive = _build_archive(pbest_pos, pbest_cost, n_archive=5, min_dist_deg=0.3)

    return (
        iters,
        opt_cost,
        counter,
        gbest_pos if np.isfinite(gbest_cost) else None,
        np.array(search_points),
        np.array(search_costs_list),
        archive,
    )


def _build_archive(
    positions: np.ndarray,
    costs: np.ndarray,
    n_archive: int = 5,
    min_dist_deg: float = 0.3,
) -> list:
    """
    Extract up to n_archive spatially distinct optima from the personal bests.

    Greedily picks the best-cost particle, then the next best that is at
    least min_dist_deg away from all already-selected particles, and so on.
    """
    order = np.argsort(costs)
    archive: list = []

    for idx in order:
        c = float(costs[idx])
        if not np.isfinite(c):
            continue
        p = positions[idx]
        too_close = False
        for entry in archive:
            if np.linalg.norm(p - entry["x"]) < min_dist_deg:
                too_close = True
                break
        if not too_close:
            archive.append({"x": p.copy(), "f": c})
        if len(archive) >= n_archive:
            break

    return archive


# =============================================================================
# PLOTTING — Cartopy style, matches BA1 exactly so results are directly
#             comparable
# =============================================================================

def plot_north_sea(
    search_points: np.ndarray,
    search_costs: np.ndarray,
    best_pos: np.ndarray | None,
    archive: list,
) -> None:
    """Plot evaluated sites, best location and archive on a North Sea map."""
    _ensure_bounds()
    pts = np.asarray(search_points, dtype=float)
    costs = np.asarray(search_costs, dtype=float)
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
        "PSO — Offshore Wind Farm LCOE Optimisation (North Sea, incl. wind)",
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
    plt.title("PSO Convergence — LCOE Minimisation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN — identical summary-statistics framework as BA1
# =============================================================================
if __name__ == "__main__":
    load_depth_raster()
    load_wind_mean_speed()
    _ensure_bounds()

    max_evals = 5000
    independentRuns = 50     # ≥10 recommended for meaningful statistics
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
    print(
        f"[PSO]   w: {W_START}→{W_END}, c1={C1}, c2={C2}, "
        f"v_clamp={V_CLAMP}, pop={populationSize}, max_evals={max_evals}"
    )

    for run in range(independentRuns):
        t0 = time.perf_counter()
        (
            iters,
            opt_cost,
            counter,
            best_pos,
            search_points,
            search_costs,
            archive,
        ) = PSO_implementation(population_size=populationSize, max_evals=max_evals)
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
                f"Pos: (no feasible site) | Iters: {iters} | Time: {t1 - t0:.1f}s"
            )
        else:
            print(
                f"Run {run + 1:3d} | LCOE: {best_cost:.2f} £/MWh | "
                f"Pos: ({best_pos[0]:.3f}°E, {best_pos[1]:.3f}°N) | "
                f"Iters: {iters} | Time: {t1 - t0:.1f}s"
            )

    # =================================================================
    # SUMMARY STATISTICS
    # =================================================================
    obj_vals = np.array(best_objective_values)
    times    = np.array(run_times)

    best_run_idx  = int(np.argmin(obj_vals))
    worst_run_idx = int(np.argmax(obj_vals))
    best_loc      = best_positions[best_run_idx]
    worst_loc     = best_positions[worst_run_idx]

    evals_per_run = np.array([len(oc) for oc in all_opt_costs])

    print("\n" + "=" * 65)
    print(f"  SUMMARY STATISTICS  —  {independentRuns} independent runs")
    print(f"  (PSO: population = {populationSize}, max evals = {max_evals})")
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

    if np.mean(obj_vals) != 0 and len(obj_vals) > 1:
        cv = (np.std(obj_vals, ddof=1) / np.mean(obj_vals)) * 100.0
        print(f"{'':>4s}{'Coeff. of variation':<22s}{cv:>13.2f}%")

    print(f"\n{'':>4s}{'Mean evaluations':<22s}{np.mean(evals_per_run):>14.1f}")
    print(f"{'':>4s}{'Total evaluations':<22s}{np.sum(evals_per_run):>14d}")

    print(f"\n{'':>4s}{'Run':>4s}{'LCOE':>12s}{'Time (s)':>10s}{'Evals':>8s}  {'Best position':<28s}")
    print(f"{'':>4s}{'-' * 62}")
    for i in range(independentRuns):
        pos_str = (
            f"({best_positions[i][0]:.3f}°E, {best_positions[i][1]:.3f}°N)"
            if best_positions[i] is not None
            else "(no feasible site)"
        )
        marker = " <-- best" if i == best_run_idx else ""
        print(
            f"{'':>4s}{i + 1:>4d}{obj_vals[i]:>12.4f}{times[i]:>10.2f}"
            f"{evals_per_run[i]:>8d}  {pos_str}{marker}"
        )

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
        algorithm="PSO",
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
                f"PSO Multi-Run Summary ({independentRuns} runs, "
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
            plt.title("PSO Convergence — All Runs Overlaid")
            plt.legend(fontsize=8, loc="upper right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
