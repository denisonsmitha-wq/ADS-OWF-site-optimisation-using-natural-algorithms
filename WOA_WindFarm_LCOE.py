# =============================================================================
# WOA_WindFarm_LCOE.py
# Whale Optimisation Algorithm for offshore wind farm siting.
# Minimises LCOE over the North Sea by optimising (longitude, latitude).
#
# Adapted from EvoloPy WOA
#
# Fixes applied:
#   Fix 1 — Leader_pos initialised to a random position within bounds rather
#            than np.zeros(dim) = (0°, 0°), which lies far outside the North
#            Sea.  If the entire initial population is infeasible, the
#            encircling and spiral updates stay in-region rather than steering
#            towards Africa.
# =============================================================================

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
import random
import math

from lcoe_model import compute_lcoe, lcoe_breakdown
from spatial_data_local import load_depth_raster, PORTS
from wind_energy import load_wind_mean_speed, coverage_bounds
from results_io import export_results

# =============================================================================
# SEARCH SPACE — bounding box for the North Sea (clipped to data coverage)
# =============================================================================
USER_LON_MIN, USER_LON_MAX = -3.0,  3.0
USER_LAT_MIN, USER_LAT_MAX = 51.0, 61.0

_LON_MIN = _LON_MAX = _LAT_MIN = _LAT_MAX = None
_lowerBound = _upperBound = _bounds = None


def _compute_search_bounds():
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
    """Thin wrapper: x = [lon, lat] -> LCOE."""
    return float(compute_lcoe(float(x[0]), float(x[1])))


# =============================================================================
# WOA IMPLEMENTATION — adapted for LCOE wind farm siting
# =============================================================================

def WOA_implementation(population_size: int, max_evals: int):
    """
    Whale Optimisation Algorithm for LCOE minimisation.

    Returns the same tuple structure as BA1_implementation:
        (iters, opt_cost, counter, best_position,
         search_points, search_costs, archive)
    """
    _ensure_bounds()

    dim = 2
    lb = _lowerBound.copy()
    ub = _upperBound.copy()

    # --- Tracking ---
    counter = 0
    search_points: list = []
    search_costs_list: list = []
    opt_cost: list = []
    best_sol = {"cost": float("inf"), "position": None}

    def eval_and_track(pos: np.ndarray) -> float:
        nonlocal counter, best_sol
        cost = float(objective(pos))
        counter += 1
        search_points.append(pos.copy())
        search_costs_list.append(cost)
        if cost < best_sol["cost"]:
            best_sol = {"cost": cost, "position": pos.copy()}
        opt_cost.append(best_sol["cost"])
        return cost

    # --- Initialise leader (best whale) ---
    # Fix 1: initialise to a random position within bounds rather than
    # np.zeros(dim) = (0°, 0°), which lies far outside the North Sea.
    # If the entire initial population is infeasible, the encircling and
    # spiral updates stay in-region rather than steering towards Africa.
    Leader_pos   = np.random.uniform(lb, ub)
    Leader_score = float("inf")

    # --- Initialise whale positions ---
    Positions = np.zeros((population_size, dim))
    for j in range(dim):
        Positions[:, j] = np.random.uniform(lb[j], ub[j], population_size)

    # Evaluate initial population
    for i in range(population_size):
        if counter >= max_evals:
            break
        fitness = eval_and_track(Positions[i, :])

        if fitness < Leader_score:
            Leader_score = fitness
            Leader_pos = Positions[i, :].copy()

    # --- Main loop ---
    max_iter = max(1, (max_evals - counter) // population_size)
    iters = 0

    for t in range(max_iter):
        if counter >= max_evals:
            break
        iters += 1

        a = 2.0 - t * (2.0 / max_iter)       # linearly decreases from 2 to 0
        a2 = -1.0 + t * (-1.0 / max_iter)     # linearly decreases from -1 to -2

        for i in range(population_size):
            if counter >= max_evals:
                break

            r1 = random.random()
            r2 = random.random()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            b = 1.0
            l = (a2 - 1.0) * random.random() + 1.0
            p = random.random()

            for j in range(dim):
                if p < 0.5:
                    if abs(A) >= 1:
                        # Exploration: move towards a random whale
                        rand_idx = int(math.floor(population_size * random.random()))
                        rand_idx = min(rand_idx, population_size - 1)
                        X_rand = Positions[rand_idx, :]
                        D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                        Positions[i, j] = X_rand[j] - A * D_X_rand
                    else:
                        # Exploitation: shrinking encircling towards leader
                        D_Leader = abs(C * Leader_pos[j] - Positions[i, j])
                        Positions[i, j] = Leader_pos[j] - A * D_Leader
                else:
                    # Spiral update
                    distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                    Positions[i, j] = (
                        distance2Leader * math.exp(b * l) * math.cos(l * 2.0 * math.pi)
                        + Leader_pos[j]
                    )

            # Clip to bounds
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)

            # Evaluate
            fitness = eval_and_track(Positions[i, :])

            if fitness < Leader_score:
                Leader_score = fitness
                Leader_pos = Positions[i, :].copy()

        

    # Build archive from best unique positions found
    # Collect all evaluated points, sort by cost, take top distinct ones
    all_costs = np.array(search_costs_list)
    all_pts = np.array(search_points)
    finite_mask = np.isfinite(all_costs)
    if finite_mask.any():
        sorted_idx = np.argsort(all_costs[finite_mask])
        finite_pts = all_pts[finite_mask]
        finite_costs = all_costs[finite_mask]

        archive = []
        for idx in sorted_idx:
            pos = finite_pts[idx]
            cost = finite_costs[idx]
            # Check it's not too close to an existing archive entry
            too_close = False
            for entry in archive:
                if np.linalg.norm(pos - entry["x"]) < 0.05:
                    too_close = True
                    break
            if not too_close:
                archive.append({"x": pos.copy(), "f": float(cost)})
            if len(archive) >= 5:
                break
    else:
        archive = []

    return (
        iters,
        opt_cost,
        counter,
        best_sol["position"],
        np.array(search_points),
        np.array(search_costs_list),
        archive,
    )


# =============================================================================
# PLOTTING  (Cartopy style — matches BA1)
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
            zorder=4, label="Top whale positions",
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
        "WOA — Offshore Wind Farm LCOE Optimisation (North Sea, incl. wind)",
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
    plt.title("WOA Convergence — LCOE Minimisation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    load_depth_raster()
    load_wind_mean_speed()
    _ensure_bounds()

    max_evals = 5000
    independentRuns = 50  # ≥10 recommended for meaningful summary statistics
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
            search_costs,
            archive,
        ) = WOA_implementation(population_size=populationSize, max_evals=max_evals)
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

    # Coefficient of variation (useful for comparing algorithm consistency)
    if np.mean(obj_vals) != 0 and len(obj_vals) > 1:
        cv = (np.std(obj_vals, ddof=1) / np.mean(obj_vals)) * 100.0
        print(f"{'':>4s}{'Coeff. of variation':<22s}{cv:>13.2f}%")

    print(f"\n{'':>4s}{'Mean evaluations':<22s}{np.mean(evals_per_run):>14.1f}")
    print(f"{'':>4s}{'Total evaluations':<22s}{np.sum(evals_per_run):>14d}")

    # Per-run detail table
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
        algorithm="WOA",
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
                            boxprops=dict(facecolor="lightsalmon"))
            axes[0].set_ylabel("Best LCOE (£/MWh)")
            axes[0].set_title("LCOE Distribution Across Runs")
            axes[0].grid(True, alpha=0.3)

            axes[1].boxplot(times, vert=True, patch_artist=True,
                            boxprops=dict(facecolor="lightyellow"))
            axes[1].set_ylabel("Run time (s)")
            axes[1].set_title("Run Time Distribution")
            axes[1].grid(True, alpha=0.3)

            fig.suptitle(
                f"WOA Multi-Run Summary ({independentRuns} runs, "
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
            plt.title("WOA Convergence — All Runs Overlaid")
            plt.legend(fontsize=8, loc="upper right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
