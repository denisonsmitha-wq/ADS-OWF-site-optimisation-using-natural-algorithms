# =============================================================================
# wind_energy.py
# Wind resource utilities for offshore wind farm siting.
#
# - Loads an ERA5-style NetCDF that contains u100 and v100 (m/s) on a lat/lon grid
#   with a time dimension (hourly).
# - Pre-computes and caches the *time-mean wind-speed magnitude* at 100 m:
#       mean_speed(lat, lon) = mean_t( sqrt(u100^2 + v100^2) )
# - Provides fast lookups of the area-mean wind speed in a 50 km "square" around a
#   candidate site (to mirror calculateEnergy.m).
#
# This module is designed to be called thousands of times from an optimisation
# loop (e.g. Bees Algorithm) without repeatedly loading the NetCDF.
# =============================================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import netCDF4 as nc

# -----------------------------------------------------------------------------
# Defaults / configuration
# -----------------------------------------------------------------------------
# You can override these paths via environment variables if desired:
#   WINDSPEED_NC_PATH=/path/to/WindSpeedData.nc
_DEFAULT_NC = Path(__file__).resolve().with_name("Windspeedupdated.nc")
WINDSPEED_NC_PATH = os.getenv("WINDSPEED_NC_PATH", str(_DEFAULT_NC))

# Mirror the MATLAB assumptions
AIR_DENSITY_KG_M3 = 1.225
CP = 0.4
TURBINE_RATED_POWER_W = 6e6
TURBINE_ROTOR_RADIUS_M = 75.0
CUT_IN_MS = 4.0
RATED_MS = 12.0
CUT_OUT_MS = 25.0

# -----------------------------------------------------------------------------
# Cached data
# -----------------------------------------------------------------------------
_mean_speed: Optional[np.ndarray] = None   # shape (lat, lon)
_lat: Optional[np.ndarray] = None          # shape (lat,)
_lon: Optional[np.ndarray] = None          # shape (lon,)
_meta: dict = {}


def _compute_time_mean_speed(ds: nc.Dataset, chunk_size: int = 744) -> np.ndarray:
    """Compute mean_t( sqrt(u^2 + v^2) ) in chunks to limit memory."""
    u = ds.variables["u100"]
    v = ds.variables["v100"]

    nt, ny, nx = u.shape
    acc = np.zeros((ny, nx), dtype=np.float64)
    n = 0

    # If the dataset has missing values, we track per-cell counts
    acc_count = np.zeros((ny, nx), dtype=np.int64)

    for start in range(0, nt, chunk_size):
        stop = min(nt, start + chunk_size)
        uu = u[start:stop, :, :].astype(np.float64)
        vv = v[start:stop, :, :].astype(np.float64)

        spd = np.sqrt(uu * uu + vv * vv)

        if np.ma.isMaskedArray(spd):
            spd = spd.filled(np.nan)

        valid = np.isfinite(spd)
        # sum over time axis
        acc += np.nansum(spd, axis=0)
        acc_count += np.sum(valid, axis=0).astype(np.int64)
        n += (stop - start)

    # Prefer per-cell mean if there were missing values
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = acc / np.where(acc_count > 0, acc_count, np.nan)

    return mean_speed.astype(np.float32)


def load_wind_mean_speed(path: str = WINDSPEED_NC_PATH, *, chunk_size: int = 744
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and cache the time-mean wind-speed magnitude (m/s) at 100 m.

    Returns
    -------
    mean_speed : (lat, lon) float32
    lats       : (lat,) float64
    lons       : (lon,) float64
    """
    global _mean_speed, _lat, _lon, _meta

    if _mean_speed is not None:
        return _mean_speed, _lat, _lon

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Wind NetCDF not found at '{path}'. "
            "Set WINDSPEED_NC_PATH env var or place WindSpeedData.nc next to this file."
        )

    ds = nc.Dataset(path, "r")
    lats = np.array(ds.variables["latitude"][:], dtype=float)
    lons = np.array(ds.variables["longitude"][:], dtype=float)

    mean_speed = _compute_time_mean_speed(ds, chunk_size=chunk_size)
    ds.close()

    _mean_speed = mean_speed
    _lat = lats
    _lon = lons
    _meta = {
        "path": path,
        "lat_min": float(np.min(lats)),
        "lat_max": float(np.max(lats)),
        "lon_min": float(np.min(lons)),
        "lon_max": float(np.max(lons)),
        "shape": tuple(mean_speed.shape),
    }

    print(f"[wind_energy] Cached time-mean wind speed from: {path}")
    print(f"  Lon range: {_meta['lon_min']:.2f} to {_meta['lon_max']:.2f}")
    print(f"  Lat range: {_meta['lat_min']:.2f} to {_meta['lat_max']:.2f}")
    print(f"  Grid shape (lat, lon): {_meta['shape']}")

    return _mean_speed, _lat, _lon


def mean_wind_speed_square_km(latitude: float, longitude: float, half_width_km: float = 50.0
                             ) -> float:
    """
    Mirror calculateEnergy.m:
    - Build a 50 km "square" around (lat, lon) (i.e., +/- half_width_km in each direction)
    - Take the mean wind speed over the selected grid cells.

    Notes
    -----
    Uses the *time-mean* wind speed magnitude at each grid cell, then averages
    spatially over the square.

    Returns NaN if no grid cells fall within the square (e.g., point outside coverage).
    """
    mean_speed, lats, lons = load_wind_mean_speed()

    # Convert km to degrees
    dlat = half_width_km / 111.32
    dlon = half_width_km / (111.32 * np.cos(np.deg2rad(latitude)))

    lat_min = latitude - dlat
    lat_max = latitude + dlat
    lon_min = longitude - dlon
    lon_max = longitude + dlon

    # lats may be descending; comparisons still work on values
    lat_lo, lat_hi = (min(lat_min, lat_max), max(lat_min, lat_max))
    lon_lo, lon_hi = (min(lon_min, lon_max), max(lon_min, lon_max))

    lat_idx = np.where((lats >= lat_lo) & (lats <= lat_hi))[0]
    lon_idx = np.where((lons >= lon_lo) & (lons <= lon_hi))[0]

    if lat_idx.size == 0 or lon_idx.size == 0:
        return float("nan")

    sub = mean_speed[np.ix_(lat_idx, lon_idx)]
    return float(np.nanmean(sub))


def turbine_power_from_speed_ms(speed_ms: float) -> float:
    """
    Single-turbine electrical power output (W) from a simplified curve, matching MATLAB.
    """
    v = float(speed_ms)
    if not np.isfinite(v):
        return float("nan")

    if v < CUT_IN_MS or v > CUT_OUT_MS:
        return 0.0

    if CUT_IN_MS <= v <= RATED_MS:
        swept_area = np.pi * (TURBINE_ROTOR_RADIUS_M ** 2)
        power = CP * 0.5 * AIR_DENSITY_KG_M3 * swept_area * (v ** 3)
        return float(min(power, TURBINE_RATED_POWER_W))

    # RATED_MS < v <= CUT_OUT_MS
    return float(TURBINE_RATED_POWER_W)


def capacity_factor_at(latitude: float, longitude: float, half_width_km: float = 50.0) -> float:
    """
    Capacity factor (0..1) for the farm, based on mean wind speed in the square.
    Because we use the same power curve for every turbine, the farm CF equals the
    turbine CF.
    """
    v = mean_wind_speed_square_km(latitude, longitude, half_width_km=half_width_km)
    p = turbine_power_from_speed_ms(v)
    if not np.isfinite(p) or TURBINE_RATED_POWER_W <= 0:
        return float("nan")
    return float(np.clip(p / TURBINE_RATED_POWER_W, 0.0, 1.0))


def energy_lifetime_mwh(latitude: float, longitude: float,
                        *, num_turbines: int = 83, years: int = 25,
                        half_width_km: float = 50.0) -> float:
  
    v = mean_wind_speed_square_km(latitude, longitude, half_width_km=half_width_km)
    p_turb = turbine_power_from_speed_ms(v)
    total_power = p_turb * float(num_turbines)  # W
    energy = total_power * 24.0 * float(years) * 365.0 * 1e-3 / 3600.0
    return float(energy)


def annual_energy_per_mw_mwh(latitude: float, longitude: float, half_width_km: float = 50.0) -> float:
    
    cf = capacity_factor_at(latitude, longitude, half_width_km=half_width_km)
    if not np.isfinite(cf) or cf <= 0:
        return float("nan")
    return float(cf * 8760.0)


def coverage_bounds() -> dict:
    """Return cached dataset bounds once loaded."""
    if _mean_speed is None:
        load_wind_mean_speed()
    return dict(_meta)
