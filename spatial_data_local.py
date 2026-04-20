# =============================================================================
# spatial_data_local.py
# Handles spatial lookups:
#   - Bathymetry (GEBCO NetCDF)
#   - Distance to nearest port (simple planar approximation)
#
# =============================================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree

# =============================================================================
# DEPTH DATA — GEBCO NetCDF
# GEBCO convention: negative = ocean depth, positive = land elevation.
#
# Configure via:
#   DEPTH_NC_PATH=/path/to/gebco_tile.nc
# =============================================================================

# Set this to your new GEBCO filename
DEFAULT_GEBCO_FILE = "gebco_2025_n61.0_s51.0_w-3.0_e3.0.nc"

def _default_gebco_path() -> str:
    here = Path(__file__).resolve().parent
    return str(here / DEFAULT_GEBCO_FILE)

DEPTH_NC_PATH = os.getenv("DEPTH_NC_PATH", _default_gebco_path())

_depth_array: Optional[np.ndarray] = None
_lon_array: Optional[np.ndarray] = None
_lat_array: Optional[np.ndarray] = None


def load_depth_raster(path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load GEBCO NetCDF bathymetry and cache it.

    Expected variables (GEBCO standard):
      - elevation (2D) with dims (lat, lon)
      - lon (1D)
      - lat (1D)
    """
    global _depth_array, _lon_array, _lat_array

    if path is None:
        path = DEPTH_NC_PATH

    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            "GEBCO NetCDF bathymetry file not found. "
            "Set DEPTH_NC_PATH env var or place a 'gebco_*.nc' file next to this module."
        )

    if _depth_array is None:
        ds = nc.Dataset(path, "r")
        _lon_array = np.array(ds.variables["lon"][:], dtype=float)
        _lat_array = np.array(ds.variables["lat"][:], dtype=float)
        _depth_array = np.array(ds.variables["elevation"][:], dtype=float)  # (lat, lon)
        ds.close()

        print(f"[spatial_data_local] Loaded GEBCO NetCDF: {path}")
        print(f"  Lon range: {_lon_array.min():.2f} to {_lon_array.max():.2f}")
        print(f"  Lat range: {_lat_array.min():.2f} to {_lat_array.max():.2f}")

    return _depth_array, _lon_array, _lat_array


def get_depth_at(lon: float, lat: float) -> float:
    """
    Return water depth in metres (positive = below sea level) at (lon, lat).

    Returns np.nan for land (elevation >= 0) or out-of-bounds points.
    Uses nearest-neighbour lookup.
    """
    arr, lons, lats = load_depth_raster()

    if lon < float(lons.min()) or lon > float(lons.max()) or lat < float(lats.min()) or lat > float(lats.max()):
        return float("nan")

    col = int(np.argmin(np.abs(lons - lon)))
    row = int(np.argmin(np.abs(lats - lat)))

    val = float(arr[row, col])

    # GEBCO: negative = ocean, positive = land
    if val >= 0:
        return float("nan")

    return float(-val)


# =============================================================================
# PORT LOCATIONS  (longitude, latitude)
# =============================================================================
PORTS = {
    "Aberdeen":         (-2.094, 57.144),
    "Dundee":           (-2.967, 56.462),
    "Blyth":            (-1.509, 55.127),
    "Port of Tyne":     (-1.440, 55.010),
    "Teesside":         (-1.177, 54.635),
    "Hartlepool":       (-1.190, 54.690),
    "Grimsby":          (-0.070, 53.575),
    "Great Yarmouth":   ( 1.730, 52.607),
    "Lowestoft":        ( 1.750, 52.475),
    "Harwich":          ( 1.285, 51.945),
}

_port_coords = np.array(list(PORTS.values()), dtype=float)
_port_names = list(PORTS.keys())
_port_tree = cKDTree(_port_coords)


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Great-circle distance in km between two points using the Haversine formula.
    Assumes a spherical Earth with mean radius 6 371 km.
    """
    R = 6371.0  # mean Earth radius in km

    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return float(R * c)


def distance_to_nearest_port_km(lon: float, lat: float) -> float:
    """
    Great-circle distance in km to the nearest port using the Haversine formula.
    The cKDTree is used only as a fast first pass to identify the nearest port
    in degree-space; the returned distance is then computed via Haversine.
    """
    _, idx = _port_tree.query([lon, lat])
    port_lon, port_lat = _port_coords[int(idx)]

    return _haversine_km(lon, lat, port_lon, port_lat)


def nearest_port_name(lon: float, lat: float) -> str:
    """Return the name of the nearest port to (lon, lat)."""
    _, idx = _port_tree.query([lon, lat])
    return _port_names[int(idx)]
