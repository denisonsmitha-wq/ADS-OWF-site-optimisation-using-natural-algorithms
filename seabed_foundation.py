# =============================================================================
# seabed_foundation.py
# Seabed substrate lookup from the BGS 250k Seabed Sediments ESRI Shapefile, with foundation-cost multipliers for bottom-fixed offshore wind farms.
#
# Uses the BGS_250k_SeaBedSediments_WGS84_v3 shapefile with the FOLK_S field
#
# 16 Folk sediment classes are mapped to continuous cost multipliers reflecting
#
# Data source: BGS Seabed Sediments 250k v3 (British Geological Survey, 2024)
#   https://www.bgs.ac.uk/download/offshore-seabed-sediments-250k-gis-shapefile-format/
#
# Author: Arthur Denison-Smith
# =============================================================================

from __future__ import annotations

import os
import struct
import bisect


# =============================================================================
# FOUNDATION COST PARAMETERS
# =============================================================================

# Base cost and depth coefficient
FOUNDATION_BASE_COST_PER_MW = 0.35   # £M / MW at 30 m reference depth
FOUNDATION_DEPTH_COEFF      = 0.012  # £M / MW per metre deviation from 30 m

# =============================================================================
# SUBSTRATE COST MULTIPLIERS — Folk classification (FOLK_S field)
#
# 16 categories ranked by installation difficulty for monopile foundations.
# Two "difficulty axes":
#   Sand → Gravel: increases driving resistance and scour protection needs
#   Sand → Mud:    decreases bearing capacity, requires longer piles
# Pure sand is the reference case (1.00). Rock requires drilling (1.40).
#
# Values are engineering estimates informed by BVG Associates (2019) and
# general geotechnical principles. A real project would use site-specific
# geotechnical survey data.
# =============================================================================
SUBSTRATE_COST_MULTIPLIERS: dict[str, float] = {
    # --- Sand-dominated (easiest installation) ---
    "S":     1.00,   # Sand — reference case, easiest monopile driving
    "(g)S":  1.00,   # Slightly gravelly sand — negligible gravel content
    "mS":    1.00,   # Muddy sand — slightly reduced bearing capacity
    "(g)mS": 1.00,   # Slightly gravelly muddy sand — mixed but sand-dominated

    # --- Gravel-influenced (higher driving resistance, scour risk) ---
    "gS":    1.00,   # Gravelly sand — significant gravel, larger hammer needed
    "gmS":   1.00,   # Gravelly muddy sand — mixed, moderate driving difficulty
    "sG":    1.00,   # Sandy gravel — gravel-dominated, pile refusal risk
    "msG":   1.00,   # Muddy sandy gravel — variable resistance
    "mG":    1.00,   # Muddy gravel — high resistance, some cohesion from mud
    "G":     1.22,   # Gravel — high driving resistance, possible pre-drilling

    # --- Mud-dominated (low bearing capacity, longer piles) ---
    "(g)sM": 1.00,   # Slightly gravelly sandy mud — reduced bearing capacity
    "sM":    1.00,   # Sandy mud — significant pile length increase
    "(g)M":  1.00,   # Slightly gravelly mud — long piles, heavy transition piece
    "gM":    1.00,   # Gravelly mud — unpredictable driving, pile deviation risk
    "M":     1.00,   # Mud — lowest bearing capacity, longest piles needed

}

# Fallback if query point is outside shapefile coverage or code not recognised
DEFAULT_MULTIPLIER = 1.15

# =============================================================================
# SHAPEFILE CONFIGURATION
# =============================================================================
_SHAPEFILE_BASENAME = "BGS_250k_SeaBedSediments_WGS84_v3"
_DBF_FIELD_NAME     = "FOLK_S"   # Folk classification short code field

# =============================================================================
# INTERNAL DATA STRUCTURES
# =============================================================================

# Each loaded polygon is stored as:
#   (xmin, ymin, xmax, ymax, [ring0, ring1, …], folk_code_str)
# where each ring is a list of (x, y) tuples.

_polygons: list = []            # flat list of all polygons
_y_sorted_indices: list = []    # polygon indices sorted by ymin
_y_mins: list = []              # parallel list of ymin values for bisect
_loaded: bool = False


# =============================================================================
# SHAPEFILE READER  (pure-Python, no external dependencies)
# =============================================================================

def _read_dbf(path: str) -> list[str]:
    """Return a list of FOLK_S values, one per DBF record."""
    with open(path, "rb") as f:
        f.read(4)  # version + date
        num_records = struct.unpack("<I", f.read(4))[0]
        header_size = struct.unpack("<H", f.read(2))[0]
        record_size = struct.unpack("<H", f.read(2))[0]
        f.read(20)

        fields: list[tuple[str, int, int]] = []  # (name, offset_in_record, length)
        offset = 1  # skip deletion flag byte
        while True:
            marker = f.read(1)
            if marker == b"\r":
                break
            name = (marker + f.read(10)).replace(b"\x00", b"").decode("ascii").strip()
            f.read(1)  # field type
            f.read(4)  # reserved
            flen = struct.unpack("B", f.read(1))[0]
            f.read(1 + 14)  # decimal count + reserved
            fields.append((name, offset, flen))
            offset += flen

        # Find the FOLK_S field
        sub_offset = sub_len = None
        for fname, foff, flen in fields:
            if fname.upper() == _DBF_FIELD_NAME.upper():
                sub_offset, sub_len = foff, flen
                break
        if sub_offset is None:
            raise ValueError(
                f"{_DBF_FIELD_NAME} field not found in DBF. "
                f"Available fields: {[f[0] for f in fields]}"
            )

        f.seek(header_size)
        substrates: list[str] = []
        for _ in range(num_records):
            rec = f.read(record_size)
            val = rec[sub_offset : sub_offset + sub_len].decode("ascii", errors="replace").strip()
            substrates.append(val)

    return substrates


def _read_shp_polygons(path: str) -> list[tuple]:
    """
    Read all Polygon records from a .shp file.

    Returns a list of (bbox, rings) where:
        bbox  = (xmin, ymin, xmax, ymax)
        rings = list of list-of-(x,y)
    """
    records: list[tuple] = []
    with open(path, "rb") as f:
        f.seek(100)  # skip file header
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            _rec_num, content_len = struct.unpack(">II", hdr)
            start_pos = f.tell()
            expected_end = start_pos + content_len * 2

            shape_type = struct.unpack("<I", f.read(4))[0]
            if shape_type == 0:
                f.seek(expected_end)
                records.append(None)
                continue

            bbox = struct.unpack("<4d", f.read(32))
            num_parts = struct.unpack("<I", f.read(4))[0]
            num_points = struct.unpack("<I", f.read(4))[0]
            parts = [struct.unpack("<I", f.read(4))[0] for _ in range(num_parts)]
            points = [struct.unpack("<2d", f.read(16)) for _ in range(num_points)]

            rings: list[list[tuple]] = []
            for i in range(num_parts):
                s = parts[i]
                e = parts[i + 1] if i + 1 < num_parts else num_points
                rings.append(points[s:e])

            records.append((bbox, rings))
            f.seek(expected_end)

    return records


# =============================================================================
# POINT-IN-POLYGON  (ray-casting algorithm)
# =============================================================================

def _point_in_ring(x: float, y: float, ring: list[tuple]) -> bool:
    """Ray-casting test: is (x, y) inside the closed ring?"""
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_polygon(x: float, y: float, rings: list) -> bool:
    """
    Shapefile polygon convention: first ring is outer boundary (CCW),
    subsequent rings are holes (CW).  A point must be inside the outer
    ring and outside every hole.
    """
    if not _point_in_ring(x, y, rings[0]):
        return False
    for hole in rings[1:]:
        if _point_in_ring(x, y, hole):
            return False
    return True


# =============================================================================
# PUBLIC API
# =============================================================================

def load_seabed(shp_dir: str | None = None) -> int:
    """
    Load the BGS 250k Seabed Sediments shapefile.

    Parameters
    ----------
    shp_dir : str or None
        Directory containing BGS_250k_SeaBedSediments_WGS84_v3.shp / .dbf / .shx.
        If None, looks in the current directory, then common upload paths.

    Returns
    -------
    int — number of polygons loaded.
    """
    global _polygons, _y_sorted_indices, _y_mins, _loaded

    if _loaded:
        return len(_polygons)

    # Resolve paths
    search_dirs = []
    if shp_dir:
        search_dirs.append(shp_dir)
    search_dirs += [
        os.getcwd(),
        os.path.join(os.path.dirname(__file__)),
        "/mnt/user-data/uploads",
    ]

    shp_path = dbf_path = None
    for d in search_dirs:
        candidate_shp = os.path.join(d, f"{_SHAPEFILE_BASENAME}.shp")
        candidate_dbf = os.path.join(d, f"{_SHAPEFILE_BASENAME}.dbf")
        if os.path.isfile(candidate_shp) and os.path.isfile(candidate_dbf):
            shp_path, dbf_path = candidate_shp, candidate_dbf
            break

    if shp_path is None:
        raise FileNotFoundError(
            f"Cannot find {_SHAPEFILE_BASENAME}.shp/.dbf in any of: "
            + ", ".join(search_dirs)
        )

    print(f"[seabed_foundation] Loading {shp_path} ...")
    substrates = _read_dbf(dbf_path)
    geoms = _read_shp_polygons(shp_path)

    if len(substrates) != len(geoms):
        raise ValueError(
            f"DBF has {len(substrates)} records but SHP has {len(geoms)} geometries"
        )

    _polygons = []
    for i, (sub, geom) in enumerate(zip(substrates, geoms)):
        if geom is None:
            continue
        bbox, rings = geom
        _polygons.append((bbox[0], bbox[1], bbox[2], bbox[3], rings, sub))

    # Build a y-sorted index for fast spatial filtering
    idx_ymin_pairs = sorted(enumerate(_polygons), key=lambda t: t[1][1])
    _y_sorted_indices = [t[0] for t in idx_ymin_pairs]
    _y_mins = [t[1][1] for t in idx_ymin_pairs]

    _loaded = True
    print(f"[seabed_foundation] Loaded {len(_polygons)} seabed polygons.")

    # Report unique substrate codes found
    unique_codes = sorted(set(p[5] for p in _polygons))
    print(f"[seabed_foundation] Unique FOLK_S codes: {unique_codes}")

    return len(_polygons)


def get_substrate(lon: float, lat: float) -> Optional[str]:
    """
    Return the Folk classification code (FOLK_S) at (lon, lat), or None
    if the point is outside all seabed polygons.

    Uses bounding-box pre-filtering + y-sorted index for speed.
    """
    if not _loaded:
        load_seabed()

    x, y = float(lon), float(lat)

    # Use bisect to skip polygons whose ymin is above our query point.
    # We scan polygons whose ymin <= y.
    right = bisect.bisect_right(_y_mins, y)

    for idx_pos in range(right):
        i = _y_sorted_indices[idx_pos]
        xmin, ymin, xmax, ymax, rings, sub = _polygons[i]
        if y < ymin or y > ymax or x < xmin or x > xmax:
            continue
        if _point_in_polygon(x, y, rings):
            return sub

    return None


def foundation_cost_multiplier(lon: float, lat: float) -> float:
    """
    Return the dimensionless foundation-cost multiplier at (lon, lat).

    Multiplier reflects the difficulty/expense of installing monopile
    foundations in the local seabed substrate, based on the Folk
    sediment classification.
    """
    sub = get_substrate(lon, lat)
    if sub is None:
        return DEFAULT_MULTIPLIER
    return SUBSTRATE_COST_MULTIPLIERS.get(sub, DEFAULT_MULTIPLIER)


def compute_foundation_cost_per_mw(depth_m: float, lon: float, lat: float) -> float:
    """
    Foundation CAPEX in £M / MW, adjusted for seabed substrate.

        base = FOUNDATION_BASE_COST_PER_MW + FOUNDATION_DEPTH_COEFF * (depth - 30)
        adjusted = base * substrate_multiplier

    Parameters
    ----------
    depth_m : float   — water depth (positive downward)
    lon, lat : float  — WGS-84 coordinates of the site

    Returns
    -------
    float — foundation cost in £M per MW installed
    """
    base = FOUNDATION_BASE_COST_PER_MW + FOUNDATION_DEPTH_COEFF * (depth_m - 30.0)
    mult = foundation_cost_multiplier(lon, lat)
    return float(base * mult)


def substrate_info(lon: float, lat: float) -> dict:
    """Convenience: return substrate code, description, multiplier and cost at 30 m."""
    sub = get_substrate(lon, lat)
    mult = foundation_cost_multiplier(lon, lat)
    cost_30m = compute_foundation_cost_per_mw(30.0, lon, lat)

    # Human-readable description of the Folk code
    folk_descriptions = {
        "S": "Sand", "(g)S": "Slightly gravelly sand",
        "mS": "Muddy sand", "(g)mS": "Slightly gravelly muddy sand",
        "gS": "Gravelly sand", "gmS": "Gravelly muddy sand",
        "sG": "Sandy gravel", "G": "Gravel",
        "mG": "Muddy gravel", "msG": "Muddy sandy gravel",
        "(g)sM": "Slightly gravelly sandy mud", "sM": "Sandy mud",
        "(g)M": "Slightly gravelly mud", "gM": "Gravelly mud",
        "M": "Mud", "-": "Rock / diamicton",
    }

    return {
        "folk_code": sub if sub else "Unknown",
        "description": folk_descriptions.get(sub, "Outside coverage") if sub else "Outside coverage",
        "multiplier": mult,
        "foundation_cost_30m_depth_GBP_M_per_MW": cost_30m,
    }


# =============================================================================
# QUICK SELF-TEST
# =============================================================================
if __name__ == "__main__":
    load_seabed()

    # Test a grid of points across the UK North Sea
    print("\nSample substrate lookups across the UK North Sea:")
    print(f"{'Lon':>8s} {'Lat':>8s}  {'Folk':>6s}  {'Description':<30s} {'Mult':>5s}  {'Cost @30m':>10s}")
    print("-" * 80)

    test_points = [
        ( 1.5,  53.0, "Dogger Bank area"),
        ( 0.0,  54.0, "Central North Sea"),
        (-1.0,  55.0, "Off NE England"),
        ( 1.0,  52.0, "Southern North Sea"),
        ( 2.0,  56.0, "Northern area"),
        (-2.0,  51.5, "English Channel approach"),
        ( 0.5,  53.5, "Humber approach"),
        ( 1.8,  52.5, "East Anglia"),
        ( 1.0,  54.5, "Hornsea zone"),
        (-1.5,  57.5, "Moray Firth"),
        (1.98559, 52.6847, "optima")
    ]

    for lon, lat, label in test_points:
        info = substrate_info(lon, lat)
        print(
            f"{lon:8.2f} {lat:8.2f}  {info['folk_code']:>6s}  {info['description']:<30s} "
            f"{info['multiplier']:>5.2f}  "
            f"£{info['foundation_cost_30m_depth_GBP_M_per_MW']:.4f}M/MW"
            f"  ({label})"
        )
