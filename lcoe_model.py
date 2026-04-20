# =============================================================================
# lcoe_model_corrected.py
#
# =============================================================================

from __future__ import annotations

import numpy as np

from spatial_data_local import get_depth_at, distance_to_nearest_port_km
from wind_energy import capacity_factor_at, mean_wind_speed_square_km
from seabed_foundation import (
    load_seabed,
    get_substrate,
    foundation_cost_multiplier,
    SUBSTRATE_COST_MULTIPLIERS,
    DEFAULT_MULTIPLIER as DEFAULT_SUBSTRATE_MULTIPLIER,
)

# =============================================================================
# WIND FARM PARAMETERS
# =============================================================================
CAPACITY_MW       = 500.0   # Farm nameplate capacity (MW)
DESIGN_LIFE_YEARS = 25      # Project lifetime (years)
DISCOUNT_RATE     = 0.08    # WACC

# Wind resource settings
USE_WIND_RESOURCE  = True
WIND_HALF_WIDTH_KM = 10.0
DEFAULT_CAPACITY_FACTOR = 0.45  # Fallback if USE_WIND_RESOURCE is False

# =============================================================================
# SITE VALIDITY CONSTRAINTS
# =============================================================================
MIN_DEPTH_M = 10.0
MAX_DEPTH_M = 60.0

# =============================================================================
# ENERGY LOSS FACTORS
# These reduce gross capacity factor to net capacity factor.
# Sources: typical industry values from ORE Catapult, Crown Estate reports.
# =============================================================================
AVAILABILITY_LOSS    = 0.07   # 7%  — turbine downtime for maintenance
WAKE_LOSS            = 0.10   # 10% — inter-turbine wake effects
ELECTRICAL_LOSS      = 0.03   # 3%  — transformer and cable losses
OTHER_LOSS           = 0.03   # 3%  — blade degradation, curtailment, grid

# Combined: net = gross * (1 - total_loss)
TOTAL_LOSS_FACTOR = 1.0 - (AVAILABILITY_LOSS + WAKE_LOSS + ELECTRICAL_LOSS + OTHER_LOSS)
# = 0.77, i.e. 23% total losses

# =============================================================================
# CAPEX COEFFICIENTS  (£M per MW installed)
# =============================================================================

# --- NEW: Turbine supply cost (nacelle, blades, tower, generator) ---
# This was MISSING from the original lcoe_model_wind.py.
# Typical range for 6 MW class offshore turbines: £1.0–1.5 M/MW
# Source: BVG Associates "Guide to an Offshore Wind Farm" / BEIS reports
TURBINE_COST_PER_MW = 1.40  # £M/MW

# --- Foundation (depth-dependent + substrate multiplier) ---
# Base cost and depth coefficient are defined here.
# Substrate multipliers (16 Folk classification categories) are defined in
# seabed_foundation.py and accessed via foundation_cost_multiplier().
FOUNDATION_BASE_COST_PER_MW = 0.35   # £M/MW at 30m depth reference
FOUNDATION_DEPTH_COEFF      = 0.012  # £M/MW per metre deviation from 30m

# --- Cable (distance-dependent) ---
CABLE_COST_PER_KM_PER_MW = 0.0015  # £M/MW/km

# --- Installation (base + distance-dependent) ---
INSTALLATION_BASE_PER_MW    = 0.60  # £M/MW
INSTALLATION_DISTANCE_COEFF = 0.0015 # £M/MW/km

# --- NEW: Decommissioning provision ---
# Typically 3-5% of total CAPEX, set aside at project start.
# Applied as a fraction of the sum of all other CAPEX components.
DECOMMISSIONING_FRACTION = 0.04  # 4% of CAPEX

# =============================================================================
# OPEX COEFFICIENTS  (£M per MW per year)
# =============================================================================
OPEX_BASE_PER_MW_YR = 0.06
OPEX_DISTANCE_COEFF = 0.001


# =============================================================================
# SUBSTRATE LOOKUP
# =============================================================================

def _substrate_multiplier(lon: float, lat: float) -> float:
    """Return the substrate cost multiplier at (lon, lat).
    
    Delegates to seabed_foundation.foundation_cost_multiplier() which uses
    the BGS 250k Folk classification (16 substrate categories with continuous
    cost multipliers ranging from 1.00 for pure sand to 1.40 for rock).
    """
    return foundation_cost_multiplier(lon, lat)


def _foundation_cost_per_mw(depth_m: float, lon: float, lat: float) -> float:
    """Foundation CAPEX (£M/MW) adjusted for depth and substrate."""
    base = FOUNDATION_BASE_COST_PER_MW + FOUNDATION_DEPTH_COEFF * (depth_m - 30.0)
    mult = _substrate_multiplier(lon, lat)
    return float(base * mult)


# =============================================================================
# COST FUNCTIONS
# =============================================================================

def compute_capex(depth_m: float, dist_km: float,
                  lon: float = 0.0, lat: float = 0.0) -> float:
    """
    Total CAPEX in £M/MW including ALL components.

    Components:
        1. Turbine supply       (was missing in original)
        2. Foundation            (depth + substrate adjusted)
        3. Export cable          (distance-dependent)
        4. Installation          (base + distance-dependent)
        5. Decommissioning       (fraction of above, was missing in original)
    """
    turbine     = TURBINE_COST_PER_MW
    foundation  = _foundation_cost_per_mw(depth_m, lon, lat)
    cable       = CABLE_COST_PER_KM_PER_MW * dist_km
    installation = INSTALLATION_BASE_PER_MW + INSTALLATION_DISTANCE_COEFF * dist_km

    subtotal = turbine + foundation + cable + installation
    decommissioning = DECOMMISSIONING_FRACTION * subtotal

    return float(subtotal + decommissioning)


def compute_opex(dist_km: float) -> float:
    """Annual OPEX in £M/MW/year given distance from port."""
    return float(OPEX_BASE_PER_MW_YR + OPEX_DISTANCE_COEFF * dist_km)


def capital_recovery_factor() -> float:
    """CRF to annualise CAPEX over the project lifetime."""
    r, n = float(DISCOUNT_RATE), int(DESIGN_LIFE_YEARS)
    return float((r * (1.0 + r) ** n) / ((1.0 + r) ** n - 1.0))


def site_capacity_factor(lon: float, lat: float) -> float:
    """
    Net capacity factor at (lon, lat), including energy loss factors.

    Gross CF comes from the wind resource (power curve applied to mean speed).
    Net CF = Gross CF × (1 - total_losses)
    """
    if not USE_WIND_RESOURCE:
        return float(DEFAULT_CAPACITY_FACTOR * TOTAL_LOSS_FACTOR)

    gross_cf = capacity_factor_at(lat, lon, half_width_km=WIND_HALF_WIDTH_KM)
    if not np.isfinite(gross_cf):
        return float("nan")

    net_cf = gross_cf * TOTAL_LOSS_FACTOR
    return float(net_cf)


# =============================================================================
# LCOE
# =============================================================================

def compute_lcoe(lon: float, lat: float) -> float:
    """
    Compute LCOE in £/MWh for a wind farm centred at (lon, lat).

    LCOE = (CAPEX × CRF + OPEX) × 1e6 / (net_CF × 8760)

    Improvements over original lcoe_model_wind.py:
        - Includes turbine supply cost in CAPEX
        - Includes decommissioning provision in CAPEX
        - Applies energy loss factors to capacity factor
        - Uses BGS 250k Folk classification (16 substrate categories)

    Returns np.inf for invalid locations.
    """
    depth_m = get_depth_at(lon, lat)
    dist_km = distance_to_nearest_port_km(lon, lat)

    if np.isnan(depth_m) or depth_m < MIN_DEPTH_M or depth_m > MAX_DEPTH_M:
        return float("inf")

    net_cf = site_capacity_factor(lon, lat)
    if not np.isfinite(net_cf) or net_cf <= 0.0:
        return float("inf")

    capex_per_mw = compute_capex(depth_m, dist_km, lon, lat)
    opex_per_mw_yr = compute_opex(dist_km)

    crf = capital_recovery_factor()
    ann_capex_per_mw = capex_per_mw * crf  # £M/MW/year

    annual_energy_per_mw = net_cf * 8760.0  # MWh/MW/year

    return float((ann_capex_per_mw + opex_per_mw_yr) * 1e6 / annual_energy_per_mw)


def lcoe_breakdown(lon: float, lat: float) -> dict:
    """Full breakdown for reporting / debugging."""
    depth_m = get_depth_at(lon, lat)
    dist_km = distance_to_nearest_port_km(lon, lat)
    ws_ms = mean_wind_speed_square_km(lat, lon, half_width_km=WIND_HALF_WIDTH_KM)

    gross_cf = capacity_factor_at(lat, lon, half_width_km=WIND_HALF_WIDTH_KM)
    net_cf = site_capacity_factor(lon, lat)

    capex = compute_capex(depth_m, dist_km, lon, lat) if np.isfinite(depth_m) else float("nan")
    opex = compute_opex(dist_km) if np.isfinite(dist_km) else float("nan")
    crf = capital_recovery_factor()
    lcoe = compute_lcoe(lon, lat)

    folk_code = get_substrate(lon, lat)
    sub_mult = _substrate_multiplier(lon, lat)

    # Individual CAPEX components for transparency
    foundation = _foundation_cost_per_mw(depth_m, lon, lat) if np.isfinite(depth_m) else float("nan")
    cable = CABLE_COST_PER_KM_PER_MW * dist_km if np.isfinite(dist_km) else float("nan")
    installation = (INSTALLATION_BASE_PER_MW + INSTALLATION_DISTANCE_COEFF * dist_km) if np.isfinite(dist_km) else float("nan")

    return {
        "longitude_deg": float(lon),
        "latitude_deg": float(lat),
        "depth_m": float(depth_m) if np.isfinite(depth_m) else float("nan"),
        "distance_to_port_km": float(dist_km),
        "mean_wind_speed_100m_ms": float(ws_ms) if np.isfinite(ws_ms) else float("nan"),
        "gross_capacity_factor": float(gross_cf) if np.isfinite(gross_cf) else float("nan"),
        "net_capacity_factor": float(net_cf) if np.isfinite(net_cf) else float("nan"),
        "total_loss_factor": float(TOTAL_LOSS_FACTOR),
        "annual_energy_per_mw_mwh": float(net_cf * 8760.0) if np.isfinite(net_cf) else float("nan"),
        "seabed_folk_code": folk_code if folk_code else "Unknown",
        "substrate_cost_multiplier": float(sub_mult),
        "capex_turbine_per_mw_GBP_M": float(TURBINE_COST_PER_MW),
        "capex_foundation_per_mw_GBP_M": float(foundation) if np.isfinite(foundation) else float("nan"),
        "capex_cable_per_mw_GBP_M": float(cable) if np.isfinite(cable) else float("nan"),
        "capex_installation_per_mw_GBP_M": float(installation) if np.isfinite(installation) else float("nan"),
        "capex_decommissioning_pct": float(DECOMMISSIONING_FRACTION * 100),
        "capex_total_per_mw_GBP_M": float(capex) if np.isfinite(capex) else float("nan"),
        "opex_per_mw_yr_GBP_M": float(opex),
        "crf": float(crf),
        "lcoe_GBP_per_MWh": float(lcoe),
    }
