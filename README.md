# ADS-OWF-site-optimisation-using-natural-algorithms
Optimising offshore wind farm location in the UK North Sea using the Single Parameter Bees Algorithm (BA1). A multi-factor LCOE model integrates ERA5 wind data, GEBCO bathymetry, port distance and BGS seabed geology. BA1 is benchmarked against GWO, WOA and PSO under identical conditions.
├── BA1_WindFarm_LCOE.py              # Single Parameter Bees Algorithm
├── GWO_WindFarm_LCOE.py              # Grey Wolf Optimiser
├── WOA_WindFarm_LCOE.py              # Whale Optimisation Algorithm
├── PSO_WindFarm_LCOE.py              # Particle Swarm Optimisation
├── lcoe_model_corrected.py           # LCOE objective function
├── spatial_data_local.py             # Bathymetry and port distance lookups
├── wind_energy.py                    # ERA5 wind resource and energy calculation
├── seabed_foundation.py              # BGS seabed substrate and foundation cost
├── results_io.py                     # CSV export of results
└── data/                             # Datasets (not included — see below)
    ├── GEBCO_bathymetry.nc
    ├── ERA5_wind.nc
    └── BGS_seabed_sediments/

Datasets need to downloaded from:
- https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2025-grid
- https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
- https://www.bgs.ac.uk/datasets/seabed-sediments-250k/
