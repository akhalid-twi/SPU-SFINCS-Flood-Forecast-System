# SPU SFINCS Flood Forecast System

Live dashboard of the South Park (Duwamish River, Seattle) compound-flood forecast run by
The Water Institute for Seattle Public Utilities: a SFINCS quadtree model of the lower Duwamish and
Green River, Elliott Bay to Auburn, forced by NOAA STOFS water levels, HRRR/GFS wind, pressure and rain,
and NWRFC/USGS river flows, run every 12 hours on the Rougarou cluster.

* `main` (this branch): the Streamlit app (`app.py`).
* `data`: a snapshot of the forecast system written every 5 minutes by the cluster (cycle status, Slurm
  queue, water level series, products, log tails). It is force-pushed as a single commit; do not edit it.

The app reads the snapshot from `https://raw.githubusercontent.com/<owner>/<repo>/data` (secret
`data_url` on Streamlit Community Cloud, or environment variable `SPU_DATA_URL`).

Water levels are in NAVD88. Flood thresholds at USGS 12113415 (Duwamish at E Marginal Way Bridge):
minor 3.50 m, major 3.93 m. Forecasts are experimental and not an official warning product.
