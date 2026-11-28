"""South Park forecast dashboard (Streamlit). Renders the snapshot written by dashboard/publish.py.

Data source, first one found:
  * st.secrets["data_url"] or env SPU_DATA_URL: base URL of the snapshot, e.g.
    https://raw.githubusercontent.com/<user>/spu-southpark-forecast/data   (Streamlit Community Cloud)
  * env SPU_DATA_DIR or ../public next to this file                          (on Rougarou)

    cd $SYS && $PY -m streamlit run dashboard/app.py --server.port 8501 --server.address 127.0.0.1
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

FT = 3.28084
STAGES = [("prep", "Preprocessing"), ("an", "Nowcast"), ("fc", "Forecast"), ("pp", "Products")]
# in order along the river, from Elliott Bay upstream
STATIONS = [("Seattle_bay", "1 · Elliott Bay, Seattle (NOAA 9447130), the ocean end"),
            ("SouthPark_PumpStation", "2 · South Park pump station, 640 S Riverside Dr (no gauge here: Seattle downstream and the USGS gauge upstream for reference)"),
            ("USGS_12113415", "3 · Duwamish at E Marginal Way Bridge, South Park (USGS 12113415), 1.5 km upstream of the pump station"),
            ("USGS_12113390", "4 · Duwamish at Golf Course, Tukwila (USGS 12113390)"),
            ("USGS_12113344", "5 · Green River at 200th St, Kent (USGS 12113344), the upstream end of the tide")]
# data: observations as dark points, models as solid Okabe-Ito lines (nowcast blue, forecast vermillion)
# thresholds: NWS/AHPS flood-category colours (action yellow, minor orange, moderate red, major purple), pale bands
C = dict(obs="#1d1d1f", an="#0072B2", fc="#D55E00", minor="#FF9900", major="#CC33FF", street="#CC79A7",
         green="#34c759", blue="#0a84ff", orange="#ff9f0a", red="#ff3b30", gray="#8e8e93")
NWS = {"action": "#FFD700", "minor": "#FF9900", "moderate": "#FF0000", "major": "#CC33FF", "streets flood (SPU log)": "#CC79A7"}
MAP = getattr(go, "Scattermap", None) or go.Scattermapbox      # plotly >= 6 renamed the map traces
MAPKEY = "map" if hasattr(go, "Scattermap") else "mapbox"
FONT = "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Inter, sans-serif"

st.set_page_config(page_title="SPU Flood Forecast System", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"""<style>
html, body, .stApp, .stMarkdown, p, h1, h2, h3, label, input, button, table, .stCaption {{ font-family: {FONT}; }}
span[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {{ font-family: "Material Symbols Rounded" !important; }}
.stApp {{ background: #f5f5f7; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1440px; }}
h1, h2 {{ letter-spacing: -0.02em; color: #1d1d1f; font-weight: 600; }} h1 {{ font-size: 30px; }} h2 {{ font-size: 20px; }}
.card {{ background: #fff; border-radius: 18px; padding: 16px 20px; box-shadow: 0 1px 2px rgba(0,0,0,.04), 0 10px 30px rgba(0,0,0,.05); margin-bottom: 14px; }}
.k {{ font-size: 12.5px; color: #6e6e73; }} .v {{ font-size: 30px; font-weight: 600; letter-spacing: -0.03em; color: #1d1d1f; line-height: 1.15; }}
.s {{ font-size: 12px; color: #86868b; }}
.pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 500; }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
.steps {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }}
.step {{ background: #f5f5f7; border-radius: 12px; padding: 10px 12px; min-height: 64px; }}
.step .n {{ font-size: 12px; color: #6e6e73; }} .step .t {{ margin: 3px 0 4px; }}
.bar {{ height: 6px; background: #e5e5ea; border-radius: 3px; overflow: hidden; margin-top: 6px; }} .bar > i {{ display: block; height: 100%; background: #0a84ff; border-radius: 3px; }}
.tiles {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }} .tiles .card {{ margin: 0; }}
@media (max-width: 900px) {{ .tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }} }}
.steplist {{ margin-top: 6px; border-top: 1px solid #e5e5ea; padding-top: 5px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); column-gap: 12px; }}
.steplist .sr {{ display: flex; justify-content: space-between; gap: 6px; font-size: 11px; color: #6e6e73; line-height: 1.55; font-variant-numeric: tabular-nums; white-space: nowrap; }} .steplist .sr span:first-child {{ overflow: hidden; text-overflow: ellipsis; }} .steplist .sr span:first-child::first-letter {{ color: #34c759; }}
.cycle-title {{ font-size: 17px; font-weight: 600; letter-spacing: -0.02em; }} .cycle-sub {{ font-size: 12.5px; color: #6e6e73; }}
/* phones and narrow windows: Streamlit already stacks columns below 640 px; tighten everything else */
@media (max-width: 900px) {{ .steps {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 640px) {{
  .block-container {{ padding: 3.4rem 0.9rem 2rem !important; }}
  h1 {{ font-size: 23px !important; line-height: 1.15; }} h2 {{ font-size: 17px !important; }}
  .card {{ padding: 12px 14px; border-radius: 14px; }}
  .v {{ font-size: 24px; }} .cycle-title {{ font-size: 15px; }} .cycle-sub {{ font-size: 12px; }}
  .step {{ padding: 8px 10px; min-height: 0; }}
  div[data-baseweb="tab-list"] {{ overflow-x: auto; scrollbar-width: none; }}
  div[data-baseweb="tab"] {{ padding-left: 8px !important; padding-right: 8px !important; }}
  .stPlotlyChart {{ margin-left: -6px; margin-right: -6px; }}
}}
@media (max-width: 640px) {{ .steplist {{ grid-template-columns: 1fr; }} }}
@media (max-width: 380px) {{ .steps {{ grid-template-columns: 1fr; }} }}
div[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"], button[data-testid="stExpandSidebarButton"] {{ display: none; }}
div[data-testid="stRadio"] label p, div[data-testid="stCheckbox"] label p {{ font-size: 13px; }}
.maplibregl-ctrl-attrib, .mapboxgl-ctrl-attrib {{ font-size: 9px !important; line-height: 1.2 !important; padding: 0 4px !important; opacity: .7; }}
.maplibregl-ctrl-attrib-button, .mapboxgl-ctrl-attrib-button {{ transform: scale(.7); }}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ snapshot access
DEFAULT_URL = "https://raw.githubusercontent.com/akhalid-twi/SPU-SFINCS-Flood-Forecast-System/data"


def _source() -> tuple[str | None, Path | None]:
    """Snapshot location: secret/env URL, else a local folder that has a snapshot, else the public data branch."""
    url = None
    try:
        url = st.secrets.get("data_url")
    except Exception:
        pass
    url = url or os.environ.get("SPU_DATA_URL")
    if url:
        return url.rstrip("/"), None
    d = Path(os.environ.get("SPU_DATA_DIR") or Path(__file__).resolve().parents[1] / "public")
    if (d / "status.json").exists():
        return None, d
    return DEFAULT_URL, None


URL, DIR = _source()


@st.cache_data(ttl=30, show_spinner=False)
def fetch(path: str, binary: bool = False):
    """One snapshot file as text (or bytes); None if missing."""
    if URL:
        r = requests.get(f"{URL}/{path}", params={"t": int(time.time() // 30)}, timeout=30)   # query string defeats CDN caching
        if r.status_code != 200:
            return None
        return r.content if binary else r.text
    f = DIR / path
    if not f.exists():
        return None
    return f.read_bytes() if binary else f.read_text(encoding="utf-8", errors="ignore")


RECORD = {}   # t0 -> set it is shown from (filled by record_cycles)


def record_cycles(sets: dict, recent: list) -> list:
    """The dashboard's one record, newest first: the GPU baseline rerun (NOAA tide + STOFS surge) up to its last cycle,
    then the live cycles after it. Live cycles inside the baseline period (CPU, STOFS tide) are left out."""
    key = lambda t: f"{pd.Timestamp(t):%Y-%m-%dT%H}"
    base = sorted(key(t) for t in sets.get("baseline", []))
    live = sorted(set(key(t) for t in sets.get("live", [])) | set(key(t) for t in recent))
    cut = base[-1] if base else ""
    RECORD.clear()
    RECORD.update({t: "baseline" for t in base})
    RECORD.update({t: "live" for t in live if t > cut})
    return sorted(RECORD, reverse=True)


@st.cache_data(ttl=600, show_spinner=False)
def archive_cycle(set_: str, t0: str):
    return fetch_json(f"archive/{set_}/{t0}.json")


def fetch_json(path):
    t = fetch(path)
    return json.loads(t) if t else None


def fetch_csv(path):
    from io import StringIO
    t = fetch(path)
    return pd.read_csv(StringIO(t)) if t else None


# live observations, fetched by the app itself so the observed line keeps growing after the snapshot was made
LIVE_OBS = {"Seattle_bay": ("coops", "9447130"), "USGS_12113415": ("usgs", "12113415"),
            "USGS_12113390": ("usgs", "12113390"), "USGS_12113344": ("usgs", "12113344")}


@st.cache_data(ttl=86400, show_spinner=False)
def usgs_datum_ft(site: str) -> float:
    """Gage datum [ft NAVD88] from the NWIS site record (0 for gauges that report NAVD88 elevations directly)."""
    t = requests.get("https://waterservices.usgs.gov/nwis/site/", params=dict(format="rdb", sites=site, siteOutput="expanded"), timeout=30).text
    rows = [ln.split("	") for ln in t.splitlines() if ln and not ln.startswith("#")]
    rec = dict(zip(rows[0], rows[2]))
    if rec.get("alt_datum_cd") != "NAVD88":
        raise ValueError(f"USGS {site} datum not NAVD88")
    return float(rec.get("alt_va") or 0)


@st.cache_data(ttl=300, show_spinner=False)
def live_obs(name: str, t0: str, t1: str) -> pd.Series:
    """Observed water level [m NAVD88] over [t0, t1] UTC from NOAA CO-OPS or USGS NWIS; empty on any failure."""
    src = LIVE_OBS.get(name)
    if not src:
        return pd.Series(dtype=float)
    a, b = pd.Timestamp(t0), pd.Timestamp(t1)
    try:
        if src[0] == "coops":
            parts = []
            for s0 in pd.date_range(a, b, freq="30D"):            # CO-OPS serves 6-min data 31 days at a time
                d = requests.get("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter", timeout=30, params=dict(
                    product="water_level", datum="NAVD", units="metric", time_zone="gmt", format="json", application="spu_flood_dashboard",
                    station=src[1], begin_date=f"{s0:%Y%m%d %H:%M}", end_date=f"{min(s0 + pd.Timedelta('30D'), b):%Y%m%d %H:%M}")).json()
                parts.append(pd.Series({pd.Timestamp(r["t"]): float(r["v"]) for r in d.get("data", []) if r.get("v") not in ("", None)}, dtype=float))
            out = pd.concat(parts) if parts else pd.Series(dtype=float)
            return out[~out.index.duplicated()].sort_index()
        datum = usgs_datum_ft(src[1])
        d = requests.get("https://waterservices.usgs.gov/nwis/iv/", timeout=60, params=dict(
            format="json", sites=src[1], parameterCd="00065", startDT=f"{a:%Y-%m-%dT%H:%MZ}", endDT=f"{b:%Y-%m-%dT%H:%MZ}")).json()
        tsr = d["value"]["timeSeries"]
        if not tsr:
            return pd.Series(dtype=float)
        nod = float(tsr[0]["variable"]["noDataValue"])
        return pd.Series({pd.Timestamp(r["dateTime"]).tz_convert("UTC").tz_localize(None): (float(r["value"]) + datum) * 0.3048
                          for r in tsr[0]["values"][0]["value"] if float(r["value"]) != nod}, dtype=float).sort_index()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=600, show_spinner=False)
def kbfi_obs(t0: str, t1: str) -> pd.DataFrame:
    """Boeing Field (NWS KBFI) observations: wind [m/s], direction [deg from], pressure [hPa], 3-h rain [mm]; UTC."""
    try:
        r = requests.get("https://api.weather.gov/stations/KBFI/observations", timeout=30,
                         params={"start": pd.Timestamp(t0).strftime("%Y-%m-%dT%H:%M:%SZ"), "end": pd.Timestamp(t1).strftime("%Y-%m-%dT%H:%M:%SZ")},
                         headers={"User-Agent": "spu-flood-dashboard (akhalid@thewaterinstitute.org)", "Accept": "application/geo+json"}).json()
    except Exception:
        return pd.DataFrame()

    def g(p, k):
        return (p.get(k) or {}).get("value")

    rows = []
    for f in r.get("features", []):
        p = f["properties"]
        ws, pr = g(p, "windSpeed"), g(p, "barometricPressure")
        rows.append({"t": pd.Timestamp(p["timestamp"]).tz_convert("UTC").tz_localize(None),
                     "wind_ms": None if ws is None else ws / 3.6, "wind_dir": g(p, "windDirection"),
                     "pressure_hpa": None if pr is None else pr / 100, "rain3h_mm": g(p, "precipitationLast3Hours")})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("t").sort_index().apply(pd.to_numeric, errors="coerce")


def met_skill(model: pd.Series, obs: pd.Series, circular: bool = False, speed=None) -> dict:
    o = obs.dropna().resample("1h").first() if circular else obs.dropna().resample("1h").mean()
    pair = pd.concat({"m": model, "o": o}, axis=1).dropna()
    if speed is not None:
        pair = pair[speed.reindex(pair.index).fillna(0) >= 2]
    if len(pair) < 6:
        return {}
    e = pair.m - pair.o
    if circular:
        e = (e + 180) % 360 - 180
        return {"n": len(pair), "mae": float(e.abs().mean())}
    return {"n": len(pair), "bias": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())), "r": float(pair.m.corr(pair.o))}


def met_fig(frc, obs, key, title, ytitle, t0, obs_col=None, bars=False, obs_rate=False):
    fig = go.Figure()
    if obs_col is not None and len(obs) and obs_col in obs and obs[obs_col].notna().any():
        o = obs[obs_col].dropna()
        if obs_rate:   # 3-h total -> mean rate over those 3 hours
            fig.add_trace(go.Bar(x=o.index, y=o.values / 3.0, name="Observed, KBFI (3-h mean)", marker_color="#8e8e93", opacity=0.45, width=3 * 3600e3))
        else:
            fig.add_trace(go.Scatter(x=o.index, y=o.values, name="Observed, KBFI", mode="markers", marker=dict(color=C["obs"], size=4, opacity=0.7)))
    for run, label, col in (("nowcast", "Nowcast (HRRR analyses)", C["an"]), ("forecast", "Forecast (HRRR, GFS)", C["fc"])):
        m = ts(frc.get(run, {}).get(key))
        if not len(m):
            continue
        if bars:
            fig.add_trace(go.Bar(x=m.index, y=m.values, name=label, marker_color=col, opacity=0.75, width=3600e3))
            continue
        if key == "wind_ms":
            d = ts(frc.get(run, {}).get("wind_dir")).reindex(m.index)
            txt = [f"{v:.1f} m/s from {int(x)}°" if pd.notna(x) else f"{v:.1f} m/s" for v, x in zip(m.values, d.values)]
            fig.add_trace(go.Scatter(x=m.index, y=m.values, name=label, mode="lines", line=dict(color=col, width=2), text=txt, hovertemplate="%{text}"))
        else:
            fig.add_trace(go.Scatter(x=m.index, y=m.values, name=label, mode="lines", line=dict(color=col, width=2)))
    fig.add_vline(x=pd.Timestamp(t0), line=dict(color="#8e8e93", dash="dot", width=1))
    fig.update_layout(title=dict(text=title, font=dict(size=15), x=0, xanchor="left", y=0.98, yanchor="top"), height=280, barmode="overlay",
                      margin=dict(l=10, r=20, t=70, b=10), paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, yanchor="bottom", x=0, font=dict(size=12)),
                      yaxis=dict(title=ytitle, gridcolor="#f2f2f7", zeroline=False), xaxis=dict(gridcolor="#f2f2f7"),
                      font=dict(family=FONT, color="#1d1d1f"))
    return fig


def met_section(ser: dict, t0):
    frc = ser.get("forcing") or {}
    st.markdown("## Weather")
    if not frc.get("nowcast") and not frc.get("forecast"):
        st.caption("Weather forcing appears here once the publisher on Rougarou has been updated.")
        return
    starts = [ts(frc.get(k, {}).get("wind_ms")).index.min() for k in ("nowcast", "forecast") if frc.get(k, {}).get("wind_ms")]
    a = min([x for x in starts if pd.notna(x)] + [pd.Timestamp(t0) - pd.Timedelta(days=3)])
    b = min(pd.Timestamp.utcnow().tz_localize(None), pd.Timestamp(t0) + pd.Timedelta(days=7))
    obs = kbfi_obs(str(a), str(b))
    st.caption("What the model was driven by at Boeing Field (nowcast: HRRR analyses; forecast: HRRR to 48 h, then GFS), against what the "
               "Boeing Field weather station (NWS KBFI, about 2 km from the pump station) measured, fetched live. Station rain is reported as 3-hour totals.")
    cfg_ = dict(displaylogo=False)
    st.plotly_chart(met_fig(frc, obs, "wind_ms", "Wind speed at Boeing Field (hover for direction)", "m/s", t0, "wind_ms"), width="stretch", config=cfg_)
    st.plotly_chart(met_fig(frc, obs, "pressure_hpa", "Mean sea level pressure at Boeing Field", "hPa", t0, "pressure_hpa"), width="stretch", config=cfg_)
    st.plotly_chart(met_fig(frc, obs, "rain_mmh", "Rain at Boeing Field", "mm/h", t0, "rain3h_mm", bars=True, obs_rate=True), width="stretch", config=cfg_)
    rows = []
    for run, label in (("nowcast", "Nowcast"), ("forecast", "Forecast (elapsed part)")):
        f = frc.get(run, {})
        if not f or obs.empty:
            continue
        ws, wd, pr = ts(f.get("wind_ms")), ts(f.get("wind_dir")), ts(f.get("pressure_hpa"))
        sp = met_skill(ws, obs["wind_ms"]) if "wind_ms" in obs else {}
        di = met_skill(wd, obs["wind_dir"], circular=True, speed=obs["wind_ms"].dropna().resample("1h").mean()) if "wind_dir" in obs else {}
        pp = met_skill(pr, obs["pressure_hpa"]) if "pressure_hpa" in obs else {}
        rr = {}
        if "rain3h_mm" in obs and f.get("rain_mmh"):
            o3 = obs["rain3h_mm"].dropna()
            m3 = ts(f.get("rain_mmh")).resample("1h").mean().rolling("3h").sum()
            m3 = m3.reindex(o3.index, method="nearest", tolerance=pd.Timedelta("40min")).dropna()
            if len(m3):
                rr = {"model": float(m3.sum()), "obs": float(o3.reindex(m3.index).sum())}

        def fmt(d, k, nd=1):
            if k not in d:
                return "—"
            return f"{d[k]:+.{nd}f}" if k == "bias" else (f"{d[k]:.2f}" if k == "r" else f"{d[k]:.{nd}f}")

        rows.append({"run": label, "hours": sp.get("n", pp.get("n", 0)),
                     "wind bias m/s": fmt(sp, "bias"), "wind RMSE m/s": fmt(sp, "rmse"), "wind r": fmt(sp, "r"),
                     "direction error °": fmt(di, "mae", 0), "pressure bias hPa": fmt(pp, "bias"), "pressure RMSE hPa": fmt(pp, "rmse"),
                     "rain model / obs mm": f"{rr['model']:.1f} / {rr['obs']:.1f}" if rr else "—"})
    if rows:
        st.markdown("**Weather skill against Boeing Field**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Bias = model minus observed. Direction error is the mean angle between model and observed wind when the observed wind "
                   "is 2 m/s or more. Rain compares 3-hour totals at the times the station reported them.")


def observed_now(ser: dict, name: str, t0) -> pd.Series:
    """Snapshot observations joined with live ones up to now (live wins where both exist)."""
    snap = ts(ser.get("observed", {}).get(name))
    starts = [ts(ser.get(k, {}).get(name)).index.min() for k in ("catchup", "forecast") if ser.get(k, {}).get(name)]
    a = min([x for x in starts if pd.notna(x)] + [pd.Timestamp(t0) - pd.Timedelta(days=3)])
    b = min(pd.Timestamp.utcnow().tz_localize(None), pd.Timestamp(t0) + pd.Timedelta(days=7)).floor("5min")
    live = live_obs(name, str(a.floor("1h")), str(b))
    if live.empty:
        return snap
    live = live.resample("15min").mean().dropna()
    if snap.empty:
        return live
    out = pd.concat([snap[snap.index < live.index.min()], live])
    return out[~out.index.duplicated(keep="last")]


def ts(points) -> pd.Series:
    if not points:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    s = pd.Series({pd.Timestamp(p[0], unit="ms"): p[1] for p in points}, dtype=float)
    return s.dropna()


# ------------------------------------------------------------------ ui pieces
def pill(state):
    col, txt = {"COMPLETED": (C["green"], "Done"), "RUNNING": (C["blue"], "Running"), "PENDING": (C["gray"], "Waiting"),
                "FAILED": (C["red"], "Failed"), "CANCELLED": (C["red"], "Cancelled"), "TIMEOUT": (C["red"], "Timed out"),
                "OUT_OF_MEMORY": (C["red"], "Out of memory")}.get(state or "", ("#c7c7cc", "Not started"))
    return f'<span class="pill" style="background:{col}22;color:{col}"><span class="dot" style="background:{col}"></span>{txt}</span>'


def stepper(rec):
    cells = []
    for key, label in STAGES:
        s = rec["stages"].get(key, {})
        st_ = s.get("state")
        bits = []
        if s.get("cpus") and key in ("an", "fc"):
            bits.append(f'{s["cpus"]} cores')
        if s.get("elapsed") and st_ in ("RUNNING", "COMPLETED", "FAILED", "TIMEOUT"):
            bits.append(s["elapsed"])
        extra = f'<div class="s">{" · ".join(bits)}</div>' if bits else ""
        if st_ == "RUNNING" and s.get("pct") is not None:
            extra += f'<div class="bar"><i style="width:{s["pct"]:.0f}%"></i></div><div class="s">SFINCS {s["pct"]:.0f}%</div>'
        elif st_ == "PENDING":
            if s.get("reason") not in (None, "", "Dependency", "None"):   # only say why when it is not just the previous stage
                extra += f'<div class="s">{s["reason"]}</div>'
        if key == "prep" and s.get("steps"):
            rows = "".join(f'<div class="sr"><span>&#10003; {x["step"]}</span><span>{x["seconds"] // 60}:{x["seconds"] % 60:02d}</span></div>' for x in s["steps"])
            extra += f'<div class="steplist">{rows}</div>'
        cells.append(f'<div class="step"><div class="n">{label}</div><div class="t">{pill(st_)}</div>{extra}</div>')
    return f'<div class="steps">{"".join(cells)}</div>'


def card(html):
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


def metric(k, v, s=""):
    return f'<div><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'


def local(t, tz):
    return pd.Timestamp(t).tz_localize("UTC").tz_convert(tz).strftime("%a %b %d, %H:%M %Z")


def wl_chart(ser, name, title, unit, t0, cfg):
    F = FT if unit == "ft" else 1.0
    corr = ser.get("hwc", {}).get(name, 0.0)
    fig = go.Figure()
    last_obs = None
    # observed solid in the back; nowcast and forecast dotted on top, so the overlap stays visible
    # the pump station has no gauge. Its neighbours: Seattle downstream (low tides drain to the bay level there) and USGS
    # 12113415 4.7 km upstream, above the river-km-11 rock outcrop that holds low tides up near 0 ft
    obs_name, obs_label = (("Seattle_bay", "Observed at Seattle, downstream") if name == "SouthPark_PumpStation" else (name, "Observed"))
    if name == "SouthPark_PumpStation":
        up = observed_now(ser, "USGS_12113415", t0)
        if len(up):
            fig.add_trace(go.Scatter(x=up.index, y=up.values * F, name="Observed at USGS 12113415, 4.7 km upstream (above the outcrop)",
                                     mode="markers", marker=dict(color="#8e8e93", size=3, opacity=0.6, symbol="circle-open")))
    for key, label, col, w in (("observed", obs_label, C["obs"], 0), ("catchup", "Nowcast (observed forcing)", C["an"], 2.0), ("forecast", "Forecast", C["fc"], 2.0)):
        s = observed_now(ser, obs_name, t0) if key == "observed" else ts(ser.get(key, {}).get(name))
        if len(s):
            if key == "observed":
                last_obs = s.index.max()
            if key == "forecast":
                s = s + corr
            if key == "observed":   # points: the model line stays visible through them, peaks are not hidden
                fig.add_trace(go.Scatter(x=s.index, y=s.values * F, name=label, mode="markers", marker=dict(color=col, size=3.5, opacity=0.7)))
            else:
                fig.add_trace(go.Scatter(x=s.index, y=s.values * F, name=label, mode="lines", line=dict(color=col, width=w)))
    # thresholds as shaded bands (AHPS style): each band runs from its level up to the next one, the top one to the chart top.
    # Seattle: the NWS categories from NWPS. South Park and the pump station: the project's minor/major levels and the
    # gate level at which streets flooded in the SPU log. Other gauges have no published thresholds.
    shade = NWS
    bands = []
    cats = (cfg.get("categories") or {}).get(name)
    if cats:
        bands = [(f"NWS {k}", cats[k]) for k in ("action", "minor", "moderate", "major") if k in cats]
    elif name in ("USGS_12113415", "SouthPark_PumpStation"):
        bands = [("minor", cfg["minor_m"]), ("major", cfg["major_m"])] + ([("streets flood (SPU log)", cfg["street_ft"] / FT)] if cfg.get("street_ft") else [])
    if bands:
        bands.sort(key=lambda b: b[1])
        vis = [d[1] for key in ("observed", "catchup", "forecast") for d in ser.get(key, {}).get(name, []) if d[1] is not None]
        top = (max(vis + [bands[-1][1]]) + 0.6) * F
        low = (min(vis) if vis else bands[0][1] - 1) * F
        close = 0.07 * (top - low)          # labels closer than ~7% of the chart height would overlap
        # labels in the right margin, each at its level but pushed up to stay at least `gap` above the one below
        gap, prev = 0.075 * (top - low), -1e9
        for i, (lab, v) in enumerate(bands):
            hi = bands[i + 1][1] * F if i + 1 < len(bands) else top
            col = shade.get(lab.replace("NWS ", ""), "#ff9d00")
            fig.add_hrect(y0=v * F, y1=hi, fillcolor=col, opacity=0.08, line_width=0, layer="below")
            fig.add_hline(y=v * F, line=dict(color=col, width=1), opacity=0.7)
            y = max(v * F, prev + gap); prev = y
            fig.add_annotation(x=1.0, xref="paper", xanchor="left", xshift=6, y=y, yanchor="middle", showarrow=False,
                               text=f"{lab} {v * F:.2f}", font=dict(size=10.5, color=col), align="left")

    fig.add_vline(x=pd.Timestamp(t0), line=dict(color="#8e8e93", dash="dot", width=1))
    if last_obs is not None:   # where the observations currently end
        fig.add_annotation(x=last_obs, y=1.0, yref="paper", text=f"observed to {pd.Timestamp(last_obs).tz_localize('UTC').tz_convert(cfg['tz']):%a %H:%M}",
                           showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=10, color="#6e6e73"))
    fig.update_layout(title=dict(text=title, font=dict(size=15), x=0, xanchor="left", y=0.98, yanchor="top"), height=340,
                      margin=dict(l=10, r=150 if bands else 20, t=78, b=10), paper_bgcolor="white",
                      plot_bgcolor="white", hovermode="x unified", legend=dict(orientation="h", y=1.08, yanchor="bottom", x=0, font=dict(size=12)),
                      yaxis=dict(title=f"{unit} NAVD88", gridcolor="#f2f2f7", zeroline=False), xaxis=dict(gridcolor="#f2f2f7"),
                      font=dict(family=FONT, color="#1d1d1f"))
    return fig


STATION_KEY = {"Seattle_bay": "COOPS_9447130", "SouthPark_PumpStation": "SP_pump_station"}


def station_map(g, domain, name):
    """Small map: the model domain, all gauges faint, this station highlighted."""
    key = STATION_KEY.get(name, name)
    row = g[g.name == key]
    fig = go.Figure()
    if domain:
        lon, lat = zip(*domain)
        fig.add_trace(MAP(lon=lon, lat=lat, mode="lines", line=dict(color="#8e8e93", width=1.5), hoverinfo="skip"))
    sc = g[(g.scored.astype(str).str.lower() == "true") & (g.name != key)]
    fig.add_trace(MAP(lon=sc.lon, lat=sc.lat, mode="markers", marker=dict(size=6, color="#c7c7cc"), text=sc.label, hoverinfo="text"))
    if len(row):
        fig.add_trace(MAP(lon=row.lon, lat=row.lat, mode="markers", marker=dict(size=16, color=C["fc"]), text=row.label, hoverinfo="text"))
        center = dict(lat=float(row.lat.iloc[0]), lon=float(row.lon.iloc[0]))
    else:
        center = dict(lat=47.50, lon=-122.28)
    fig.update_layout(**{MAPKEY: dict(style="carto-positron", center=center, zoom=9.6)}, height=300, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig


def gauge_map(g, domain, skill, aoi):
    sk = skill.set_index("gauge") if skill is not None and len(skill) else pd.DataFrame()
    scored = g.scored.astype(str).str.lower() == "true"
    col, txt = [], []
    for r in g.itertuples():
        s = sk.loc[r.name] if r.name in sk.index else None
        v = None if s is None else (s.get("rmse_m") if r.datum == "navd88" else s.get("rmse_anomaly_m"))
        v = None if v is None or pd.isna(v) else float(v)
        col.append(C["gray"] if v is None else C["green"] if v < 0.10 else C["orange"] if v < 0.25 else C["red"])
        txt.append(f"<b>{r.label}</b><br>{str(r.source).upper()} · {'NAVD88' if r.datum == 'navd88' else 'local datum' if r.scored else 'output point'}"
                   + ("" if v is None else f"<br>RMSE {v * 100:.1f} cm (nowcast)"))
    fig = go.Figure()
    if domain:
        lon, lat = zip(*domain)
        fig.add_trace(MAP(lon=lon, lat=lat, mode="lines", line=dict(color="#8e8e93", width=1.5), name="Model domain", hoverinfo="skip"))
    fig.add_trace(MAP(lon=g.lon[~scored], lat=g.lat[~scored], mode="markers", marker=dict(size=5, color="#c7c7cc"), name="Output points",
                                   text=[t for t, s in zip(txt, scored) if not s], hoverinfo="text"))
    fig.add_trace(MAP(lon=g.lon[scored], lat=g.lat[scored], mode="markers", marker=dict(size=13, color=[c for c, s in zip(col, scored) if s]),
                                   name="Gauges", text=[t for t, s in zip(txt, scored) if s], hoverinfo="text"))
    if aoi and aoi.get("lat"):
        fig.add_trace(MAP(lon=[aoi["lon"]], lat=[aoi["lat"]], mode="markers", marker=dict(size=18, color=C["orange"]), name="Pump station",
                                       text=[f"<b>{aoi.get('name') or 'Pump station'}</b>"], hoverinfo="text"))
    fig.update_layout(**{MAPKEY: dict(style="carto-positron", center=dict(lat=47.50, lon=-122.28), zoom=10.3)}, height=560, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(255,255,255,.85)"), font=dict(family=FONT))
    return fig


# ------------------------------------------------------------------ page
status = fetch_json("status.json")
if not status:
    st.title("SPU Flood Forecast System")
    st.warning(f"No snapshot found at {URL or DIR}. The cluster publishes one every 5 minutes; try again shortly.")
    st.stop()
cfg, recs, tz = status["config"], status["cycles"], status["config"]["tz"]
F = lambda u: FT if u == "ft" else 1.0

st.markdown("# SPU Flood Forecast System")
st.markdown(f'<div class="cycle-sub">South Park, Duwamish River, Seattle · SFINCS compound flood model · data as of {local(status["updated"], tz)}</div>', unsafe_allow_html=True)
# controls in a row under the title (no sidebar)
c_sel, c_ctl = st.columns([3.2, 3.5], vertical_alignment="bottom")
with c_sel:
    _idx = fetch_json("archive/index.json") or {}
    _sets = _idx.get("sets") or {}
    _opts = [f"live/{t}" for t in record_cycles(_sets, [r["t0"] for r in recs])]

    def _cyc_label(o):
        set_, t = o.split("/", 1)
        return f"{pd.Timestamp(t):%a %b %d, %H}z · {local(t, tz)}" + ("  · baseline rerun" if set_ == "baseline" else "")
    sel_full = st.selectbox("Cycle", _opts, format_func=_cyc_label) if _opts else None
    sel_set, sel = sel_full.split("/", 1) if sel_full else ("live", None)
    sel_set = RECORD.get(sel, "live")
with c_ctl:
    # one row that wraps on a phone instead of stacking
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        unit = st.radio("Units", ["ft", "m"], horizontal=True, label_visibility="collapsed")
        _view = st.session_state.get("view") or "Overview"   # the view chosen on the previous run (selector is below)
        auto = st.toggle("Auto-refresh", value=_view == "Forecast" and os.environ.get("SPU_NO_AUTO") != "1",
                         key=f"auto_{_view}", help="Reload the data every 60 s (on by default on the Forecast view)")
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()
st.caption(f"Cycles at {', '.join(f'{h:02d}z' for h in cfg['hours'])} UTC · thresholds at USGS 12113415: minor {cfg['minor_m'] * FT:.2f} ft, "
           f"major {cfg['major_m'] * FT:.2f} ft NAVD88 · snapshot {local(status['updated'], tz)}")
view = st.segmented_control("View", ["Overview", "Forecast", "Analysis", "Stations", "Jobs & logs", "About"], key="view", label_visibility="collapsed") or "Overview"

if view == "Overview":
    if not recs:
        st.info("No cycles yet.")
    else:
        latest = next((r for r in recs if r.get("has_products")), None)
        u = unit
        if latest:
            pk = fetch_csv(f"cycles/{pd.Timestamp(latest['t0']):%Y%m%dT%H}/peaks.csv")
            top = pk.loc[pk.south_park_m.idxmax()] if pk is not None and len(pk) else None
            vals = [("Latest forecast", f"{pd.Timestamp(latest['t0']):%b %d %Hz}", local(latest["t0"], tz)),
                    ("Peak at South Park", "—" if top is None else f"{top.south_park_m * F(u):.2f} {u}", "" if top is None else f"{top.day} {top.time} PST"),
                    ("Flood class", "None" if top is None or not isinstance(top.flood, str) or not top.flood else top.flood.title(),
                     f"minor {cfg['minor_m'] * F(u):.2f}, major {cfg['major_m'] * F(u):.2f} {u}"),
                    ("Cycles", f"{len(recs)}", f"{sum(1 for r in recs if r.get('has_products'))} with products")]
        else:
            vals = [("Latest forecast", "running", "no products yet"), ("Peak at South Park", "—", ""), ("Flood class", "—", ""), ("Cycles", f"{len(recs)}", "")]
        # one wrapping grid: four across on a desktop, two across on a phone
        st.markdown('<div class="tiles">' + "".join(f'<div class="card tile">{metric(k, v, s)}</div>' for k, v, s in vals) + '</div>', unsafe_allow_html=True)
        def cycle_card(r):
            row = r.get("row", {})
            sub = " · ".join(x for x in (f"STOFS {row['stofs_cycle']}" if row.get("stofs_cycle") else "", f"HRRR {row['hrrr_cycle']}" if row.get("hrrr_cycle") else "",
                                          f"GFS {row['gfs_cycle']}" if row.get("gfs_cycle") else "", "backfill" if str(row.get("backfill", "")).lower() == "true" else "",
                                          row.get("warm_from", "")) if x)
            card(f'<div class="cycle-title">{pd.Timestamp(r["t0"]):%a %b %d, %H}z <span class="cycle-sub">· {local(r["t0"], tz)}</span></div>'
                 f'<div class="cycle-sub" style="margin-bottom:10px">{sub or "preparing…"}</div>{stepper(r)}')

        # the newest cycle only, plus any older one still running; the rest folded away
        active = lambda r: any(v.get("state") in ("RUNNING", "PENDING") for v in r["stages"].values())
        show = [recs[0]] + [r for r in recs[1:] if active(r)]
        st.markdown("## Latest cycle")
        for r in show:
            cycle_card(r)
        older = [r for r in recs if r not in show]
        if older:
            with st.expander(f"Earlier cycles ({len(older)})"):
                for r in older:
                    cycle_card(r)

if view == "Forecast":
    if sel:
        key = f"{pd.Timestamp(sel):%Y%m%dT%H}"
        r = next((x for x in recs if x["t0"] == sel), None) if sel_set == "live" else None
        if sel_set == "baseline":
            st.caption("Baseline rerun on the GPU (NOAA tide + STOFS surge), prepared with the data available 5.5 h after the cycle time; "
                       "past river forecasts are not archived, so rivers use observed flow held constant.")
        if r is not None:
            card(stepper(r))
            ser = fetch_json(f"cycles/{key}/series.json") or {}
        else:   # older cycle: the compact archive copy (forecast, nowcast, forcing); observations come live
            arc = archive_cycle(sel_set, key) or {}
            r = {"t0": sel, "stages": {}, "has_products": False}
            ser = {"forecast": arc.get("forecast", {}), "catchup": arc.get("nowcast", {}), "observed": {}, "forcing": arc.get("forcing", {}),
                   "hwc": {"USGS_12113415": (fetch_json("archive/index.json") or {}).get("hwc_m", cfg.get("hwc_m", 0.0))}}
            src = (arc.get("meta") or {}).get("tide_source", "")
            if sel_set == "live":
                st.caption(f"Archived cycle{' · ' + src if src else ''}. Run details, reports and gauge tables are kept for the newest cycles only.")
        if not ser.get("forecast") and not ser.get("catchup"):
            st.info("No model output yet for this cycle.")
        st.caption("Stations from the ocean upstream: Elliott Bay, the pump station, the South Park gauge, Tukwila, Kent. The map beside each chart shows where it sits. "
                   "Observations are fetched live from NOAA and USGS every 5 minutes, so the observed line keeps extending past the forecast start.")
        g_all, dom = fetch_csv("gauges.csv"), fetch_json("domain.json")
        for name, title in STATIONS:
            if ser.get("forecast", {}).get(name) or ser.get("catchup", {}).get(name):
                cc, cm_ = st.columns([3, 1], vertical_alignment="center")
                with cc:
                    st.plotly_chart(wl_chart(ser, name, title, unit, sel, cfg), width="stretch", config=dict(displaylogo=False))
                with cm_:
                    if g_all is not None:
                        st.plotly_chart(station_map(g_all, dom, name), width="stretch", config=dict(displayModeBar=False, scrollZoom=False))
        met_section(ser, sel)
        if r.get("has_products"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("## Daily high water")
                pk = fetch_csv(f"cycles/{key}/peaks.csv")
                if pk is not None:
                    st.dataframe(pk, hide_index=True, width="stretch")
            with c2:
                st.markdown("## Gauge skill (nowcast)")
                sa = fetch_csv(f"cycles/{key}/skill_analysis.csv")
                if sa is None:
                    st.caption("Scored once the products job of this cycle has run.")
                else:
                    cols = [c for c in ("label", "n", "bias_m", "rmse_m", "storm_peak_error_m", "rmse_anomaly_m", "r", "note") if c in sa]
                    st.dataframe(sa[cols].rename(columns={"label": "gauge"}), hide_index=True, width="stretch")
            with st.expander("Full report: 2 m flood map, forcing, skill by lead time"):
                html = fetch(f"cycles/{key}/forecast_report.html")
                if html:
                    st.components.v1.html(html, height=1400, scrolling=True)

if view == "Stations":
    g = fetch_csv("gauges.csv")
    if g is not None:
        sa = fetch_csv(f"cycles/{pd.Timestamp(sel):%Y%m%dT%H}/skill_analysis.csv") if sel else None
        st.markdown("## Stations and gauges")
        st.caption("Gauges coloured by nowcast RMSE of the selected cycle: green under 10 cm, orange 10–25 cm, red over 25 cm, grey not scored yet. "
                   "Small dots are output-only points around the pump station.")
        st.plotly_chart(gauge_map(g, fetch_json("domain.json"), sa, cfg.get("aoi")), width="stretch", config=dict(displaylogo=False))
        st.dataframe(g[g.scored.astype(str).str.lower() == "true"][["name", "label", "source", "datum"]], hide_index=True, width="stretch")

if view == "Jobs & logs":
    st.markdown("## Slurm queue")
    q = pd.DataFrame(status.get("queue", []))
    if len(q):
        what = {"prep": "download forcing, prepare runs", "an": "nowcast run (observed forcing)", "fc": "7-day forecast run", "pp": "skill, 2 m map, report"}
        q["task"] = q.name.str.extract(r"cyc_(\w+?)_")[0].map(what)
        st.dataframe(q[["job", "name", "task", "state", "elapsed", "node", "cpus", "reason"]], hide_index=True, width="stretch")
    else:
        st.caption("No forecast jobs in the queue.")
    if sel:
        st.markdown(f"## Logs · {pd.Timestamp(sel):%Y-%m-%d %H}z")
        logs = fetch_json(f"cycles/{pd.Timestamp(sel):%Y%m%dT%H}/logs.json") or {}
        groups = [k for k in ("Preparation", "Nowcast SFINCS", "Forecast SFINCS", "Products") if logs.get(k)]
        if groups:
            which = st.radio("Log", groups, horizontal=True, label_visibility="collapsed")
            for fname, text in logs[which].items():
                st.markdown(f"**{fname}**")
                st.code(text or "(empty)", language="text")
        else:
            st.caption("Nothing written yet.")
        if logs.get("run.json"):
            with st.expander("Forecast inputs (run.json)"):
                st.json(logs["run.json"])

# ------------------------------------------------------------------ analysis (skill over one or many cycles)
AN_VARS = {"Water level": "wl", "Wind speed": "wind_ms", "Wind direction": "wind_dir", "Pressure": "pressure_hpa", "Rain": "rain_mmh"}
AN_STATIONS = {"South Park, USGS 12113415": "USGS_12113415", "Seattle, NOAA 9447130": "Seattle_bay",
               "Tukwila Golf Course, USGS 12113390": "USGS_12113390", "Kent, USGS 12113344": "USGS_12113344"}


def pair_metrics(m: pd.Series, o: pd.Series, circular=False) -> dict:
    p = pd.concat({"m": m, "o": o}, axis=1).dropna()
    if len(p) < 6:
        return {"n": len(p)}
    e = p.m - p.o
    if circular:
        e = (e + 180) % 360 - 180
        return {"n": len(p), "mae": float(e.abs().mean()), "bias": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean()))}
    ss = float(((p.o - p.o.mean()) ** 2).sum())
    return {"n": len(p), "mae": float(e.abs().mean()), "bias": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r2": float(1 - (e ** 2).sum() / ss) if ss > 0 else np.nan}


def analysis_tab():
    idx = fetch_json("archive/index.json")
    if not idx or not idx.get("sets"):
        st.info("The analysis archive appears once the publisher on Rougarou has been updated.")
        return
    sets = {k: v for k, v in idx["sets"].items() if v}
    sets = {"record": sorted(record_cycles(idx["sets"], []))}   # one record: GPU reforecasts + live GPU cycles
    hwc_m = idx.get("hwc_m", cfg.get("hwc_m", 0.0))
    st.markdown("## Forecast analysis")
    st.caption("Skill of one forecast cycle or a range of cycles against observations: water level at the gauges (NOAA, USGS, fetched live), "
               "wind, pressure and rain at Boeing Field (NWS KBFI). All cycles run on the GPU with the NOAA tide + STOFS surge boundary "
               "(Nov 12–26 reforecast with the data available at the time, then the live cycles). Each cycle is scored on its forecast "
               "from the cycle time up to the latest observation; fix the lead window (e.g. 0–48 h) to compare cycles like for like.")
    set_ = "record"
    c2, c3 = st.columns(2)
    t0s = sorted(f"{pd.Timestamp(t):%Y-%m-%dT%H}" if set_ != "record" else t for t in sets[set_])
    lab = lambda t: f"{pd.Timestamp(t):%Y-%m-%d %H}:00 UTC"
    with c2:
        t_from = st.selectbox("From cycle", t0s, index=max(0, len(t0s) - 28), format_func=lab)
    with c3:
        t_to = st.selectbox("To cycle", t0s, index=len(t0s) - 1, format_func=lab)
    c4, c5, c6, c7 = st.columns([1.2, 1.6, 1.0, 1.6])
    with c4:
        var_lab = st.selectbox("Variable", list(AN_VARS))
    var = AN_VARS[var_lab]
    with c5:
        stn_lab = st.selectbox("Station", list(AN_STATIONS) if var == "wl" else ["Boeing Field, NWS KBFI"])
    with c6:
        run_ = st.radio("Run", ["Forecast", "Nowcast"], horizontal=True)
    with c7:
        lead = st.slider("Forecast lead (h)", 0, 168, (0, 168), step=6, disabled=run_ == "Nowcast")
    sel = [t for t in t0s if t_from <= t <= t_to]
    if not sel:
        st.warning("The From cycle is after the To cycle.")
        return
    stn = AN_STATIONS.get(stn_lab)
    # observations over the whole range, fetched once
    first, last = pd.Timestamp(sel[0]) - pd.Timedelta("3D"), min(pd.Timestamp.utcnow().tz_localize(None), pd.Timestamp(sel[-1]) + pd.Timedelta("7D"))
    with st.spinner("Fetching observations…"):
        if var == "wl":
            obs = live_obs(stn, str(first.floor("1h")), str(last.floor("5min")))
            obs = obs.resample("15min").mean().dropna() if len(obs) else obs
        else:
            k = kbfi_obs(str(first), str(last))
            col = {"rain_mmh": "rain3h_mm"}.get(var, var)
            obs = k[col].dropna() if len(k) and col in k else pd.Series(dtype=float)
            if var == "rain_mmh" and len(obs):
                obs = obs / 3.0                       # 3-h totals -> mean rate
            elif len(obs):
                obs = obs.resample("1h").first() if var == "wind_dir" else obs.resample("1h").mean()
    unit, scale = {"wl": ("cm", 100.0), "wind_ms": ("m/s", 1.0), "wind_dir": ("°", 1.0), "pressure_hpa": ("hPa", 1.0), "rain_mmh": ("mm/h", 1.0)}[var]
    rows, pairs, windows = [], [], []
    from concurrent.futures import ThreadPoolExecutor
    _keys = [(RECORD.get(t0, "live") if set_ == "record" else set_, f"{pd.Timestamp(t0):%Y%m%dT%H}") for t0 in sel]   # archive files are <YYYYMMDDTHH>.json
    with st.spinner(f"Loading {len(sel)} cycles…"), ThreadPoolExecutor(max_workers=12) as ex:
        _recs = dict(zip(sel, ex.map(lambda k: archive_cycle(*k), _keys)))
    for t0 in sel:
        rec = _recs.get(t0)
        if not rec:
            continue
        key = "forecast" if run_ == "Forecast" else "nowcast"
        if var == "wl":
            m = ts(rec.get(key, {}).get(stn))
            if stn == "USGS_12113415" and key == "forecast":
                m = m + hwc_m
        else:
            m = ts((rec.get("forcing", {}).get(key) or {}).get(var))
        if m.empty:
            continue
        T = pd.Timestamp(t0)
        if key == "forecast":
            m = m[(m.index >= T + pd.Timedelta(hours=lead[0])) & (m.index <= T + pd.Timedelta(hours=lead[1]))]
        else:
            m = m[m.index < T]
        if var == "rain_mmh":
            m = m.resample("1h").mean().rolling("3h").sum().reindex(obs.index, method="nearest", tolerance=pd.Timedelta("40min")) / 3.0 if len(obs) else m
        o = obs.reindex(m.index, method="nearest", tolerance=pd.Timedelta("10min" if var == "wl" else "40min")) if len(obs) else pd.Series(index=m.index, dtype=float)
        past = m[m.index <= obs.index.max()] if len(obs) else m.iloc[:0]
        met = pair_metrics(m, o, circular=var == "wind_dir")
        cov = met.get("n", 0) / max(1, len(past))
        row = {"cycle": lab(t0), "n": met.get("n", 0), "coverage": cov}
        for k in ("mae", "bias", "rmse", "r2"):
            if k in met:
                row[k] = met[k] * (scale if k != "r2" else 1)
        if var == "wl" and met.get("n", 0):
            e = (m - o).dropna()
            row["within_4cm"] = float((e.abs() <= 0.04).mean())
        rows.append(row)
        pairs.append(pd.DataFrame({"model": m, "obs": o}).dropna().assign(cycle=t0))
        windows.append((t0, m))
    df = pd.DataFrame(rows)
    good = df[(df.n >= 6) & (df.coverage >= 0.5)] if len(df) and "mae" in df else df.iloc[:0]
    if var == "rain_mmh" and good.empty and len(df):
        st.info("Too few rain reports to score: Boeing Field reports rain only as 3-hour totals, and only when it rains. Choose a longer range of cycles.")
        return
    if good.empty:
        st.info("No scored cycles yet in this range: the forecasts have to be in the past to be compared with observations.")
        return
    # aggregated statistics
    st.markdown(f"**{stn_lab} · {var_lab} · {run_.lower()}{'' if run_ == 'Nowcast' else f', lead {lead[0]}–{lead[1]} h'} · {len(good)} of {len(df)} cycles scored**")
    tiles = [("Cycles scored", f"{len(good)} / {len(df)}", "≥50% of the elapsed forecast observed"),
             (f"MAE, mean", f"{good.mae.mean():.1f} {unit}", f"median {good.mae.median():.1f} {unit}"),
             ("Best cycle", f"{good.mae.min():.1f} {unit}", good.loc[good.mae.idxmin(), "cycle"]),
             ("Worst cycle", f"{good.mae.max():.1f} {unit}", good.loc[good.mae.idxmax(), "cycle"])]
    if "r2" in good:
        tiles.append(("R², mean", f"{good.r2.mean():.3f}", f"median {good.r2.median():.3f}"))
    if "within_4cm" in good:
        tiles.append(("Within 4 cm", f"{good.within_4cm.mean() * 100:.0f}%", "of all paired values (client target)"))
    st.markdown('<div class="tiles">' + "".join(f'<div class="card tile">{metric(k, v, s_)}</div>' for k, v, s_ in tiles[:4]) + '</div>', unsafe_allow_html=True)
    if len(tiles) > 4:
        st.markdown('<div class="tiles">' + "".join(f'<div class="card tile">{metric(k, v, s_)}</div>' for k, v, s_ in tiles[4:]) + '</div>', unsafe_allow_html=True)
    # metrics by cycle
    x = pd.to_datetime(good.cycle.str.replace(" UTC", ""))
    figs = [("mae", f"MAE ({unit})")] + ([("r2", "R²")] if "r2" in good else []) + ([("within_4cm", "Within 4 cm (%)")] if "within_4cm" in good else []) + ([("bias", f"Bias ({unit})")])
    cols = st.columns(len(figs))
    for c, (k, t) in zip(cols, figs):
        y = good[k] * (100 if k == "within_4cm" else 1)
        f = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=C["blue"], width=2), marker=dict(size=4)))
        if k == "mae" and var == "wl":
            f.add_hline(y=4, line=dict(color=C["red"], dash="dot", width=1), annotation_text="4 cm target", annotation_position="top left")
        f.update_layout(title=dict(text=f"{t} by forecast cycle", font=dict(size=13)), height=260, margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="white", plot_bgcolor="white", xaxis=dict(gridcolor="#f2f2f7"), yaxis=dict(gridcolor="#f2f2f7"), font=dict(family=FONT))
        with c:
            st.plotly_chart(f, width="stretch", config=dict(displaylogo=False))
    # time series: observations and each cycle's forecast over the chosen lead window
    if windows:
        st.markdown("**Time series: observed and each cycle's forecast over the selected lead window**")
        sc_ = (FT if unit_ == "ft" else 1.0) if var == "wl" else 1.0
        ax_unit = ("ft NAVD88" if unit_ == "ft" else "m NAVD88") if var == "wl" else unit
        f = go.Figure()
        t_lo, t_hi = min(w.index.min() for _, w in windows), max(w.index.max() for _, w in windows)
        ob = obs[t_lo:t_hi] if len(obs) else obs
        if len(ob):
            f.add_trace(go.Scatter(x=ob.index, y=ob.values * sc_, name="Observed", mode="markers",
                                   marker=dict(color=C["obs"], size=3.5, opacity=0.75)))
        n = len(windows)
        for i, (t0, w) in enumerate(windows):
            # one hue, light (oldest) to dark (newest): vermillion ramp
            frac = 0.25 + 0.75 * (i / max(1, n - 1))
            col = f"rgba(213, 94, 0, {frac:.2f})"
            f.add_trace(go.Scatter(x=w.index, y=w.values * sc_, name=lab(t0), mode="markers" if var == "wind_dir" else "lines",
                                   line=dict(color=col, width=1.4), marker=dict(color=col, size=3),
                                   showlegend=False, hovertemplate=f"cycle {lab(t0)}<br>%{{x|%b %d %H:%M}}: %{{y:.2f}} {ax_unit}<extra></extra>"))
        f.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="rgba(213, 94, 0, 0.9)", width=2),
                               name=f"Forecasts, {n} cycles (darker = newer), lead {lead[0]}–{lead[1]} h" if run_ == "Forecast" else f"Nowcasts, {n} cycles"))
        if var == "wl" and stn in ("USGS_12113415",):
            for v, labl, col in ((cfg["minor_m"], "minor", NWS["minor"]), (cfg["major_m"], "major", NWS["major"])):
                f.add_hline(y=v * sc_, line=dict(color=col, width=1, dash="dash"), opacity=0.7)
        f.update_layout(height=380, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor="white", plot_bgcolor="white", hovermode="closest",
                        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12)), xaxis=dict(gridcolor="#f2f2f7"),
                        yaxis=dict(title=ax_unit, gridcolor="#f2f2f7"), font=dict(family=FONT, color="#1d1d1f"))
        st.plotly_chart(f, width="stretch", config=dict(displaylogo=False))
    # scatter of all pairs
    allp = pd.concat(pairs) if pairs else pd.DataFrame(columns=["model", "obs"])
    allp = allp.dropna()
    cA, cB = st.columns([1, 1.3])
    with cA:
        if len(allp):
            sc = 1.0 if var != "wl" else FT if unit_ == "ft" else 1.0
            pm = pair_metrics(allp.model, allp.obs, circular=var == "wind_dir")
            st.markdown(f"**Matched values:** {len(allp):,} · **MAE:** {pm.get('mae', np.nan) * scale:.1f} {unit} · **bias:** {pm.get('bias', np.nan) * scale:+.1f} {unit}"
                        + (f" · **R²:** {pm['r2']:.3f}" if "r2" in pm else ""))
            sample = allp.sample(min(len(allp), 20000), random_state=0)
            ax_unit = ("ft NAVD88" if unit_ == "ft" else "m NAVD88") if var == "wl" else unit
            f = go.Figure(go.Scattergl(x=sample.obs * sc, y=sample.model * sc, mode="markers", marker=dict(size=4, color=C["blue"], opacity=0.45), name="matched values"))
            lo_, hi_ = float(min(sample.obs.min(), sample.model.min()) * sc), float(max(sample.obs.max(), sample.model.max()) * sc)
            f.add_trace(go.Scatter(x=[lo_, hi_], y=[lo_, hi_], mode="lines", line=dict(color="#8e8e93", dash="dash"), name="1:1"))
            f.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white", plot_bgcolor="white",
                            xaxis=dict(title=f"Observed ({ax_unit})", gridcolor="#f2f2f7"), yaxis=dict(title=f"Model ({ax_unit})", gridcolor="#f2f2f7", scaleanchor="x"),
                            legend=dict(orientation="h", y=1.05, x=0), font=dict(family=FONT))
            st.plotly_chart(f, width="stretch", config=dict(displaylogo=False))
    with cB:
        st.markdown("**By forecast cycle**")
        show = good.drop(columns=["coverage"]).copy()
        if "within_4cm" in show:
            show["within_4cm"] = (show.within_4cm * 100).round(0).astype(int).astype(str) + "%"
        ren = {"mae": f"MAE {unit}", "bias": f"bias {unit}", "rmse": f"RMSE {unit}", "r2": "R²", "within_4cm": "within 4 cm", "n": "matched values"}
        st.dataframe(show.iloc[::-1].rename(columns=ren).round(3), hide_index=True, width="stretch", height=430)
        step = "15-minute" if var == "wl" else ("3-hour" if var == "rain_mmh" else "hourly")
        st.caption(f"A matched value is one {step} {'forecast' if run_ == 'Forecast' else 'nowcast'} value and the observation at the same time "
                   f"(within {'10 min' if var == 'wl' else '40 min'}), inside the lead window and up to the latest observation. "
                   "All errors are model minus observed.")
        gaps = good[good.coverage < 0.9]
        if len(gaps):
            st.caption("Observations incomplete (under 90% of the elapsed forecast) for: " + ", ".join(gaps.cycle) + ".")



if view == "Analysis":
    unit_ = unit
    analysis_tab()

if view == "About":
    st.markdown("## What this is")
    st.markdown("A compound-flood forecast for South Park on the Duwamish River in Seattle, run for Seattle Public Utilities by The Water Institute. "
                "The model is SFINCS, a fast hydrodynamic model of the lower Duwamish and Green River from Elliott Bay to Auburn, on cells of 12.5 to 50 m "
                "with 2 m terrain detail inside each cell. It is driven by the ocean, the rivers and the weather, and it predicts water levels along the river "
                "and over the land for the next 7 days. Water levels are in feet or metres above NAVD88.")
    c1, c2 = st.columns(2)
    with c1:
        card('<div class="k" style="color:#2b6f7c;font-weight:600">NOWCAST</div><div style="margin-top:6px">The model run <b>up to now</b> with what was actually observed: '
             'NOAA tide gauges at Seattle and Tacoma for the ocean, gauged river flows, and the hourly HRRR weather analyses. It brings the model state up to the cycle time and shows how '
             'well the model reproduces the last days (its skill under perfect forcing). In the charts it is the <span style="color:#5ac8fa;font-weight:600">teal</span> line, ending at the cycle time.</div>')
    with c2:
        card('<div class="k" style="color:#c85a17;font-weight:600">FORECAST</div><div style="margin-top:6px">The model run <b>7 days ahead</b> from the end state of the nowcast, driven by forecasts: '
             'NOAA STOFS ocean levels (tide plus surge, corrected against the gauges), the Northwest River Forecast Center flow for the Green River, rain-based estimates for the Black River, '
             'and HRRR (first 48 h) then GFS weather: wind, pressure and rain. In the charts it is the <span style="color:#ff9f0a;font-weight:600">orange</span> line, starting at the cycle time. '
             'The South Park forecast includes a +3.9 cm high-water correction from the model calibration.</div>')
    st.markdown("## A forecast cycle")
    st.markdown("Cycles are named after the NOAA model cycle time they use, 00z and 12z (UTC). NOAA posts the ocean forecast about 5 to 6 hours after that time, so the "
                "cycle then runs in four stages, shown on the Overview: **Preprocessing** (download all inputs, build the two runs), **Nowcast** (10 min, or up to an hour after a cold start), "
                "**Forecast** (3 to 4 hours), **Products** (gauge skill, 2 m flood map, report). Each cycle's nowcast continues from the previous one, so the model state carries forward "
                "from cycle to cycle. Products for the 00z cycle are ready around 10 to 11 UTC (2 to 3 am Pacific); for 12z around 10 to 11 pm UTC.")
    st.markdown("## Reading the charts")
    st.markdown("Black is observed. Where an older cycle's orange line overlaps the black line, the gap is forecast error at that lead time; the gap between teal and black is the model's own error. "
                "Dashed lines at South Park are the minor and major flood levels at USGS gauge 12113415 and, in purple, the gate level at which streets flooded in the SPU pump-station log (11.6 ft). "
                "The 2 m flood map in each cycle's report shows where the forecast maximum water level stands above the ground around the pump station: blue from the river and bay, green where rain ponds higher than the river could reach.")
    st.markdown("## Where the numbers come from")
    st.markdown("Ocean: NOAA STOFS-2D-Global and CO-OPS gauges 9447130 (Seattle), 9446484 (Tacoma). Rivers: NWS/NWRFC forecast point AUBW1 (Green River near Auburn), USGS 12113344 (Kent), "
                "12113346 (Springbrook Creek), King County gauges 03G, 03u, 03V. Weather: NOAA HRRR and GFS. Skill: 26 USGS, NOAA and King County gauges inside the model domain (Stations tab). "
                "The system runs on The Water Institute's Rougarou cluster; this page reads a snapshot it publishes every 5 minutes, so it is 5 to 10 minutes behind.")
    st.caption("Experimental guidance, not an official flood warning. For warnings see the National Weather Service Seattle office.")

if auto:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
