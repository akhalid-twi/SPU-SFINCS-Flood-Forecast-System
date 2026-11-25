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

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

FT = 3.28084
STAGES = [("prep", "Prepare"), ("an", "Nowcast"), ("fc", "Forecast"), ("pp", "Products")]
STATIONS = [("USGS_12113415", "Duwamish at South Park (USGS 12113415)"), ("SouthPark_PumpStation", "South Park pump station"),
            ("Seattle_bay", "Elliott Bay, Seattle (NOAA 9447130)"), ("USGS_12113390", "Duwamish at Golf Course, Tukwila (USGS 12113390)"),
            ("USGS_12113344", "Green River at 200th St, Kent (USGS 12113344)")]
C = dict(obs="#1d1d1f", an="#5ac8fa", fc="#ff9f0a", minor="#ffcc00", major="#ff3b30", street="#af52de",
         green="#34c759", blue="#0a84ff", orange="#ff9f0a", red="#ff3b30", gray="#8e8e93")
MAP = getattr(go, "Scattermap", None) or go.Scattermapbox      # plotly >= 6 renamed the map traces
MAPKEY = "map" if hasattr(go, "Scattermap") else "mapbox"
FONT = "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', Inter, sans-serif"

st.set_page_config(page_title="South Park forecast", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")
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
.steps {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
.step {{ background: #f5f5f7; border-radius: 12px; padding: 10px 12px; min-height: 64px; }}
.step .n {{ font-size: 12px; color: #6e6e73; }} .step .t {{ margin: 3px 0 4px; }}
.bar {{ height: 6px; background: #e5e5ea; border-radius: 3px; overflow: hidden; margin-top: 6px; }} .bar > i {{ display: block; height: 100%; background: #0a84ff; border-radius: 3px; }}
.cycle-title {{ font-size: 17px; font-weight: 600; letter-spacing: -0.02em; }} .cycle-sub {{ font-size: 12.5px; color: #6e6e73; }}
div[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"], button[data-testid="stExpandSidebarButton"] {{ display: none; }}
div[data-testid="stRadio"] label p, div[data-testid="stCheckbox"] label p {{ font-size: 13px; }}
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


def fetch_json(path):
    t = fetch(path)
    return json.loads(t) if t else None


def fetch_csv(path):
    from io import StringIO
    t = fetch(path)
    return pd.read_csv(StringIO(t)) if t else None


def ts(points) -> pd.Series:
    if not points:
        return pd.Series(dtype=float)
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
        extra = ""
        if s.get("state") == "RUNNING":
            extra = f'<div class="s">{s.get("elapsed", "")} on {s.get("node", "")}</div>'
            if s.get("pct") is not None:
                extra += f'<div class="bar"><i style="width:{s["pct"]:.0f}%"></i></div><div class="s">SFINCS {s["pct"]:.0f}%</div>'
        elif s.get("state") == "PENDING":
            extra = f'<div class="s">{s.get("reason", "")}</div>'
        elif s.get("state") == "COMPLETED" and s.get("elapsed"):
            extra = f'<div class="s">{s["elapsed"]}</div>'
        cells.append(f'<div class="step"><div class="n">{label}</div><div class="t">{pill(s.get("state"))}</div>{extra}</div>')
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
    for key, label, col, w in (("observed", "Observed", C["obs"], 1.6), ("catchup", "Nowcast (observed forcing)", C["an"], 2), ("forecast", "Forecast", C["fc"], 2.4)):
        s = ts(ser.get(key, {}).get(name))
        if len(s):
            if key == "forecast":
                s = s + corr
            fig.add_trace(go.Scatter(x=s.index, y=s.values * F, name=label, line=dict(color=col, width=w)))
    if name in ("USGS_12113415", "SouthPark_PumpStation"):
        for v, lab, col in ((cfg["minor_m"], "minor", C["minor"]), (cfg["major_m"], "major", C["major"])):
            fig.add_hline(y=v * F, line=dict(color=col, dash="dash", width=1.2), annotation_text=f"{lab} {v * F:.2f}", annotation_position="right")
        if cfg.get("street_ft"):
            v = cfg["street_ft"] if unit == "ft" else cfg["street_ft"] / FT
            fig.add_hline(y=v, line=dict(color=C["street"], dash="dot", width=1.2), annotation_text="streets flood (SPU log)", annotation_position="top left")
    fig.add_vline(x=pd.Timestamp(t0), line=dict(color="#8e8e93", dash="dot", width=1))
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=320, margin=dict(l=10, r=70, t=40, b=10), paper_bgcolor="white",
                      plot_bgcolor="white", hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0, font=dict(size=12)),
                      yaxis=dict(title=f"{unit} NAVD88", gridcolor="#f2f2f7", zeroline=False), xaxis=dict(gridcolor="#f2f2f7"),
                      font=dict(family=FONT, color="#1d1d1f"))
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
    st.title("South Park flood forecast")
    st.warning(f"No snapshot found at {URL or DIR}. The cluster publishes one every 5 minutes; try again shortly.")
    st.stop()
cfg, recs, tz = status["config"], status["cycles"], status["config"]["tz"]
F = lambda u: FT if u == "ft" else 1.0

st.markdown("# South Park flood forecast")
st.markdown(f'<div class="cycle-sub">Duwamish River, Seattle · SFINCS compound flood model · data as of {local(status["updated"], tz)}</div>', unsafe_allow_html=True)
# controls in a row under the title (no sidebar)
c_sel, c_unit, c_auto, c_btn = st.columns([3.2, 1.1, 1.3, 1.1], vertical_alignment="bottom")
with c_sel:
    sel = st.selectbox("Cycle", [r["t0"] for r in recs], format_func=lambda t: f"{pd.Timestamp(t):%a %b %d, %H}z · {local(t, tz)}") if recs else None
with c_unit:
    unit = st.radio("Units", ["ft", "m"], horizontal=True)
with c_auto:
    auto = st.toggle("Auto-refresh (60 s)", value=os.environ.get("SPU_NO_AUTO") != "1")
with c_btn:
    if st.button("Refresh now", width="stretch"):
        st.cache_data.clear()
        st.rerun()
st.caption(f"Cycles at {', '.join(f'{h:02d}z' for h in cfg['hours'])} UTC · thresholds at USGS 12113415: minor {cfg['minor_m'] * FT:.2f} ft, "
           f"major {cfg['major_m'] * FT:.2f} ft NAVD88 · snapshot {local(status['updated'], tz)}")
tab_over, tab_fc, tab_map, tab_jobs, tab_about = st.tabs(["Overview", "Forecast", "Stations", "Jobs & logs", "About"])

with tab_over:
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
        for c, (k, v, s) in zip(st.columns(4), vals):
            with c:
                card(metric(k, v, s))
        st.markdown("## Cycles")
        for r in recs:
            row = r.get("row", {})
            sub = " · ".join(x for x in (f"STOFS {row['stofs_cycle']}" if row.get("stofs_cycle") else "", f"HRRR {row['hrrr_cycle']}" if row.get("hrrr_cycle") else "",
                                          f"GFS {row['gfs_cycle']}" if row.get("gfs_cycle") else "", "backfill" if str(row.get("backfill", "")).lower() == "true" else "",
                                          row.get("warm_from", "")) if x)
            card(f'<div class="cycle-title">{pd.Timestamp(r["t0"]):%a %b %d, %H}z <span class="cycle-sub">· {local(r["t0"], tz)}</span></div>'
                 f'<div class="cycle-sub" style="margin-bottom:10px">{sub or "preparing…"}</div>{stepper(r)}')

with tab_fc:
    if sel:
        r = next(x for x in recs if x["t0"] == sel)
        key = f"{pd.Timestamp(sel):%Y%m%dT%H}"
        card(stepper(r))
        ser = fetch_json(f"cycles/{key}/series.json") or {}
        if not ser.get("forecast") and not ser.get("catchup"):
            st.info("No model output yet for this cycle.")
        for name, title in STATIONS:
            if ser.get("forecast", {}).get(name) or ser.get("catchup", {}).get(name):
                st.plotly_chart(wl_chart(ser, name, title, unit, sel, cfg), width="stretch", config=dict(displaylogo=False))
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

with tab_map:
    g = fetch_csv("gauges.csv")
    if g is not None:
        sa = fetch_csv(f"cycles/{pd.Timestamp(sel):%Y%m%dT%H}/skill_analysis.csv") if sel else None
        st.markdown("## Stations and gauges")
        st.caption("Gauges coloured by nowcast RMSE of the selected cycle: green under 10 cm, orange 10–25 cm, red over 25 cm, grey not scored yet. "
                   "Small dots are output-only points around the pump station.")
        st.plotly_chart(gauge_map(g, fetch_json("domain.json"), sa, cfg.get("aoi")), width="stretch", config=dict(displaylogo=False))
        st.dataframe(g[g.scored.astype(str).str.lower() == "true"][["name", "label", "source", "datum"]], hide_index=True, width="stretch")

with tab_jobs:
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

with tab_about:
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
                "cycle then runs in four stages, shown on the Overview: **Prepare** (download all inputs, build the two runs), **Nowcast** (10 min, or up to an hour after a cold start), "
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
