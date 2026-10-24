"""
Outage Energy-Loss Estimator · prototype for the BayWa r.e. challenge, Munich Climate Week 2026
Main user: the asset / operations manager who documents an outage and prepares the insurance claim.
Run with:  streamlit run app.py
"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model as M

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ASSETS = ROOT / "assets"

GREEN, LIME, YELLOW, INK, GREY = "#3D8A4C", "#87B73B", "#F3DA49", "#2F2F2F", "#7A807C"
TURB_COL = [GREEN, LIME, "#C9A800", "#5C6F64", "#A3C985"]
BAND = "rgba(61,138,76,0.15)"
UNC_PCT = round(M.MIN_ENERGY_UNC * 100)

icon = ASSETS / "icon.png"
st.set_page_config(page_title="Outage Energy-Loss Estimator", page_icon=str(icon) if icon.exists() else None,
                   layout="wide")
PANEL = "rgba(255,255,255,0.72)"
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600&display=swap');
  html, body, [class*="css"], .stMarkdown, p, li, label, input, textarea, button {{
      font-family: 'Barlow', 'Segoe UI', Arial, sans-serif;}}
  .stApp {{background: linear-gradient(115deg, #EAF4E4 0%, #F1F3F7 38%, #EEEAF8 62%, #F8EEF3 100%) fixed;}}
  [data-testid="stHeader"] {{background: transparent;}}
  [data-testid="stSidebar"] {{background: rgba(255,255,255,0.55); backdrop-filter: blur(10px);
                              border-right: 1px solid rgba(61,138,76,0.15);}}
  .block-container {{padding-top: 1.4rem; max-width: 1400px;}}
  h1, h2, h3, h4, h1 *, h2 *, h3 *, h4 * {{color: {INK}; font-family: 'Barlow', 'Segoe UI', Arial, sans-serif !important; font-weight: 500 !important;
                   letter-spacing: -0.01em;}}
  div[data-testid="stMetric"] {{background: {PANEL}; border: 1px solid #CBD8C6 !important; border-radius: 22px;
                                padding: 16px 20px;}}
  div[data-testid="stMetricValue"] {{font-size: 1.55rem; font-weight: 500; color: {INK};}}
  div[data-testid="stMetricValue"] > div {{overflow: visible; text-overflow: clip; white-space: nowrap;}}
                                       border-radius: 999px; border: 1px solid #D7E0D3; width: fit-content;}}
  .stTabs [aria-selected="true"] p {{color: {GREEN} !important; font-weight: 600;}}
  .stTabs [data-baseweb="tab-highlight"] {{background-color: {GREEN} !important; height: 3px;}}
  .sectionlabel {{font-size: 0.8rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
                 color: {GREY}; margin: 8px 0 10px 4px;}}
  .process {{display:flex; align-items:stretch; gap:0; background: rgba(214,222,212,0.55);
            border: 1.5px solid #A9BBA4; border-radius: 28px; padding: 18px 14px; margin-bottom: 22px;}}
  .pstep {{flex:1; display:flex; flex-direction:column; align-items:center; text-align:center; gap:6px;
          padding: 6px 8px; color:{INK};}}
  .pstep svg {{width:46px; height:46px; color:#3B4A40;}}
  .pstep.ok svg {{color:{GREEN};}} .pstep.warn svg {{color:#B89A00;}} .pstep.bad svg {{color:#C0392B;}}
  .pstep .t {{font-size:0.95rem; font-weight:600;}}
  .pstep .s {{font-size:0.85rem; color:{GREY};}}
  .pstep.warn .s {{color:#8A7300;}} .pstep.bad .s {{color:#A93226;}}
  .parrow {{align-self:center; width:56px; height:2px; background:#3B4A40; position:relative; flex:none;}}
  .parrow:after {{content:''; position:absolute; right:-1px; top:-5px; border-left:9px solid #3B4A40;
                 border-top:6px solid transparent; border-bottom:6px solid transparent;}}
  .timeline {{background: rgba(214,222,212,0.45); border:1.5px solid #A9BBA4; border-radius:24px; padding:18px 22px; margin-bottom:16px;}}
  .tl-row {{display:flex; align-items:center; gap:10px; margin:6px 0;}}
  .tl-lab {{width:150px; font-size:0.85rem; color:{GREY}; flex:none;}}
  .tl-bar {{flex:1; display:flex; gap:4px;}}
  .seg {{border-radius:4px; font-size:0.78rem; padding:5px 6px; text-align:center; white-space:nowrap;
        overflow:hidden; text-overflow:ellipsis;}}
  .seg.train {{background: rgba(255,255,255,0.85); color:#4A554E;}}
  .seg.blind {{background:#3A3F3B; color:#FFFFFF; font-weight:600;}}
  .seg.test {{background:{YELLOW}; color:#3A3200; font-weight:600;}}
  .tl-notes {{display:flex; gap:4px; margin-left:160px; font-size:0.8rem; color:{INK};}}
  .status {{border-radius:20px; padding:14px 18px; margin:10px 0 14px 0; border:1.5px solid #CBD8C6; background:{PANEL};}}
  .status.ok {{border-color:{GREEN}; background:rgba(234,244,236,0.85);}}
  .status.warn {{border-color:#D9C24A; background:rgba(255,249,219,0.85);}}
  .status.bad {{border-color:#C0392B; background:rgba(251,234,234,0.85);}}
  .status b {{display:block; font-size:0.75rem; color:{GREY}; text-transform:uppercase; letter-spacing:.08em;}}
  .card {{background:{PANEL}; border:1px solid #D7E0D3; border-radius:18px; padding:14px 18px; margin-bottom:10px; color:{INK};}}
  .muted {{color:{GREY}; font-size:0.88rem;}}
  .deflist {{background:{PANEL}; border:1px solid #D7E0D3; border-radius:18px; padding:8px 18px;}}
  .deflist div {{padding:7px 0; border-bottom:1px solid #E6ECE3; font-size:0.92rem; color:{INK};}}
  .deflist div:last-child {{border-bottom:none;}}
  .deflist b {{display:inline-block; min-width:210px; font-weight:600;}}
  [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {{background:{PANEL}; border-radius:18px;
      border:1px solid #D7E0D3; padding:8px;}}
</style>""", unsafe_allow_html=True)

ICONS = {
    "turbine": '<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="24" cy="17" r="2.6"/><path d="M24 14.4 21.5 3.5M26.3 18.4 36.5 24M21.7 18.4 11.5 24M24 19.6V44M17 44h14"/></svg>',
    "data": '<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="7" width="38" height="27" rx="2.5"/><path d="M18 42h12M24 34v8M11 27l7-8 6 5 7-9 6 6"/></svg>',
    "model": '<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M31 5a9 9 0 0 0-8 12L6 34a3.5 3.5 0 0 0 5 5l17-17a9 9 0 0 0 12-8l-6 6-6-1-1-6z"/></svg>',
    "energy": '<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M27 4 10 27h13l-2 17 17-24H25z"/></svg>',
    "shield": '<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M24 4 8 10v12c0 10 7 18 16 22 9-4 16-12 16-22V10z"/><path d="m17 24 5 5 9-10"/></svg>',
}


def process_flow(steps):
    parts = []
    for i, (icon_key, title, sub, cls) in enumerate(steps):
        if i:
            parts.append("<div class='parrow'></div>")
        parts.append(f"<div class='pstep {cls}'>{ICONS[icon_key]}<div class='t'>{title}</div><div class='s'>{sub}</div></div>")
    st.markdown("<div class='process'>" + "".join(parts) + "</div>", unsafe_allow_html=True)


def timeline(index, parks, blind, test):
    """BayWa-style data split: training/validation, blind test (outage) and test period."""
    t0, t1 = index.min(), index.max()
    span = (t1 - t0).total_seconds()
    cuts = sorted([(blind[0], blind[1], "blind", "Blind test"), (test[0], test[1], "test", "Test")])
    segs, cur = [], t0
    for s, e, cls, lab in cuts:
        if s - cur > pd.Timedelta(hours=12):
            segs.append((cur, s, "train", "Training + validation"))
        segs.append((s, e, cls, lab)); cur = e
    if t1 - cur > pd.Timedelta(hours=12):
        segs.append((cur, t1, "train", "Training"))
    w = lambda s, e, c: max((e - s).total_seconds() / span, 0.09 if c != "train" else 0.04)
    def bar(label):
        cells = "".join(f"<div class='seg {c}' style='flex:{w(s, e, c):.4f}'>{lab}</div>"
                        for s, e, c, lab in segs)
        return f"<div class='tl-row'><div class='tl-lab'>{label}</div><div class='tl-bar'>{cells}</div></div>"
    notes = "".join(
        f"<div style='flex:{w(s, e, c):.4f};text-align:center'>"
        + ({"blind": "↑ energy loss", "test": "↑ RMSE &amp; MAE"}.get(c, "")) + "</div>" for s, e, c, _ in segs)
    st.markdown("<div class='timeline'>" + "".join(bar(p) for p in parks)
                + f"<div class='tl-notes'>{notes}</div></div>", unsafe_allow_html=True)


# ============================================================================ loading
@st.cache_data(show_spinner="Reading turbine data…")
def load_folder(folder: str):
    ts, curves = {}, {}
    for p in sorted(Path(folder).glob("*")):
        if p.suffix.lower() not in (".xlsx", ".xls", ".csv"):
            continue
        if "power_curve" in p.name.lower():
            t, c = M.load_power_curve(str(p), p.name); curves[t] = c
        else:
            t, d = M.load_timeseries(str(p), p.name); ts[t] = d
    return ts, curves


def load_uploads(files):
    ts, curves = {}, {}
    for f in files:
        if "power_curve" in f.name.lower():
            t, c = M.load_power_curve(f, f.name); curves[t] = c
        else:
            t, d = M.load_timeseries(f, f.name); ts[t] = d
    return ts, curves


def find_asset(stem):
    for ext in (".png", ".svg", ".jpg", ".jpeg"):
        if (ASSETS / f"{stem}{ext}").exists():
            return ASSETS / f"{stem}{ext}"
    return None


def deflist(items):
    st.markdown("<div class='deflist'>" + "".join(f"<div><b>{a}</b>{b}</div>" for a, b in items) + "</div>",
                unsafe_allow_html=True)


# ============================================================================ sidebar
with st.sidebar:
    st.markdown("### 1 · Data")
    src = st.radio("Source", ["Challenge dataset (built in)", "Upload files"], label_visibility="collapsed")
    if src.startswith("Upload"):
        files = st.file_uploader("Turbine data and power curves (.csv / .xlsx)", accept_multiple_files=True,
                                 type=["csv", "xlsx", "xls"])
        if not files:
            st.info("Upload one file per turbine (timestamp, wind speed) and one *_power_curve file per affected turbine.")
            st.stop()
        TS, CURVES = load_uploads(files)
    else:
        TS, CURVES = load_folder(str(DATA_DIR))

if not TS:
    st.error("No turbine data found."); st.stop()

ws = pd.DataFrame({t: d["wind_speed"] for t, d in TS.items()}).sort_index()
full_idx = pd.date_range(ws.index.min(), ws.index.max(), freq="10min")
ws = ws.reindex(full_idx)
wd = pd.DataFrame({t: d["wind_direction"] for t, d in TS.items() if "wind_direction" in d}).reindex(full_idx)
all_turbines = sorted(ws.columns)
parks = sorted({t[0] for t in all_turbines})

with st.sidebar:
    st.markdown("### 2 · Affected wind park")
    default_park = "T" if "T" in parks else parks[0]
    target_park = st.selectbox("Wind park", parks, index=parks.index(default_park))
    targets_avail = [t for t in all_turbines if t.startswith(target_park) and t in CURVES]
    if not targets_avail:
        st.error(f"No power curve found for wind park {target_park}."); st.stop()
    targets = st.multiselect("Turbines", targets_avail, default=targets_avail)
    ref_pool = [t for t in all_turbines if not t.startswith(target_park)]

    st.markdown("### 3 · Outage period")
    det_s, det_e = M.detect_outage(ws, targets) if targets else (None, None)
    if det_s is None:
        det_s, det_e = ws.index[-144], ws.index[-1]
        st.warning("No outage detected. Please enter the period manually.")
    else:
        st.success(f"Detected automatically:\n\n{det_s:%d.%m.%Y %H:%M} → {det_e:%d.%m.%Y %H:%M}")
    c1, c2 = st.columns([3, 2])
    o_sd = c1.date_input("Start", det_s.date(), format="DD.MM.YYYY")
    o_st = c2.time_input("Time", det_s.time(), step=600, key="st")
    c1, c2 = st.columns([3, 2])
    o_ed = c1.date_input("End", det_e.date(), format="DD.MM.YYYY")
    o_et = c2.time_input("Time", det_e.time(), step=600, key="et")
    outage_start, outage_end = datetime.combine(o_sd, o_st), datetime.combine(o_ed, o_et)

    st.markdown("### 4 · Valuation")
    price = st.number_input("Electricity price (€/MWh)", value=109.40, step=5.0,
                            help="Used to convert lost energy into lost revenue. Default: monthly EEG market value "
                                 "for onshore wind, September 2026 (10.94 ct/kWh, netztransparenz.de). "
                                 "Replace with the wind farm's contractual value where known.")

    with st.expander("Advanced settings"):
        n_refs = st.slider("Reference turbines per affected turbine", 1, max(1, len(ref_pool)),
                           min(3, len(ref_pool)),
                           help="For each affected turbine, the neighbouring turbines whose wind speed correlates "
                                "most closely with it are used. Three gave the best accuracy in testing.")
        manual_refs = st.multiselect("Manual selection of reference turbines (optional)", ref_pool,
                                     help="Overrides the automatic selection for all affected turbines.")
        has_dir = not wd.empty and wd.notna().any().any()
        use_dir = st.toggle("Account for wind direction", value=has_dir, disabled=not has_dir,
                            help="The relationship between the parks varies with the wind direction because of "
                                 "terrain and turbine positions. Recommended.")
        use_shift = st.toggle("Account for wind travel time (±30 min)", value=True,
                              help="Weather fronts and gusts reach neighbouring parks a few minutes apart. The model "
                                   "may use reference measurements from up to 30 minutes earlier or later. "
                                   "Slightly improves accuracy. Recommended.")
        val_len = st.slider("Test period after the outage (days)", 2, 10, 6,
                            help="Number of days after the outage, with known measurements, used to test the method. "
                                 "These days are excluded when the model is calibrated for the test.")
        availability = st.slider("Technical availability (%)", 80, 100, 100,
                                 help="Share of the outage period during which the turbines would have been able to "
                                      "operate (e.g. excluding planned maintenance). 100% assumes full operation; "
                                      "a lower value reduces the estimated loss proportionally.")

if not targets:
    st.warning("Please select at least one turbine."); st.stop()
if outage_end <= outage_start:
    st.error("The end of the outage must be after its start."); st.stop()

o_start, o_end = pd.Timestamp(outage_start), pd.Timestamp(outage_end)
in_outage = (ws.index >= o_start) & (ws.index <= o_end)
known = ~in_outage
v_start = o_end + pd.Timedelta(minutes=10)
v_end = min(v_start + pd.Timedelta(days=val_len) - pd.Timedelta(minutes=10), ws.index.max())
if (ws.index > o_end).sum() < 288:  # under two days after the outage -> test on the days before
    v_end = o_start - pd.Timedelta(minutes=10)
    v_start = v_end - pd.Timedelta(days=val_len) + pd.Timedelta(minutes=10)

# references per turbine: best-correlated neighbours (mentor feedback)
corr_tbl = pd.DataFrame({t: M.rank_references(ws, t, ref_pool, known) for t in targets})
refs_by_t, cfgs = {}, {}
for t in targets:
    refs = manual_refs or list(corr_tbl[t].sort_values(ascending=False).index[:n_refs])
    dir_src = next((r for r in refs if r in wd.columns and wd[r].notna().any()), None)
    refs_by_t[t] = refs
    cfgs[t] = M.ModelConfig(refs, bool(use_dir and dir_src), use_shift, dir_src)

res = M.run(ws, wd, CURVES, targets, cfgs, o_start, o_end, v_start, v_end)
f_av = availability / 100
for t in targets:
    for k in ("energy_kwh", "energy_lo_kwh", "energy_hi_kwh"):
        res[t][k] *= f_av
    res[t]["power_kw"] = res[t]["power_kw"] * f_av

pred_park_kw = sum(res[t]["power_kw"] for t in targets)
gross_kwh = sum(res[t]["energy_kwh"] for t in targets)
hours = (o_end - o_start).total_seconds() / 3600 + 1 / 6

# ============================================================================ outage verification
meter = None
meter_file = st.session_state.get("meter_file")
if meter_file is not None:
    try:
        meter_file.seek(0)
        meter = M.load_meter(meter_file, meter_file.name).reindex(pred_park_kw.index)
    except Exception as e:  # noqa: BLE001
        st.session_state.meter_error = str(e)

if meter is not None and meter.notna().sum() > 0:
    metered_kwh = float(meter.fillna(0).clip(lower=0).sum() * M.STEP_HOURS)
    share = metered_kwh / gross_kwh if gross_kwh else 0
    claim_kwh = max(gross_kwh - metered_kwh, 0.0)
    if share < 0.05:
        status, s_cls = "Verified: no feed-in during the outage", "ok"
        s_text = (f"The grid meter recorded {metered_kwh:,.0f} kWh during the outage ({share:.0%} of the expected "
                  "production). The turbines did not feed electricity into the grid.")
    elif share > 0.8:
        status, s_cls = "Communication failure only: feed-in continued", "bad"
        claim_kwh = 0.0
        s_text = (f"The grid meter recorded {metered_kwh:,.0f} kWh ({share:.0%} of the expected production). "
                  "Only the data connection was interrupted; no material energy loss occurred.")
    else:
        status, s_cls = "Partial outage", "warn"
        s_text = (f"The grid meter recorded {metered_kwh:,.0f} kWh ({share:.0%} of the expected production). "
                  "Only the difference between expected production and metered feed-in is claimable.")
    confirmation = "meter"
elif st.session_state.get("manual_ok"):
    metered_kwh, claim_kwh = None, gross_kwh
    status, s_cls = "Confirmed by the operator", "ok"
    s_text = f"Evidence: {st.session_state.get('evidence') or 'reference to be provided'}."
    confirmation = "operator"
else:
    metered_kwh, claim_kwh = None, gross_kwh
    status, s_cls = "Pending verification", "warn"
    s_text = "Upload grid meter data or record the operator's confirmation in the ‘Outage verification’ tab."
    confirmation = None

unc_share = (sum(res[t]["energy_hi_kwh"] for t in targets) - gross_kwh) / gross_kwh if gross_kwh else M.MIN_ENERGY_UNC
claim_lo, claim_hi = claim_kwh * (1 - unc_share), claim_kwh * (1 + unc_share)
val_dev = {t: res[t]["val_metrics"]["energy_dev_pct"] for t in targets}
avg_dev = float(np.mean([abs(v) for v in val_dev.values()]))
claim_ready = bool(confirmation) and s_cls != "bad"

# ============================================================================ header
logo, event_logo = find_asset("logo"), find_asset("event_logo")
h1, h2, h3 = st.columns([1.3, 5, 1], vertical_alignment="center")
if logo:
    h1.image(str(logo), width=190)
if event_logo:
    h3.image(str(event_logo), width=120)
with h2:
    st.markdown("## Outage Energy-Loss Estimator")
    st.markdown(f"<span class='muted'>Wind park <b>{target_park}</b> · turbines {', '.join(targets)} · "
                f"outage {o_start:%d.%m.%Y %H:%M} – {o_end:%d.%m.%Y %H:%M}</span>", unsafe_allow_html=True)
st.markdown("<div class='sectionlabel'>Process</div>", unsafe_allow_html=True)
n_refs_used = len({r for t in targets for r in refs_by_t[t]})
process_flow([
    ("turbine", "Neighbouring wind parks", f"{n_refs_used} reference turbines", "ok"),
    ("data", "Outage detection", f"{hours:.0f} h without data", "ok"),
    ("model", "Model", f"{len(refs_by_t[targets[0]])} references per turbine", "ok"),
    ("energy", "Missing wind & energy loss", f"{claim_kwh:,.0f} kWh", "ok" if s_cls != "bad" else "bad"),
    ("shield", "Verification & claim", status if confirmation else "Verification pending",
     "ok" if claim_ready else ("bad" if s_cls == "bad" else "warn")),
])

st.markdown("<div class='sectionlabel'>Key figures</div>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Estimated energy loss", f"{claim_kwh:,.0f} kWh", border=True,
          help=f"Electricity the turbines would have fed into the grid during the outage. The expected range "
               f"(±{unc_share:.0%}) reflects the typical deviation observed when the method was tested on past periods.")
k1.caption(f"Expected range: {claim_lo:,.0f} – {claim_hi:,.0f} kWh (±{unc_share:.0%})")
k2.metric("Estimated revenue loss", f"€ {claim_kwh / 1000 * price:,.0f}", border=True,
          help="Estimated energy loss (MWh) × electricity price. Policy deductibles, waiting periods and saved "
               "costs are not deducted.")
k2.caption(f"At {price:,.2f} €/MWh")
k3.metric("Outage duration", f"{hours:.0f} h", border=True,
          help="Period in which all selected turbines were without data, detected automatically.")
k3.caption(f"{o_start:%d.%m.} – {o_end:%d.%m.%Y}")
k4.metric("Tested accuracy", f"± {avg_dev:.1f} %", border=True,
          help=f"Average deviation of the estimated energy from the measured energy on the test period "
               f"({v_start:%d.%m.} – {v_end:%d.%m.%Y}), in which real values are known but were not used to calibrate "
               f"the model. Across 28 historical three-day periods the typical deviation was about ±{UNC_PCT}%.")
k4.caption("Energy estimate vs. measured values")

tab_o, tab_v, tab_e, tab_a, tab_r, tab_m = st.tabs(
    ["Overview", "Outage verification", "Energy loss", "Accuracy", "Insurance claim", "Methodology"])


def base_layout(fig, h, title=None, ytitle=None):
    fig.update_layout(height=h, title=title, yaxis_title=ytitle, hovermode="x unified",
                      plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)",
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      font=dict(color=INK, family="Barlow, Segoe UI, Arial, sans-serif"),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1EF"); fig.update_yaxes(showgrid=True, gridcolor="#EEF1EF")
    return fig


# ============================================================================ Overview
with tab_o:
    t_sel = st.radio("Turbine", targets, horizontal=True, key="ov_t")
    r = res[t_sel]
    pad = pd.Timedelta(days=1)
    win = (ws.index >= o_start - pad) & (ws.index <= o_end + pad)
    fig = go.Figure()
    fig.add_vrect(x0=o_start, x1=o_end, fillcolor="rgba(243,218,73,0.18)", line_width=0,
                  annotation_text="Outage", annotation_position="top left")
    fig.add_trace(go.Scatter(x=ws.index[win], y=ws[t_sel][win], name="Measured", line=dict(color=INK, width=1.3)))
    fig.add_trace(go.Scatter(x=list(r["pred_hi"].index) + list(r["pred_lo"].index[::-1]),
                             y=list(r["pred_hi"]) + list(r["pred_lo"][::-1]), fill="toself", fillcolor=BAND,
                             line=dict(width=0), name="Likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r["pred"].index, y=r["pred"], name="Estimated", line=dict(color=GREEN, width=2.2)))
    st.plotly_chart(base_layout(fig, 400, f"Wind speed at turbine {t_sel}", "m/s"), width="stretch")
    st.caption("Measured wind speed before and after the outage, and the estimated wind speed during the outage. "
               "The shaded band indicates the range within which the actual values most likely fell.")
    bar = go.Figure(go.Bar(x=targets, y=[res[t]["energy_kwh"] for t in targets],
                           marker_color=[TURB_COL[i % 5] for i in range(len(targets))],
                           text=[f"{res[t]['energy_kwh']:,.0f} kWh" for t in targets], textposition="outside"))
    bar.update_yaxes(range=[0, max(res[t]["energy_kwh"] for t in targets) * 1.2])
    st.plotly_chart(base_layout(bar, 300, "Estimated energy loss per turbine", "kWh"), width="stretch")

# ============================================================================ Outage verification
with tab_v:
    st.markdown("#### Outage verification with grid meter data")
    st.markdown("Missing turbine data does not necessarily mean that production stopped: the data connection may "
                "have failed while the turbines continued to feed in. The grid operator's meter at the connection "
                "point records the actual feed-in independently of the turbine data and is therefore used to verify "
                "the outage before a claim is submitted.")
    st.markdown(f"<div class='status {s_cls}'><b>Status</b>{status}<br><span class='muted'>{s_text}</span></div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Option A · Grid meter data**")
        st.file_uploader("Timestamp and power (kW) or energy (kWh) per 10 minutes, .csv or .xlsx",
                         type=["csv", "xlsx", "xls"], key="meter_file")
        if st.session_state.get("meter_error"):
            st.error(f"The meter file could not be read: {st.session_state.pop('meter_error')}")
        st.caption("Synthetic demonstration files are provided in the folder ‘demo’.")
    with c2:
        st.markdown("**Option B · Operator confirmation**")
        st.checkbox("The operator confirms that no electricity was fed into the grid during the outage",
                    key="manual_ok")
        st.text_input("Evidence reference (e.g. grid operator statement, service report number)", key="evidence")
    if meter is not None and meter.notna().sum() > 0:
        w2 = (pred_park_kw.index >= o_start - pd.Timedelta(hours=12)) & (pred_park_kw.index <= o_end + pd.Timedelta(hours=12))
        mf = go.Figure()
        mf.add_vrect(x0=o_start, x1=o_end, fillcolor="rgba(243,218,73,0.18)", line_width=0)
        mf.add_trace(go.Scatter(x=pred_park_kw.index, y=pred_park_kw, name="Expected production (estimate)",
                                line=dict(color=GREEN, width=2)))
        mf.add_trace(go.Scatter(x=meter.index[w2], y=meter[w2], name="Grid meter", line=dict(color=INK, width=1.4)))
        st.plotly_chart(base_layout(mf, 340, "Expected production vs. metered feed-in (whole wind park)", "kW"),
                        width="stretch")

# ============================================================================ Energy loss
with tab_e:
    st.markdown("#### Estimated energy loss per turbine")
    tbl = pd.DataFrame({
        "Turbine": targets,
        "Estimated loss (kWh)": [res[t]["energy_kwh"] for t in targets],
        "Lower bound (kWh)": [res[t]["energy_lo_kwh"] for t in targets],
        "Upper bound (kWh)": [res[t]["energy_hi_kwh"] for t in targets],
        "Revenue loss (€)": [res[t]["energy_kwh"] / 1000 * price for t in targets],
    })
    tot = pd.DataFrame([{"Turbine": "Total", "Estimated loss (kWh)": gross_kwh,
                         "Lower bound (kWh)": sum(res[t]["energy_lo_kwh"] for t in targets),
                         "Upper bound (kWh)": sum(res[t]["energy_hi_kwh"] for t in targets),
                         "Revenue loss (€)": gross_kwh / 1000 * price}])
    st.dataframe(pd.concat([tbl, tot]).style.format({c: "{:,.0f}" for c in tbl.columns[1:]}),
                 hide_index=True, width="stretch")
    deflist([
        ("Estimated loss", "Sum over all 10-minute intervals of the outage: estimated wind speed → power from the "
                           "manufacturer's power curve → power × 10 minutes."),
        ("Lower / upper bound", f"Estimated loss −/+ {unc_share:.0%}: the typical deviation observed when the method was "
                                "tested on past periods with known values."),
        ("Revenue loss", f"Estimated loss in MWh × electricity price ({price:,.2f} €/MWh)."),
    ])
    if metered_kwh is not None:
        st.markdown(f"<div class='card' style='margin-top:12px'>Expected production <b>{gross_kwh:,.0f} kWh</b> − "
                    f"metered feed-in <b>{metered_kwh:,.0f} kWh</b> = claimable loss <b>{claim_kwh:,.0f} kWh</b></div>",
                    unsafe_allow_html=True)

    st.markdown("#### Cumulative energy loss over the outage")
    cum = go.Figure()
    for i, t in enumerate(targets):
        e = res[t]["power_kw"].fillna(0).cumsum() * M.STEP_HOURS
        cum.add_trace(go.Scatter(x=e.index, y=e, name=t, stackgroup="one", line=dict(color=TURB_COL[i % 5])))
    st.plotly_chart(base_layout(cum, 340, None, "kWh"), width="stretch")
    st.caption("Running total of the lost energy from the start of the outage, stacked by turbine. The height at the "
               "end equals the total estimated loss. Steep sections indicate periods of strong wind, flat sections "
               "periods of weak wind with little lost production.")

    st.markdown("#### Estimated power output during the outage")
    pw = go.Figure()
    for i, t in enumerate(targets):
        pw.add_trace(go.Scatter(x=res[t]["power_kw"].index, y=res[t]["power_kw"], name=t,
                                line=dict(color=TURB_COL[i % 5], width=1.4)))
    st.plotly_chart(base_layout(pw, 300, None, "kW"), width="stretch")
    st.caption("Power each turbine would have produced in every 10-minute interval (maximum 4,500 kW per turbine).")

# ============================================================================ Accuracy
with tab_a:
    st.markdown("#### Accuracy on the test period")
    timeline(ws.index, [f"Wind park {target_park} (affected)", "Reference parks"], (o_start, o_end), (v_start, v_end))
    st.markdown(f"The method is tested on **{v_start:%d.%m.} – {v_end:%d.%m.%Y}**, a period in which the actual "
                "measurements are known but were not used to calibrate the model. The estimate is then compared "
                "with the measurements.")
    metric_rows = [
        ("Wind speed MAE (m/s)", "wind_mae", "{:.2f}", "Average absolute deviation of the wind speed per 10-minute interval."),
        ("Wind speed RMSE (m/s)", "wind_rmse", "{:.2f}", "As MAE, but larger deviations are weighted more strongly."),
        ("Power MAE (kW)", "power_mae", "{:,.0f}", "Average absolute deviation of the power per 10-minute interval."),
        ("Power RMSE (kW)", "power_rmse", "{:,.0f}", "As power MAE, with larger deviations weighted more strongly."),
        ("Measured energy (kWh)", "energy_true_kwh", "{:,.0f}", "Energy actually produced in the test period, from measured wind speeds."),
        ("Estimated energy (kWh)", "energy_pred_kwh", "{:,.0f}", "Energy estimated by the method for the same period."),
        ("Energy deviation (%)", "energy_dev_pct", "{:+.1f}", "Estimated vs. measured energy. Most relevant figure for a claim."),
    ]
    acc = pd.DataFrame({"Metric": [m[0] for m in metric_rows]})
    for t in targets:
        acc[t] = [m[2].format(res[t]["val_metrics"][m[1]]) for m in metric_rows]
    acc["Definition"] = [m[3] for m in metric_rows]
    st.dataframe(acc, hide_index=True, width="stretch",
                 column_config={"Definition": st.column_config.TextColumn(width="large")})
    st.caption("Individual 10-minute values deviate more than the energy total, because over- and underestimates "
               f"largely offset each other over many intervals. Across 28 historical three-day periods, the energy "
               f"total typically deviated by about ±{UNC_PCT}%, with a slight tendency to underestimate.")
    t_a = st.radio("Turbine", targets, horizontal=True, key="acc_t")
    vp = res[t_a]["val_pred"]; vt = ws[t_a].reindex(vp.index)
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=vt.index, y=vt, name="Measured", line=dict(color=INK, width=1.2)))
    f1.add_trace(go.Scatter(x=vp.index, y=vp, name="Estimated", line=dict(color=GREEN, width=1.8)))
    st.plotly_chart(base_layout(f1, 330, f"Estimated vs. measured wind speed at turbine {t_a}", "m/s"),
                    width="stretch")

# ============================================================================ Insurance claim
with tab_r:
    c1, c2 = st.columns(2)
    claimant = c1.text_input("Operator", "BayWa r.e. Wind GmbH")
    policy = c2.text_input("Policy / claim reference", "BI-2026-____")
    conf_line = {
        None: "Verification of the outage against grid meter data is pending; the figures below are therefore preliminary.",
        "meter": f"The outage has been verified against the grid operator's meter data at the connection point ({status.lower()}).",
        "operator": f"The outage has been confirmed by the operator (reference: {st.session_state.get('evidence') or 'to be provided'}).",
    }[confirmation]
    per_t = "\n".join(f"   – {t}: {res[t]['energy_kwh']:,.0f} kWh" for t in targets)
    meter_line = (f"Expected production during the outage amounted to {gross_kwh:,.0f} kWh, of which {metered_kwh:,.0f} kWh "
                  "were fed into the grid according to the meter data.\n" if metered_kwh is not None else "")
    body = f"""Subject: Notification of business interruption – wind park {target_park}, {o_start:%d.%m.%Y}–{o_end:%d.%m.%Y} – {policy}

Dear Sir or Madam,

We hereby notify you of a business interruption at wind park {target_park}, operated by {claimant}, and submit a preliminary quantification of the resulting loss.

1. Event
Between {o_start:%d.%m.%Y, %H:%M} and {o_end:%d.%m.%Y, %H:%M} ({hours:.0f} hours), turbines {', '.join(targets)} were unable to feed electricity into the grid. {conf_line}

2. Quantification of the energy loss
{meter_line}Estimated energy loss: {claim_kwh:,.0f} kWh ({claim_kwh/1000:,.1f} MWh); expected range {claim_lo:,.0f} – {claim_hi:,.0f} kWh.
Expected production per turbine:
{per_t}

Estimated revenue loss: EUR {claim_kwh/1000*price:,.0f}, based on an electricity price of {price:,.2f} EUR/MWh. Deductibles, waiting periods and saved costs in accordance with the policy terms have not been applied.

3. Methodology
As the turbines' own measurements are unavailable for the outage period, wind speeds were reconstructed from neighbouring turbines with a high correlation to each affected turbine. The relationship was calibrated on measurements outside the outage period, and the reconstructed wind speeds were converted into power using the manufacturer's power curves.

4. Reliability of the estimate
The method was tested on {v_start:%d.%m.}–{v_end:%d.%m.%Y}, a period with known measurements not used for calibration; the estimated energy deviated from the measured energy by {avg_dev:.1f}% on average. In a back-test on 28 historical three-day periods, the deviation was typically within ±{UNC_PCT}%, with a slight tendency to underestimate the loss.

The underlying time series and calculations are enclosed. We will be pleased to provide any further information you may require.

Yours faithfully,
{claimant}
"""
    if s_cls == "bad":
        st.error("The meter data shows that feed-in continued during the outage. A claim does not appear to be justified.")
    elif not confirmation:
        st.warning("The outage has not been verified yet. The draft therefore marks the figures as preliminary.")
    st.text_area("Draft notification", body, height=520)
    export = pd.concat([pd.DataFrame({"timestamp": res[t]["pred"].index, "turbine_name": t,
                                      "wind_speed_pred": res[t]["pred"].round(3).values,
                                      "power_pred_kw": res[t]["power_kw"].round(1).values}) for t in targets])
    d1, d2 = st.columns(2)
    d1.download_button("Download notification (.txt)", body.encode(), "insurance_notification.txt", width="stretch")
    d2.download_button("Download supporting time series (.csv)", export.to_csv(index=False).encode(),
                       f"supporting_timeseries_{o_start:%Y%m%d}.csv", "text/csv", width="stretch")

# ============================================================================ Methodology
with tab_m:
    st.markdown("#### Data split")
    timeline(ws.index, [f"Wind park {target_park} (affected)", "Reference parks"], (o_start, o_end), (v_start, v_end))
    st.markdown("#### Methodology in four steps")
    for i, txt in enumerate([
        "**Outage detection** – the longest period in which all selected turbines are without data at the same time.",
        "**Calibration** – neighbouring wind parks experience the same weather. For each affected turbine, the "
        "neighbouring turbines with the highest correlation are selected, and the relationship between their wind "
        "speeds is calibrated on the periods before and after the outage"
        + (", taking the wind direction into account" if use_dir else "") + ".",
        "**Reconstruction** – the reference turbines continued to measure during the outage; their measurements are "
        "used to estimate the wind speed at each affected turbine.",
        "**Energy calculation** – each 10-minute wind speed is converted into power with the manufacturer's power "
        "curve and summed to the energy loss in kWh."], 1):
        html_txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
        st.markdown(f"<div class='card'><b>{i}.</b> {html_txt}</div>", unsafe_allow_html=True)

    st.markdown("#### Selection of reference turbines")
    st.caption("Correlation between the wind speed of each neighbouring turbine and each affected turbine "
               "(1 = identical behaviour). Selected references are marked with ✓.")
    hm = corr_tbl.loc[corr_tbl.mean(axis=1).sort_values(ascending=False).index]
    heat = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, zmin=0.5, zmax=1,
                                colorscale=[[0, "#F2F6F3"], [0.5, LIME], [1, GREEN]],
                                text=[[("✓ " if r_ in refs_by_t[t] else "") + f"{hm.loc[r_, t]:.2f}" for t in hm.columns]
                                      for r_ in hm.index], texttemplate="%{text}", showscale=False))
    heat.update_layout(height=60 + 34 * len(hm), margin=dict(l=10, r=10, t=10, b=10), font=dict(color=INK),
                       yaxis=dict(autorange="reversed"), xaxis=dict(side="top"))
    c1, c2 = st.columns(2)
    c1.plotly_chart(heat, width="stretch")
    with c2:
        for t in targets:
            st.markdown(f"<div class='card'><b>{t}</b>: {', '.join(refs_by_t[t])}</div>", unsafe_allow_html=True)
        st.markdown("<span class='muted'>Several references are combined because each turbine is subject to local "
                    "turbulence. In testing, a single reference turbine produced clearly less accurate results.</span>",
                    unsafe_allow_html=True)

    st.markdown("#### Uncertainty and assumptions")
    deflist([
        ("Individual values", "A single 10-minute wind speed can deviate by more than 1 m/s due to local gusts."),
        ("Energy total", f"Deviations largely offset each other; across 28 historical three-day periods the total "
                         f"typically deviated by about ±{UNC_PCT}%."),
        ("Direction of error", "The method tends to underestimate slightly, as a smoothed estimate does not capture "
                               "short gusts. The estimate is therefore conservative."),
        ("Assumptions", "Normal operation during the outage: no planned maintenance and no curtailment by the grid operator."),
    ])

st.markdown("<div class='muted' style='margin-top:28px'>Prototype developed for the BayWa r.e. challenge · "
            "Munich Climate Week 2026</div>", unsafe_allow_html=True)
