"""
Outage Energy-Loss Estimator  ·  prototype for the BayWa r.e. challenge, Munich Climate Week 2026
Main user: the asset / operations manager who has to document an outage and prepare the insurance claim.
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

# Brand palette (sampled from BayWa r.e. branding)
GREEN, LIME, YELLOW, INK, GREY = "#3D8A4C", "#87B73B", "#F3DA49", "#2F2F2F", "#8A8F8C"
TURB_COL = [GREEN, LIME, "#C9A800", "#5C6F64", "#A3C985"]
BAND = "rgba(61,138,76,0.15)"

st.set_page_config(page_title="Outage Energy-Loss Estimator", page_icon="🌬️", layout="wide")
st.markdown(f"""
<style>
  .block-container {{padding-top: 1.2rem; max-width: 1400px;}}
  h1, h2, h3 {{color: {INK};}}
  div[data-testid="stMetricValue"] {{font-size: 1.65rem; color: {INK};}}
  div[data-testid="stMetric"] {{border-left: 5px solid {GREEN} !important;}}
  .brandbar {{height: 6px; border-radius: 3px; margin: 2px 0 14px 0;
             background: linear-gradient(90deg, {GREEN} 0 60%, {LIME} 60% 85%, {YELLOW} 85% 100%);}}
  .steps {{display:flex; gap:8px; flex-wrap:wrap; margin: 4px 0 16px 0;}}
  .step {{flex:1; min-width:150px; border-radius:10px; padding:9px 12px; font-size:0.86rem;
         background:#F2F6F3; color:{INK}; border:1px solid #E1E8E3;}}
  .step b {{display:block; font-size:0.78rem; color:{GREY}; text-transform:uppercase; letter-spacing:.04em;}}
  .ok {{border-color:{GREEN}; background:#EAF4EC;}}
  .warn {{border-color:{YELLOW}; background:#FFF9DB;}}
  .bad {{border-color:#C0392B; background:#FBEAEA;}}
  .card {{background:#F2F6F3; border-radius:10px; padding:12px 16px; margin-bottom:8px; color:{INK};}}
  .muted {{color:{GREY}; font-size:0.85rem;}}
</style>""", unsafe_allow_html=True)


# ============================================================================ loading
@st.cache_data(show_spinner="Reading turbine files…")
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


def find_logo(stem="logo"):
    for ext in (".png", ".svg", ".jpg", ".jpeg"):
        if (ASSETS / f"{stem}{ext}").exists():
            return ASSETS / f"{stem}{ext}"
    return None


# ============================================================================ sidebar
with st.sidebar:
    st.markdown("### 1 · Data")
    src = st.radio("Source", ["Challenge dataset (built in)", "Upload files"], label_visibility="collapsed")
    if src.startswith("Upload"):
        files = st.file_uploader("Turbine data + power curves (.csv / .xlsx)", accept_multiple_files=True,
                                 type=["csv", "xlsx", "xls"])
        if not files:
            st.info("One file per turbine (timestamp, wind speed …) and one *_power_curve file per affected turbine.")
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
        st.error(f"No power curve found for park {target_park}."); st.stop()
    targets = st.multiselect("Turbines", targets_avail, default=targets_avail)
    ref_pool = [t for t in all_turbines if not t.startswith(target_park)]

    st.markdown("### 3 · Outage period")
    det_s, det_e = M.detect_outage(ws, targets) if targets else (None, None)
    if det_s is None:
        det_s, det_e = ws.index[-144], ws.index[-1]
        st.warning("No outage found automatically — please set it.")
    else:
        st.success(f"Detected automatically:\n\n{det_s:%d.%m.%Y %H:%M} → {det_e:%d.%m.%Y %H:%M}")
    c1, c2 = st.columns([3, 2])
    o_sd = c1.date_input("Start", det_s.date(), format="DD.MM.YYYY")
    o_st = c2.time_input("Time", det_s.time(), step=600, key="st")
    c1, c2 = st.columns([3, 2])
    o_ed = c1.date_input("End", det_e.date(), format="DD.MM.YYYY")
    o_et = c2.time_input("Time", det_e.time(), step=600, key="et")
    outage_start, outage_end = datetime.combine(o_sd, o_st), datetime.combine(o_ed, o_et)

    st.markdown("### 4 · Claim")
    price = st.number_input("Remuneration (€/MWh)", value=80.0, step=5.0,
                            help="Value of one MWh for this wind farm (EEG award value / marketing contract).")

    with st.expander("Advanced settings"):
        n_refs = st.slider("Reference turbines per turbine", 1, max(1, len(ref_pool)), min(3, len(ref_pool)),
                           help="For each affected turbine, the neighbours whose wind follows it most closely are used.")
        manual_refs = st.multiselect("…or choose references manually (applies to all turbines)", ref_pool)
        has_dir = not wd.empty and wd.notna().any().any()
        use_dir = st.toggle("Use wind direction", value=has_dir, disabled=not has_dir,
                            help="The relationship between parks changes with where the wind comes from.")
        use_shift = st.toggle("Allow ±30 min time shift", value=True,
                              help="A gust reaches the parks a few minutes apart.")
        val_len = st.slider("Accuracy check: days after the outage", 2, 10, 6)
        availability = st.slider("Assumed technical availability (%)", 80, 100, 100)

if not targets:
    st.warning("Select at least one turbine."); st.stop()
if outage_end <= outage_start:
    st.error("The outage end must be after the start."); st.stop()

o_start, o_end = pd.Timestamp(outage_start), pd.Timestamp(outage_end)
in_outage = (ws.index >= o_start) & (ws.index <= o_end)
known = ~in_outage
v_start = o_end + pd.Timedelta(minutes=10)
v_end = min(v_start + pd.Timedelta(days=val_len) - pd.Timedelta(minutes=10), ws.index.max())
if (ws.index > o_end).sum() < 288:  # less than 2 days after the outage -> check on the days before
    v_end = o_start - pd.Timedelta(minutes=10)
    v_start = v_end - pd.Timedelta(days=val_len) + pd.Timedelta(minutes=10)

# ---- references per turbine (mentor feedback: use the best-correlated neighbours for each turbine)
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

# ============================================================================ outage confirmation (meter data)
if "meter_mode" not in st.session_state:
    st.session_state.meter_mode = "none"

meter = None
meter_file = st.session_state.get("meter_file")
if meter_file is not None:
    try:
        meter_file.seek(0)
        meter = M.load_meter(meter_file, meter_file.name).reindex(pred_park_kw.index)
    except Exception as e:  # noqa: BLE001
        st.session_state.meter_error = str(e)
        meter = None

if meter is not None and meter.notna().sum() > 0:
    metered_kwh = float(meter.fillna(0).clip(lower=0).sum() * M.STEP_HOURS)
    share = metered_kwh / gross_kwh if gross_kwh else 0
    # claimable = expected production minus what really reached the grid (on totals: interval noise cancels out)
    claim_kwh = max(gross_kwh - metered_kwh, 0.0)
    if share < 0.05:
        status, s_cls = "Confirmed: no electricity reached the grid", "ok"
        s_text = (f"The grid meter recorded only {metered_kwh:,.0f} kWh during the outage "
                  f"({share:.0%} of the expected production). The turbines really stopped feeding in.")
    elif share > 0.8:
        status, s_cls = "Communication gap only — turbines kept producing", "bad"
        claim_kwh = 0.0
        s_text = (f"The grid meter recorded {metered_kwh:,.0f} kWh ({share:.0%} of the expected production). "
                  "Only the data link failed; there is little or no energy loss to claim.")
    else:
        status, s_cls = "Partial outage", "warn"
        s_text = (f"The grid meter recorded {metered_kwh:,.0f} kWh ({share:.0%} of the expected production). "
                  "Only the missing part is claimable.")
    confirmation = "grid meter data"
elif st.session_state.get("manual_ok"):
    metered_kwh, claim_kwh = None, gross_kwh
    status, s_cls = "Confirmed by operator", "ok"
    s_text = f"Confirmed manually: {st.session_state.get('evidence') or 'no evidence reference given'}."
    confirmation = "operator confirmation"
else:
    metered_kwh, claim_kwh = None, gross_kwh
    status, s_cls = "Not yet confirmed", "warn"
    s_text = "Upload grid meter data or confirm the outage in the ‘Outage check’ tab."
    confirmation = None

unc_share = (sum(res[t]["energy_hi_kwh"] for t in targets) - gross_kwh) / gross_kwh if gross_kwh else 0.1
claim_lo, claim_hi = claim_kwh * (1 - unc_share), claim_kwh * (1 + unc_share)
val_dev = {t: res[t]["val_metrics"]["energy_dev_pct"] for t in targets}

# ============================================================================ header
logo, event_logo = find_logo("logo"), find_logo("event_logo")
h1, h2, h3 = st.columns([1.3, 5, 1], vertical_alignment="center")
if logo:
    h1.image(str(logo), width=190)
if event_logo:
    h3.image(str(event_logo), width=120)
with h2:
    st.markdown("## Outage Energy-Loss Estimator")
    st.markdown(f"<span class='muted'>Wind park <b>{target_park}</b> · turbines {', '.join(targets)} · "
                f"{o_start:%d.%m.%Y %H:%M} → {o_end:%d.%m.%Y %H:%M} ({hours:.0f} h)</span>", unsafe_allow_html=True)
st.markdown("<div class='brandbar'></div>", unsafe_allow_html=True)

steps = [
    ("1 · Data", f"{len(TS)} turbines loaded", "ok"),
    ("2 · Outage", f"{hours:.0f} h detected", "ok"),
    ("3 · Confirmation", status, s_cls),
    ("4 · Energy loss", f"{claim_kwh:,.0f} kWh", "ok" if s_cls != "bad" else "bad"),
    ("5 · Claim", "Ready to send" if confirmation and s_cls != "bad" else "Waiting for confirmation",
     "ok" if confirmation and s_cls != "bad" else "warn"),
]
st.markdown("<div class='steps'>" + "".join(f"<div class='step {c}'><b>{a}</b>{b}</div>" for a, b, c in steps)
            + "</div>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Lost energy", f"{claim_kwh:,.0f} kWh", border=True)
k1.caption(f"Range {claim_lo:,.0f} – {claim_hi:,.0f} kWh")
k2.metric("Lost revenue", f"€ {claim_kwh / 1000 * price:,.0f}", border=True)
k2.caption(f"At {price:,.0f} €/MWh")
k3.metric("Outage duration", f"{hours:.0f} h", border=True)
k3.caption(f"{int(in_outage.sum())} ten-minute intervals")
k4.metric("Tested accuracy", f"± {np.mean([abs(v) for v in val_dev.values()]):.0f} %", border=True)
k4.caption("Energy total vs. real values on the check days")

tab_o, tab_c, tab_e, tab_a, tab_m, tab_r = st.tabs(
    ["Overview", "Outage check", "Energy loss", "Accuracy", "How it works", "Insurance claim"])


def base_layout(fig, h, title=None, ytitle=None):
    fig.update_layout(height=h, title=title, yaxis_title=ytitle, hovermode="x unified", plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10), font=dict(color=INK),
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
                  annotation_text="outage", annotation_position="top left")
    fig.add_trace(go.Scatter(x=ws.index[win], y=ws[t_sel][win], name="measured", line=dict(color=INK, width=1.3)))
    fig.add_trace(go.Scatter(x=list(r["pred_hi"].index) + list(r["pred_lo"].index[::-1]),
                             y=list(r["pred_hi"]) + list(r["pred_lo"][::-1]), fill="toself", fillcolor=BAND,
                             line=dict(width=0), name="likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r["pred"].index, y=r["pred"], name="estimated", line=dict(color=GREEN, width=2.2)))
    st.plotly_chart(base_layout(fig, 400, f"Wind speed at {t_sel}", "m/s"), width="stretch")
    st.caption("Black = what the turbine measured. Green = our estimate for the missing period, "
               "with the range where real values most likely were.")

    bar = go.Figure(go.Bar(x=targets, y=[res[t]["energy_kwh"] for t in targets],
                           marker_color=[TURB_COL[i % 5] for i in range(len(targets))],
                           text=[f"{res[t]['energy_kwh']:,.0f} kWh" for t in targets], textposition="outside"))
    st.plotly_chart(base_layout(bar, 300, "Lost energy per turbine", "kWh"), width="stretch")

# ============================================================================ Outage check
with tab_c:
    st.markdown("#### Did the turbines really stop, or was it only a communication problem?")
    st.markdown("The turbines' own data is missing, but the **grid operator's meter** at the connection point keeps "
                "counting what is fed into the grid. If it shows production during the outage, only the data link "
                "failed and there is no (or less) loss to claim.")
    st.markdown(f"<div class='step {s_cls}' style='margin:10px 0'><b>Status</b>{status}<br>"
                f"<span class='muted'>{s_text}</span></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.file_uploader("Grid meter data (.csv / .xlsx): timestamp + power (kW) or energy (kWh)",
                         type=["csv", "xlsx", "xls"], key="meter_file")
        if st.session_state.get("meter_error"):
            st.error(f"Could not read the meter file: {st.session_state.pop('meter_error')}")
        st.caption("Demo files (synthetic, for testing only) are in the folder `demo/`.")
    with c2:
        st.checkbox("No meter data available — the operator confirms that no electricity was fed in",
                    key="manual_ok")
        st.text_input("Evidence reference (e.g. grid operator statement, service report no.)", key="evidence")
    if meter is not None and meter.notna().sum() > 0:
        win = (pred_park_kw.index >= o_start - pd.Timedelta(hours=12)) & (pred_park_kw.index <= o_end + pd.Timedelta(hours=12))
        mf = go.Figure()
        mf.add_vrect(x0=o_start, x1=o_end, fillcolor="rgba(243,218,73,0.18)", line_width=0)
        mf.add_trace(go.Scatter(x=pred_park_kw.index, y=pred_park_kw, name="expected production (estimate)",
                                line=dict(color=GREEN, width=2)))
        mf.add_trace(go.Scatter(x=meter.index[win] if len(meter) else [], y=meter[win] if len(meter) else [],
                                name="grid meter", line=dict(color=INK, width=1.4)))
        st.plotly_chart(base_layout(mf, 340, "Expected production vs. grid meter (whole park)", "kW"), width="stretch")

# ============================================================================ Energy loss
with tab_e:
    c1, c2 = st.columns([1, 2])
    with c1:
        tbl = pd.DataFrame({
            "Turbine": targets,
            "Lost energy (kWh)": [res[t]["energy_kwh"] for t in targets],
            "Low (kWh)": [res[t]["energy_lo_kwh"] for t in targets],
            "High (kWh)": [res[t]["energy_hi_kwh"] for t in targets],
            "Revenue (€)": [res[t]["energy_kwh"] / 1000 * price for t in targets],
        })
        tot = pd.DataFrame([{"Turbine": "Total", "Lost energy (kWh)": gross_kwh,
                             "Low (kWh)": sum(res[t]["energy_lo_kwh"] for t in targets),
                             "High (kWh)": sum(res[t]["energy_hi_kwh"] for t in targets),
                             "Revenue (€)": gross_kwh / 1000 * price}])
        st.dataframe(pd.concat([tbl, tot]).style.format({c: "{:,.0f}" for c in tbl.columns[1:]}),
                     hide_index=True, width="stretch")
        if metered_kwh is not None:
            st.markdown(f"<div class='card'>Expected production <b>{gross_kwh:,.0f} kWh</b><br>"
                        f"− fed in according to grid meter <b>{metered_kwh:,.0f} kWh</b><br>"
                        f"= claimable loss <b>{claim_kwh:,.0f} kWh</b></div>", unsafe_allow_html=True)
        st.caption("Low / High: the estimate can be about ±10% off over a 3-day outage "
                   "(typical error when we tested the method on 28 past periods).")
    with c2:
        cum = go.Figure()
        for i, t in enumerate(targets):
            e = res[t]["power_kw"].fillna(0).cumsum() * M.STEP_HOURS
            cum.add_trace(go.Scatter(x=e.index, y=e, name=t, stackgroup="one", line=dict(color=TURB_COL[i % 5])))
        st.plotly_chart(base_layout(cum, 330, "Lost energy adding up over the outage", "kWh"), width="stretch")
    pw = go.Figure()
    for i, t in enumerate(targets):
        pw.add_trace(go.Scatter(x=res[t]["power_kw"].index, y=res[t]["power_kw"], name=t,
                                line=dict(color=TURB_COL[i % 5], width=1.4)))
    st.plotly_chart(base_layout(pw, 290, "Electricity the turbines would have produced", "kW"), width="stretch")

# ============================================================================ Accuracy
with tab_a:
    st.markdown(f"We test the method on **{v_start:%d.%m.} – {v_end:%d.%m.%Y}**, days where the real values are "
                "known but which the model did **not** use to learn. Then we compare.")
    rows = []
    for t in targets:
        m = res[t]["val_metrics"]
        rows.append({"Turbine": t, "Wind RMSE (m/s)": m["wind_rmse"], "Wind MAE (m/s)": m["wind_mae"],
                     "Power RMSE (kW)": m["power_rmse"], "Power MAE (kW)": m["power_mae"],
                     "Energy measured (kWh)": m["energy_true_kwh"], "Energy estimated (kWh)": m["energy_pred_kwh"],
                     "Energy deviation (%)": m["energy_dev_pct"]})
    st.dataframe(pd.DataFrame(rows).style.format({
        "Wind RMSE (m/s)": "{:.2f}", "Wind MAE (m/s)": "{:.2f}", "Power RMSE (kW)": "{:,.0f}",
        "Power MAE (kW)": "{:,.0f}", "Energy measured (kWh)": "{:,.0f}", "Energy estimated (kWh)": "{:,.0f}",
        "Energy deviation (%)": "{:+.1f}"}), hide_index=True, width="stretch")
    st.markdown("<span class='muted'><b>MAE</b> = average miss per 10 minutes. <b>RMSE</b> = same, but big misses "
                "count extra. <b>Energy deviation</b> = how far the total was off — the number that matters for a "
                "claim, because over many intervals small misses cancel out.</span>", unsafe_allow_html=True)
    t_a = st.radio("Turbine", targets, horizontal=True, key="acc_t")
    vp = res[t_a]["val_pred"]; vt = ws[t_a].reindex(vp.index)
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=vt.index, y=vt, name="measured", line=dict(color=INK, width=1.2)))
    f1.add_trace(go.Scatter(x=vp.index, y=vp, name="estimated", line=dict(color=GREEN, width=1.8)))
    st.plotly_chart(base_layout(f1, 330, f"Estimated vs. measured wind at {t_a}", "m/s"), width="stretch")

# ============================================================================ How it works
with tab_m:
    st.markdown("#### Four steps")
    for i, txt in enumerate([
        "**Find the outage** — the longest period in which all affected turbines are silent at the same time.",
        "**Learn from the neighbours** — wind farms nearby feel the same weather. For each turbine we pick the "
        "neighbouring turbines whose wind follows it most closely, and learn the relationship from the weeks "
        "before and after the outage" + (", including the wind direction" if use_dir else "") + ".",
        "**Fill the gap** — the neighbours kept measuring during the outage, so we use their wind to estimate ours.",
        "**Wind → electricity** — each 10-minute wind value is converted with the manufacturer's power curve "
        "and added up to kWh."], 1):
        html_txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
        st.markdown(f"<div class='card'><b>{i}.</b> {html_txt}</div>", unsafe_allow_html=True)

    st.markdown("#### Which neighbours are used")
    st.caption("How closely each neighbour's wind follows each turbine (1 = perfectly in step). "
               "The best ones per turbine are used.")
    hm = corr_tbl.loc[corr_tbl.mean(axis=1).sort_values(ascending=False).index]
    heat = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, zmin=0.5, zmax=1,
                                colorscale=[[0, "#F2F6F3"], [0.5, LIME], [1, GREEN]],
                                text=[[("✓ " if r_ in refs_by_t[t] else "") + f"{hm.loc[r_, t]:.2f}" for t in hm.columns]
                                      for r_ in hm.index], texttemplate="%{text}", showscale=False))
    heat.update_layout(height=60 + 34 * len(hm), margin=dict(l=10, r=10, t=10, b=10), font=dict(color=INK),
                       yaxis=dict(autorange="reversed"), xaxis=dict(side="top"))
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(heat, width="stretch")
    with c2:
        for t in targets:
            st.markdown(f"<div class='card'><b>{t}</b> uses {', '.join(refs_by_t[t])}</div>", unsafe_allow_html=True)
        st.markdown("<span class='muted'>Why not just the single best neighbour? Every turbine has its own local "
                    "gusts. Combining a few of the best ones cancels part of that noise; using only one made the "
                    "estimate clearly worse in our tests.</span>", unsafe_allow_html=True)

    st.markdown("#### How sure are we?")
    st.markdown(
        "- Single 10-minute values can be off by more than 1 m/s — local gusts no neighbour can see.\n"
        "- Over a whole outage those misses largely cancel out. When we hid 28 different 3-day periods and "
        "estimated them, the energy total was typically within **about ±10%**.\n"
        "- The method tends to be slightly **conservative** (a few % too low), because a smoothed estimate misses "
        "short gusts — useful for a claim that should not be inflated.\n"
        "- The estimate assumes the turbines would have run normally (no planned maintenance or grid curtailment).")

# ============================================================================ Insurance claim
with tab_r:
    claimant = st.text_input("Operator", "BayWa r.e. Wind GmbH")
    policy = st.text_input("Policy / claim reference", "BI-2026-____")
    conf_line = {None: "The outage has not yet been confirmed with grid meter data; this estimate is preliminary.",
                 "grid meter data": f"Grid meter data at the connection point confirms the outage: {status.lower()}.",
                 "operator confirmation": f"The outage was confirmed by the operator ({st.session_state.get('evidence') or 'reference to follow'})."}[confirmation]
    lines = "\n".join(f"  • {t}: {res[t]['energy_kwh']:,.0f} kWh" for t in targets)
    meter_line = (f"Expected production {gross_kwh:,.0f} kWh, of which {metered_kwh:,.0f} kWh were fed in according to "
                  "the grid meter.\n" if metered_kwh is not None else "")
    body = f"""Subject: Business interruption claim – wind park {target_park}, {o_start:%d.%m.%Y}–{o_end:%d.%m.%Y} ({policy})

Dear Sir or Madam,

we hereby notify you of a business interruption at wind park {target_park}, operated by {claimant}.

1. Event
From {o_start:%d.%m.%Y %H:%M} to {o_end:%d.%m.%Y %H:%M} ({hours:.0f} hours) turbines {', '.join(targets)} could not feed electricity into the grid. {conf_line}

2. Estimated energy loss
{meter_line}Claimed energy loss: {claim_kwh:,.0f} kWh ({claim_kwh/1000:,.1f} MWh), estimated range {claim_lo:,.0f}–{claim_hi:,.0f} kWh.
Per turbine (expected production):
{lines}

Estimated revenue loss: € {claim_kwh/1000*price:,.0f} (at {price:,.0f} €/MWh). Deductibles, waiting periods and saved costs according to the policy are to be applied.

3. Method
Wind speeds during the outage were reconstructed from neighbouring turbines whose wind closely follows each affected turbine, calibrated on measurements outside the outage, and converted to power with the manufacturer's power curves.

4. Accuracy
Tested on {v_start:%d.%m.}–{v_end:%d.%m.%Y}, where real values exist, the method reproduced the measured energy within {np.mean([abs(v) for v in val_dev.values()]):.1f}% on average. Across 28 simulated outages the typical deviation was about ±10%, with a slight tendency to underestimate.

Time series and calculations are attached.

Kind regards,
{claimant}
"""
    if s_cls == "bad":
        st.error("The grid meter shows the turbines kept producing — a claim is probably not justified.")
    elif not confirmation:
        st.warning("Tip: confirm the outage first (Outage check tab). Until then the e-mail marks the estimate as preliminary.")
    st.text_area("Draft e-mail", body, height=480)
    export = pd.concat([pd.DataFrame({"timestamp": res[t]["pred"].index, "turbine_name": t,
                                      "wind_speed_pred": res[t]["pred"].round(3).values,
                                      "power_pred_kw": res[t]["power_kw"].round(1).values}) for t in targets])
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ E-mail (.txt)", body.encode(), "insurance_claim_email.txt", width="stretch")
    c2.download_button("⬇️ Supporting time series (.csv)", export.to_csv(index=False).encode(),
                       f"predictions_{o_start:%Y%m%d}.csv", "text/csv", width="stretch")

st.markdown(f"<div class='muted' style='margin-top:24px'>Prototype developed for the BayWa r.e. challenge · "
            f"Munich Climate Week 2026</div>", unsafe_allow_html=True)
