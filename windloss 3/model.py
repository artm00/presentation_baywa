"""
Core logic for the wind-farm outage energy-loss tool.

Pipeline (kept deliberately simple and explainable):
  1. load      -> read turbine time series (.xlsx or .csv) and power curves
  2. detect    -> find the outage = longest block of missing wind speed at the target turbines
  3. fit       -> linear regression: target wind speed ~ reference wind speeds
                  (+ optional wind-direction effect and short time shifts)
  4. predict   -> apply the regression to reference data during the outage
  5. energy    -> predicted wind -> power curve -> kWh lost
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

STEP_HOURS = 10 / 60  # 10-minute values -> 1/6 hour per row
MIN_ENERGY_UNC = 0.10  # minimum +/- range on lost energy (typical error in 28 simulated 3-day outages)


# ----------------------------------------------------------------------------- loading
def _read_any(src, name: str) -> pd.DataFrame:
    """Read an .xlsx or .csv from a path or an uploaded file object."""
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(src)
    raw = src.read() if hasattr(src, "read") else open(src, "rb").read()
    text = raw.decode("utf-8-sig", errors="replace")
    # Decide the separator from the header line: German files use ";" (because "," is the decimal mark)
    header = text.splitlines()[0] if text else ""
    sep = ";" if header.count(";") >= max(1, header.count(",")) else ("\t" if "\t" in header else ",")
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str)


def _to_number(s: pd.Series) -> pd.Series:
    """Numbers like '4,87' (German) or '4.87' -> 4.87; empty / 'NaN' -> missing (never 0)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s = s.astype(str).str.strip().str.replace(" ", "", regex=False)
    german = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
    s = s.where(~german, s.str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def _to_time(s: pd.Series) -> pd.Series:
    """Timestamps like '01.07.2026 00:00' (day first), '2026-07-01 00:00' or with a UTC offset -> Berlin local time."""
    if pd.api.types.is_datetime64_any_dtype(s):
        t = s
    else:
        s = s.astype(str).str.strip()
        day_first = s.str.match(r"^\d{1,2}[./]\d{1,2}[./]\d{4}").mean() > 0.5
        t = pd.to_datetime(s, dayfirst=day_first, errors="coerce", utc=s.str.contains(r"[+-]\d\d:?\d\d$|Z$").any())
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    return t


def _turbine_from_name(name: str) -> str:
    stem = name.split("/")[-1].rsplit(".", 1)[0]
    m = re.search(r"([A-Z]\d+)", stem)
    return m.group(1) if m else stem


def load_timeseries(src, name: str) -> tuple[str, pd.DataFrame]:
    df = _read_any(src, name)
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Minimal format support: "timestamp | wind speed" (challenge spec) or the full README format
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    if "wind_speed" not in df.columns:
        ws_col = [c for c in df.columns if ("wind" in c and "speed" in c and "std" not in c) or "windgeschw" in c]
        df = df.rename(columns={(ws_col[0] if ws_col else df.columns[1]): "wind_speed"})
    turbine = str(df["turbine_name"].dropna().iloc[0]) if "turbine_name" in df else _turbine_from_name(name)
    df["timestamp"] = _to_time(df["timestamp"])
    df = df[df["timestamp"].notna()]
    for c in df.columns:
        if c not in ("timestamp", "turbine_name", "test_set"):
            df[c] = _to_number(df[c])
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return turbine, df


def load_power_curve(src, name: str) -> tuple[str, pd.DataFrame]:
    df = _read_any(src, name)
    df.columns = [str(c).strip().lower() for c in df.columns]
    turbine = str(df["turbine_name"].iloc[0]) if "turbine_name" in df else _turbine_from_name(name)
    rest = [c for c in df.columns if c != "turbine_name"]
    w = next((c for c in rest if "wind" in c), rest[0])
    p = next((c for c in rest if "power" in c or "leistung" in c), [c for c in rest if c != w][0])
    df = df.rename(columns={w: "wind_speed", p: "power"})
    df["wind_speed"], df["power"] = _to_number(df["wind_speed"]), _to_number(df["power"])
    return turbine, df[["wind_speed", "power"]].dropna().sort_values("wind_speed").reset_index(drop=True)


def power_from_wind(wind, curve: pd.DataFrame) -> np.ndarray:
    """Linear interpolation on the power curve. Below the first point and above cut-out -> 0 kW."""
    w = np.asarray(wind, dtype=float)
    p = np.interp(w, curve["wind_speed"], curve["power"], left=0.0, right=0.0)
    return np.where(np.isnan(w), np.nan, p)


# ----------------------------------------------------------------------------- outage detection
def detect_outage(ws: pd.DataFrame, targets: list[str], min_steps: int = 6):
    """Longest run of consecutive NaN where ALL target turbines are missing."""
    missing = ws[targets].isna().all(axis=1)
    best, start, run_start = (None, None, 0), None, None
    for i, (ts, m) in enumerate(missing.items()):
        if m and start is None:
            start, run_start = ts, i
        if (not m or i == len(missing) - 1) and start is not None:
            end_i = i if m else i - 1
            length = end_i - run_start + 1
            if best[0] is None or length > best[2]:
                best = (start, missing.index[end_i], length)
            start = None
    if best[0] is None or best[2] < min_steps:
        return None, None
    return best[0], best[1]


# ----------------------------------------------------------------------------- features + model
@dataclass
class ModelConfig:
    references: list[str]
    use_direction: bool = True
    use_time_shift: bool = True
    direction_source: str | None = None  # reference turbine whose direction is used


def build_features(ws: pd.DataFrame, wd: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    # Fill short gaps in references (max 30 min), then any remaining gaps with the other references' mean
    X = ws[cfg.references].interpolate(limit=3, limit_area="inside")
    ref_mean = X.mean(axis=1)
    X = X.apply(lambda c: c.fillna(ref_mean))
    feats = X.copy()
    if cfg.use_direction:
        src = cfg.direction_source or cfg.references[0]
        d = np.deg2rad(wd[src].interpolate(limit=6, limit_area="inside").ffill().bfill())
        feats["dir_sin"], feats["dir_cos"] = np.sin(d), np.cos(d)
        feats["mean_x_sin"], feats["mean_x_cos"] = ref_mean * np.sin(d), ref_mean * np.cos(d)
    if cfg.use_time_shift:
        for lag in (-3, -1, 1, 3):  # wind front reaches the parks up to +/- 30 min apart
            feats[f"mean_shift_{lag*10:+d}min"] = ref_mean.shift(lag)
    return feats


@dataclass
class FitResult:
    model: LinearRegression
    features: list[str]
    resid_std: float
    n_train: int


def fit(feats: pd.DataFrame, y: pd.Series, mask: pd.Series) -> FitResult:
    m = mask & feats.notna().all(axis=1) & y.notna()
    lr = LinearRegression().fit(feats[m], y[m])
    resid = y[m] - lr.predict(feats[m])
    return FitResult(lr, list(feats.columns), float(resid.std()), int(m.sum()))


def predict(fr: FitResult, feats: pd.DataFrame) -> pd.Series:
    f = feats[fr.features].ffill().bfill()
    return pd.Series(np.clip(fr.model.predict(f), 0, None), index=feats.index)


# ----------------------------------------------------------------------------- metrics
def metrics(y_true: pd.Series, y_pred: pd.Series, curve: pd.DataFrame) -> dict:
    m = y_true.notna() & y_pred.notna()
    yt, yp = y_true[m], y_pred[m]
    e = yp - yt
    pt, pp = power_from_wind(yt, curve), power_from_wind(yp, curve)
    pe = pp - pt
    e_true, e_pred = pt.sum() * STEP_HOURS, pp.sum() * STEP_HOURS
    return {
        "n": int(m.sum()),
        "wind_rmse": float(np.sqrt((e ** 2).mean())),
        "wind_mae": float(e.abs().mean()),
        "wind_bias": float(e.mean()),
        "corr": float(np.corrcoef(yt, yp)[0, 1]),
        "power_rmse": float(np.sqrt((pe ** 2).mean())),
        "power_mae": float(np.abs(pe).mean()),
        "energy_true_kwh": float(e_true),
        "energy_pred_kwh": float(e_pred),
        "energy_dev_pct": float(100 * (e_pred - e_true) / e_true) if e_true else float("nan"),
    }


# ----------------------------------------------------------------------------- full run
def rank_references(ws: pd.DataFrame, target: str, pool: list[str], known: np.ndarray) -> pd.Series:
    """Correlation of every candidate reference with one target turbine, best first."""
    return ws.loc[known, pool].corrwith(ws.loc[known, target]).sort_values(ascending=False)


def load_meter(src, name: str) -> pd.Series:
    """Grid feed-in meter at the park's connection point -> power in kW per 10-minute step.
    Accepts a power column (kW) or an energy column (kWh per step, converted to kW)."""
    df = _read_any(src, name)
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in df.columns if "time" in c or "zeit" in c or "date" in c or "datum" in c), df.columns[0])
    rest = [c for c in df.columns if c != tcol]
    ecol = next((c for c in rest if "kwh" in c or "energy" in c or "energie" in c), None)
    vcol = ecol or next((c for c in rest if "kw" in c or "power" in c or "leistung" in c), rest[0])
    t = _to_time(df[tcol])
    v = _to_number(df[vcol])
    s = pd.Series(v.values, index=t.values).dropna().sort_index()
    s = s[~s.index.duplicated()]
    return s / STEP_HOURS if ecol else s


def run(ws, wd, curves, targets, cfgs, outage_start, outage_end, val_start, val_end):
    """Validate on [val_start, val_end], then refit on all known data and predict the outage.
    cfgs: one ModelConfig for all turbines, or a dict {turbine: ModelConfig} (own references per turbine)."""
    idx = ws.index
    in_outage = (idx >= outage_start) & (idx <= outage_end)
    in_val = (idx >= val_start) & (idx <= val_end)
    out = {}
    for t in targets:
        cfg = cfgs[t] if isinstance(cfgs, dict) else cfgs
        feats = build_features(ws, wd, cfg)
        y = ws[t]
        # 1) honest validation: train without outage + validation period
        fr_val = fit(feats, y, pd.Series(~in_outage & ~in_val, index=idx))
        val_pred = predict(fr_val, feats[in_val])
        val_m = metrics(y[in_val], val_pred, curves[t])
        # 2) final model: all known data outside the outage
        fr = fit(feats, y, pd.Series(~in_outage, index=idx))
        pred = predict(fr, feats[in_outage])
        lo = (pred - fr.resid_std).clip(lower=0)
        hi = pred + fr.resid_std
        power = power_from_wind(pred, curves[t])
        energy = float(np.nansum(power) * STEP_HOURS)
        # Energy uncertainty = how far off the energy total was on the validation period (empirical, easy to explain)
        # Floor of 10% = typical energy error per 3-day outage in our backtest of 28 simulated outages,
        # so one lucky validation week cannot make the range look unrealistically narrow.
        unc = max(abs(val_m["energy_dev_pct"]) / 100 if np.isfinite(val_m["energy_dev_pct"]) else 0.0, MIN_ENERGY_UNC)
        out[t] = {
            "fit": fr, "refs": list(cfg.references), "val_metrics": val_m, "val_pred": val_pred,
            "pred": pred, "pred_lo": lo, "pred_hi": hi, "power_kw": pd.Series(power, index=pred.index),
            "energy_kwh": energy, "energy_unc_pct": 100 * unc,
            "energy_lo_kwh": energy * (1 - unc), "energy_hi_kwh": energy * (1 + unc),
        }
    return out
