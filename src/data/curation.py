"""Build curated frames (hourly DatetimeIndex 'date', float cols, no NaN)
for the real-data grid. Each builder returns a DataFrame with column 'y'
(target) plus exogenous covariate columns valid AT TARGET TIME (lead-matched
where applicable) and caches to curated/<name>.parquet.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
CURATED = os.path.join(ROOT, "curated")
GEF = os.path.join(CURATED, "raw", "gefcom", "GEFCom2014 Data")
KPX = os.path.join(CURATED, "raw", "kpx")
KMA = os.path.join(CURATED, "raw", "kma")


def _cache(name: str, build) -> pd.DataFrame:
    path = os.path.join(CURATED, f"{name}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = build()
    assert df.index.name == "date" and df.index.is_monotonic_increasing
    assert not df.isna().any().any(), f"{name}: NaN in curated frame"
    df.to_parquet(path)
    return df


def _wide_kpx_to_series(csv_path: str) -> pd.Series:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")
    df = df.dropna(how="any")
    date_col = df.columns[0]
    hours = df.columns[1:25]
    long = df.melt(id_vars=[date_col], value_vars=list(hours),
                   var_name="h", value_name="v")
    long["h"] = long["h"].str.replace("시", "").astype(int)
    # '1시'..'24시' are the hour ENDING; hour 24 = next day 00:00
    ts = pd.to_datetime(long[date_col]) + pd.to_timedelta(long["h"], unit="h")
    out = pd.Series(long["v"].astype(float).values, index=ts).sort_index()
    return out[~out.index.duplicated(keep="last")]


# --------------------------------------------------------------- Jeju wind
def build_jeju_wind() -> pd.DataFrame:
    """Target: KPX Jeju hourly wind MWh. Covariates: lead-matched KMA NWP
    (mean over wind-farm grid cells): ws_da (h<=24-valid), ws_d2 (h<=48-valid).
    Restricted to the span where both sources exist."""

    def build():
        csv = glob.glob(os.path.join(KPX, "jeju_gen", "*풍력*.csv"))[0]
        y = _wide_kpx_to_series(csv).rename("y")

        farms = ["hangyeong", "sangmyeong", "eoeum", "hanlim_os", "gimnyeong",
                 "dongbok", "seongsan", "gasiri", "samdal"]
        nwp = pd.concat([pd.read_parquet(p) for p in
                         sorted(glob.glob(os.path.join(KMA, "nwp_*.parquet")))])
        nwp["tmef_dt"] = pd.to_datetime(nwp["tmef"], format="%Y%m%d%H")
        nwp["tmfc_dt"] = pd.to_datetime(nwp["tmfc"], format="%Y%m%d%H")
        nwp["lead"] = (nwp["tmef_dt"] - nwp["tmfc_dt"]).dt.total_seconds() / 3600
        wsd = nwp[nwp["var"] == "WSD"].copy()
        wsd["ws_mean"] = wsd[farms].where(wsd[farms] > -90).mean(axis=1)
        # issue (D-2)23h covers day D at leads 26..49h (valid for any h<=24);
        # issue (D-3)23h covers it at leads 50..73h (valid for any h<=48)
        band1 = (wsd[wsd["lead"] <= 49].drop_duplicates("tmef_dt")
                 .set_index("tmef_dt")["ws_mean"])
        band2 = (wsd[wsd["lead"] > 49].drop_duplicates("tmef_dt")
                 .set_index("tmef_dt")["ws_mean"])
        cov = pd.DataFrame({"ws_da": band1, "ws_d2": band2})

        df = pd.concat([y, cov], axis=1, join="inner").dropna()
        df.index.name = "date"
        return df.asfreq("h").dropna() if df.index.is_unique else df

    return _cache("jeju_wind", build)


# --------------------------------------------------------------- GEFCom2014
def build_gefcom_wind() -> pd.DataFrame:
    """Mean capacity-normalized power over the 10 zones; covariates = mean
    forecast wind speeds at 10 m / 100 m (competition-provided, lead-matched)."""

    def build():
        zones = []
        for f in sorted(glob.glob(os.path.join(
                GEF, "Wind", "Task 15", "Task15_W_Zone1_10", "Task15_W_Zone*.csv"))):
            z = pd.read_csv(f)
            z["date"] = pd.to_datetime(z["TIMESTAMP"], format="%Y%m%d %H:%M")
            z["ws10"] = np.hypot(z["U10"], z["V10"])
            z["ws100"] = np.hypot(z["U100"], z["V100"])
            zones.append(z.set_index("date")[["TARGETVAR", "ws10", "ws100"]])
        m = sum(zones) / len(zones)
        m.columns = ["y", "ws10", "ws100"]
        return m.dropna()

    return _cache("gefcom_wind", build)


def _parse_gef_load_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    wcols = [c for c in df.columns if c.startswith("w")]
    df["temp"] = df[wcols].mean(axis=1)
    return df[["TIMESTAMP", "LOAD", "temp"]]


def build_gefcom_load() -> pd.DataFrame:
    """Aggregate load (single zone) + mean of the 25 temperature stations.
    The mangled M/D/YYYY timestamps are ambiguous, so the hourly index is
    reconstructed from the known start (2001-01-01 01:00) and validated
    against the last stamp of the last increment (12/1/2011 00:00)."""

    def build():
        parts = [_parse_gef_load_file(os.path.join(GEF, "Load", "Task 1",
                                                   "L1-train.csv"))]
        for k in range(2, 16):
            parts.append(_parse_gef_load_file(
                os.path.join(GEF, "Load", f"Task {k}", f"L{k}-train.csv")))
        df = pd.concat(parts, ignore_index=True)
        idx = pd.date_range("2001-01-01 01:00", periods=len(df), freq="h")
        assert idx[-1] == pd.Timestamp("2011-12-01 00:00"), idx[-1]
        out = pd.DataFrame({"y": df["LOAD"].values, "temp": df["temp"].values},
                           index=idx)
        out.index.name = "date"
        return out.dropna()  # 2001-2004 has temperature only (no load)

    return _cache("gefcom_load", build)


def build_gefcom_solar() -> pd.DataFrame:
    """Mean normalized power over the 3 plants + competition NWP predictors
    (mean over plants). Solar task period: 2012-04-01 .. 2014-07-01."""

    def build():
        tr = pd.read_csv(os.path.join(GEF, "Solar", "Task 15", "train15.csv"))
        pv = pd.read_csv(os.path.join(GEF, "Solar", "Task 15", "predictors15.csv"))
        tr["date"] = pd.to_datetime(tr["TIMESTAMP"], format="%Y%m%d %H:%M")
        pv["date"] = pd.to_datetime(pv["TIMESTAMP"], format="%Y%m%d %H:%M")
        y = tr.groupby("date")["POWER"].mean().rename("y")
        var_cols = [c for c in pv.columns if c.startswith("VAR")]
        cov = pv.groupby("date")[var_cols].mean()
        df = pd.concat([y, cov], axis=1).dropna()
        df.index.name = "date"
        return df

    return _cache("gefcom_solar", build)


# ------------------------------------------------------- KPX demand (short)
def build_kpx_demand(which: str = "national") -> pd.DataFrame:
    def build():
        tag = "national_demand" if which == "national" else "jeju_demand"
        csv = glob.glob(os.path.join(KPX, tag, "*.csv"))[0]
        df = _wide_kpx_to_series(csv).rename("y").to_frame()
        df.index.name = "date"
        return df.dropna()

    return _cache(f"kpx_demand_{which}", build)


BUILDERS = {
    "jeju_wind": build_jeju_wind,
    "gefcom_wind": build_gefcom_wind,
    "gefcom_load": build_gefcom_load,
    "gefcom_solar": build_gefcom_solar,
    "kpx_demand_national": lambda: build_kpx_demand("national"),
    "kpx_demand_jeju": lambda: build_kpx_demand("jeju"),
}
