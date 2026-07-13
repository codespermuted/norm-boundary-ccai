"""KMA API-hub NWP collector: short-term forecast grid -> Jeju site series.

Lead-matching contract (operationally airtight, day-ahead-market style):
  For every hour of target day D we store forecasts from fixed past issues
    band1: issued (D-2) 23:00 KST, leads +26..+49h  -> valid for any h <= 24
    band2: issued (D-3) 23:00 KST, leads +50..+73h  -> valid for any h <= 48
  A forecast used for target time tau at origin t = tau - h satisfies
  issue <= t by construction. Variables: WSD (band1+2), TMP (band1).

Budget: each grid call ~341 KB; per-key daily limits 20,000 calls / 5 GB.
KeyManager rotates KMA_API_KEY1..12 from .env and persists per-day usage.

Spatial mapping: DFS Lambert-conformal grid (149x253, 5 km). Wind-farm and
ASOS coordinates are converted with the standard KMA dfs conversion; the full
grid is fetched, the ~13 Jeju points are extracted, the grid is discarded.

Usage:
  uv run python -m src.data.collectors.kma_nwp [--start 2021-07-01]
      [--end 2023-12-31] [--workers 6] [--dry-run]
Resumable: done-keys ledger + monthly parquet chunks in curated/raw/kma/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
OUT_DIR = os.path.join(ROOT, "curated", "raw", "kma")
STATE_PATH = os.path.join(OUT_DIR, "key_state.json")
DONE_PATH = os.path.join(OUT_DIR, "done_keys.txt")

URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"
NX, NY = 149, 253
CALL_CAP, BYTE_CAP = 19_500, int(4.8 * 2**30)  # per key per day, with margin

# ---- DFS Lambert conformal conic (standard KMA constants) ------------------
_RE, _GRID = 6371.00877, 5.0
_SLAT1, _SLAT2 = math.radians(30.0), math.radians(60.0)
_OLON, _OLAT = math.radians(126.0), math.radians(38.0)
_XO, _YO = 43, 136


def dfs_xy(lat: float, lon: float) -> tuple[int, int]:
    re = _RE / _GRID
    sn = math.log(math.cos(_SLAT1) / math.cos(_SLAT2)) / math.log(
        math.tan(math.pi / 4 + _SLAT2 / 2) / math.tan(math.pi / 4 + _SLAT1 / 2)
    )
    sf = (math.tan(math.pi / 4 + _SLAT1 / 2) ** sn) * math.cos(_SLAT1) / sn
    ro = re * sf / (math.tan(math.pi / 4 + _OLAT / 2) ** sn)
    ra = re * sf / (math.tan(math.pi / 4 + math.radians(lat) / 2) ** sn)
    theta = math.radians(lon) - _OLON
    if theta > math.pi:
        theta -= 2 * math.pi
    if theta < -math.pi:
        theta += 2 * math.pi
    theta *= sn
    # official KMA JS: floor(value + origin + 0.5), result is the 1-based index
    x = math.floor(ra * math.sin(theta) + _XO + 0.5)
    y = math.floor(ro - ra * math.cos(theta) + _YO + 0.5)
    return int(x), int(y)


# Jeju wind farms (approx. centroids) + ASOS stations (validation/demand)
SITES = {
    "hangyeong":  (33.350, 126.170),   # 한경풍력 (W coast)
    "sangmyeong": (33.363, 126.257),   # 상명풍력
    "eoeum":      (33.400, 126.330),   # 어음풍력
    "hanlim_os":  (33.420, 126.240),   # 한림해상풍력
    "gimnyeong":  (33.552, 126.755),   # 김녕/월정 (NE coast)
    "dongbok":    (33.530, 126.700),   # 동복·북촌풍력
    "seongsan":   (33.440, 126.905),   # 성산풍력 (E)
    "gasiri":     (33.383, 126.760),   # 가시리/표선 (SE)
    "samdal":     (33.370, 126.850),   # 삼달풍력
    "asos_jeju":     (33.514, 126.530),  # ASOS 184
    "asos_gosan":    (33.294, 126.163),  # ASOS 185
    "asos_seongsan": (33.387, 126.880),  # ASOS 188
    "asos_seogwipo": (33.246, 126.565),  # ASOS 189
}


def site_indices() -> dict[str, tuple[int, int]]:
    """site -> (nx, ny), deduplicated grid cells kept per site name."""
    return {name: dfs_xy(lat, lon) for name, (lat, lon) in SITES.items()}


# ---- key rotation with persisted daily budgets ------------------------------
class KeyManager:
    def __init__(self):
        self.keys = []
        for line in open(os.path.join(ROOT, ".env")):
            line = line.strip()
            if line.startswith("KMA_API_KEY") and "=" in line:
                # keys must be whitespace-free; .env lines may wrap/pad them
                self.keys.append("".join(line.split("=", 1)[1].split()))
        if not self.keys:
            raise RuntimeError("no KMA_API_KEY* in .env")
        self.lock = threading.Lock()
        today = date.today().isoformat()
        self.state = {str(i): {"date": today, "calls": 0, "bytes": 0}
                      for i in range(len(self.keys))}
        if os.path.exists(STATE_PATH):
            saved = json.load(open(STATE_PATH))
            for k, v in saved.items():
                if k in self.state and v.get("date") == today:
                    self.state[k] = v
        self.idx = 0

    def _usable(self, i: int) -> bool:
        s = self.state[str(i)]
        return s["calls"] < CALL_CAP and s["bytes"] < BYTE_CAP

    def acquire(self) -> tuple[int, str]:
        with self.lock:
            for off in range(len(self.keys)):
                i = (self.idx + off) % len(self.keys)
                if self._usable(i):
                    self.idx = i
                    return i, self.keys[i]
        raise RuntimeError("all keys exhausted for today")

    def record(self, i: int, nbytes: int, save_every: int = 200):
        with self.lock:
            s = self.state[str(i)]
            s["calls"] += 1
            s["bytes"] += nbytes
            if s["calls"] % save_every == 0:
                self._save()

    def exhaust(self, i: int):
        """Mark key dead for today (server said limit exceeded)."""
        with self.lock:
            self.state[str(i)]["calls"] = CALL_CAP
            self._save()

    def _save(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(self.state, f)

    def totals(self) -> str:
        c = sum(s["calls"] for s in self.state.values())
        g = sum(s["bytes"] for s in self.state.values()) / 2**30
        return f"{c} calls, {g:.2f} GiB today"


# ---- fetching ---------------------------------------------------------------
class LimitExceeded(Exception):
    pass


def fetch_points(km: KeyManager, tmfc: str, tmef: str, var: str,
                 sites: dict[str, tuple[int, int]], retries: int = 4) -> dict:
    last_exc = None
    for attempt in range(retries):
        ki, key = km.acquire()
        try:
            # authKey must go raw: the server matches it before URL-decoding,
            # so requests' percent-encoding of +/= breaks auth (curl-raw works)
            r = requests.get(
                f"{URL}?tmfc={tmfc}&tmef={tmef}&vars={var}&authKey={key}",
                timeout=90)
            km.record(ki, len(r.content))
            txt = r.text
            if r.status_code != 200 or txt.lstrip().startswith("{"):
                if "한도" in txt or "limit" in txt.lower() or r.status_code == 403:
                    km.exhaust(ki)
                    raise LimitExceeded(txt[:80])
                raise RuntimeError(f"http={r.status_code} {txt[:80]}")
            vals = np.fromiter(
                (float(s) for s in txt.replace("\n", ",").split(",")
                 if s.strip()), dtype=float)
            if vals.size != NX * NY:
                raise RuntimeError(f"grid size {vals.size} != {NX * NY}")
            grid = vals.reshape(NY, NX)  # row 0 = ny=1 (south)
            return {name: float(grid[ny - 1, nx - 1])
                    for name, (nx, ny) in sites.items()}
        except LimitExceeded:
            last_exc = None
            continue
        except Exception as exc:  # transient net/parse errors
            last_exc = exc
            time.sleep(1.5 * (attempt + 1) + random.random())
    if last_exc:
        raise last_exc
    raise RuntimeError("retries exhausted (key limits)")


def jobs_for_day(d: date) -> list[tuple[str, str, str]]:
    """(tmfc, tmef, var) triples for target day d per the band schedule."""
    out = []
    for band_offset, vars_ in ((2, ("WSD", "TMP")), (3, ("WSD",))):
        tmfc = (d - timedelta(days=band_offset)).strftime("%Y%m%d") + "23"
        for hour in range(1, 25):
            tmef = (datetime(d.year, d.month, d.day)
                    + timedelta(hours=hour)).strftime("%Y%m%d%H")
            for v in vars_:
                out.append((tmfc, tmef, v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sites = site_indices()
    print("site grid points:", sites, flush=True)

    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    all_jobs = [j for d in days for j in jobs_for_day(d)]

    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(DONE_PATH):
        done = set(open(DONE_PATH).read().split())
    jobs = [j for j in all_jobs if "|".join(j) not in done]
    print(f"jobs: {len(jobs)} todo / {len(all_jobs)} total "
          f"(~{len(jobs) * 0.34 / 1024:.1f} GiB)", flush=True)
    if args.dry_run:
        return

    km = KeyManager()
    done_f = open(DONE_PATH, "a")
    buf, buf_lock = [], threading.Lock()
    fail = ok = 0

    def flush_buf():
        nonlocal buf
        with buf_lock:
            rows, buf = buf, []
        if rows:
            df = pd.DataFrame(rows)
            for ym, g in df.groupby(df["tmef"].str[:6]):
                path = os.path.join(OUT_DIR, f"nwp_{ym}.parquet")
                if os.path.exists(path):
                    g = pd.concat([pd.read_parquet(path), g])
                g.drop_duplicates(["tmfc", "tmef", "var"]).to_parquet(path,
                                                                      index=False)

    def work(job):
        tmfc, tmef, var = job
        pts = fetch_points(km, tmfc, tmef, var, sites)
        return job, pts

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                (tmfc, tmef, var), pts = fut.result()
                with buf_lock:
                    buf.append({"tmfc": tmfc, "tmef": tmef, "var": var, **pts})
                done_f.write("|".join((tmfc, tmef, var)) + "\n")
                ok += 1
            except Exception as exc:
                fail += 1
                if fail <= 20 or fail % 100 == 0:
                    print(f"FAIL {futs[fut]}: {str(exc)[:80]}", flush=True)
                if isinstance(exc, RuntimeError) and "keys exhausted" in str(exc):
                    print("ALL KEYS EXHAUSTED — stopping; rerun tomorrow",
                          flush=True)
                    break
            if n % 500 == 0:
                flush_buf()
                done_f.flush()
                print(f"[{n}/{len(jobs)}] ok={ok} fail={fail} | {km.totals()}",
                      flush=True)
    flush_buf()
    done_f.close()
    print(f"finished: ok={ok} fail={fail} | {km.totals()}", flush=True)


if __name__ == "__main__":
    main()
