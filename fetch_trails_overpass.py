#!/usr/bin/env python3
"""
Fetch OSM trail data (nationwide via state-by-state Overpass queries),
write per-state GeoJSON + one merged GeoJSON, and optionally load to PostGIS.

Trail definition:
  - ways with highway in {path, footway, track}
  - ways that belong to relation route=hiking (to catch named hiking routes)

Run:
  python3 fetch_trails_overpass.py --states US-SC US-NC  # just SC + NC
  python3 fetch_trails_overpass.py --all                 # all 50 states + DC & territories (config below)
  python3 fetch_trails_overpass.py --all --load         # also loads into PostGIS (requires GDAL/ogr2ogr)
"""

import argparse
import json
import os
import sys
import time
import gzip
from pathlib import Path
from typing import List, Dict, Any
import subprocess

import requests

# ---------- CONFIG ----------
# Overpass API endpoints (tries in order if one is busy)
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# ISO3166-2 codes you want. Use --states to pass a subset or --all for the full list.
# Full US (50 states + DC); you can add territories as needed.
US_STATE_CODES = [
    "US-AL","US-AK","US-AZ","US-AR","US-CA","US-CO","US-CT","US-DE","US-FL","US-GA",
    "US-HI","US-ID","US-IL","US-IN","US-IA","US-KS","US-KY","US-LA","US-ME","US-MD",
    "US-MA","US-MI","US-MN","US-MS","US-MO","US-MT","US-NE","US-NV","US-NH","US-NJ",
    "US-NM","US-NY","US-NC","US-ND","US-OH","US-OK","US-OR","US-PA","US-RI","US-SC",
    "US-SD","US-TN","US-TX","US-UT","US-VT","US-VA","US-WA","US-WV","US-WI","US-WY",
    "US-DC",
]

# Output folders/files
OUT_DIR = Path("out_trails")
PER_STATE_DIR = OUT_DIR / "states"
MERGED_GEOJSON = OUT_DIR / "us_trails.geojson"          # merged nationwide
MERGED_GEOJSON_GZ = OUT_DIR / "us_trails.geojson.gz"    # gzipped (smaller)

# Which tags to carry into properties (add/remove to taste)
KEEP_TAGS = {
    "name","access","foot","bicycle","horse","surface","sac_scale",
    "trail_visibility","operator","informal","incline","highway"
}

# PostGIS load (if --load)
PG_DATABASE = "trailsdb"      # change if needed
PG_SCHEMA   = "public"
PG_TABLE    = "ways_trail_overpass"  # separate from osm2pgsql tables
# ---------------------------


def build_overpass_query(iso_code: str) -> str:
    """Overpass QL: area by ISO3166-2, then trail-like ways + ways in hiking relations."""
    # We fetch both ways with desired highway=* and ways that are members of hiking route relations.
    # 'out geom' returns node coords along the line for easy GeoJSON creation.
    return f"""
[out:json][timeout:900];
area["ISO3166-2"="{iso_code}"]->.a;
(
  way["highway"~"^(path|footway|track)$"](area.a);
  relation["route"="hiking"](area.a);
  way(r);
);
out tags geom;
"""


def call_overpass(query: str, max_retries: int = 4, sleep_s: int = 10) -> Dict[str, Any]:
    """Try multiple Overpass mirrors with retries/backoff."""
    last_exc = None
    for attempt in range(max_retries):
        for url in OVERPASS_URLS:
            try:
                r = requests.post(url, data={'data': query}, timeout=600)
                if r.status_code == 429 or "rate_limited" in r.text.lower():
                    # server busy; sleep and try next
                    time.sleep(sleep_s)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                time.sleep(sleep_s)
        # exponential-ish backoff
        sleep_s = min(sleep_s * 2, 120)
    raise RuntimeError(f"Overpass failed after retries: {last_exc}")


def overpass_to_geojson(overpass_json: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Overpass JSON (with out geom) to a GeoJSON FeatureCollection of LineStrings."""
    elements = overpass_json.get("elements", [])
    features = []
    seen_way_ids = set()

    for el in elements:
        if el.get("type") != "way":
            continue
        wid = el.get("id")
        if wid in seen_way_ids:
            continue
        seen_way_ids.add(wid)

        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue

        coords = [[pt["lon"], pt["lat"]] for pt in geom]
        tags = el.get("tags", {})
        props = {k: v for k, v in tags.items() if k in KEEP_TAGS}
        props["osm_id"] = wid

        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def save_geojson(fc: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)


def merge_geojson_files(paths: List[Path], out_path: Path):
    merged = {"type": "FeatureCollection", "features": []}
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            fc = json.load(f)
            merged["features"].extend(fc.get("features", []))
    save_geojson(merged, out_path)


def gzip_file(path_in: Path, path_out: Path):
    with path_in.open("rb") as fin, gzip.open(path_out, "wb") as fout:
        fout.writelines(fin)


def ogr2ogr_load(geojson_path: Path, table: str):
    """
    Load GeoJSON into PostGIS using ogr2ogr (requires gdal-bin).
    - Creates table if not exists, writes geometry in EPSG:4326.
    - Adds a GIST spatial index.
    """
    # Create or replace table:
    cmd = [
        "ogr2ogr", "-f", "PostgreSQL",
        f"PG:dbname={PG_DATABASE}",
        str(geojson_path),
        "-nln", f"{PG_SCHEMA}.{table}",
        "-nlt", "MULTILINESTRING",
        "-lco", "GEOMETRY_NAME=geom",
        "-lco", "FID=gid",
        "-t_srs", "EPSG:4326",
        "-overwrite"
    ]
    print("-> Loading into PostGIS with ogr2ogr:", " ".join(cmd))
    subprocess.check_call(cmd)

    # Add spatial index for speed
    psql = [
        "psql", PG_DATABASE, "-c",
        f"CREATE INDEX IF NOT EXISTS {table}_gix ON {PG_SCHEMA}.{table} USING GIST(geom);"
    ]
    subprocess.check_call(psql)


def main():
    parser = argparse.ArgumentParser(description="Fetch OSM trails from Overpass (state-by-state).")
    parser.add_argument("--states", nargs="+", help="ISO3166-2 codes, e.g., US-SC US-NC")
    parser.add_argument("--all", action="store_true", help="Fetch all states in US_STATE_CODES")
    parser.add_argument("--load", action="store_true", help="Load merged GeoJSON into PostGIS via ogr2ogr")
    parser.add_argument("--table", default=PG_TABLE, help="PostGIS table name (default ways_trail_overpass)")
    parser.add_argument("--sleep", type=int, default=5, help="Sleep (s) between state requests")
    args = parser.parse_args()

    if not args.states and not args.all:
        print("Provide --states US-SC US-NC ... or --all", file=sys.stderr)
        sys.exit(1)

    states = US_STATE_CODES if args.all else args.states

    PER_STATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_state_files = []

    for iso in states:
        out_path = PER_STATE_DIR / f"{iso}_trails.geojson"
        if out_path.exists():
            print(f"[skip] {iso} (already exists)")
            per_state_files.append(out_path)
            continue

        print(f"[fetch] {iso} ...")
        q = build_overpass_query(iso)
        try:
            data = call_overpass(q)
            fc = overpass_to_geojson(data)
            print(f"  -> {len(fc['features'])} features")
            save_geojson(fc, out_path)
            per_state_files.append(out_path)
        except Exception as e:
            print(f"  !! Failed {iso}: {e}")
        time.sleep(args.sleep)

    if not per_state_files:
        print("No per-state files created; exiting.")
        sys.exit(2)

    print("[merge] Building nationwide GeoJSON …")
    merge_geojson_files(per_state_files, MERGED_GEOJSON)
    print(f"  -> {MERGED_GEOJSON} ({MERGED_GEOJSON.stat().st_size/1_000_000:.1f} MB)")

    print("[gzip] Compressing …")
    gzip_file(MERGED_GEOJSON, MERGED_GEOJSON_GZ)
    print(f"  -> {MERGED_GEOJSON_GZ} ({MERGED_GEOJSON_GZ.stat().st_size/1_000_000:.1f} MB)")

    if args.load:
        try:
            ogr2ogr_load(MERGED_GEOJSON, args.table)
            print("PostGIS load complete ✅")
        except FileNotFoundError:
            print("ogr2ogr not found. Install gdal-bin or remove --load.")
        except subprocess.CalledProcessError as e:
            print(f"ogr2ogr failed: {e}")
            sys.exit(3)


if __name__ == "__main__":
    main()
