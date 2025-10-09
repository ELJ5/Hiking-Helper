#!/usr/bin/env python3
"""
Fetch hiking-trail-like features from OpenStreetMap via Overpass API,
optionally compute elevation gain via Open-Elevation, and export JSON.

Highlights
- Mirror rotation, retries, backoff, and timeouts for Overpass.
- Broad trail filters (paths/footways/tracks/bridleways + hiking relations).
- Requests tags+geometry directly (no node join).
- Optional tiling for large regions.
- Distance + difficulty score (Easy/Moderate/Hard).
- Optional elevation gain using Open-Elevation (sampled along line).

Usage examples
--------------
# Small bbox (Greenville, SC test)
python fetch_trails_overpass.py --bbox 34.75 -82.5 35.0 -82.2 -o greenville.json

# Center+radius (km) + elevation sampling every 50 m
python fetch_trails_overpass.py --center 35.06 -82.73 --radius-km 12 --elevation --elev-sample-m 50 -o table_rock.json

# Large bbox, auto-tiling (4x3 grid) with elevation (be patient & polite)
python fetch_trails_overpass.py --bbox 34.0 -84.5 36.5 -80.0 --tiles 4 3 --elevation -o region.json
"""

import argparse
import json
import math
import time
from typing import Dict, Any, List, Tuple, Optional

import requests
from geopy.distance import geodesic

# ------------------ CONFIG ------------------

# Multiple Overpass endpoints: rotate through on failure/429/busy responses
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

INITIAL_SLEEP_S = 6
MAX_SLEEP_S = 120
REQUEST_TIMEOUT_S = 300   # requests-level timeout
OVERPASS_TIMEOUT_S = 240  # Overpass [timeout:...]

# Distance threshold (miles) to treat start≈end as a loop
LOOP_CLOSURE_MI = 0.05

# ------------------ Difficulty model ------------------

def compute_difficulty(distance_miles: float, elevation_gain_ft: float, surface: str = "unknown") -> Tuple[str, float]:
    """
    Difficulty Score (rough heuristic):
        base = 0.3 * miles + 1.2 * (elev_ft / 800)
        score = base * surface_factor
    """
    surface_factor_map = {
        "paved": 0.9, "asphalt": 0.9, "concrete": 0.9,
        "gravel": 1.0, "compacted": 1.0,
        "dirt": 1.1, "ground": 1.1,
        "rocky": 1.3, "stone": 1.2,
        "unknown": 1.0,
    }
    sf = surface_factor_map.get((surface or "unknown").lower(), 1.0)
    base = (0.3 * distance_miles) + 1.2 * (elevation_gain_ft / 800.0)
    score = base * sf

    if score < 2:
        label = "Easy"
    elif score < 4:
        label = "Moderate"
    else:
        label = "Hard"
    return label, round(score, 1)

def compute_distance(coords_latlon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Geodesic polyline length (miles, km)."""
    if len(coords_latlon) < 2:
        return 0.0, 0.0
    miles = 0.0
    for i in range(1, len(coords_latlon)):
        miles += geodesic(coords_latlon[i-1], coords_latlon[i]).miles
    return miles, miles * 1.60934

def identify_trail_type(coords_latlon: List[Tuple[float, float]]) -> str:
    if len(coords_latlon) < 2:
        return "point-to-point"
    start, end = coords_latlon[0], coords_latlon[-1]
    if geodesic(start, end).miles < LOOP_CLOSURE_MI:
        return "loop"
    return "point-to-point"  # out-and-back needs path merging beyond a single way

# ------------------ Overpass query ------------------

def build_overpass_query_bbox(bbox: Tuple[float, float, float, float]) -> str:
    """Return a query that fetches hiking relations + standalone trail-like ways with tags+geometry."""
    s, w, n, e = bbox
    return f"""
[out:json][timeout:{OVERPASS_TIMEOUT_S}];

// Hiking route relations within bbox
rel["route"~"^(hiking|foot)$"]({s},{w},{n},{e});
way(r);
out tags geom;

// Standalone trail-like ways (accept missing 'foot'; exclude explicit no/private)
(
  way["highway"~"^(path|footway|track|bridleway)$"]["foot"!~"^(no|private)$"]({s},{w},{n},{e});
  way["highway"~"^(path|footway|track|bridleway)$"][!"foot"]({s},{w},{n},{e});
);
out tags geom;
"""

def call_overpass(query: str, max_retries: int = 4, initial_sleep: int = INITIAL_SLEEP_S) -> Dict[str, Any]:
    """Try multiple mirrors with retries/backoff. Returns JSON with 'elements' on success."""
    sleep_s = initial_sleep
    last_err = None

    for attempt in range(max_retries):
        for url in OVERPASS_URLS:
            try:
                if attempt > 0:
                    time.sleep(sleep_s)  # pacing between attempts
                r = requests.post(url, data={'data': query}, timeout=REQUEST_TIMEOUT_S)
                if r.status_code == 429 or "rate_limited" in r.text.lower():
                    last_err = RuntimeError(f"rate-limited by {url}")
                    continue
                r.raise_for_status()
                j = r.json()
                if not j.get("elements"):
                    last_err = RuntimeError(f"no elements from {url}")
                    continue
                return j
            except Exception as e:
                last_err = e
        sleep_s = min(sleep_s * 2, MAX_SLEEP_S)

    raise RuntimeError(f"Overpass failed after retries: {last_err}")

# ------------------ Tiling helpers ------------------

def bbox_from_center_radius_km(lat: float, lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    """
    Approximate circle as bbox using ~111.32 km/deg latitude and cos(lat) for longitude.
    """
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(0.01, math.cos(math.radians(lat))))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)

def tile_bbox(bbox: Tuple[float, float, float, float], nx: int, ny: int) -> List[Tuple[float, float, float, float]]:
    s, w, n, e = bbox
    tiles = []
    dlat = (n - s) / ny
    dlon = (e - w) / nx
    for iy in range(ny):
        for ix in range(nx):
            tile_s = s + iy * dlat
            tile_n = s + (iy + 1) * dlat
            tile_w = w + ix * dlon
            tile_e = w + (ix + 1) * dlon
            tiles.append((tile_s, tile_w, tile_n, tile_e))
    return tiles

# ------------------ Elevation helpers ------------------

def interpolate_along_line(coords: List[Tuple[float, float]], sample_meters: float) -> List[Tuple[float, float]]:
    """Resample a polyline at roughly equal intervals in meters (lat, lon)."""
    if len(coords) < 2:
        return coords[:]
    # Positions are (lat, lon) for geopy; compute cumulative distances
    dists = [0.0]
    for i in range(1, len(coords)):
        d = geodesic(coords[i-1], coords[i]).meters
        dists.append(dists[-1] + d)
    total = dists[-1]
    if total == 0:
        return [coords[0], coords[-1]]
    num_samples = max(2, int(total // sample_meters) + 1)
    target = [i * (total / (num_samples - 1)) for i in range(num_samples)]

    # linear interpolation in lat/lon space (okay for short steps)
    out: List[Tuple[float, float]] = []
    j = 0
    for t in target:
        while j < len(dists) - 2 and dists[j+1] < t:
            j += 1
        # interpolate between j and j+1
        seg_len = dists[j+1] - dists[j] if dists[j+1] > dists[j] else 1e-9
        ratio = (t - dists[j]) / seg_len
        lat = coords[j][0] + ratio * (coords[j+1][0] - coords[j][0])
        lon = coords[j][1] + ratio * (coords[j+1][1] - coords[j][1])
        out.append((lat, lon))
    return out

def fetch_elevations_open_elevation(
    points: List[Tuple[float, float]],
    url: str,
    batch_size: int = 100,
    max_retries: int = 4,
    initial_sleep: float = 1.5,
    timeout_s: int = 60
) -> List[Optional[float]]:
    """
    Fetch elevations (meters) for points [(lat, lon), ...] via Open-Elevation.
    Returns list of elevations (meters) or None where unavailable.
    """
    elevations: List[Optional[float]] = [None] * len(points)
    sleep_s = initial_sleep
    idx = 0

    while idx < len(points):
        batch = points[idx: idx + batch_size]
        locations = [{"latitude": lat, "longitude": lon} for (lat, lon) in batch]
        payload = {"locations": locations}

        success = False
        last_err: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                # Be gentle: a little pacing helps avoid rate limits
                if attempt > 0:
                    time.sleep(sleep_s)
                r = requests.post(url, json=payload, timeout=timeout_s)
                if r.status_code == 429:
                    last_err = RuntimeError("Open-Elevation rate limited (429)")
                    time.sleep(sleep_s)
                    sleep_s = min(sleep_s * 2, 30)
                    continue
                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])
                if len(results) != len(batch):
                    last_err = RuntimeError(f"Open-Elevation returned {len(results)} results for {len(batch)} points")
                    time.sleep(sleep_s)
                    sleep_s = min(sleep_s * 2, 30)
                    continue
                for i, res in enumerate(results):
                    elevations[idx + i] = float(res.get("elevation")) if res.get("elevation") is not None else None
                success = True
                break
            except Exception as e:
                last_err = e
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 2, 30)

        if not success:
            # Leave Nones for this batch and proceed, rather than failing the whole trail
            # (You could also choose to raise here.)
            pass
        idx += batch_size

    return elevations

def calc_total_elevation_gain(elev_m: List[Optional[float]]) -> Tuple[int, int]:
    """
    Sum positive deltas. Returns (feet, meters) as integers.
    If any Nones, they are skipped conservatively.
    """
    gain_m = 0.0
    last = None
    for v in elev_m:
        if v is None:
            continue
        if last is not None and v > last:
            gain_m += (v - last)
        last = v
    gain_ft = gain_m * 3.28084
    return int(round(gain_ft)), int(round(gain_m))

# ------------------ Transformation ------------------

def elements_to_trails(
    elements: List[Dict[str, Any]],
    use_elevation: bool = False,
    elev_sample_m: float = 50.0,
    elev_batch: int = 100,
    elev_url: str = "https://api.open-elevation.com/api/v1/lookup"
) -> List[Dict[str, Any]]:
    trails = []
    for el in elements:
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue

        coords_latlon = [(pt["lat"], pt["lon"]) for pt in geom]
        tags = el.get("tags", {}) or {}

        name = tags.get("name", "Unnamed Trail")
        surface = tags.get("surface", "unknown")
        description = tags.get("description", "")

        amenities = [k for k in ["parking", "toilets", "drinking_water"] if tags.get(k)]

        dist_mi, dist_km = compute_distance(coords_latlon)

        # Elevation gain (optional)
        elev_ft, elev_m = 0, 0
        if use_elevation and len(coords_latlon) >= 2:
            sample_pts = interpolate_along_line(coords_latlon, elev_sample_m)
            elevs_m = fetch_elevations_open_elevation(
                sample_pts, url=elev_url, batch_size=elev_batch
            )
            elev_ft, elev_m = calc_total_elevation_gain(elevs_m)

        trail_type = identify_trail_type(coords_latlon)
        difficulty, difficulty_score = compute_difficulty(dist_mi, elev_ft, surface)

        trails.append({
            "id": str(el["id"]),
            "name": name,
            "coordinates": [[lat, lon] for (lat, lon) in coords_latlon],
            "distance_miles": round(dist_mi, 2),
            "distance_km": round(dist_km, 2),
            "elevation_gain_feet": elev_ft,
            "elevation_gain_meters": elev_m,
            "difficulty": difficulty,
            "difficulty_score": difficulty_score,
            "trail_type": trail_type,
            "surface": surface,
            "amenities": amenities,
            "description": description
        })
    return trails

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser(description="Fetch OSM hiking trails with Overpass; optional elevation via Open-Elevation.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
                   help="Bounding box: south west north east")
    g.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"),
                   help="Center point for a radius search (paired with --radius-km)")

    ap.add_argument("--radius-km", type=float, default=None,
                    help="Radius in kilometers (requires --center)")
    ap.add_argument("--tiles", nargs=2, type=int, metavar=("NX", "NY"),
                    help="Split bbox into NX x NY tiles to avoid timeouts")
    ap.add_argument("-o", "--output", default="osm_trails.json", help="Output JSON file")

    # Elevation flags
    ap.add_argument("--elevation", action="store_true", help="Enable elevation gain via Open-Elevation")
    ap.add_argument("--elev-sample-m", type=float, default=50.0, help="Sampling interval along trail polyline (meters)")
    ap.add_argument("--elev-batch", type=int, default=100, help="Points per Open-Elevation request (<=100 recommended)")
    ap.add_argument("--elev-url", type=str, default="https://api.open-elevation.com/api/v1/lookup", help="Open-Elevation endpoint")

    args = ap.parse_args()

    if args.center and args.radius_km is None:
        ap.error("--center requires --radius-km")

    if args.center:
        bbox = bbox_from_center_radius_km(args.center[0], args.center[1], args.radius_km)
    else:
        bbox = tuple(args.bbox)  # type: ignore

    # Tiles to query
    tile_list = [bbox]
    if args.tiles:
        nx, ny = args.tiles
        tile_list = tile_bbox(bbox, nx, ny)

    all_trails: List[Dict[str, Any]] = []

    for i, bb in enumerate(tile_list, 1):
        print(f"[{i}/{len(tile_list)}] Fetching tile bbox={bb} …")
        q = build_overpass_query_bbox(bb)
        data = call_overpass(q)
        ways_count = sum(1 for e in data.get("elements", []) if e.get("type") == "way")
        print(f"  -> got {ways_count} ways")
        tile_trails = elements_to_trails(
            data.get("elements", []),
            use_elevation=args.elevation,
            elev_sample_m=args.elev_sample_m,
            elev_batch=args.elev_batch,
            elev_url=args.elev_url
        )
        print(f"  -> parsed {len(tile_trails)} trail features")
        all_trails.extend(tile_trails)

    # De-duplicate by way id (tiles may overlap slightly)
    uniq: Dict[str, Dict[str, Any]] = {}
    for tr in all_trails:
        uniq[tr["id"]] = tr
    trails = list(uniq.values())

    out = {"trails": trails}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"✅ Saved {len(trails)} trails to {args.output}")

if __name__ == "__main__":
    main()
