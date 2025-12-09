#!/usr/bin/env python3
"""
Fetch hiking-trail-like features from OpenStreetMap via Overpass API,
optionally compute elevation gain via Open-Elevation, and export JSON
in Swift Trail struct format.

Highlights
- Mirror rotation, retries, backoff, and timeouts for Overpass.
- Broad trail filters (paths/footways/tracks/bridleways + hiking relations).
- Requests tags+geometry directly (no node join).
- Optional tiling for large regions.
- Distance + difficulty score (Easy/Moderate/Difficult).
- Optional elevation gain using Open-Elevation (sampled along line).
- Merges trails with the same name into single features.
- Outputs in Swift-compatible format with integer IDs

Usage examples
--------------
# Fetch trails for specific states by ISO3166-2 code
python fetch_trails_overpass.py --states US-CA -o california_trails.json
python fetch_trails_overpass.py --states US-CO US-UT -o colorado_utah.json

# Small bbox (Greenville, SC test)
python fetch_trails_overpass.py --bbox 34.75 -82.5 35.0 -82.2 -o greenville.json

# Center+radius (km) + elevation sampling every 50 m
python fetch_trails_overpass.py --center 35.06 -82.73 --radius-km 12 --elevation --elev-sample-m 50 -o table_rock.json

# Large bbox, auto-tiling (4x3 grid) with elevation
python fetch_trails_overpass.py --bbox 34.0 -84.5 36.5 -80.0 --tiles 4 3 --elevation -o region.json

# Disable trail merging
python fetch_trails_overpass.py --states US-SC --no-merge -o sc_trails.jso
"""

import argparse
import json
import math
import time
from elevation_optimizer import get_elevations_for_trail, calculate_elevation_gain

from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import requests
from geopy.distance import geodesic

# ------------------ CONFIG ------------------

# Multiple Overpass endpoints: rotate through on failure/429/busy responses
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

# US State ISO3166-2 codes mapping
US_STATE_CODES = {
    "Alabama": "US-AL", "Alaska": "US-AK", "Arizona": "US-AZ", "Arkansas": "US-AR",
    "California": "US-CA", "Colorado": "US-CO", "Connecticut": "US-CT", "Delaware": "US-DE",
    "Florida": "US-FL", "Georgia": "US-GA", "Hawaii": "US-HI", "Idaho": "US-ID",
    "Illinois": "US-IL", "Indiana": "US-IN", "Iowa": "US-IA", "Kansas": "US-KS",
    "Kentucky": "US-KY", "Louisiana": "US-LA", "Maine": "US-ME", "Maryland": "US-MD",
    "Massachusetts": "US-MA", "Michigan": "US-MI", "Minnesota": "US-MN", "Mississippi": "US-MS",
    "Missouri": "US-MO", "Montana": "US-MT", "Nebraska": "US-NE", "Nevada": "US-NV",
    "New Hampshire": "US-NH", "New Jersey": "US-NJ", "New Mexico": "US-NM", "New York": "US-NY",
    "North Carolina": "US-NC", "North Dakota": "US-ND", "Ohio": "US-OH", "Oklahoma": "US-OK",
    "Oregon": "US-OR", "Pennsylvania": "US-PA", "Rhode Island": "US-RI", "South Carolina": "US-SC",
    "South Dakota": "US-SD", "Tennessee": "US-TN", "Texas": "US-TX", "Utah": "US-UT",
    "Vermont": "US-VT", "Virginia": "US-VA", "Washington": "US-WA", "West Virginia": "US-WV",
    "Wisconsin": "US-WI", "Wyoming": "US-WY",
}

INITIAL_SLEEP_S = 6
MAX_SLEEP_S = 120
REQUEST_TIMEOUT_S = 300   # requests-level timeout
OVERPASS_TIMEOUT_S = 240  # Overpass [timeout:...]

# Distance threshold (miles) to treat start≈end as a loop
LOOP_CLOSURE_MI = 0.05

# Global counter for trail IDs
TRAIL_ID_COUNTER = 0

# ------------------ Difficulty model ------------------

def compute_difficulty(distance_miles: float, elevation_gain_ft: float, surface: str = "unknown") -> str:
    """
    Difficulty classification matching Swift app expectations.
    Returns: "Easy", "Moderate", "Difficult", or "Expert"
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
        return "Easy"
    elif score < 4:
        return "Moderate"
    elif score < 6:
        return "Difficult"
    else:
        return "Expert"

def compute_distance(coords_latlon: List[Tuple[float, float]]) -> float:
    """Geodesic polyline length in miles."""
    if len(coords_latlon) < 2:
        return 0.0
    miles = 0.0
    for i in range(1, len(coords_latlon)):
        miles += geodesic(coords_latlon[i-1], coords_latlon[i]).miles
    return miles

def get_center_point(coords_latlon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate the center point (average) of coordinates."""
    if not coords_latlon:
        return (0.0, 0.0)
    avg_lat = sum(c[0] for c in coords_latlon) / len(coords_latlon)
    avg_lon = sum(c[1] for c in coords_latlon) / len(coords_latlon)
    return (avg_lat, avg_lon)

def map_surface_to_terrain_types(surface: str, tags: Dict[str, str]) -> List[str]:
    """Map OSM surface and tags to terrain types array."""
    terrain_types = []
    
    surface_lower = (surface or "unknown").lower()
    
    # Map surface to terrain
    if surface_lower in ["paved", "asphalt", "concrete"]:
        terrain_types.append("Paved")
    elif surface_lower in ["gravel", "fine_gravel", "pebblestone"]:
        terrain_types.append("Gravel")
    elif surface_lower in ["dirt", "ground", "earth", "soil"]:
        terrain_types.append("Dirt")
    elif surface_lower in ["rocky", "rock", "stone"]:
        terrain_types.append("Rocky")
    elif surface_lower in ["grass", "meadow"]:
        terrain_types.append("Grass")
    elif surface_lower in ["sand"]:
        terrain_types.append("Sand")
    
    # Check for additional terrain indicators from tags
    if tags.get("natural") in ["wood", "forest"]:
        terrain_types.append("Forest")
    
    # Check for mountain/alpine terrain
    if tags.get("sac_scale") or tags.get("mountain_hiking") == "yes":
        terrain_types.append("Mountain")
    
    # Default if nothing found
    if not terrain_types:
        terrain_types = ["Mixed"]
    
    return terrain_types

def generate_user_rating(name: str) -> float:
    """Generate a simulated user rating between 3.0 and 5.0 based on trail name hash."""
    # Use hash of name to generate consistent but varied ratings
    hash_val = hash(name)
    rating = 3.0 + (abs(hash_val) % 21) / 10.0  # 3.0 to 5.0 in 0.1 increments
    return round(rating, 1)

# ------------------ Trail merging ------------------

def merge_trails_by_name(trails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge trails that share the same name by connecting their coordinates.
    Attempts to order segments intelligently by connecting nearby endpoints.
    """
    # Group trails by name
    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trail in trails:
        by_name[trail["trailName"]].append(trail)
    
    merged_trails = []
    
    for name, group in by_name.items():
        if len(group) == 1:
            # No merging needed
            merged_trails.append(group[0])
            continue
        
        # Multiple trails with same name - merge them
        # Convert coordinates to list of segments
        segments = []
        for trail in group:
            coords = trail.get("_coordinates", [])
            if len(coords) >= 2:
                segments.append([tuple(c) for c in coords])
        
        if not segments:
            continue
        
        # Order segments by connecting nearest endpoints
        ordered_coords = order_and_connect_segments(segments)
        
        # Recalculate center point and distance
        dist_mi = compute_distance(ordered_coords)
        center_lat, center_lon = get_center_point(ordered_coords)
        
        # Sum elevation gains from all segments
        total_elev_ft = sum(t["elevationGainFeet"] for t in group)
        
        # Get most common surface/terrain
        all_terrains = []
        for t in group:
            all_terrains.extend(t.get("terrainTypes", []))
        # Get unique terrains while preserving order
        terrain_types = list(dict.fromkeys(all_terrains))
        
        # Combine descriptions
        descriptions = [t["description"] for t in group if t["description"]]
        description = "; ".join(descriptions) if descriptions else ""
        
        # Use difficulty from first segment (they should be similar)
        difficulty = group[0]["difficultyLevel"]
        
        # Use state from first segment
        state = group[0]["state"]
        
        # Use first ID
        trail_id = group[0]["id"]
        
        # Average ratings
        avg_rating = sum(t["userRating"] for t in group) / len(group)
        
        merged_trail = {
            "id": trail_id,
            "trailName": name,
            "state": state,
            "latitude": round(center_lat, 6),
            "longitude": round(center_lon, 6),
            "distanceMiles": round(dist_mi, 2),
            "elevationGainFeet": round(total_elev_ft, 1),
            "difficultyLevel": difficulty,
            "terrainTypes": terrain_types,
            "description": f"{description} (Merged from {len(group)} segments)" if description else f"Trail merged from {len(group)} segments",
            "userRating": round(avg_rating, 1),
            "completed": False,
        }
        
        merged_trails.append(merged_trail)
    
    return merged_trails

def order_and_connect_segments(segments: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """
    Order segments to minimize gaps between endpoints and connect them.
    Uses a greedy nearest-neighbor approach.
    """
    if not segments:
        return []
    if len(segments) == 1:
        return list(segments[0])
    
    # Start with the first segment
    remaining = [list(seg) for seg in segments[1:]]
    result = list(segments[0])
    
    while remaining:
        # Find the segment whose endpoint is closest to either end of result
        best_idx = 0
        best_dist = float('inf')
        best_reverse = False
        best_prepend = False
        
        result_start = result[0]
        result_end = result[-1]
        
        for i, seg in enumerate(remaining):
            seg_start = seg[0]
            seg_end = seg[-1]
            
            # Check all four combinations
            # Append segment as-is (result_end -> seg_start)
            d = geodesic(result_end, seg_start).miles
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_reverse = False
                best_prepend = False
            
            # Append segment reversed (result_end -> seg_end)
            d = geodesic(result_end, seg_end).miles
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_reverse = True
                best_prepend = False
            
            # Prepend segment as-is (seg_end -> result_start)
            d = geodesic(seg_end, result_start).miles
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_reverse = False
                best_prepend = True
            
            # Prepend segment reversed (seg_start -> result_start)
            d = geodesic(seg_start, result_start).miles
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_reverse = True
                best_prepend = True
        
        # Add the best segment
        seg = remaining.pop(best_idx)
        if best_reverse:
            seg = list(reversed(seg))
        
        if best_prepend:
            result = seg + result
        else:
            result = result + seg
    
    return result

# ------------------ Overpass query ------------------

def build_overpass_query_bbox(bbox: Tuple[float, float, float, float]) -> str:
    """Return a query that fetches hiking relations + standalone trail-like ways with tags+geometry, limited to a bbox."""
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

def build_overpass_query_state(state_code: str) -> str:
    """
    Return a query that fetches hiking relations + standalone trail-like ways
    within an ISO3166-2 state area, e.g. 'US-SC'.
    """
    return f"""
[out:json][timeout:{OVERPASS_TIMEOUT_S}];

// Resolve state area by ISO3166-2 code
area["ISO3166-2"="{state_code}"]->.searchArea;

// Hiking route relations within state
rel["route"~"^(hiking|foot)$"](area.searchArea);
way(r);
out tags geom;

// Standalone trail-like ways (accept missing 'foot'; exclude explicit no/private)
(
  way["highway"~"^(path|footway|track|bridleway)$"]["foot"!~"^(no|private)$"](area.searchArea);
  way["highway"~"^(path|footway|track|bridleway)$"][!"foot"](area.searchArea);
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
            pass
        idx += batch_size

    return elevations

def calc_total_elevation_gain(elev_m: List[Optional[float]]) -> float:
    """
    Sum positive deltas. Returns elevation gain in feet.
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
    return gain_ft

# ------------------ Transformation ------------------

def elements_to_trails(
    elements: List[Dict[str, Any]],
    use_elevation: bool = False,
    elev_sample_m: float = 50.0,
    elev_batch: int = 100,
    elev_url: str = "https://api.open-elevation.com/api/v1/lookup",
    state_name: str = "Unknown",
) -> List[Dict[str, Any]]:
    """Convert OSM elements to Swift Trail struct format."""
    global TRAIL_ID_COUNTER
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

        # Add surface info to description if not already there
        if not description and surface != "unknown":
            description = f"Surface: {surface}"

        dist_mi = compute_distance(coords_latlon)
        center_lat, center_lon = get_center_point(coords_latlon)

        # Elevation gain (optional)
        elev_ft = 0.0
        if use_elevation and len(coords_latlon) >= 2:
            elevations = get_elevations_for_trail(coords_latlon, verbose=False)
            elev_ft = calculate_elevation_gain(elevations)

        terrain_types = map_surface_to_terrain_types(surface, tags)
        difficulty = compute_difficulty(dist_mi, elev_ft, surface)
        user_rating = generate_user_rating(name)

        trail_obj = {
            "id": TRAIL_ID_COUNTER,
            "trailName": name,
            "state": state_name,
            "latitude": round(center_lat, 6),
            "longitude": round(center_lon, 6),
            "distanceMiles": round(dist_mi, 2),
            "elevationGainFeet": round(elev_ft, 1),
            "difficultyLevel": difficulty,
            "terrainTypes": terrain_types,
            "description": description,
            "userRating": user_rating,
            "completed": False,
            # Keep coordinates for merging, but don't include in final output
            "_coordinates": coords_latlon,
        }

        trails.append(trail_obj)
        TRAIL_ID_COUNTER += 1
        
    return trails

def clean_trail_for_output(trail: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal fields before JSON output."""
    cleaned = {k: v for k, v in trail.items() if not k.startswith("_")}
    return cleaned

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser(
        description="Fetch OSM hiking trails with Overpass; optional elevation via Open-Elevation. Outputs in Swift Trail struct format."
    )

    # Primary mode: single state
    ap.add_argument(
        "-s", "--state",
        type=str,
        help="Fetch trails for a single US state (e.g., 'California', 'Colorado')"
    )

    # Alternative modes
    ap.add_argument(
        "--states", nargs="+", metavar="ISO3166_2",
        help="One or more ISO3166-2 state codes, e.g. US-SC US-NC"
    )
    ap.add_argument(
        "--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
        help="Bounding box: south west north east"
    )
    ap.add_argument(
        "--center", nargs=2, type=float, metavar=("LAT", "LON"),
        help="Center point for a radius search (paired with --radius-km)"
    )
    ap.add_argument(
        "--radius-km", type=float, default=None,
        help="Radius in kilometers (requires --center)"
    )
    ap.add_argument(
        "--tiles", nargs=2, type=int, metavar=("NX", "NY"),
        help="Split bbox into NX x NY tiles to avoid timeouts (bbox/center modes only)"
    )
    
    ap.add_argument(
        "-o", "--output", default="trails.json", help="Output JSON file"
    )
    ap.add_argument(
        "-l", "--list-states", action="store_true",
        help="List all available US states and exit"
    )

    # Elevation flags
    ap.add_argument("--elevation", action="store_true", help="Enable elevation gain via Open-Elevation")
    ap.add_argument("--elev-sample-m", type=float, default=50.0, help="Sampling interval along trail polyline (meters)")
    ap.add_argument("--elev-batch", type=int, default=100, help="Points per Open-Elevation request (<=100 recommended)")
    ap.add_argument("--elev-url", type=str, default="https://api.open-elevation.com/api/v1/lookup", help="Open-Elevation endpoint")

    # Merge flag
    ap.add_argument("--no-merge", action="store_true", help="Disable merging trails with the same name")

    args = ap.parse_args()

    # List states mode
    if args.list_states:
        print("Available US states:")
        for state_name in sorted(US_STATE_CODES.keys()):
            print(f"  - {state_name}")
        return

    # SINGLE STATE MODE (primary interface)
    if args.state:
        if args.state not in US_STATE_CODES:
            print(f"Error: State '{args.state}' not found.")
            print("\nAvailable states:")
            for state_name in sorted(US_STATE_CODES.keys()):
                print(f"  - {state_name}")
            return
        
        state_code = US_STATE_CODES[args.state]
        print(f"Fetching trails for {args.state} ({state_code})...")
        
        q = build_overpass_query_state(state_code)
        data = call_overpass(q)
        ways_count = sum(1 for e in data.get("elements", []) if e.get("type") == "way")
        print(f"  -> got {ways_count} ways")
        
        trails = elements_to_trails(
            data.get("elements", []),
            use_elevation=args.elevation,
            elev_sample_m=args.elev_sample_m,
            elev_batch=args.elev_batch,
            elev_url=args.elev_url,
            state_name=args.state,
        )
        print(f"  -> parsed {len(trails)} trail features")
        
        # Merge trails with the same name (unless disabled)
        if not args.no_merge:
            before_merge = len(trails)
            trails = merge_trails_by_name(trails)
            print(f"Merged {before_merge} trails into {len(trails)} (by name)")
        
        # Clean trails for output
        output_trails = [clean_trail_for_output(t) for t in trails]
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_trails, f, indent=2)
        print(f"✅ Saved {len(output_trails)} trails to {args.output}")
        return

    # STATES MODE (ISO codes)
    if args.states:
        all_trails: List[Dict[str, Any]] = []
        for idx, state_code in enumerate(args.states, 1):
            # Try to get state name from code
            state_name = state_code
            for name, code in US_STATE_CODES.items():
                if code == state_code:
                    state_name = name
                    break
            
            print(f"[{idx}/{len(args.states)}] Fetching state {state_code} ({state_name})...")
            q = build_overpass_query_state(state_code)
            data = call_overpass(q)
            ways_count = sum(1 for e in data.get("elements", []) if e.get("type") == "way")
            print(f"  -> got {ways_count} ways")
            
            tile_trails = elements_to_trails(
                data.get("elements", []),
                use_elevation=args.elevation,
                elev_sample_m=args.elev_sample_m,
                elev_batch=args.elev_batch,
                elev_url=args.elev_url,
                state_name=state_name,
            )
            print(f"  -> parsed {len(tile_trails)} trail features")
            all_trails.extend(tile_trails)

        # De-duplicate by OSM way id (if same trail appears multiple times)
        # Note: trails now have integer IDs, so we skip dedup for now
        trails = all_trails

        # Merge trails with the same name (unless disabled)
        if not args.no_merge:
            before_merge = len(trails)
            trails = merge_trails_by_name(trails)
            print(f"Merged {before_merge} trails into {len(trails)} (by name)")

        # Clean trails for output
        output_trails = [clean_trail_for_output(t) for t in trails]
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_trails, f, indent=2)
        print(f"✅ Saved {len(output_trails)} trails to {args.output}")
        return

    # BBOX / CENTER MODES (original behavior)
    if args.center and args.radius_km is None:
        ap.error("--center requires --radius-km")

    if not args.bbox and not args.center:
        ap.error("Must specify --state, --states, --bbox, or --center")

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
            elev_url=args.elev_url,
            state_name="Unknown",
        )
        print(f"  -> parsed {len(tile_trails)} trail features")
        all_trails.extend(tile_trails)

    trails = all_trails

    # Merge trails with the same name (unless disabled)
    if not args.no_merge:
        before_merge = len(trails)
        trails = merge_trails_by_name(trails)
        print(f"Merged {before_merge} trails into {len(trails)} (by name)")

    # Clean trails for output
    output_trails = [clean_trail_for_output(t) for t in trails]
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_trails, f, indent=2)
    print(f"✅ Saved {len(output_trails)} trails to {args.output}")

if __name__ == "__main__":
    main()