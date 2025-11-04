#!/usr/bin/env python3
"""
Trail Data Collector with Segment Connection
==============================================
Fetches trail data from OpenStreetMap and Open-Elevation API,
connects unnamed segments to named trails, and saves per-state JSON.

Usage:
  python collect_trails_async.py --states US-SC
  python collect_trails_async.py --all --compress
"""

import asyncio, aiohttp, json, gzip, time, math, hashlib, logging, argparse
from datetime import datetime
from pathlib import Path
from aiohttp.client_exceptions import ClientError, ServerTimeoutError
from typing import List, Dict, Any, Tuple, Optional

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
CONFIG = {
    "overpass_url": "https://overpass-api.de/api/interpreter",
    "elevation_url": "https://api.open-elevation.com/api/v1/lookup",
    "max_concurrent_requests": 12,
    "elevation_batch_trails": 15,
    "max_points_per_request": 1000,
    "timeout": 45,
    "retry_attempts": 3,
    "connection_threshold_meters": 30,  # Distance threshold for connecting segments
    "user_agent": "TrailDataCollector/2.4 (https://yourapp.com)",
    "version": "2.4",
}

OUTPUT_DIR = Path("output_async")
OUTPUT_DIR.mkdir(exist_ok=True)

ALL_US_STATES = [
    "US-AL","US-AK","US-AZ","US-AR","US-CA","US-CO","US-CT","US-DE","US-FL",
    "US-GA","US-HI","US-ID","US-IL","US-IN","US-IA","US-KS","US-KY","US-LA",
    "US-ME","US-MD","US-MA","US-MI","US-MN","US-MS","US-MO","US-MT","US-NE",
    "US-NV","US-NH","US-NJ","US-NM","US-NY","US-NC","US-ND","US-OH","US-OK",
    "US-OR","US-PA","US-RI","US-SC","US-SD","US-TN","US-TX","US-UT","US-VT",
    "US-VA","US-WA","US-WV","US-WI","US-WY"
]

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

# -------------------------------------------------------
# UTILITIES
# -------------------------------------------------------

def haversine_distance(coords: list) -> float:
    """Compute trail distance in miles."""
    if len(coords) < 2:
        return 0.0
    R = 3958.8
    d = 0
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i-1]
        lat2, lon2 = coords[i]
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        d += 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return round(d, 2)


def point_distance_meters(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate distance between two lat/lon points in meters."""
    R = 6371000  # Earth radius in meters
    lat1, lon1 = p1
    lat2, lon2 = p2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


def estimate_difficulty(distance_miles: float, elevation_gain_feet: int) -> str:
    if not distance_miles or not elevation_gain_feet:
        return "Moderate"
    score = distance_miles * elevation_gain_feet
    if score < 1500: return "Easy"
    elif score < 4500: return "Moderate"
    elif score < 8000: return "Hard"
    else: return "Expert"


def validate_trail(trail: Dict[str, Any]) -> bool:
    try:
        assert isinstance(trail["trailName"], str)
        assert isinstance(trail["distanceMiles"], (int, float))
        assert isinstance(trail["elevationGainFeet"], int)
        assert trail["difficultyLevel"] in {"Easy","Moderate","Hard","Expert"}
        return True
    except Exception:
        return False


def deduplicate(trails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, uniq = set(), []
    for t in trails:
        tid = hashlib.md5((t["trailName"]+str(t["latitude"])+str(t["longitude"])).encode()).hexdigest()
        if tid not in seen:
            seen.add(tid)
            uniq.append(t)
    return uniq


def save_json(data: Dict[str, Any], path: Path, compress=False):
    if compress:
        with gzip.open(path.with_suffix(".json.gz"), "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"💾 Compressed to {path.with_suffix('.json.gz')}")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"💾 Saved {path}")

# -------------------------------------------------------
# SEGMENT CONNECTION LOGIC
# -------------------------------------------------------

def is_unnamed_trail(trail: Dict[str, Any]) -> bool:
    """Check if a trail is unnamed (auto-generated name)."""
    name = trail["trailName"]
    return name.startswith("Unnamed Trail") or not name or name.lower() in ["unnamed", "unknown"]


def get_endpoints(geometry: list) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Get start and end points of a trail geometry."""
    return (geometry[0], geometry[-1])


def find_connection_point(unnamed_trail: Dict[str, Any], 
                          named_trail: Dict[str, Any],
                          threshold_meters: float) -> Optional[Tuple[str, str]]:
    """
    Find if unnamed trail connects to named trail.
    Returns: (unnamed_end, named_end) where end is 'start' or 'end'
    """
    unnamed_start, unnamed_end = get_endpoints(unnamed_trail["geometry"])
    named_start, named_end = get_endpoints(named_trail["geometry"])
    
    connections = []
    
    # Check all 4 possible connection points
    dist_us_ns = point_distance_meters(unnamed_start, named_start)
    dist_us_ne = point_distance_meters(unnamed_start, named_end)
    dist_ue_ns = point_distance_meters(unnamed_end, named_start)
    dist_ue_ne = point_distance_meters(unnamed_end, named_end)
    
    if dist_us_ns < threshold_meters:
        connections.append((dist_us_ns, 'start', 'start'))
    if dist_us_ne < threshold_meters:
        connections.append((dist_us_ne, 'start', 'end'))
    if dist_ue_ns < threshold_meters:
        connections.append((dist_ue_ns, 'end', 'start'))
    if dist_ue_ne < threshold_meters:
        connections.append((dist_ue_ne, 'end', 'end'))
    
    if not connections:
        return None
    
    # Return the closest connection
    connections.sort(key=lambda x: x[0])
    return (connections[0][1], connections[0][2])


def merge_geometries(base_geometry: list, 
                     addition_geometry: list,
                     base_connection_end: str,
                     addition_connection_end: str) -> list:
    """
    Merge two trail geometries based on their connection points.
    
    Args:
        base_geometry: The main trail geometry
        addition_geometry: The geometry to add
        base_connection_end: 'start' or 'end' - where on base trail to connect
        addition_connection_end: 'start' or 'end' - where on addition to connect
    """
    # Make copies to avoid modifying originals
    base = list(base_geometry)
    addition = list(addition_geometry)
    
    # Reverse addition if needed to maintain proper direction
    if addition_connection_end == 'end':
        addition = list(reversed(addition))
    
    # Merge based on connection point
    if base_connection_end == 'end':
        # Add to end of base trail
        # Skip first point of addition to avoid duplication at connection
        return base + addition[1:]
    else:
        # Add to start of base trail
        # Reverse addition if connecting to start
        if addition_connection_end == 'start':
            addition = list(reversed(addition))
        return addition[:-1] + base


def connect_unnamed_segments(trails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Connect unnamed trail segments to named trails they physically touch.
    Returns list with merged trails.
    """
    # Separate named and unnamed trails
    named_trails = []
    unnamed_trails = []
    
    for trail in trails:
        if is_unnamed_trail(trail):
            unnamed_trails.append(trail)
        else:
            named_trails.append(trail)
    
    logging.info(f"🔗 Connecting segments: {len(named_trails)} named, {len(unnamed_trails)} unnamed")
    
    # Track which unnamed segments have been merged
    merged_unnamed_ids = set()
    
    # For each named trail, try to find and merge connecting unnamed segments
    for named_trail in named_trails:
        merged_count = 0
        max_iterations = 10  # Prevent infinite loops
        
        for iteration in range(max_iterations):
            found_connection = False
            
            for unnamed_trail in unnamed_trails:
                if unnamed_trail["id"] in merged_unnamed_ids:
                    continue
                
                connection = find_connection_point(
                    unnamed_trail, 
                    named_trail, 
                    CONFIG["connection_threshold_meters"]
                )
                
                if connection:
                    # Merge the geometries
                    unnamed_end, named_end = connection
                    named_trail["geometry"] = merge_geometries(
                        named_trail["geometry"],
                        unnamed_trail["geometry"],
                        named_end,
                        unnamed_end
                    )
                    
                    # Mark as merged
                    merged_unnamed_ids.add(unnamed_trail["id"])
                    merged_count += 1
                    found_connection = True
                    break  # Start over to find more connections
            
            if not found_connection:
                break  # No more connections found
        
        if merged_count > 0:
            logging.info(f"  ✓ '{named_trail['trailName']}': merged {merged_count} segments")
    
    # Keep standalone unnamed trails that weren't merged
    standalone_unnamed = [t for t in unnamed_trails if t["id"] not in merged_unnamed_ids]
    
    logging.info(f"📊 Connection results:")
    logging.info(f"   - Named trails: {len(named_trails)}")
    logging.info(f"   - Merged unnamed segments: {len(merged_unnamed_ids)}")
    logging.info(f"   - Standalone unnamed: {len(standalone_unnamed)}")
    
    return named_trails + standalone_unnamed

# -------------------------------------------------------
# FETCHING TRAILS
# -------------------------------------------------------

async def fetch_overpass_trails(session: aiohttp.ClientSession, state: str) -> List[Dict[str, Any]]:
    """Fetch trail data from Overpass API."""
    logging.info(f"Fetching trails for {state}...")
    query = f"""
    [out:json][timeout:180];
    relation["boundary"="administrative"]["admin_level"="4"]["name"="{state}"];
    convert area ::id = id();
    (
      way["highway"~"path|footway|track"]["motor_vehicle"!="yes"]["access"!="private"](area);
      way(r)["route"="hiking"](area);
    );
    out body geom tags;
    """

    try:
        async with session.post(CONFIG["overpass_url"], data={"data": query}, timeout=CONFIG["timeout"]) as resp:
            if resp.status != 200:
                logging.warning(f"Overpass query failed for {state}: HTTP {resp.status}")
                return []
            data = await resp.json()
            trails = []
            for el in data.get("elements", []):
                if "geometry" not in el:
                    continue
                coords = [(pt["lat"], pt["lon"]) for pt in el["geometry"]]
                trails.append({
                    "id": str(el["id"]),
                    "trailName": el["tags"].get("name", f"Unnamed Trail {el['id']}"),
                    "state": state.split("-")[-1],
                    "geometry": coords,
                    "terrainTypes": ["Forest"],
                    "description": el["tags"].get("description", "No description available"),
                    "userRating": 0.0,
                    "source": "OpenStreetMap",
                })
            return trails
    except Exception as e:
        logging.error(f"Overpass fetch failed for {state}: {e}")
        return []

# -------------------------------------------------------
# ELEVATION BATCHING
# -------------------------------------------------------

async def fetch_elevation_batch(session: aiohttp.ClientSession, batch_payload: list) -> list:
    """Send one batched elevation API request."""
    for attempt in range(CONFIG["retry_attempts"]):
        try:
            async with session.post(CONFIG["elevation_url"], json={"locations": batch_payload},
                                    timeout=CONFIG["timeout"]) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [r["elevation"] for r in data.get("results", [])]
        except (ClientError, ServerTimeoutError, asyncio.TimeoutError):
            await asyncio.sleep(2)
    logging.warning("⚠️ Batch elevation request failed after retries.")
    return []


async def compute_elevations_for_trails(session: aiohttp.ClientSession, trails: list) -> Dict[str, int]:
    """Batch elevation lookups for multiple trails (no caching)."""
    results = {}
    batch_size = CONFIG["elevation_batch_trails"]
    batches = [trails[i:i + batch_size] for i in range(0, len(trails), batch_size)]
    total_batches = len(batches)
    start_time = time.time()

    for i, batch in enumerate(batches, start=1):
        sampled_points = []
        trail_point_indices = []
        total_points = 0

        for t in batch:
            coords = t["geometry"]
            step = max(1, len(coords)//100)
            sampled = coords[::step]
            if not sampled:
                continue
            if total_points + len(sampled) > CONFIG["max_points_per_request"]:
                break
            trail_point_indices.append((t["id"], total_points, total_points + len(sampled)))
            sampled_points.extend([{"latitude": lat, "longitude": lon} for lat, lon in sampled])
            total_points += len(sampled)

        logging.info(f"Processing batch {i}/{total_batches} ({len(batch)} trails, {total_points} pts)...")
        elevations = await fetch_elevation_batch(session, sampled_points)

        for trail_id, start, end in trail_point_indices:
            subset = elevations[start:end]
            if len(subset) < 2:
                results[trail_id] = 0
                continue
            gain = sum(max(0, subset[j] - subset[j-1]) for j in range(1, len(subset)))
            results[trail_id] = int(gain * 3.281)

        elapsed = time.time() - start_time
        rate = i/elapsed if elapsed else 0
        remaining = (total_batches - i)/rate if rate else 0
        logging.info(f"⏳ ETA: {remaining:.1f}s remaining")

    return results

# -------------------------------------------------------
# STATE COLLECTION
# -------------------------------------------------------

async def collect_trails_for_state(session: aiohttp.ClientSession, state: str, compress: bool):
    raw_trails = await fetch_overpass_trails(session, state)
    if not raw_trails:
        logging.warning(f"No trails found for {state}.")
        return

    # NEW: Connect unnamed segments to named trails BEFORE calculating metrics
    raw_trails = connect_unnamed_segments(raw_trails)

    # Calculate distance (after merging geometries)
    for t in raw_trails:
        t["distanceMiles"] = haversine_distance(t["geometry"])

    # Elevation batching (recalculated for merged trails)
    elevation_results = await compute_elevations_for_trails(session, raw_trails)

    for t in raw_trails:
        elev = elevation_results.get(t["id"], 0)
        t["elevationGainFeet"] = elev
        t["difficultyLevel"] = estimate_difficulty(t["distanceMiles"], elev)
        mid = t["geometry"][len(t["geometry"])//2]
        t["latitude"], t["longitude"] = mid

    valid = [t for t in raw_trails if validate_trail(t)]
    unique = deduplicate(valid)
    output_path = OUTPUT_DIR / f"{state.split('-')[-1]}_trails.json"

    result = {
        "version": CONFIG["version"],
        "collectionDate": datetime.utcnow().strftime("%Y-%m-%d"),
        "trailCount": len(unique),
        "trails": unique,
    }
    save_json(result, output_path, compress=compress)

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

async def main(states, compress):
    logging.info("🚀 Starting Trail Data Collection with segment connection...")
    start = time.time()

    connector = aiohttp.TCPConnector(limit_per_host=CONFIG["max_concurrent_requests"])
    async with aiohttp.ClientSession(connector=connector, headers={"User-Agent": CONFIG["user_agent"]}) as session:
        tasks = [collect_trails_for_state(session, s, compress) for s in states]
        await asyncio.gather(*tasks)

    logging.info(f"✅ Completed {len(states)} states in {time.time()-start:.1f}s")

# -------------------------------------------------------
# CLI
# -------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trail data collection with segment connection.")
    parser.add_argument("--states", nargs="+", help="List of ISO3166-2 codes (e.g., US-SC)")
    parser.add_argument("--all", action="store_true", help="Process all 50 states")
    parser.add_argument("--compress", action="store_true", help="Compress JSON output")
    args = parser.parse_args()

    if args.all:
        selected = ALL_US_STATES
    elif args.states:
        selected = args.states
    else:
        print("❌ Please provide --states or --all")
        exit(1)

    asyncio.run(main(selected, args.compress))