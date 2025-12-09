#!/usr/bin/env python3
"""
Add Elevation to Trail JSON (Works with Merged or Unmerged Trails)

This version can handle:
1. Unmerged trails with _coordinates (accurate elevation gain)
2. Merged trails with only lat/lon (estimates based on distance/difficulty)

Usage:
    python add_elevation_flexible.py input.json -o output.json
    python add_elevation_flexible.py input.json -o output.json --verbose
    python add_elevation_flexible.py input.json -o output.json --force-single-point
"""

import json
import argparse
import sys
from elevation_optimizer import get_elevations_for_trail, calculate_elevation_gain, get_elevation_for_point, print_stats, reset_stats

def estimate_elevation_gain(distance_miles, difficulty, terrain_types):
    """
    Estimate elevation gain based on distance and difficulty when coordinates unavailable.
    This is approximate but better than nothing.
    """
    # Base estimation: elevation gain per mile by difficulty
    difficulty_factors = {
        "Easy": 50,      # ~50 ft/mile
        "Moderate": 150, # ~150 ft/mile
        "Difficult": 300,# ~300 ft/mile
        "Expert": 500,   # ~500 ft/mile
    }
    
    base_gain_per_mile = difficulty_factors.get(difficulty, 100)
    
    # Adjust for terrain
    if "Mountain" in terrain_types:
        base_gain_per_mile *= 1.5
    elif "Rocky" in terrain_types:
        base_gain_per_mile *= 1.2
    elif "Paved" in terrain_types:
        base_gain_per_mile *= 0.5
    
    estimated_gain = distance_miles * base_gain_per_mile
    return round(estimated_gain, 1)

def add_elevation_flexible(input_file, output_file, verbose=False, force_single_point=False):
    """
    Add elevation data to trails in JSON file.
    Works with both merged (lat/lon only) and unmerged (_coordinates) trails.
    
    Args:
        input_file: Input JSON file with trails
        output_file: Output JSON file with elevation added
        verbose: Print detailed progress
        force_single_point: Force single-point mode even if coordinates exist
    """
    print(f"Loading trails from {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            trails = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {input_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON: {e}")
        return False
    
    if not isinstance(trails, list):
        print("❌ Error: JSON must be an array of trails")
        return False
    
    print(f"Found {len(trails)} trails")
    
    # Detect format
    has_coords = any(trail.get('_coordinates') for trail in trails)
    has_latlng = any(trail.get('latitude') and trail.get('longitude') for trail in trails)
    
    if force_single_point:
        print("\n⚠️  FORCED SINGLE-POINT MODE")
        print("Using center lat/lon for approximate elevation")
        mode = "single"
    elif not has_coords and has_latlng:
        print("\n⚠️  MERGED TRAIL FORMAT DETECTED")
        print("Trails have lat/lon but no _coordinates")
        print("Will estimate elevation gain based on distance/difficulty")
        mode = "estimate"
    elif has_coords:
        print("\n✅ FULL COORDINATE DATA DETECTED")
        print("Will calculate accurate elevation gain from path")
        mode = "accurate"
    else:
        print("\n❌ ERROR: Trails have neither _coordinates nor lat/lon")
        return False
    
    print()
    
    # Reset statistics
    reset_stats()
    
    # Process each trail
    trails_with_elevation = 0
    trails_estimated = 0
    trails_failed = 0
    
    for i, trail in enumerate(trails):
        if verbose:
            print(f"[{i+1}/{len(trails)}] Processing: {trail.get('trailName', 'Unknown')}")
        
        try:
            if mode == "accurate" and not force_single_point:
                # Use full coordinates for accurate elevation gain
                coords = trail.get('_coordinates', [])
                if not coords or len(coords) < 2:
                    if verbose:
                        print(f"  ⚠️  No coordinates, using estimation")
                    # Fall back to estimation
                    elev_gain = estimate_elevation_gain(
                        trail.get('distanceMiles', 0),
                        trail.get('difficultyLevel', 'Moderate'),
                        trail.get('terrainTypes', [])
                    )
                    trail['elevationGainFeet'] = elev_gain
                    trail['elevationEstimated'] = True
                    trails_estimated += 1
                else:
                    # Get accurate elevation gain from path
                    elevations = get_elevations_for_trail(coords, verbose=False)
                    gain = calculate_elevation_gain(elevations)
                    trail['elevationGainFeet'] = round(gain, 1)
                    trail['elevationEstimated'] = False
                    
                    if verbose:
                        print(f"  ✅ Elevation gain: {gain:.1f} feet (accurate)")
                    
                    trails_with_elevation += 1
            
            elif mode == "single" or force_single_point:
                # Single point mode: get elevation at center, estimate gain
                lat = trail.get('latitude')
                lon = trail.get('longitude')
                
                if lat and lon:
                    # Get elevation at this single point
                    elev_at_point = get_elevation_for_point(lat, lon, verbose=False)
                    
                    # Estimate gain based on distance/difficulty
                    elev_gain = estimate_elevation_gain(
                        trail.get('distanceMiles', 0),
                        trail.get('difficultyLevel', 'Moderate'),
                        trail.get('terrainTypes', [])
                    )
                    
                    trail['elevationGainFeet'] = round(elev_gain, 1)
                    trail['elevationAtCenter'] = round(elev_at_point, 1)
                    trail['elevationEstimated'] = True
                    
                    if verbose:
                        print(f"  ⚠️  Elevation: {elev_at_point:.1f}ft at center, gain: {elev_gain:.1f}ft (estimated)")
                    
                    trails_estimated += 1
                else:
                    if verbose:
                        print(f"  ❌ No lat/lon available")
                    trail['elevationGainFeet'] = 0.0
                    trail['elevationEstimated'] = True
                    trails_failed += 1
            
            else:  # mode == "estimate"
                # Merged trails: estimate based on difficulty/distance
                elev_gain = estimate_elevation_gain(
                    trail.get('distanceMiles', 0),
                    trail.get('difficultyLevel', 'Moderate'),
                    trail.get('terrainTypes', [])
                )
                trail['elevationGainFeet'] = round(elev_gain, 1)
                trail['elevationEstimated'] = True
                
                if verbose:
                    print(f"  ⚠️  Elevation gain: {elev_gain:.1f} feet (estimated from distance/difficulty)")
                
                trails_estimated += 1
                
        except Exception as e:
            if verbose:
                print(f"  ❌ Error: {e}")
            trail['elevationGainFeet'] = 0.0
            trail['elevationEstimated'] = True
            trails_failed += 1
    
    # Save output
    print()
    print(f"Saving trails to {output_file}...")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(trails, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(trails)} trails")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False
    
    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total trails:                {len(trails)}")
    print(f"Accurate elevation (path):   {trails_with_elevation}")
    print(f"Estimated elevation:         {trails_estimated}")
    print(f"Failed/No data:              {trails_failed}")
    
    if trails_estimated > 0:
        print()
        print("⚠️  NOTE: Some elevations are ESTIMATED based on:")
        print("   - Trail distance")
        print("   - Difficulty level")
        print("   - Terrain type")
        print("   These are approximate and may not match actual elevation gain.")
    
    print()
    
    # Print elevation query statistics (only if we made API calls)
    if trails_with_elevation > 0 or mode == "single":
        print_stats()
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Add elevation data to trail JSON (works with merged or unmerged trails)"
    )
    parser.add_argument("input", help="Input JSON file with trails")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Print detailed progress for each trail")
    parser.add_argument("--force-single-point", action="store_true",
                       help="Force single-point mode (faster but less accurate)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ELEVATION TOOL - Flexible Mode")
    print("=" * 60)
    print()
    
    success = add_elevation_flexible(
        args.input, 
        args.output, 
        args.verbose,
        args.force_single_point
    )
    
    if success:
        print()
        print("=" * 60)
        print("✅ SUCCESS! Elevation data added")
        print("=" * 60)
        sys.exit(0)
    else:
        print()
        print("=" * 60)
        print("❌ FAILED - See errors above")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()