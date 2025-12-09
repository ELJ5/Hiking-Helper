#!/usr/bin/env python3
"""
Add Elevation to Existing Trail JSON

Takes a trail JSON file and adds optimized elevation data to each trail.
Works with any JSON that has trail coordinates.

Usage:
    python add_elevation_to_trails.py input.json -o output.json
    python add_elevation_to_trails.py input.json -o output.json --verbose
"""

import json
import argparse
import sys
from elevation_optimizer import get_elevations_for_trail, calculate_elevation_gain, print_stats, reset_stats

def add_elevation_to_trails(input_file, output_file, verbose=False):
    """
    Add elevation data to trails in JSON file.
    
    Args:
        input_file: Input JSON file with trails
        output_file: Output JSON file with elevation added
        verbose: Print detailed progress
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
    print()
    
    # Reset statistics
    reset_stats()
    
    # Process each trail
    trails_with_elevation = 0
    trails_without_coords = 0
    
    for i, trail in enumerate(trails):
        if verbose:
            print(f"[{i+1}/{len(trails)}] Processing: {trail.get('trailName', 'Unknown')}")
        
        # Check if trail has coordinates
        coords = trail.get('_coordinates', [])
        if not coords:
            # Try alternative coordinate field names
            coords = trail.get('coordinates', [])
        
        if not coords or len(coords) == 0:
            if verbose:
                print(f"  ⚠️  No coordinates found, skipping")
            trails_without_coords += 1
            trail['elevationGainFeet'] = 0.0
            continue
        
        # Get elevations (fast batch method!)
        try:
            elevations = get_elevations_for_trail(coords, verbose=verbose)
            gain = calculate_elevation_gain(elevations)
            trail['elevationGainFeet'] = round(gain, 1)
            
            if verbose:
                print(f"  ✅ Elevation gain: {gain:.1f} feet")
            
            trails_with_elevation += 1
            
        except Exception as e:
            if verbose:
                print(f"  ❌ Error getting elevation: {e}")
            trail['elevationGainFeet'] = 0.0
    
    # Save output
    print()
    print(f"Saving trails with elevation to {output_file}...")
    
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
    print(f"Total trails:              {len(trails)}")
    print(f"Trails with elevation:     {trails_with_elevation}")
    print(f"Trails without coords:     {trails_without_coords}")
    print(f"Success rate:              {trails_with_elevation/len(trails)*100:.1f}%")
    print()
    
    # Print elevation query statistics
    print_stats()
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Add optimized elevation data to trail JSON"
    )
    parser.add_argument("input", help="Input JSON file with trails")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Print detailed progress for each trail")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ELEVATION OPTIMIZER - Add Elevation to Trails")
    print("=" * 60)
    print()
    
    success = add_elevation_to_trails(args.input, args.output, args.verbose)
    
    if success:
        print()
        print("=" * 60)
        print("✅ SUCCESS! Elevation data added to all trails")
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