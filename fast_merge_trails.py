#!/usr/bin/env python3
"""
Fast Trail Merger - Combines trails with the same name
WITHOUT the slow segment reordering algorithm.

This is 100x+ faster than the built-in merge but doesn't try to 
intelligently connect trail segments. It just combines the data.

Usage:
    # Step 1: Get trails WITHOUT merging (fast)
    python fetch_trails_overpass.py --bbox 32.0 -83.4 35.2 -78.5 --tiles 3 2 --no-merge -o unmerged.json
    
    # Step 2: Run this fast merge script
    python fast_merge_trails.py unmerged.json -o merged.json
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, Any, List

def fast_merge_trails(trails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fast merge: Combines trails with same name by summing distances/elevation
    and averaging coordinates. No complex segment reordering.
    
    Runtime: O(n) instead of O(n²) or worse
    """
    # Group by trail name
    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trail in trails:
        by_name[trail["trailName"]].append(trail)
    
    merged = []
    
    for name, group in by_name.items():
        if len(group) == 1:
            # No merging needed - just one segment
            merged.append(group[0])
            continue
        
        # Multiple segments with same name - combine them
        
        # Sum distances and elevation gains
        total_distance = sum(t["distanceMiles"] for t in group)
        total_elevation = sum(t["elevationGainFeet"] for t in group)
        
        # Average center coordinates (weighted by distance)
        total_dist_weight = total_distance if total_distance > 0 else len(group)
        if total_distance > 0:
            avg_lat = sum(t["latitude"] * t["distanceMiles"] for t in group) / total_distance
            avg_lon = sum(t["longitude"] * t["distanceMiles"] for t in group) / total_distance
        else:
            avg_lat = sum(t["latitude"] for t in group) / len(group)
            avg_lon = sum(t["longitude"] for t in group) / len(group)
        
        # Combine terrain types (unique)
        all_terrains = []
        for t in group:
            all_terrains.extend(t.get("terrainTypes", []))
        terrain_types = list(dict.fromkeys(all_terrains))  # Preserve order, remove duplicates
        
        # Use most common difficulty or first one
        difficulties = [t["difficultyLevel"] for t in group]
        difficulty = max(set(difficulties), key=difficulties.count)
        
        # Average user ratings
        avg_rating = sum(t["userRating"] for t in group) / len(group)
        
        # Combine descriptions
        descriptions = [t["description"] for t in group if t.get("description")]
        if descriptions:
            # Take longest description or combine if they're different
            unique_descriptions = list(dict.fromkeys(descriptions))
            if len(unique_descriptions) == 1:
                description = unique_descriptions[0]
            else:
                description = "; ".join(unique_descriptions[:3])  # Max 3 to avoid too long
                if len(unique_descriptions) > 3:
                    description += f" (and {len(unique_descriptions) - 3} more segments)"
        else:
            description = f"Trail merged from {len(group)} segments"
        
        # Use first ID and state
        trail_id = group[0]["id"]
        state = group[0]["state"]
        
        # Create merged trail
        merged_trail = {
            "id": trail_id,
            "trailName": name,
            "state": state,
            "latitude": round(avg_lat, 6),
            "longitude": round(avg_lon, 6),
            "distanceMiles": round(total_distance, 2),
            "elevationGainFeet": round(total_elevation, 1),
            "difficultyLevel": difficulty,
            "terrainTypes": terrain_types,
            "description": description,
            "userRating": round(avg_rating, 1),
            "completed": False,
        }
        
        merged.append(merged_trail)
    
    return merged

def main():
    parser = argparse.ArgumentParser(
        description="Fast trail merger - combines trails with same name"
    )
    parser.add_argument("input", help="Input JSON file (from --no-merge)")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show merge statistics")
    
    args = parser.parse_args()
    
    # Load trails
    print(f"Loading trails from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        trails = json.load(f)
    
    print(f"Loaded {len(trails)} trails")
    
    # Count unique names before merging
    unique_names_before = len(set(t["trailName"] for t in trails))
    
    # Fast merge
    print("Merging trails with same name (fast algorithm)...")
    merged = fast_merge_trails(trails)
    
    print(f"Merged {len(trails)} trails into {len(merged)} unique trails")
    
    # Statistics
    if args.verbose:
        unique_names_after = len(set(t["trailName"] for t in merged))
        print(f"\nStatistics:")
        print(f"  Unique trail names before: {unique_names_before}")
        print(f"  Unique trail names after:  {unique_names_after}")
        print(f"  Total entries before:      {len(trails)}")
        print(f"  Total entries after:       {len(merged)}")
        print(f"  Segments merged:           {len(trails) - len(merged)}")
        
        # Find trails with most segments
        by_name = defaultdict(int)
        for t in trails:
            by_name[t["trailName"]] += 1
        top_segmented = sorted(by_name.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top 5 most segmented trails:")
        for name, count in top_segmented:
            print(f"    - {name}: {count} segments")
    
    # Save merged trails
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)
    
    print(f"✅ Done! Saved {len(merged)} trails to {args.output}")

if __name__ == "__main__":
    main()