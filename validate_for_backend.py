#!/usr/bin/env python3
"""
Trail JSON Validator & Backend Converter

Ensures your trail JSON is properly formatted and ready for your backend.
Validates structure, data types, and required fields.
Can fix common issues automatically.

Usage:
    python validate_for_backend.py trails.json -o validated_trails.json
    python validate_for_backend.py trails.json --check-only
"""

import argparse
import json
import sys
from typing import Dict, Any, List, Optional

# Expected schema based on your backend requirements
REQUIRED_FIELDS = {
    "id": int,
    "trailName": str,
    "state": str,
    "latitude": (int, float),
    "longitude": (int, float),
    "distanceMiles": (int, float),
    "elevationGainFeet": (int, float),
    "difficultyLevel": str,
    "terrainTypes": list,
    "description": str,
    "userRating": (int, float),
    "completed": bool,
}

VALID_DIFFICULTIES = ["Easy", "Moderate", "Difficult", "Expert"]

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_trail(trail: Dict[str, Any], index: int, strict: bool = False) -> List[str]:
    """
    Validate a single trail entry.
    Returns list of issues found (empty if valid).
    """
    issues = []
    
    # Check required fields exist
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in trail:
            issues.append(f"Trail {index}: Missing required field '{field}'")
            continue
        
        # Check type
        value = trail[field]
        if value is None:
            issues.append(f"Trail {index}: Field '{field}' is null")
            continue
        
        # Handle tuple of types (e.g., int or float)
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                issues.append(f"Trail {index}: Field '{field}' has wrong type (expected {expected_type}, got {type(value).__name__})")
        else:
            if not isinstance(value, expected_type):
                issues.append(f"Trail {index}: Field '{field}' has wrong type (expected {expected_type.__name__}, got {type(value).__name__})")
    
    # Validate specific fields
    if "latitude" in trail:
        lat = trail["latitude"]
        if not (-90 <= lat <= 90):
            issues.append(f"Trail {index}: Latitude {lat} out of valid range (-90 to 90)")
    
    if "longitude" in trail:
        lon = trail["longitude"]
        if not (-180 <= lon <= 180):
            issues.append(f"Trail {index}: Longitude {lon} out of valid range (-180 to 180)")
    
    if "distanceMiles" in trail:
        dist = trail["distanceMiles"]
        if dist < 0:
            issues.append(f"Trail {index}: Distance {dist} is negative")
        if dist > 1000:
            issues.append(f"Trail {index}: Distance {dist} seems unusually high (>1000 miles)")
    
    if "elevationGainFeet" in trail:
        elev = trail["elevationGainFeet"]
        if elev < 0:
            issues.append(f"Trail {index}: Elevation gain {elev} is negative")
    
    if "difficultyLevel" in trail:
        diff = trail["difficultyLevel"]
        if diff not in VALID_DIFFICULTIES:
            issues.append(f"Trail {index}: Invalid difficulty '{diff}' (must be one of {VALID_DIFFICULTIES})")
    
    if "terrainTypes" in trail:
        terrain = trail["terrainTypes"]
        if not isinstance(terrain, list):
            issues.append(f"Trail {index}: terrainTypes must be an array")
        elif len(terrain) == 0:
            issues.append(f"Trail {index}: terrainTypes array is empty")
        elif not all(isinstance(t, str) for t in terrain):
            issues.append(f"Trail {index}: All terrainTypes must be strings")
    
    if "userRating" in trail:
        rating = trail["userRating"]
        if not (0 <= rating <= 5):
            issues.append(f"Trail {index}: User rating {rating} out of valid range (0-5)")
    
    if "trailName" in trail:
        name = trail["trailName"]
        if not name or name.strip() == "":
            issues.append(f"Trail {index}: Trail name is empty")
    
    # Check for coordinate arrays (should NOT be present)
    if "coordinates" in trail or "_coordinates" in trail or "geometry" in trail:
        issues.append(f"Trail {index}: Contains coordinate arrays (only single lat/lon allowed)")
    
    # Check latitude/longitude are single values, not arrays
    if "latitude" in trail and isinstance(trail["latitude"], (list, tuple)):
        issues.append(f"Trail {index}: latitude must be a single number, not an array")
    if "longitude" in trail and isinstance(trail["longitude"], (list, tuple)):
        issues.append(f"Trail {index}: longitude must be a single number, not an array")
    
    # Check for unexpected extra fields
    if strict:
        extra_fields = set(trail.keys()) - set(REQUIRED_FIELDS.keys())
        if extra_fields:
            issues.append(f"Trail {index}: Unexpected fields: {extra_fields}")
    
    return issues

def fix_trail(trail: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Attempt to automatically fix common issues in a trail entry.
    Returns fixed trail.
    """
    fixed = trail.copy()
    
    # Ensure all required fields exist with defaults
    defaults = {
        "id": index,
        "trailName": "Unnamed Trail",
        "state": "Unknown",
        "latitude": 0.0,
        "longitude": 0.0,
        "distanceMiles": 0.0,
        "elevationGainFeet": 0.0,
        "difficultyLevel": "Moderate",
        "terrainTypes": ["Mixed"],
        "description": "",
        "userRating": 3.0,
        "completed": False,
    }
    
    for field, default in defaults.items():
        if field not in fixed or fixed[field] is None:
            fixed[field] = default
    
    # Fix types
    try:
        fixed["id"] = int(fixed["id"])
    except (ValueError, TypeError):
        fixed["id"] = index
    
    # Ensure numeric fields are numeric
    for field in ["latitude", "longitude", "distanceMiles", "elevationGainFeet", "userRating"]:
        try:
            fixed[field] = float(fixed[field])
        except (ValueError, TypeError):
            fixed[field] = defaults[field]
    
    # Round to reasonable precision
    fixed["latitude"] = round(fixed["latitude"], 6)
    fixed["longitude"] = round(fixed["longitude"], 6)
    fixed["distanceMiles"] = round(fixed["distanceMiles"], 2)
    fixed["elevationGainFeet"] = round(fixed["elevationGainFeet"], 1)
    fixed["userRating"] = round(fixed["userRating"], 1)
    
    # Clamp values to valid ranges
    fixed["latitude"] = max(-90, min(90, fixed["latitude"]))
    fixed["longitude"] = max(-180, min(180, fixed["longitude"]))
    fixed["distanceMiles"] = max(0, fixed["distanceMiles"])
    fixed["elevationGainFeet"] = max(0, fixed["elevationGainFeet"])
    fixed["userRating"] = max(0, min(5, fixed["userRating"]))
    
    # Fix difficulty
    if fixed["difficultyLevel"] not in VALID_DIFFICULTIES:
        fixed["difficultyLevel"] = "Moderate"
    
    # Ensure terrainTypes is a list of strings
    if not isinstance(fixed["terrainTypes"], list):
        fixed["terrainTypes"] = ["Mixed"]
    else:
        fixed["terrainTypes"] = [str(t) for t in fixed["terrainTypes"] if t]
        if not fixed["terrainTypes"]:
            fixed["terrainTypes"] = ["Mixed"]
    
    # Ensure strings are strings
    for field in ["trailName", "state", "description"]:
        fixed[field] = str(fixed[field]) if fixed[field] else defaults[field]
    
    # Ensure completed is boolean
    fixed["completed"] = bool(fixed["completed"])
    
    # Remove coordinate arrays (only single lat/lon allowed)
    if "coordinates" in fixed:
        del fixed["coordinates"]
    if "_coordinates" in fixed:
        del fixed["_coordinates"]
    if "geometry" in fixed:
        del fixed["geometry"]
    
    # Ensure latitude/longitude are single numbers, not arrays
    if isinstance(fixed.get("latitude"), (list, tuple)):
        fixed["latitude"] = float(fixed["latitude"][0]) if fixed["latitude"] else 0.0
    if isinstance(fixed.get("longitude"), (list, tuple)):
        fixed["longitude"] = float(fixed["longitude"][0]) if fixed["longitude"] else 0.0
    
    # Remove any extra fields not in schema
    fixed = {k: v for k, v in fixed.items() if k in REQUIRED_FIELDS}
    
    return fixed

def validate_json(trails: List[Dict[str, Any]], strict: bool = False) -> tuple[List[str], bool]:
    """
    Validate entire JSON array.
    Returns (list of issues, is_valid).
    """
    all_issues = []
    
    if not isinstance(trails, list):
        all_issues.append("ERROR: Root element must be an array of trails")
        return all_issues, False
    
    if len(trails) == 0:
        all_issues.append("WARNING: No trails found in file")
    
    # Check for duplicate IDs
    ids = [t.get("id") for t in trails if "id" in t]
    duplicate_ids = [id_ for id_ in ids if ids.count(id_) > 1]
    if duplicate_ids:
        all_issues.append(f"ERROR: Duplicate trail IDs found: {set(duplicate_ids)}")
    
    # Validate each trail
    for i, trail in enumerate(trails):
        issues = validate_trail(trail, i, strict)
        all_issues.extend(issues)
    
    is_valid = len(all_issues) == 0
    return all_issues, is_valid

def convert_for_backend(input_file: str, output_file: str, fix_issues: bool = True, strict: bool = False):
    """
    Main conversion function.
    """
    print(f"Loading trails from {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            trails = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON file: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {input_file}")
        return False
    
    if not isinstance(trails, list):
        print(f"❌ ERROR: Root element must be an array")
        return False
    
    print(f"Loaded {len(trails)} trails")
    print()
    
    # Validate
    print("Validating trails...")
    issues, is_valid = validate_json(trails, strict=strict)
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issue(s):")
        for issue in issues[:20]:  # Show first 20
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")
        print()
    
    if is_valid:
        print("✅ All trails are valid!")
        if output_file:
            # Just copy with pretty formatting
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(trails, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved validated trails to {output_file}")
        return True
    
    if not fix_issues:
        print("❌ Validation failed. Use --fix to automatically correct issues.")
        return False
    
    # Fix issues
    print("Attempting to fix issues...")
    fixed_trails = []
    
    for i, trail in enumerate(trails):
        fixed = fix_trail(trail, i)
        fixed_trails.append(fixed)
    
    # Validate fixed trails
    issues_after, is_valid_after = validate_json(fixed_trails, strict=False)
    
    if is_valid_after:
        print(f"✅ Successfully fixed all issues!")
    else:
        print(f"⚠️  Fixed most issues, but {len(issues_after)} remain:")
        for issue in issues_after[:10]:
            print(f"  - {issue}")
    
    # Save
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_trails, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved {len(fixed_trails)} trails to {output_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total trails:     {len(fixed_trails)}")
    print(f"Issues found:     {len(issues)}")
    print(f"Issues fixed:     {len(issues) - len(issues_after)}")
    print(f"Issues remaining: {len(issues_after)}")
    
    # Stats
    difficulties = {}
    for t in fixed_trails:
        d = t.get("difficultyLevel", "Unknown")
        difficulties[d] = difficulties.get(d, 0) + 1
    
    print(f"\nTrails by difficulty:")
    for d in ["Easy", "Moderate", "Difficult", "Expert"]:
        if d in difficulties:
            print(f"  {d}: {difficulties[d]}")
    
    return is_valid_after

def main():
    parser = argparse.ArgumentParser(
        description="Validate and convert trail JSON for backend compatibility"
    )
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("-o", "--output", help="Output JSON file (validated/fixed)")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only check validation, don't save output")
    parser.add_argument("--no-fix", action="store_true",
                       help="Don't attempt to fix issues automatically")
    parser.add_argument("--strict", action="store_true",
                       help="Strict mode: flag any extra fields")
    
    args = parser.parse_args()
    
    if args.check_only:
        args.output = None
        args.no_fix = True
    
    if not args.output and not args.check_only:
        print("ERROR: Must specify --output or use --check-only")
        sys.exit(1)
    
    success = convert_for_backend(
        args.input,
        args.output,
        fix_issues=not args.no_fix,
        strict=args.strict
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()