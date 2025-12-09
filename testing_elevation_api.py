#!/usr/bin/env python
"""
Test if elevation API is being accessed correctly.

This script:
1. Tests direct API connectivity
2. Fetches a small area with elevation
3. Verifies elevation data in output
4. Shows timing information

Usage:
    python test_elevation_api.py
"""

import json
import requests
import time
from datetime import datetime

def test_api_connectivity():
    """Test if Open-Elevation API is accessible."""
    print("=" * 70)
    print("TEST 1: Open-Elevation API Connectivity")
    print("=" * 70)
    
    test_coords = [
        (35.0651, -82.7339),  # Table Rock, SC
        (34.8526, -82.3940),  # Greenville, SC
        (33.4152, -79.8431),  # Myrtle Beach, SC
    ]
    
    for lat, lon in test_coords:
        try:
            url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
            print(f"\nTesting: ({lat}, {lon})")
            print(f"URL: {url}")
            
            start_time = time.time()
            response = requests.get(url, timeout=10)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                elevation = data['results'][0]['elevation']
                print(f"✅ SUCCESS - Elevation: {elevation}m ({elevation * 3.28084:.1f}ft)")
                print(f"   Response time: {elapsed:.2f} seconds")
            else:
                print(f"❌ FAILED - Status code: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"❌ TIMEOUT - API took longer than 10 seconds")
            return False
        except Exception as e:
            print(f"❌ ERROR - {e}")
            return False
    
    print("\n" + "=" * 70)
    print("✅ API Connectivity Test PASSED")
    print("=" * 70)
    return True

def test_batch_elevation():
    """Test batch elevation lookup (multiple points at once)."""
    print("\n" + "=" * 70)
    print("TEST 2: Batch Elevation Lookup")
    print("=" * 70)
    
    # Multiple points along a trail
    locations = [
        (35.0651, -82.7339),
        (35.0655, -82.7340),
        (35.0660, -82.7342),
    ]
    
    locations_str = "|".join([f"{lat},{lon}" for lat, lon in locations])
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations_str}"
    
    print(f"Testing {len(locations)} points...")
    print(f"URL: {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=15)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ SUCCESS - Got {len(data['results'])} elevation points")
            print(f"   Response time: {elapsed:.2f} seconds")
            
            print("\nElevation profile:")
            for i, result in enumerate(data['results']):
                elev_m = result['elevation']
                elev_ft = elev_m * 3.28084
                print(f"   Point {i+1}: {elev_m}m ({elev_ft:.1f}ft)")
            
            # Calculate elevation gain
            elevations = [r['elevation'] * 3.28084 for r in data['results']]
            gain = sum([elevations[i+1] - elevations[i] for i in range(len(elevations)-1) if elevations[i+1] > elevations[i]])
            print(f"\nCalculated elevation gain: {gain:.1f} feet")
            
            return True
        else:
            print(f"❌ FAILED - Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR - {e}")
        return False

def check_trail_json_for_elevation(filename):
    """Check if a trail JSON file has elevation data."""
    print("\n" + "=" * 70)
    print(f"TEST 3: Checking '{filename}' for Elevation Data")
    print("=" * 70)
    
    try:
        with open(filename, 'r') as f:
            trails = json.load(f)
        
        print(f"\nTotal trails: {len(trails)}")
        
        # Check elevation data
        trails_with_elevation = [t for t in trails if t.get('elevationGainFeet', 0) > 0]
        trails_without = len(trails) - len(trails_with_elevation)
        
        print(f"Trails with elevation data: {len(trails_with_elevation)}")
        print(f"Trails without elevation: {trails_without}")
        
        if len(trails_with_elevation) > 0:
            print("\n✅ Elevation data found!")
            print("\nSample trails with elevation:")
            for trail in trails_with_elevation[:5]:
                name = trail['trailName']
                elev = trail['elevationGainFeet']
                dist = trail['distanceMiles']
                print(f"   • {name}: {elev:.1f}ft gain over {dist:.1f} miles")
            
            # Statistics
            elevations = [t['elevationGainFeet'] for t in trails_with_elevation]
            avg_elev = sum(elevations) / len(elevations)
            max_elev = max(elevations)
            max_trail = [t for t in trails_with_elevation if t['elevationGainFeet'] == max_elev][0]
            
            print(f"\nElevation statistics:")
            print(f"   Average gain: {avg_elev:.1f} feet")
            print(f"   Maximum gain: {max_elev:.1f} feet ({max_trail['trailName']})")
            
            return True
        else:
            print("\n⚠️  No elevation data found in trails")
            print("This means either:")
            print("   1. Trails were fetched without --elevation flag")
            print("   2. Elevation API was unavailable during fetch")
            print("   3. All trails are perfectly flat (unlikely)")
            return False
            
    except FileNotFoundError:
        print(f"\n❌ File not found: {filename}")
        print("Run a fetch with --elevation flag first:")
        print(f"   python fetch_trails_overpass.py --bbox 34.85 -82.4 34.87 -82.38 --elevation -o {filename}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR reading file: {e}")
        return False

def run_quick_elevation_test():
    """Run a quick elevation test by fetching a tiny area."""
    print("\n" + "=" * 70)
    print("TEST 4: Quick Fetch with Elevation (Optional)")
    print("=" * 70)
    print("\nThis test will fetch a tiny area with elevation data.")
    print("It will take about 1-2 minutes.")
    print()
    
    response = input("Run this test? (y/N): ")
    if response.lower() != 'y':
        print("Skipping fetch test.")
        return None
    
    import subprocess
    import sys
    
    print("\nFetching tiny area around Greenville, SC...")
    print("Command: python fetch_trails_overpass.py --bbox 34.85 -82.4 34.87 -82.38 --elevation -o test_elevation.json")
    
    try:
        result = subprocess.run([
            sys.executable,
            'fetch_trails_overpass.py',
            '--bbox', '34.85', '-82.4', '34.87', '-82.38',
            '--elevation',
            '-o', 'test_elevation.json'
        ], capture_output=True, text=True, timeout=180)
        
        print("\n" + result.stdout)
        
        if result.returncode == 0:
            print("✅ Fetch completed successfully!")
            return 'test_elevation.json'
        else:
            print(f"❌ Fetch failed with error:")
            print(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Fetch timed out after 3 minutes")
        return None
    except FileNotFoundError:
        print("❌ fetch_trails_overpass.py not found in current directory")
        return None
    except Exception as e:
        print(f"❌ Error running fetch: {e}")
        return None

def main():
    print("\n" + "=" * 70)
    print("ELEVATION API TEST SUITE")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = []
    
    # Test 1: API connectivity
    results.append(("API Connectivity", test_api_connectivity()))
    
    # Test 2: Batch elevation
    if results[0][1]:  # Only if API is accessible
        results.append(("Batch Elevation", test_batch_elevation()))
    else:
        print("\n⚠️  Skipping batch test since API is not accessible")
        results.append(("Batch Elevation", False))
    
    # Test 3: Check existing file for elevation data
    test_file = input("\nEnter trail JSON file to check (or press Enter to skip): ").strip()
    if test_file:
        results.append(("Trail JSON Check", check_trail_json_for_elevation(test_file)))
    
    # Test 4: Optional quick fetch
    if results[0][1]:  # Only if API is accessible
        fetch_output = run_quick_elevation_test()
        if fetch_output:
            results.append(("Quick Fetch Test", check_trail_json_for_elevation(fetch_output)))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all([r[1] for r in results if r[1] is not None])
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Elevation API is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Check the errors above for details.")
    print("=" * 70)

if __name__ == "__main__":
    main()