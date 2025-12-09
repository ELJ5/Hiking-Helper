#!/usr/bin/env python3
"""
Optimized Elevation Fetching Module

This module provides fast, reliable elevation data fetching with:
- Batch requests (100 points at once instead of 1)
- Automatic retry with exponential backoff
- Rate limit handling
- Progress tracking
- Up to 10x faster than single-point queries

Usage:
    from elevation_optimizer import get_elevations_for_trail
    
    coordinates = [(35.0, -82.0), (35.1, -82.1), ...]
    elevations = get_elevations_for_trail(coordinates)
"""

import requests
import time
from typing import List, Tuple, Optional
import sys

# Configuration
BATCH_SIZE = 100  # Open-Elevation can handle ~100 points per request
REQUEST_TIMEOUT = 30  # Longer timeout for batch requests
RETRY_ATTEMPTS = 3
BASE_DELAY = 0.5  # Base delay between requests (seconds)
MAX_DELAY = 10.0  # Maximum delay for exponential backoff

# Statistics tracking
stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'retries': 0,
    'total_points': 0,
    'points_with_data': 0,
}


def get_elevations_batch(coordinates: List[Tuple[float, float]], 
                        verbose: bool = False) -> List[float]:
    """
    Get elevations for multiple points in a single API request.
    
    Args:
        coordinates: List of (latitude, longitude) tuples
        verbose: Print progress messages
        
    Returns:
        List of elevations in feet (same order as input)
        Returns 0.0 for failed points
    """
    if not coordinates:
        return []
    
    # Limit to batch size
    coordinates = coordinates[:BATCH_SIZE]
    
    # Format locations for API: "lat1,lon1|lat2,lon2|..."
    locations_str = "|".join([f"{lat},{lon}" for lat, lon in coordinates])
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations_str}"
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            stats['total_requests'] += 1
            
            if verbose and attempt > 0:
                print(f"    Retry attempt {attempt + 1}/{RETRY_ATTEMPTS}...", end=' ')
            
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            
            # Handle rate limiting
            if response.status_code == 429:
                stats['retries'] += 1
                wait_time = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                if verbose:
                    print(f"Rate limited, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            
            # Success
            if response.status_code == 200:
                stats['successful_requests'] += 1
                data = response.json()
                
                # Convert meters to feet
                elevations = [result['elevation'] * 3.28084 for result in data['results']]
                stats['points_with_data'] += len(elevations)
                
                if verbose and attempt > 0:
                    print("Success!")
                
                return elevations
            
            # Other error
            if verbose:
                print(f"Error: Status {response.status_code}")
            
        except requests.exceptions.Timeout:
            stats['retries'] += 1
            if verbose:
                print(f"Timeout on attempt {attempt + 1}/{RETRY_ATTEMPTS}")
            
            if attempt < RETRY_ATTEMPTS - 1:
                wait_time = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
                time.sleep(wait_time)
                continue
                
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            break
    
    # Failed after all retries
    stats['failed_requests'] += 1
    return [0.0] * len(coordinates)


def get_elevations_for_trail(coordinates: List[Tuple[float, float]], 
                             verbose: bool = False) -> List[float]:
    """
    Get elevations for all points of a trail with optimal batching.
    
    Args:
        coordinates: List of (latitude, longitude) tuples
        verbose: Print progress messages
        
    Returns:
        List of elevations in feet (same order as input)
    """
    if not coordinates:
        return []
    
    stats['total_points'] += len(coordinates)
    all_elevations = []
    
    # Process in batches
    num_batches = (len(coordinates) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(coordinates), BATCH_SIZE):
        batch = coordinates[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        if verbose:
            print(f"  Batch {batch_num}/{num_batches} ({len(batch)} points)...", end=' ')
        
        elevations = get_elevations_batch(batch, verbose=False)
        all_elevations.extend(elevations)
        
        if verbose:
            success_count = sum([1 for e in elevations if e > 0])
            print(f"✓ ({success_count}/{len(batch)} successful)")
        
        # Small delay between batches to avoid rate limiting
        if i + BATCH_SIZE < len(coordinates):
            time.sleep(BASE_DELAY)
    
    return all_elevations


def calculate_elevation_gain(elevations: List[float]) -> float:
    """
    Calculate total elevation gain from elevation profile.
    Only counts uphill sections (positive changes).
    
    Args:
        elevations: List of elevations in feet
        
    Returns:
        Total elevation gain in feet
    """
    if not elevations or len(elevations) < 2:
        return 0.0
    
    total_gain = 0.0
    for i in range(len(elevations) - 1):
        change = elevations[i + 1] - elevations[i]
        if change > 0:  # Only count uphill
            total_gain += change
    
    return total_gain


def get_elevation_for_point(lat: float, lon: float, 
                            verbose: bool = False) -> float:
    """
    Get elevation for a single point (with retry logic).
    For compatibility with old code.
    
    Args:
        lat: Latitude
        lon: Longitude
        verbose: Print messages
        
    Returns:
        Elevation in feet (0.0 if failed)
    """
    elevations = get_elevations_for_trail([(lat, lon)], verbose=verbose)
    return elevations[0] if elevations else 0.0


def print_stats():
    """Print statistics about elevation queries."""
    print("\n" + "=" * 60)
    print("ELEVATION QUERY STATISTICS")
    print("=" * 60)
    print(f"Total API requests:    {stats['total_requests']}")
    print(f"Successful:            {stats['successful_requests']}")
    print(f"Failed:                {stats['failed_requests']}")
    print(f"Retries:               {stats['retries']}")
    print(f"Total points queried:  {stats['total_points']}")
    print(f"Points with data:      {stats['points_with_data']}")
    
    if stats['total_points'] > 0:
        success_rate = (stats['points_with_data'] / stats['total_points']) * 100
        print(f"Success rate:          {success_rate:.1f}%")
    
    if stats['successful_requests'] > 0:
        avg_points = stats['points_with_data'] / stats['successful_requests']
        print(f"Avg points per request: {avg_points:.1f}")
    
    print("=" * 60)


def reset_stats():
    """Reset statistics counters."""
    for key in stats:
        stats[key] = 0


# Example usage and testing
if __name__ == "__main__":
    print("Testing Optimized Elevation Module")
    print("=" * 60)
    
    # Test 1: Single point
    print("\nTest 1: Single point")
    elevation = get_elevation_for_point(35.0651, -82.7339, verbose=True)
    print(f"Elevation: {elevation:.1f} feet")
    
    # Test 2: Small batch
    print("\nTest 2: Small batch (10 points)")
    coords = [(35.0 + i*0.01, -82.0) for i in range(10)]
    elevations = get_elevations_for_trail(coords, verbose=True)
    print(f"Elevations: {[f'{e:.1f}' for e in elevations]}")
    gain = calculate_elevation_gain(elevations)
    print(f"Total gain: {gain:.1f} feet")
    
    # Test 3: Large batch
    print("\nTest 3: Large batch (250 points - will split into batches)")
    coords = [(35.0 + i*0.001, -82.0) for i in range(250)]
    print(f"Fetching {len(coords)} elevations...")
    start_time = time.time()
    elevations = get_elevations_for_trail(coords, verbose=True)
    elapsed = time.time() - start_time
    gain = calculate_elevation_gain(elevations)
    print(f"Completed in {elapsed:.1f} seconds")
    print(f"Total gain: {gain:.1f} feet")
    
    # Print statistics
    print_stats()
    
    # Calculate speedup
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    old_method_time = len(coords) * 2.0  # ~2 seconds per point
    print(f"Old method (1 point/request): ~{old_method_time:.0f} seconds")
    print(f"New method (batch requests):   {elapsed:.1f} seconds")
    speedup = old_method_time / elapsed
    print(f"Speedup: {speedup:.1f}x faster!")
    print("=" * 60)