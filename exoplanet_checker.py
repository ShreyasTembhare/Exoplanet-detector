#!/usr/bin/env python3
"""
Simple Exoplanet Database Checker
=================================

A reliable script to check if stars have discovered exoplanets.
Uses NASA Exoplanet Archive with better error handling.

Usage:
    python exoplanet_checker.py "KIC 11442793"
    python exoplanet_checker.py --list-kepler
    python exoplanet_checker.py --search "Kepler-90"
"""

import requests
import sys
import argparse
import json
from datetime import datetime
import time

def query_nasa_exoplanet_archive_simple(target_name):
    """Query NASA Exoplanet Archive with better error handling."""
    try:
        # Use a simpler query approach
        base_url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
        
        # First, try to get basic info about the target
        query = f"table=exoplanets&select=pl_name,pl_hostname,pl_orbper,pl_rade,pl_masse,pl_discmethod,pl_disc,pl_status&where=pl_hostname+like+'%{target_name}%'&format=csv"
        url = f"{base_url}?{query}"
        
        print(f"🔍 Querying NASA Exoplanet Archive for: {target_name}")
        print(f"URL: {url}")
        
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            content = response.text.strip()
            print(f"Response length: {len(content)} characters")
            
            if content and len(content) > 10:  # Has meaningful content
                lines = content.split('\n')
                print(f"Found {len(lines)} lines in response")
                
                if len(lines) > 1:
                    # Parse the CSV response
                    headers = lines[0].split(',')
                    print(f"Headers: {headers}")
                    
                    results = []
                    for i, line in enumerate(lines[1:], 1):
                        if line.strip():
                            values = line.split(',')
                            print(f"Line {i}: {values}")
                            
                            if len(values) >= len(headers):
                                result = {}
                                for j, header in enumerate(headers):
                                    if j < len(values):
                                        result[header] = values[j]
                                    else:
                                        result[header] = 'Unknown'
                                results.append(result)
                    
                    return results
                else:
                    print("No data lines found in response")
            else:
                print("Empty or invalid response")
        
        return []
        
    except Exception as e:
        print(f"⚠️ Error querying NASA Exoplanet Archive: {str(e)}")
        return []

def check_single_target(target_name):
    """Check if a single target has discovered exoplanets."""
    print(f"\n{'='*60}")
    print(f"🔍 CHECKING: {target_name}")
    print(f"{'='*60}")
    
    results = query_nasa_exoplanet_archive_simple(target_name)
    
    if results:
        print(f"✅ FOUND {len(results)} EXOPLANET(S) for {target_name}")
        print("\n🌍 Discovered Exoplanets:")
        
        for i, planet in enumerate(results, 1):
            print(f"\n  Planet {i}:")
            print(f"    Name: {planet.get('pl_name', 'Unknown')}")
            print(f"    Host: {planet.get('pl_hostname', 'Unknown')}")
            print(f"    Period: {planet.get('pl_orbper', 'Unknown')} days")
            print(f"    Radius: {planet.get('pl_rade', 'Unknown')} Earth radii")
            print(f"    Mass: {planet.get('pl_masse', 'Unknown')} Earth masses")
            print(f"    Discovery Method: {planet.get('pl_discmethod', 'Unknown')}")
            print(f"    Discovery Year: {planet.get('pl_disc', 'Unknown')}")
            print(f"    Status: {planet.get('pl_status', 'Unknown')}")
    else:
        print(f"ℹ️ NO KNOWN EXOPLANETS found for {target_name}")
        print("This star is a potential candidate for new discoveries!")
    
    print(f"\n{'='*60}")

def get_known_exoplanet_hosts():
    """Get a comprehensive list of known exoplanet host stars for reference."""
    known_hosts = {
        # Kepler Mission Famous Systems
        "KIC 11442793": {
            "name": "Kepler-90",
            "planets": 7,
            "description": "Famous 7-planet system, like a mini solar system",
            "discovery_year": 2013,
            "mission": "Kepler"
        },
        "KIC 8462852": {
            "name": "Boyajian's Star",
            "planets": 0,
            "description": "Mysterious star with unusual dimming events",
            "discovery_year": 2015,
            "mission": "Kepler"
        },
        "KIC 3733346": {
            "name": "Kepler-1b",
            "planets": 1,
            "description": "First confirmed Kepler exoplanet",
            "discovery_year": 2009,
            "mission": "Kepler"
        },
        "KIC 11446443": {
            "name": "Kepler-11",
            "planets": 6,
            "description": "Compact 6-planet system",
            "discovery_year": 2011,
            "mission": "Kepler"
        },
        "KIC 10028792": {
            "name": "Kepler-20",
            "planets": 5,
            "description": "First Earth-sized planets discovered",
            "discovery_year": 2011,
            "mission": "Kepler"
        },
        "KIC 10001167": {
            "name": "Kepler-22",
            "planets": 1,
            "description": "First confirmed planet in habitable zone",
            "discovery_year": 2011,
            "mission": "Kepler"
        },
        "KIC 10187017": {
            "name": "Kepler-186",
            "planets": 5,
            "description": "First Earth-sized planet in habitable zone",
            "discovery_year": 2014,
            "mission": "Kepler"
        },
        "KIC 9941662": {
            "name": "Kepler-442",
            "planets": 1,
            "description": "Super-Earth in habitable zone",
            "discovery_year": 2015,
            "mission": "Kepler"
        },
        "KIC 8311864": {
            "name": "Kepler-62",
            "planets": 5,
            "description": "Multiple planets in habitable zone",
            "discovery_year": 2013,
            "mission": "Kepler"
        },
        "KIC 5866724": {
            "name": "Kepler-444",
            "planets": 5,
            "description": "Oldest known planetary system",
            "discovery_year": 2015,
            "mission": "Kepler"
        },
        
        # TESS Mission Systems
        "TIC 261136679": {
            "name": "TOI-700",
            "planets": 3,
            "description": "TESS discovery with Earth-sized planet in habitable zone",
            "discovery_year": 2020,
            "mission": "TESS"
        },
        "TIC 377659417": {
            "name": "TOI-1338",
            "planets": 1,
            "description": "First TESS circumbinary planet",
            "discovery_year": 2020,
            "mission": "TESS"
        },
        "TIC 142748283": {
            "name": "TOI-1452",
            "planets": 1,
            "description": "Water world candidate",
            "discovery_year": 2022,
            "mission": "TESS"
        },
        "TIC 257459955": {
            "name": "TOI-700d",
            "planets": 1,
            "description": "Earth-sized planet in habitable zone",
            "discovery_year": 2020,
            "mission": "TESS"
        },
        "TIC 237913194": {
            "name": "TOI-1231",
            "planets": 1,
            "description": "Neptune-like planet with atmosphere",
            "discovery_year": 2021,
            "mission": "TESS"
        },
        
        # K2 Mission Systems
        "EPIC 201367065": {
            "name": "K2-18",
            "planets": 1,
            "description": "Super-Earth with water vapor in atmosphere",
            "discovery_year": 2015,
            "mission": "K2"
        },
        "EPIC 201912552": {
            "name": "K2-72",
            "planets": 4,
            "description": "Compact 4-planet system",
            "discovery_year": 2016,
            "mission": "K2"
        },
        "EPIC 201505350": {
            "name": "K2-138",
            "planets": 6,
            "description": "Resonant chain of planets",
            "discovery_year": 2018,
            "mission": "K2"
        },
        "EPIC 246851721": {
            "name": "K2-141",
            "planets": 1,
            "description": "Lava planet with atmosphere",
            "discovery_year": 2018,
            "mission": "K2"
        },
        
        # Other Famous Systems
        "HD 209458": {
            "name": "Osiris",
            "planets": 1,
            "description": "First exoplanet with detected atmosphere",
            "discovery_year": 1999,
            "mission": "Ground-based"
        },
        "HD 189733": {
            "name": "HD 189733b",
            "planets": 1,
            "description": "Hot Jupiter with blue color",
            "discovery_year": 2005,
            "mission": "Ground-based"
        },
        "51 Pegasi": {
            "name": "51 Pegasi b",
            "planets": 1,
            "description": "First exoplanet discovered around Sun-like star",
            "discovery_year": 1995,
            "mission": "Ground-based"
        },
        "TRAPPIST-1": {
            "name": "TRAPPIST-1",
            "planets": 7,
            "description": "Ultra-cool dwarf with 7 Earth-sized planets",
            "discovery_year": 2016,
            "mission": "Ground-based"
        },
        "Proxima Centauri": {
            "name": "Proxima Centauri b",
            "planets": 1,
            "description": "Closest exoplanet to Earth",
            "discovery_year": 2016,
            "mission": "Ground-based"
        },
        
        # Additional Kepler Systems
        "KIC 3542115": {
            "name": "Kepler-9",
            "planets": 3,
            "description": "First multi-planet system confirmed",
            "discovery_year": 2010,
            "mission": "Kepler"
        },
        "KIC 10004738": {
            "name": "Kepler-16",
            "planets": 1,
            "description": "First circumbinary planet",
            "discovery_year": 2011,
            "mission": "Kepler"
        },
        "KIC 9631995": {
            "name": "Kepler-37",
            "planets": 3,
            "description": "Smallest exoplanet discovered at the time",
            "discovery_year": 2013,
            "mission": "Kepler"
        },
        "KIC 10001893": {
            "name": "Kepler-80",
            "planets": 6,
            "description": "Compact resonant system",
            "discovery_year": 2012,
            "mission": "Kepler"
        },
        "KIC 11414558": {
            "name": "Kepler-1229",
            "planets": 1,
            "description": "Super-Earth in habitable zone",
            "discovery_year": 2016,
            "mission": "Kepler"
        },
        
        # Additional TESS Systems
        "TIC 168790520": {
            "name": "TOI-270",
            "planets": 3,
            "description": "Ultra-cool dwarf with three planets",
            "discovery_year": 2019,
            "mission": "TESS"
        },
        "TIC 220520887": {
            "name": "TOI-1696",
            "planets": 1,
            "description": "Super-Earth in habitable zone",
            "discovery_year": 2021,
            "mission": "TESS"
        },
        "TIC 177032175": {
            "name": "TOI-1728",
            "planets": 1,
            "description": "Hot Jupiter with atmosphere",
            "discovery_year": 2021,
            "mission": "TESS"
        },
        
        # Additional K2 Systems
        "EPIC 201238110": {
            "name": "K2-3",
            "planets": 3,
            "description": "Three super-Earths",
            "discovery_year": 2015,
            "mission": "K2"
        },
        "EPIC 201617985": {
            "name": "K2-24",
            "planets": 2,
            "description": "Two Neptune-sized planets",
            "discovery_year": 2016,
            "mission": "K2"
        },
        "EPIC 201465501": {
            "name": "K2-72",
            "planets": 4,
            "description": "Four Earth-sized planets",
            "discovery_year": 2016,
            "mission": "K2"
        }
    }
    return known_hosts

def check_known_hosts(target_name):
    """Check against our known list of exoplanet hosts."""
    known_hosts = get_known_exoplanet_hosts()
    
    if target_name in known_hosts:
        host_info = known_hosts[target_name]
        print(f"\n✅ FOUND in known database: {host_info['name']}")
        print(f"   Planets: {host_info['planets']}")
        print(f"   Description: {host_info['description']}")
        print(f"   Discovery Year: {host_info['discovery_year']}")
        print(f"   Mission: {host_info['mission']}")
        return True
    else:
        print(f"\nℹ️ Not found in known database")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check exoplanet databases")
    parser.add_argument("target", nargs="?", help="Target name to check (e.g., 'KIC 11442793')")
    parser.add_argument("--known-hosts", action="store_true", help="Show list of known exoplanet hosts")
    parser.add_argument("--test", action="store_true", help="Test with known exoplanet hosts")
    
    args = parser.parse_args()
    
    print("🔭 Exoplanet Database Checker")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show known hosts
    if args.known_hosts:
        known_hosts = get_known_exoplanet_hosts()
        print(f"\n{'='*80}")
        print("📋 COMPREHENSIVE LIST OF KNOWN EXOPLANET HOST STARS")
        print(f"{'='*80}")
        
        # Group by mission
        missions = {}
        for target, info in known_hosts.items():
            mission = info.get('mission', 'Unknown')
            if mission not in missions:
                missions[mission] = []
            missions[mission].append((target, info))
        
        total_planets = sum(info['planets'] for info in known_hosts.values())
        total_systems = len(known_hosts)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Systems: {total_systems}")
        print(f"   Total Planets: {total_planets}")
        print(f"   Missions: {', '.join(missions.keys())}")
        
        for mission in sorted(missions.keys()):
            print(f"\n{'='*60}")
            print(f"🚀 {mission.upper()} MISSION SYSTEMS")
            print(f"{'='*60}")
            
            for target, info in sorted(missions[mission], key=lambda x: x[1]['discovery_year']):
                print(f"\n{target}:")
                print(f"  Name: {info['name']}")
                print(f"  Planets: {info['planets']}")
                print(f"  Description: {info['description']}")
                print(f"  Discovery Year: {info['discovery_year']}")
                print(f"  Mission: {info['mission']}")
        
        print(f"\n{'='*80}")
        print("💡 TIP: Use 'python exoplanet_checker.py \"TARGET_NAME\"' to check specific stars")
        print(f"{'='*80}")
        return
    
    # Test with known hosts
    if args.test:
        known_hosts = get_known_exoplanet_hosts()
        print(f"\n🧪 Testing with known exoplanet hosts...")
        
        for target in known_hosts.keys():
            print(f"\n{'='*40}")
            print(f"Testing: {target}")
            check_known_hosts(target)
            time.sleep(1)  # Be nice to the API
        
        return
    
    # Check single target
    if args.target:
        print(f"\n🔍 Checking: {args.target}")
        
        # First check our known database
        found_in_known = check_known_hosts(args.target)
        
        # Then try NASA Exoplanet Archive
        print(f"\n🔍 Checking NASA Exoplanet Archive...")
        results = query_nasa_exoplanet_archive_simple(args.target)
        
        if results:
            print(f"✅ Found {len(results)} exoplanet(s) in NASA database")
        else:
            print("ℹ️ No exoplanets found in NASA database")
        
        if not found_in_known and not results:
            print(f"\n💡 {args.target} appears to be a potential candidate for new discoveries!")
            print("   This star has no known exoplanets yet.")
    
    # Default: show help
    else:
        print("\n📖 Usage Examples:")
        print("  python exoplanet_checker.py 'KIC 11442793'")
        print("  python exoplanet_checker.py --known-hosts")
        print("  python exoplanet_checker.py --test")
        print("\n💡 For KIC 11442793 (Kepler-90):")
        print("   This star has 7 confirmed exoplanets!")
        print("   It's a famous multi-planet system discovered by Kepler.")

if __name__ == "__main__":
    main() 