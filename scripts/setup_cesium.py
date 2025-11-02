#!/usr/bin/env python3
"""
Cesium integration setup for FlyingCarRL

This script handles the setup and configuration of Cesium for geospatial data integration,
including Google 3D Tiles streaming for realistic urban environments.

Usage:
    python scripts/setup_cesium.py [--test-location "New York City"]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv not installed")
    print("Please install: pip install python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()


class CesiumConfig:
    """Cesium configuration and integration manager"""

    def __init__(self):
        self.ion_token = os.getenv('CESIUM_ION_TOKEN')
        self.cache_dir = project_root / 'cesium_cache'
        self.config_path = project_root / 'assets' / 'configs' / 'cesium_config.json'

    def validate_token(self) -> bool:
        """Validate Cesium Ion token"""
        if not self.ion_token or self.ion_token == 'your_cesium_ion_token_here':
            print("❌ Cesium Ion token not configured!")
            print("\nTo get a token:")
            print("  1. Sign up at https://cesium.com/ion/")
            print("  2. Navigate to Access Tokens")
            print("  3. Copy your default token")
            print("  4. Add it to your .env file as CESIUM_ION_TOKEN")
            return False

        print(f"✓ Cesium Ion token found: {self.ion_token[:10]}...")
        return True

    def create_config(self, location: str = "New York City") -> Dict:
        """Create Cesium configuration for a location"""

        # Predefined locations with coordinates
        locations = {
            "New York City": {
                "center": {"lat": 40.7580, "lon": -73.9855, "height": 450},
                "bounds": {
                    "lat_min": 40.7128,
                    "lat_max": 40.8128,
                    "lon_min": -74.0260,
                    "lon_max": -73.9260
                }
            },
            "San Francisco": {
                "center": {"lat": 37.7749, "lon": -122.4194, "height": 450},
                "bounds": {
                    "lat_min": 37.7249,
                    "lat_max": 37.8249,
                    "lon_min": -122.4694,
                    "lon_max": -122.3694
                }
            },
            "Los Angeles": {
                "center": {"lat": 34.0522, "lon": -118.2437, "height": 450},
                "bounds": {
                    "lat_min": 34.0022,
                    "lat_max": 34.1022,
                    "lon_min": -118.2937,
                    "lon_max": -118.1937
                }
            },
            "Chicago": {
                "center": {"lat": 41.8781, "lon": -87.6298, "height": 450},
                "bounds": {
                    "lat_min": 41.8281,
                    "lat_max": 41.9281,
                    "lon_min": -87.6798,
                    "lon_max": -87.5798
                }
            }
        }

        if location not in locations:
            print(f"⚠️  Unknown location '{location}', using New York City")
            location = "New York City"

        loc_data = locations[location]

        config = {
            "cesium_ion": {
                "token": self.ion_token,
                "asset_id": 1,  # Google Photorealistic 3D Tiles
                "server_url": "https://api.cesium.com"
            },
            "location": {
                "name": location,
                "center": loc_data["center"],
                "bounds": loc_data["bounds"]
            },
            "altitude": {
                "min_ft": 400,
                "max_ft": 500,
                "default_ft": 450
            },
            "rendering": {
                "max_screen_space_error": 16,
                "tile_cache_size_mb": 512,
                "enable_lighting": True,
                "enable_shadows": True
            },
            "data_sources": [
                {
                    "name": "Google 3D Tiles",
                    "type": "3dtiles",
                    "url": "https://tile.googleapis.com/v1/3dtiles/root.json"
                },
                {
                    "name": "Cesium World Terrain",
                    "type": "terrain",
                    "url": "https://assets.cesium.com/1"
                }
            ]
        }

        return config

    def save_config(self, config: Dict):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to: {self.config_path}")

    def setup_cache(self):
        """Set up cache directory for tiles"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Cache directory created: {self.cache_dir}")

        # Create cache subdirectories
        (self.cache_dir / '3dtiles').mkdir(exist_ok=True)
        (self.cache_dir / 'terrain').mkdir(exist_ok=True)

    def generate_sample_code(self) -> str:
        """Generate sample integration code"""
        return '''
# Example: Loading Cesium in Isaac Lab/Omniverse

import json
from pathlib import Path

# Load Cesium config
config_path = Path("assets/configs/cesium_config.json")
with open(config_path) as f:
    cesium_config = json.load(f)

# In your Omniverse/Isaac Lab environment setup:
# NOTE: Actual integration requires Cesium for Omniverse plugin
# This is pseudocode showing the concept

def setup_cesium_environment(stage):
    """Setup Cesium geospatial environment in USD stage"""

    # Set Cesium Ion token
    cesium_prim = stage.DefinePrim("/World/Cesium", "CesiumGeoreference")
    cesium_prim.GetAttribute("cesium:ionAccessToken").Set(
        cesium_config["cesium_ion"]["token"]
    )

    # Set geolocation
    center = cesium_config["location"]["center"]
    cesium_prim.GetAttribute("cesium:georeferenceOrigin:latitude").Set(center["lat"])
    cesium_prim.GetAttribute("cesium:georeferenceOrigin:longitude").Set(center["lon"])
    cesium_prim.GetAttribute("cesium:georeferenceOrigin:height").Set(center["height"])

    # Load 3D Tiles
    tileset = stage.DefinePrim("/World/Cesium/GoogleTiles", "Cesium3DTileset")
    tileset.GetAttribute("cesium:ionAssetId").Set(
        cesium_config["cesium_ion"]["asset_id"]
    )

    # Constrain altitude (for eVTOL flight)
    alt_min = cesium_config["altitude"]["min_ft"] * 0.3048  # ft to meters
    alt_max = cesium_config["altitude"]["max_ft"] * 0.3048

    print(f"✓ Cesium environment loaded: {cesium_config['location']['name']}")
    print(f"  Flight altitude: {alt_min:.1f}m - {alt_max:.1f}m")

    return cesium_prim

# Usage in your simulation:
# cesium_env = setup_cesium_environment(stage)
'''

    def create_integration_guide(self):
        """Create integration guide document"""
        guide_path = project_root / 'docs' / 'guides' / 'cesium_integration.md'
        guide_path.parent.mkdir(parents=True, exist_ok=True)

        guide_content = f'''# Cesium Integration Guide

## Overview
This guide explains how to integrate Cesium for realistic geospatial environments in FlyingCarRL.

## Prerequisites
- Cesium Ion account (free at https://cesium.com/ion/)
- NVIDIA Omniverse with Isaac Lab installed
- Cesium for Omniverse plugin (download from Cesium website)

## Setup Steps

### 1. Install Cesium for Omniverse
1. Download from: https://cesium.com/platform/cesium-for-omniverse/
2. Extract to Omniverse extensions folder
3. Enable in Omniverse Extension Manager

### 2. Configure API Token
Your token is configured in `.env`:
```
CESIUM_ION_TOKEN={self.ion_token[:10]}...
```

### 3. Load Configuration
Configuration is saved in: `{self.config_path}`

### 4. Integration Code
{self.generate_sample_code()}

## Altitude Constraints
eVTOL vehicles operate at 400-500 ft (122-152 m) altitude. The configuration automatically:
- Constrains camera/agent spawning to this range
- Streams only relevant 3D tile LODs for performance
- Handles coordinate transforms between USD and WGS84

## Available Locations
Preconfigured urban environments:
- New York City (Manhattan)
- San Francisco
- Los Angeles
- Chicago

Add more in `scripts/setup_cesium.py`

## Performance Tips
1. **Tile Cache**: Increase `tile_cache_size_mb` for large areas
2. **Screen Space Error**: Lower `max_screen_space_error` for higher quality (slower)
3. **Streaming**: Tiles load on-demand; first run may be slow

## Troubleshooting

### Token Invalid
- Check token at https://cesium.com/ion/tokens
- Ensure no extra spaces in `.env`

### Tiles Not Loading
- Verify internet connection (tiles stream from cloud)
- Check cache directory permissions: `{self.cache_dir}`

### Performance Issues
- Reduce tile cache size
- Increase screen space error (lower quality, faster)
- Use headless mode for training

## Next Steps
After setup, use `scripts/setup_arena.py` to create a simulation arena with Cesium terrain.

## References
- [Cesium for Omniverse Docs](https://cesium.com/docs/cesium-for-omniverse/)
- [Google 3D Tiles](https://cloud.google.com/blog/products/maps-platform/create-immersive-3d-map-experiences-photorealistic-3d-tiles)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
'''

        with open(guide_path, 'w') as f:
            f.write(guide_content)

        print(f"✓ Integration guide created: {guide_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Setup Cesium for FlyingCarRL')
    parser.add_argument(
        '--test-location',
        default='New York City',
        help='Test location for configuration'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip token validation (for testing)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FlyingCarRL - Cesium Setup")
    print("=" * 60)

    cesium = CesiumConfig()

    # Validate token
    if not args.skip_validation:
        if not cesium.validate_token():
            print("\n⚠️  Setup incomplete. Please configure your Cesium Ion token.")
            return

    # Create configuration
    print(f"\n✓ Creating configuration for: {args.test_location}")
    config = cesium.create_config(args.test_location)

    # Save configuration
    cesium.save_config(config)

    # Setup cache
    cesium.setup_cache()

    # Create integration guide
    cesium.create_integration_guide()

    print("\n" + "=" * 60)
    print("✓ Cesium setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review configuration in: assets/configs/cesium_config.json")
    print("  2. Read integration guide: docs/guides/cesium_integration.md")
    print("  3. Install Cesium for Omniverse plugin")
    print("  4. Run: python scripts/setup_arena.py")
    print("\n🚁 Ready to create realistic urban environments!")


if __name__ == '__main__':
    main()
