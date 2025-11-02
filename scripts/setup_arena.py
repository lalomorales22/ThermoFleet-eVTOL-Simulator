#!/usr/bin/env python3
"""
Arena setup script for FlyingCarRL

This script creates and configures simulation arenas with urban terrain,
buildings, and environmental elements for eVTOL training.

Usage:
    python scripts/setup_arena.py --name "NYC_Arena" --location "New York City"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

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


class ArenaBuilder:
    """Build and configure simulation arenas"""

    def __init__(self, name: str, location: str = "New York City"):
        self.name = name
        self.location = location
        self.config_dir = project_root / 'assets' / 'configs' / 'arenas'
        self.scene_dir = project_root / 'assets' / 'scenes'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.scene_dir.mkdir(parents=True, exist_ok=True)

    def generate_arena_config(self) -> Dict:
        """Generate arena configuration"""

        # Predefined locations
        locations = {
            "New York City": {
                "bounds": {
                    "lat_min": 40.7128, "lat_max": 40.8128,
                    "lon_min": -74.0260, "lon_max": -73.9260
                },
                "spawn_zones": [
                    {"lat": 40.7580, "lon": -73.9855, "alt_ft": 450},  # Times Square
                    {"lat": 40.7614, "lon": -73.9776, "alt_ft": 450},  # Central Park South
                    {"lat": 40.7489, "lon": -73.9680, "alt_ft": 450},  # UN Headquarters
                ],
                "no_fly_zones": [
                    {"lat": 40.7484, "lon": -73.9857, "radius_m": 500, "reason": "Empire State"},
                ]
            },
            "San Francisco": {
                "bounds": {
                    "lat_min": 37.7249, "lat_max": 37.8249,
                    "lon_min": -122.4694, "lon_max": -122.3694
                },
                "spawn_zones": [
                    {"lat": 37.7749, "lon": -122.4194, "alt_ft": 450},  # Downtown
                    {"lat": 37.8024, "lon": -122.4058, "alt_ft": 450},  # Fisherman's Wharf
                ],
                "no_fly_zones": [
                    {"lat": 37.7749, "lon": -122.4194, "radius_m": 300, "reason": "Downtown"},
                ]
            },
            "Los Angeles": {
                "bounds": {
                    "lat_min": 34.0022, "lat_max": 34.1022,
                    "lon_min": -118.2937, "lon_max": -118.1937
                },
                "spawn_zones": [
                    {"lat": 34.0522, "lon": -118.2437, "alt_ft": 450},  # Downtown
                ],
                "no_fly_zones": []
            }
        }

        if self.location not in locations:
            print(f"⚠️  Unknown location '{self.location}', using New York City")
            self.location = "New York City"

        loc_data = locations[self.location]

        config = {
            "metadata": {
                "name": self.name,
                "location": self.location,
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            },
            "bounds": loc_data["bounds"],
            "altitude": {
                "min_ft": 400,
                "max_ft": 500,
                "spawn_default_ft": 450
            },
            "spawn_zones": loc_data["spawn_zones"],
            "no_fly_zones": loc_data["no_fly_zones"],
            "environment": {
                "time_of_day": "noon",  # dawn, noon, dusk, night
                "weather": {
                    "condition": "clear",  # clear, cloudy, rainy, foggy
                    "wind_speed_mph": 10,
                    "wind_direction_deg": 270,
                    "temperature_f": 70,
                    "visibility_miles": 10
                },
                "lighting": {
                    "sun_intensity": 1.0,
                    "ambient_intensity": 0.3,
                    "enable_shadows": True
                }
            },
            "physics": {
                "gravity_m_s2": -9.81,
                "air_density_kg_m3": 1.225,
                "wind_turbulence_enabled": True,
                "turbulence_intensity": 0.1,  # 0-1 scale
                "timestep_s": 0.01
            },
            "obstacles": {
                "buildings_from_cesium": True,
                "procedural_buildings": False,
                "dynamic_obstacles": {
                    "enabled": False,
                    "birds": {"count": 0, "spawn_rate": 0},
                    "drones": {"count": 0, "spawn_rate": 0}
                }
            },
            "sensors": {
                "enable_cameras": True,
                "enable_lidar": True,
                "enable_imu": True,
                "enable_gps": True,
                "gps_noise_std_m": 5.0,
                "imu_noise_std": 0.01
            },
            "safety": {
                "collision_detection": True,
                "altitude_enforcement": True,
                "geofencing_enabled": True,
                "emergency_landing_zones": []
            }
        }

        return config

    def save_config(self, config: Dict):
        """Save arena configuration"""
        config_path = self.config_dir / f"{self.name}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Arena config saved: {config_path}")
        return config_path

    def generate_usd_scene(self) -> str:
        """Generate USD scene description (pseudocode for now)"""

        usd_content = f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    doc = """FlyingCarRL Arena: {self.name}"""
)

def Xform "World"
{{
    # Cesium Georeference
    def CesiumGeoreference "Cesium"
    {{
        double3 cesium:georeferenceOrigin:latitude = 40.7580
        double3 cesium:georeferenceOrigin:longitude = -73.9855
        double3 cesium:georeferenceOrigin:height = 137.0

        # Google 3D Tiles
        def Cesium3DTileset "GoogleTiles"
        {{
            int cesium:ionAssetId = 1
            string cesium:ionAccessToken = "{os.getenv('CESIUM_ION_TOKEN', 'YOUR_TOKEN')}"
        }}
    }}

    # Lighting
    def DistantLight "SunLight"
    {{
        float intensity = 1000
        float3 xformOp:rotateXYZ = (315, 45, 0)
    }}

    # Physics Scene
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}

    # Flight Zone (400-500 ft altitude constraint)
    def Cube "FlightZoneBounds"
    {{
        float3 xformOp:scale = (10000, 10000, 30.48)  # 100ft height in meters
        double3 xformOp:translate = (0, 0, 137)  # ~450ft in meters
        bool visibility = false  # Hidden, just for reference
    }}

    # Spawn Points
    def Scope "SpawnZones"
    {{
        def Xform "SpawnPoint_0"
        {{
            double3 xformOp:translate = (0, 0, 137)
        }}
    }}
}}
'''

        scene_path = self.scene_dir / f"{self.name}.usda"
        with open(scene_path, 'w') as f:
            f.write(usd_content)

        print(f"✓ USD scene created: {scene_path}")
        return str(scene_path)

    def generate_python_loader(self) -> str:
        """Generate Python script to load the arena in Isaac Lab"""

        loader_content = f'''"""
Arena loader for {self.name}

This script loads the arena configuration and USD scene into Isaac Lab/Omniverse.
"""

import json
from pathlib import Path
from omni.isaac.kit import SimulationApp

# Configuration
ARENA_NAME = "{self.name}"
CONFIG_PATH = Path(__file__).parent.parent / "assets/configs/arenas/{self.name}.json"
SCENE_PATH = Path(__file__).parent.parent / "assets/scenes/{self.name}.usda"

def load_arena_config():
    """Load arena configuration"""
    with open(CONFIG_PATH) as f:
        return json.load(f)

def setup_arena(simulation_app: SimulationApp):
    """Setup arena in Isaac Lab"""

    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import open_stage

    print(f"Loading arena: {{ARENA_NAME}}")

    # Load configuration
    config = load_arena_config()

    # Open USD scene
    open_stage(str(SCENE_PATH))

    # Create world
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config['physics']['timestep_s'],
        rendering_dt=config['physics']['timestep_s']
    )

    # Configure physics
    from omni.isaac.core.utils.prims import get_prim_at_path
    physics_scene = get_prim_at_path("/World/PhysicsScene")

    # Set gravity
    gravity = config['physics']['gravity_m_s2']
    world.get_physics_context().set_gravity(gravity)

    # Configure environment
    env_config = config['environment']
    print(f"  Time: {{env_config['time_of_day']}}")
    print(f"  Weather: {{env_config['weather']['condition']}}")
    print(f"  Wind: {{env_config['weather']['wind_speed_mph']}} mph")

    # Altitude constraints
    alt_min_m = config['altitude']['min_ft'] * 0.3048
    alt_max_m = config['altitude']['max_ft'] * 0.3048
    print(f"  Flight zone: {{alt_min_m:.1f}}m - {{alt_max_m:.1f}}m")

    # No-fly zones
    no_fly = config['no_fly_zones']
    if no_fly:
        print(f"  No-fly zones: {{len(no_fly)}}")

    print(f"✓ Arena '{{ARENA_NAME}}' loaded successfully")

    return world, config

# Example usage:
# simulation_app = SimulationApp({{"headless": False}})
# world, config = setup_arena(simulation_app)
# world.reset()
'''

        loader_path = self.scene_dir / f"load_{self.name}.py"
        with open(loader_path, 'w') as f:
            f.write(loader_content)

        print(f"✓ Python loader created: {loader_path}")
        return str(loader_path)

    def validate_cesium_config(self) -> bool:
        """Check if Cesium is configured"""
        cesium_config_path = project_root / 'assets' / 'configs' / 'cesium_config.json'

        if not cesium_config_path.exists():
            print("⚠️  Cesium not configured yet")
            print("   Run: python scripts/setup_cesium.py")
            return False

        with open(cesium_config_path) as f:
            config = json.load(f)
            token = config.get('cesium_ion', {}).get('token', '')

            if not token or token == 'your_cesium_ion_token_here':
                print("⚠️  Cesium Ion token not set")
                return False

        print("✓ Cesium configuration found")
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Setup simulation arena for FlyingCarRL')
    parser.add_argument(
        '--name',
        required=True,
        help='Arena name (e.g., NYC_Arena)'
    )
    parser.add_argument(
        '--location',
        default='New York City',
        help='Location name (e.g., New York City, San Francisco)'
    )
    parser.add_argument(
        '--skip-usd',
        action='store_true',
        help='Skip USD scene generation (config only)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FlyingCarRL - Arena Setup")
    print("=" * 60)

    # Create arena builder
    builder = ArenaBuilder(args.name, args.location)

    # Validate Cesium setup
    builder.validate_cesium_config()

    # Generate configuration
    print(f"\n✓ Creating arena: {args.name}")
    print(f"  Location: {args.location}")

    config = builder.generate_arena_config()
    config_path = builder.save_config(config)

    # Generate USD scene
    if not args.skip_usd:
        scene_path = builder.generate_usd_scene()
        loader_path = builder.generate_python_loader()

    print("\n" + "=" * 60)
    print("✓ Arena setup complete!")
    print("=" * 60)
    print("\nFiles created:")
    print(f"  Config: {config_path}")
    if not args.skip_usd:
        print(f"  Scene:  {scene_path}")
        print(f"  Loader: {loader_path}")

    print("\nArena details:")
    print(f"  Bounds: {config['bounds']}")
    print(f"  Altitude: {config['altitude']['min_ft']}-{config['altitude']['max_ft']} ft")
    print(f"  Spawn zones: {len(config['spawn_zones'])}")
    print(f"  No-fly zones: {len(config['no_fly_zones'])}")

    print("\nNext steps:")
    print("  1. Review arena config in assets/configs/arenas/")
    print("  2. Load in Omniverse: Open the .usda file")
    print("  3. Or use Python loader: python assets/scenes/load_*.py")
    print("\n🚁 Ready to spawn eVTOL vehicles!")


if __name__ == '__main__':
    main()
