"""
Vehicle module for FlyingCarRL

Provides eVTOL vehicle classes, configurations, physics, and spawning systems.
"""

from .evtol_config import (
    VehicleType,
    VehicleBlueprint,
    SMALL_EVTOL,
    MEDIUM_EVTOL,
    LARGE_EVTOL,
    VEHICLE_REGISTRY,
    get_vehicle_blueprint,
    get_vehicle_blueprint_by_name
)

from .physics import (
    AerodynamicsModel,
    WindTurbulenceModel,
    BatteryModel,
    WindConditions,
    AIR_DENSITY,
    GRAVITY
)

from .evtol import eVTOL

from .spawner import (
    VehicleSpawner,
    SpawnPattern
)

__all__ = [
    # Vehicle types and configs
    'VehicleType',
    'VehicleBlueprint',
    'SMALL_EVTOL',
    'MEDIUM_EVTOL',
    'LARGE_EVTOL',
    'VEHICLE_REGISTRY',
    'get_vehicle_blueprint',
    'get_vehicle_blueprint_by_name',

    # Physics
    'AerodynamicsModel',
    'WindTurbulenceModel',
    'BatteryModel',
    'WindConditions',
    'AIR_DENSITY',
    'GRAVITY',

    # Vehicle class
    'eVTOL',

    # Spawning
    'VehicleSpawner',
    'SpawnPattern',
]
