"""
eVTOL Vehicle Configuration System

Defines blueprint configurations for different vehicle types:
- Small (100kg): Sphere-like, agile scout
- Medium (500kg): Winged, balanced passenger transport
- Large (1000kg): Boxy, heavy-lift cargo
"""

from dataclasses import dataclass
from typing import Tuple
from enum import Enum


class VehicleType(Enum):
    """Enumeration of available eVTOL vehicle types"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class VehicleBlueprint:
    """
    Blueprint configuration for an eVTOL vehicle.

    Attributes:
        name: Vehicle type name
        mass: Mass in kilograms
        shape: Geometric shape descriptor
        dimensions: (length, width, height) in meters
        thrust_vectors: Number of thrust vector points
        max_thrust: Maximum thrust in Newtons
        drag_coefficient: Aerodynamic drag coefficient (Cd)
        lift_coefficient: Aerodynamic lift coefficient (Cl)
        frontal_area: Frontal area in square meters for drag calculation
        wing_area: Wing/rotor area in square meters for lift calculation
        battery_capacity: Battery capacity in kWh
        power_consumption_base: Base power consumption in kW
        max_speed: Maximum speed in m/s
        max_angular_velocity: Maximum angular velocity in rad/s
    """
    name: str
    mass: float
    shape: str
    dimensions: Tuple[float, float, float]
    thrust_vectors: int
    max_thrust: float
    drag_coefficient: float
    lift_coefficient: float
    frontal_area: float
    wing_area: float
    battery_capacity: float
    power_consumption_base: float
    max_speed: float
    max_angular_velocity: float

    @property
    def weight(self) -> float:
        """Calculate weight force (N) from mass"""
        return self.mass * 9.81  # g = 9.81 m/s²

    @property
    def thrust_to_weight_ratio(self) -> float:
        """Calculate thrust-to-weight ratio"""
        return self.max_thrust / self.weight


# Small eVTOL: 100kg, sphere-like, agile scout
SMALL_EVTOL = VehicleBlueprint(
    name="Scout-S100",
    mass=100.0,  # kg
    shape="sphere",
    dimensions=(1.2, 1.2, 0.8),  # Compact, nearly spherical (L, W, H in meters)
    thrust_vectors=4,  # Quad-rotor configuration
    max_thrust=1500.0,  # 1500N (T/W ratio ~1.53 for agility)
    drag_coefficient=0.47,  # Sphere drag coefficient
    lift_coefficient=0.3,  # Minimal lift from body
    frontal_area=1.13,  # π * r² for sphere approximation
    wing_area=2.0,  # Small rotor disc area
    battery_capacity=5.0,  # 5 kWh
    power_consumption_base=2.0,  # 2 kW base consumption
    max_speed=30.0,  # 30 m/s (~67 mph)
    max_angular_velocity=3.0,  # 3 rad/s (very agile)
)

# Medium eVTOL: 500kg, winged, balanced passenger transport
MEDIUM_EVTOL = VehicleBlueprint(
    name="Passenger-M500",
    mass=500.0,  # kg
    shape="winged",
    dimensions=(4.5, 5.0, 1.8),  # Winged vehicle (L, W, H in meters)
    thrust_vectors=6,  # Hexa-rotor configuration
    max_thrust=6500.0,  # 6500N (T/W ratio ~1.32 for stability)
    drag_coefficient=0.25,  # Streamlined body
    lift_coefficient=1.2,  # Wings provide significant lift
    frontal_area=3.5,  # Streamlined frontal area
    wing_area=12.0,  # Wing + rotor area
    battery_capacity=30.0,  # 30 kWh
    power_consumption_base=8.0,  # 8 kW base consumption
    max_speed=50.0,  # 50 m/s (~112 mph)
    max_angular_velocity=1.5,  # 1.5 rad/s (moderate agility)
)

# Large eVTOL: 1000kg, boxy, heavy-lift cargo
LARGE_EVTOL = VehicleBlueprint(
    name="Cargo-L1000",
    mass=1000.0,  # kg
    shape="box",
    dimensions=(6.0, 4.0, 2.5),  # Boxy cargo vehicle (L, W, H in meters)
    thrust_vectors=8,  # Octo-rotor configuration for stability
    max_thrust=13000.0,  # 13000N (T/W ratio ~1.33 for heavy lift)
    drag_coefficient=0.8,  # Boxy shape has high drag
    lift_coefficient=0.5,  # Minimal lift, relies on thrust
    frontal_area=10.0,  # Large frontal area
    wing_area=18.0,  # Large rotor disc area
    battery_capacity=60.0,  # 60 kWh
    power_consumption_base=15.0,  # 15 kW base consumption
    max_speed=35.0,  # 35 m/s (~78 mph, slower due to mass/drag)
    max_angular_velocity=0.8,  # 0.8 rad/s (less agile, stable)
)


# Vehicle type registry
VEHICLE_REGISTRY = {
    VehicleType.SMALL: SMALL_EVTOL,
    VehicleType.MEDIUM: MEDIUM_EVTOL,
    VehicleType.LARGE: LARGE_EVTOL,
}


def get_vehicle_blueprint(vehicle_type: VehicleType) -> VehicleBlueprint:
    """
    Get vehicle blueprint by type.

    Args:
        vehicle_type: VehicleType enum value

    Returns:
        VehicleBlueprint configuration

    Raises:
        ValueError: If vehicle type not found in registry
    """
    if vehicle_type not in VEHICLE_REGISTRY:
        raise ValueError(f"Unknown vehicle type: {vehicle_type}")
    return VEHICLE_REGISTRY[vehicle_type]


def get_vehicle_blueprint_by_name(name: str) -> VehicleBlueprint:
    """
    Get vehicle blueprint by name string.

    Args:
        name: Vehicle type name ("small", "medium", or "large")

    Returns:
        VehicleBlueprint configuration

    Raises:
        ValueError: If vehicle type name is invalid
    """
    try:
        vehicle_type = VehicleType(name.lower())
        return get_vehicle_blueprint(vehicle_type)
    except ValueError:
        raise ValueError(f"Invalid vehicle type name: {name}. Must be one of: small, medium, large")
