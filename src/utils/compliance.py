"""
Compliance and safety features for FlyingCarRL.

This module implements virtual geofencing, no-fly zones, and simulated
FAA regulations for urban air mobility.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NoFlyZone:
    """
    Represents a no-fly zone (NFZ) in the simulation.

    Attributes:
        name: Zone identifier
        center: Center position [x, y, z]
        radius: Radius in meters
        altitude_range: Tuple of (min_altitude, max_altitude) or None for all altitudes
        zone_type: Type of zone ('airport', 'restricted', 'temporary', 'emergency')
        active: Whether the zone is currently active
    """
    name: str
    center: np.ndarray
    radius: float
    altitude_range: Optional[Tuple[float, float]] = None
    zone_type: str = 'restricted'
    active: bool = True

    def contains_point(self, position: np.ndarray) -> bool:
        """
        Check if a position is inside this no-fly zone.

        Args:
            position: Position [x, y, z]

        Returns:
            True if position is inside the zone
        """
        if not self.active:
            return False

        # Check horizontal distance
        horizontal_dist = np.linalg.norm(position[:2] - self.center[:2])
        if horizontal_dist > self.radius:
            return False

        # Check altitude if specified
        if self.altitude_range is not None:
            min_alt, max_alt = self.altitude_range
            altitude = position[2]
            if altitude < min_alt or altitude > max_alt:
                return False

        return True


@dataclass
class GeofenceCorridor:
    """
    Represents an approved flight corridor.

    Attributes:
        name: Corridor identifier
        waypoints: List of waypoint positions
        width: Corridor width in meters
        altitude_range: (min_altitude, max_altitude)
        max_speed: Maximum allowed speed in m/s
    """
    name: str
    waypoints: List[np.ndarray]
    width: float
    altitude_range: Tuple[float, float]
    max_speed: float = 30.0  # m/s

    def distance_to_corridor(self, position: np.ndarray) -> float:
        """
        Calculate minimum distance from position to corridor.

        Args:
            position: Position [x, y, z]

        Returns:
            Distance in meters (0 if inside corridor)
        """
        # Check altitude
        altitude = position[2]
        min_alt, max_alt = self.altitude_range
        if altitude < min_alt or altitude > max_alt:
            return float('inf')

        # Find closest point on corridor
        min_dist = float('inf')

        for i in range(len(self.waypoints) - 1):
            # Distance to line segment
            p1 = self.waypoints[i][:2]
            p2 = self.waypoints[i + 1][:2]
            pos_2d = position[:2]

            # Vector from p1 to p2
            segment = p2 - p1
            segment_length = np.linalg.norm(segment)

            if segment_length == 0:
                dist = np.linalg.norm(pos_2d - p1)
            else:
                # Project position onto segment
                t = np.clip(np.dot(pos_2d - p1, segment) / (segment_length ** 2), 0, 1)
                projection = p1 + t * segment
                dist = np.linalg.norm(pos_2d - projection)

            min_dist = min(min_dist, dist)

        # Return distance beyond corridor width
        return max(0, min_dist - self.width / 2)


class ComplianceManager:
    """
    Manages compliance with simulated aviation regulations.
    """

    def __init__(self):
        self.no_fly_zones: List[NoFlyZone] = []
        self.corridors: List[GeofenceCorridor] = []
        self.altitude_limits = (400, 500)  # FAA low-altitude limits (feet converted to meters ~122-152m)
        self.max_speed = 50.0  # m/s (~112 mph)
        self.min_separation = 50.0  # meters between vehicles
        self.violations_log = []

    def add_no_fly_zone(self, zone: NoFlyZone):
        """Add a no-fly zone."""
        self.no_fly_zones.append(zone)
        logger.info(f"Added no-fly zone: {zone.name} at {zone.center}")

    def add_corridor(self, corridor: GeofenceCorridor):
        """Add an approved flight corridor."""
        self.corridors.append(corridor)
        logger.info(f"Added corridor: {corridor.name} with {len(corridor.waypoints)} waypoints")

    def create_default_zones(self, arena_bounds: Dict[str, float]):
        """
        Create default no-fly zones based on arena.

        Args:
            arena_bounds: Dictionary with x_min, x_max, y_min, y_max, z_min, z_max
        """
        # Example: Airport zone
        airport_zone = NoFlyZone(
            name="Airport_NFZ",
            center=np.array([0, 0, 425]),
            radius=200.0,
            altitude_range=(350, 500),
            zone_type='airport'
        )
        self.add_no_fly_zone(airport_zone)

        # Example: Restricted area (stadium, government building)
        restricted_zone = NoFlyZone(
            name="Restricted_Area",
            center=np.array([500, 500, 450]),
            radius=150.0,
            zone_type='restricted'
        )
        self.add_no_fly_zone(restricted_zone)

    def check_compliance(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        nearby_vehicles: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Check if vehicle is compliant with all regulations.

        Args:
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            nearby_vehicles: List of nearby vehicle positions

        Returns:
            Dictionary with compliance status and violations
        """
        violations = []
        compliant = True

        # Check altitude limits
        altitude = position[2]
        if altitude < self.altitude_limits[0] or altitude > self.altitude_limits[1]:
            violations.append({
                'type': 'altitude_violation',
                'severity': 'high',
                'message': f'Altitude {altitude:.1f}m outside limits {self.altitude_limits}'
            })
            compliant = False

        # Check speed limit
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            violations.append({
                'type': 'speed_violation',
                'severity': 'medium',
                'message': f'Speed {speed:.1f}m/s exceeds limit {self.max_speed}m/s'
            })
            compliant = False

        # Check no-fly zones
        for nfz in self.no_fly_zones:
            if nfz.contains_point(position):
                violations.append({
                    'type': 'no_fly_zone_violation',
                    'severity': 'critical',
                    'zone_name': nfz.name,
                    'zone_type': nfz.zone_type,
                    'message': f'Inside no-fly zone: {nfz.name} ({nfz.zone_type})'
                })
                compliant = False

        # Check minimum separation
        if nearby_vehicles is not None:
            for other_pos in nearby_vehicles:
                dist = np.linalg.norm(position - other_pos)
                if dist < self.min_separation:
                    violations.append({
                        'type': 'separation_violation',
                        'severity': 'high',
                        'distance': dist,
                        'message': f'Separation {dist:.1f}m below minimum {self.min_separation}m'
                    })
                    compliant = False

        # Check corridor compliance (if corridors are defined)
        if self.corridors:
            in_corridor = False
            for corridor in self.corridors:
                if corridor.distance_to_corridor(position) == 0:
                    in_corridor = True
                    # Check corridor speed limit
                    if speed > corridor.max_speed:
                        violations.append({
                            'type': 'corridor_speed_violation',
                            'severity': 'medium',
                            'corridor': corridor.name,
                            'message': f'Speed {speed:.1f}m/s exceeds corridor limit {corridor.max_speed}m/s'
                        })
                        compliant = False
                    break

            if not in_corridor:
                violations.append({
                    'type': 'corridor_violation',
                    'severity': 'medium',
                    'message': 'Outside designated flight corridors'
                })
                compliant = False

        # Log violations
        if violations:
            self.violations_log.extend(violations)

        return {
            'compliant': compliant,
            'violations': violations,
            'total_violations': len(violations)
        }

    def calculate_compliance_penalty(self, violations: List[Dict[str, Any]]) -> float:
        """
        Calculate penalty for compliance violations.

        Args:
            violations: List of violation dictionaries

        Returns:
            Penalty value (negative reward)
        """
        penalty = 0.0

        severity_penalties = {
            'critical': -100.0,
            'high': -50.0,
            'medium': -20.0,
            'low': -5.0
        }

        for violation in violations:
            severity = violation.get('severity', 'medium')
            penalty += severity_penalties.get(severity, -10.0)

        return penalty

    def get_compliance_stats(self) -> Dict[str, Any]:
        """
        Get statistics about compliance violations.

        Returns:
            Dictionary with violation statistics
        """
        if not self.violations_log:
            return {
                'total_violations': 0,
                'by_type': {},
                'by_severity': {}
            }

        # Count by type
        by_type = {}
        by_severity = {}

        for violation in self.violations_log:
            vtype = violation.get('type', 'unknown')
            severity = violation.get('severity', 'unknown')

            by_type[vtype] = by_type.get(vtype, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            'total_violations': len(self.violations_log),
            'by_type': by_type,
            'by_severity': by_severity
        }

    def reset_violations_log(self):
        """Clear the violations log."""
        self.violations_log = []

    def emergency_ground_all(self):
        """Emergency procedure: activate all no-fly zones."""
        for nfz in self.no_fly_zones:
            nfz.active = True
        logger.warning("EMERGENCY: All no-fly zones activated")


class ComplianceRewardWrapper:
    """
    Wrapper to add compliance penalties to RL rewards.
    """

    def __init__(self, compliance_manager: ComplianceManager):
        self.compliance_manager = compliance_manager

    def calculate_reward(
        self,
        base_reward: float,
        position: np.ndarray,
        velocity: np.ndarray,
        nearby_vehicles: Optional[List[np.ndarray]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate reward with compliance penalties.

        Args:
            base_reward: Original reward from environment
            position: Current position
            velocity: Current velocity
            nearby_vehicles: Nearby vehicle positions

        Returns:
            Tuple of (modified_reward, compliance_info)
        """
        compliance_result = self.compliance_manager.check_compliance(
            position, velocity, nearby_vehicles
        )

        penalty = self.compliance_manager.calculate_compliance_penalty(
            compliance_result['violations']
        )

        modified_reward = base_reward + penalty

        return modified_reward, compliance_result
