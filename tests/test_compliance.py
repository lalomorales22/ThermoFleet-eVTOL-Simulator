"""
Unit tests for compliance and safety features.
"""
import pytest
import numpy as np
from src.utils.compliance import NoFlyZone, GeofenceCorridor, ComplianceManager


@pytest.mark.unit
class TestNoFlyZone:
    """Test no-fly zone functionality."""

    def test_nfz_contains_point(self):
        """Test if point detection works."""
        nfz = NoFlyZone(
            name="test_zone",
            center=np.array([0, 0, 450]),
            radius=100.0
        )

        # Inside zone
        assert nfz.contains_point(np.array([50, 50, 450]))

        # Outside zone
        assert not nfz.contains_point(np.array([150, 0, 450]))

    def test_nfz_altitude_range(self):
        """Test altitude-specific no-fly zones."""
        nfz = NoFlyZone(
            name="altitude_zone",
            center=np.array([0, 0, 450]),
            radius=100.0,
            altitude_range=(400, 500)
        )

        # Inside altitude range
        assert nfz.contains_point(np.array([0, 0, 450]))

        # Outside altitude range
        assert not nfz.contains_point(np.array([0, 0, 350]))

    def test_nfz_active_status(self):
        """Test active/inactive zones."""
        nfz = NoFlyZone(
            name="test_zone",
            center=np.array([0, 0, 450]),
            radius=100.0,
            active=False
        )

        # Should not contain points when inactive
        assert not nfz.contains_point(np.array([0, 0, 450]))


@pytest.mark.unit
class TestComplianceManager:
    """Test compliance manager."""

    def test_altitude_violation(self):
        """Test altitude limit violations."""
        manager = ComplianceManager()
        manager.altitude_limits = (400, 500)

        # Compliant altitude
        result = manager.check_compliance(
            position=np.array([0, 0, 450]),
            velocity=np.array([10, 0, 0])
        )
        assert result['compliant']

        # Too low
        result = manager.check_compliance(
            position=np.array([0, 0, 300]),
            velocity=np.array([10, 0, 0])
        )
        assert not result['compliant']
        assert any(v['type'] == 'altitude_violation' for v in result['violations'])

    def test_speed_violation(self):
        """Test speed limit violations."""
        manager = ComplianceManager()
        manager.max_speed = 50.0

        # Compliant speed
        result = manager.check_compliance(
            position=np.array([0, 0, 450]),
            velocity=np.array([30, 0, 0])
        )
        assert result['compliant']

        # Too fast
        result = manager.check_compliance(
            position=np.array([0, 0, 450]),
            velocity=np.array([60, 0, 0])
        )
        assert not result['compliant']
        assert any(v['type'] == 'speed_violation' for v in result['violations'])

    def test_no_fly_zone_violation(self):
        """Test no-fly zone violations."""
        manager = ComplianceManager()

        nfz = NoFlyZone(
            name="test_zone",
            center=np.array([0, 0, 450]),
            radius=100.0,
            zone_type='restricted'
        )
        manager.add_no_fly_zone(nfz)

        # Outside NFZ
        result = manager.check_compliance(
            position=np.array([200, 0, 450]),
            velocity=np.array([10, 0, 0])
        )
        assert result['compliant']

        # Inside NFZ
        result = manager.check_compliance(
            position=np.array([50, 0, 450]),
            velocity=np.array([10, 0, 0])
        )
        assert not result['compliant']
        assert any(v['type'] == 'no_fly_zone_violation' for v in result['violations'])

    def test_separation_violation(self):
        """Test minimum separation violations."""
        manager = ComplianceManager()
        manager.min_separation = 50.0

        # Safe separation
        result = manager.check_compliance(
            position=np.array([0, 0, 450]),
            velocity=np.array([10, 0, 0]),
            nearby_vehicles=[np.array([100, 0, 450])]
        )
        assert result['compliant']

        # Too close
        result = manager.check_compliance(
            position=np.array([0, 0, 450]),
            velocity=np.array([10, 0, 0]),
            nearby_vehicles=[np.array([30, 0, 450])]
        )
        assert not result['compliant']
        assert any(v['type'] == 'separation_violation' for v in result['violations'])

    def test_compliance_penalty_calculation(self):
        """Test penalty calculation."""
        manager = ComplianceManager()

        violations = [
            {'severity': 'critical', 'type': 'test'},
            {'severity': 'high', 'type': 'test'},
            {'severity': 'medium', 'type': 'test'}
        ]

        penalty = manager.calculate_compliance_penalty(violations)
        assert penalty < 0  # Penalties are negative
        assert penalty == -170.0  # -100 + -50 + -20

    def test_compliance_stats(self):
        """Test compliance statistics."""
        manager = ComplianceManager()

        # Generate some violations
        manager.check_compliance(
            position=np.array([0, 0, 300]),  # altitude violation
            velocity=np.array([60, 0, 0])     # speed violation
        )

        stats = manager.get_compliance_stats()
        assert stats['total_violations'] == 2
        assert 'altitude_violation' in stats['by_type']
        assert 'speed_violation' in stats['by_type']
