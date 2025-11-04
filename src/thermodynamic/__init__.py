"""
Thermodynamic Computing Module for ThermoFleet eVTOL Simulator

This module implements thermodynamic computing principles using energy-based models
for efficient decision-making, path planning, and multi-agent coordination.

Based on THRML (Thermodynamic HypergRaphical Model Library) concepts from Extropic AI.
"""

from .energy_based_planner import EnergyBasedPathPlanner
from .probabilistic_decision import ThermodynamicDecisionMaker
from .collision_avoidance import EnergyBasedCollisionAvoidance
from .multi_agent_coordinator import ThermodynamicCoordinator

__all__ = [
    'EnergyBasedPathPlanner',
    'ThermodynamicDecisionMaker',
    'EnergyBasedCollisionAvoidance',
    'ThermodynamicCoordinator',
]
