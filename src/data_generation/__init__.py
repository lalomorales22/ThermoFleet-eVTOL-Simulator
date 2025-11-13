"""
Data generation module for ThermoFleet-eVTOL-Simulator.
Provides synthetic scenario generation for diverse training data.
"""

from .scenario_generator import (
    ScenarioGenerator,
    WeatherConditionGenerator,
    TrafficPatternGenerator,
    FailureModeInjector,
    EdgeCaseLibrary,
    Scenario,
    WeatherCondition,
    TrafficPattern,
    FailureMode,
    EdgeCase,
    WeatherType,
    TrafficDensity,
    FailureType,
    EdgeCaseType
)

__all__ = [
    'ScenarioGenerator',
    'WeatherConditionGenerator',
    'TrafficPatternGenerator',
    'FailureModeInjector',
    'EdgeCaseLibrary',
    'Scenario',
    'WeatherCondition',
    'TrafficPattern',
    'FailureMode',
    'EdgeCase',
    'WeatherType',
    'TrafficDensity',
    'FailureType',
    'EdgeCaseType'
]

