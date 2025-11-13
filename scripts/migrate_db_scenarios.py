#!/usr/bin/env python3
"""
Database migration script to add scenario tracking support.

This adds tables to track:
- Generated scenarios (weather, traffic, failures, edge cases)
- Scenario-episode associations
- Performance metrics per scenario type

Usage:
    python scripts/migrate_db_scenarios.py [--db-type sqlite|mysql]
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    from sqlalchemy import (
        create_engine, Column, Integer, Float, String, Text,
        DateTime, JSON, ForeignKey, Boolean, Index, Table
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required packages: pip install sqlalchemy python-dotenv pymysql")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Import existing base and models
from scripts.init_db import Base, Episode, get_database_url


# ===========================
# New Scenario Tracking Tables
# ===========================

class ScenarioTemplate(Base):
    """
    Scenario templates for training data generation.
    Stores generated scenarios with all their parameters.
    """
    __tablename__ = 'scenario_templates'
    
    id = Column(Integer, primary_key=True)
    scenario_id = Column(String(100), unique=True, nullable=False)
    
    # Difficulty and metadata
    difficulty = Column(Float, nullable=False)  # 0-1 scale
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Weather conditions (JSON)
    weather_config = Column(JSON, nullable=False)
    # Fields: weather_type, wind_speed, wind_direction, wind_gusts, turbulence_level,
    #         precipitation, visibility, temperature, temperature_gradient, dynamic_changes, thermal_activity
    
    # Traffic pattern (JSON)
    traffic_config = Column(JSON, nullable=False)
    # Fields: density, time_of_day, num_vehicles, hotspot_locations, emergency_vehicles,
    #         delivery_routes, congestion_zones
    
    # Failure modes (JSON array)
    failures_config = Column(JSON)
    # Array of: {failure_type, severity, start_timestep, duration, recovery_possible, degradation_rate, affected_components}
    
    # Edge cases (JSON array)
    edge_cases_config = Column(JSON)
    # Array of: {case_type, position, velocity, size, timestep, duration, danger_level, avoidable}
    
    # Arena configuration
    arena_bounds = Column(JSON)  # {x_min, x_max, y_min, y_max, z_min, z_max}
    max_timesteps = Column(Integer, default=1000)
    
    # Generation metadata
    generator_seed = Column(Integer)
    generator_version = Column(String(50), default='v1.0')
    
    # Statistics (filled after episodes using this scenario)
    times_used = Column(Integer, default=0)
    avg_success_rate = Column(Float)
    avg_reward = Column(Float)
    
    # Relationships
    episodes = relationship("EpisodeScenario", back_populates="scenario")
    
    # Indexes
    __table_args__ = (
        Index('idx_scenario_difficulty', 'difficulty'),
        Index('idx_scenario_created', 'created_at'),
    )


class EpisodeScenario(Base):
    """
    Association table between episodes and scenarios.
    Tracks which scenario was used for each episode.
    """
    __tablename__ = 'episode_scenarios'
    
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer, ForeignKey('episodes.id'), nullable=False)
    scenario_id = Column(Integer, ForeignKey('scenario_templates.id'), nullable=False)
    
    # Performance on this specific scenario instance
    completion_time = Column(Float)
    success = Column(Boolean)
    reward = Column(Float)
    
    # Scenario effectiveness metrics
    challenge_rating = Column(Float)  # How challenging was it (0-1)
    learning_value = Column(Float)  # How much did agent improve (0-1)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    episode = relationship("Episode", backref="scenario_association")
    scenario = relationship("ScenarioTemplate", back_populates="episodes")
    
    # Indexes
    __table_args__ = (
        Index('idx_episode_scenario_episode', 'episode_id'),
        Index('idx_episode_scenario_scenario', 'scenario_id'),
    )


class ScenarioMetrics(Base):
    """
    Aggregated metrics for scenario performance.
    Tracks how different scenario types affect training.
    """
    __tablename__ = 'scenario_metrics'
    
    id = Column(Integer, primary_key=True)
    
    # Scenario characteristics
    weather_type = Column(String(50))  # clear, windy, rainy, foggy, snowy, stormy
    traffic_density = Column(String(50))  # low, medium, high, rush_hour, emergency
    has_failures = Column(Boolean, default=False)
    has_edge_cases = Column(Boolean, default=False)
    difficulty_bin = Column(String(50))  # easy, medium, hard, extreme
    
    # Performance statistics
    total_episodes = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    success_rate = Column(Float)
    
    avg_reward = Column(Float)
    avg_duration = Column(Float)
    avg_collisions = Column(Float)
    avg_altitude_violations = Column(Float)
    
    # Updated timestamp
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_scenario_metrics_weather', 'weather_type'),
        Index('idx_scenario_metrics_traffic', 'traffic_density'),
        Index('idx_scenario_metrics_difficulty', 'difficulty_bin'),
    )


def migrate_database(db_type=None, dry_run=False):
    """
    Run database migration to add scenario tracking.
    
    Args:
        db_type: Database type ('sqlite' or 'mysql')
        dry_run: If True, only show what would be created
    """
    db_url = get_database_url(db_type)
    print(f"Connecting to database: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    
    engine = create_engine(db_url, echo=True)
    
    if dry_run:
        print("\n🔍 DRY RUN - Would create the following tables:")
        print("  - scenario_templates")
        print("  - episode_scenarios")
        print("  - scenario_metrics")
        return
    
    print("\n✓ Creating new scenario tracking tables...")
    
    # Create only the new tables (won't affect existing ones)
    ScenarioTemplate.__table__.create(engine, checkfirst=True)
    EpisodeScenario.__table__.create(engine, checkfirst=True)
    ScenarioMetrics.__table__.create(engine, checkfirst=True)
    
    print("\n✓ Migration completed successfully!")
    print("\nCreated tables:")
    print("  - scenario_templates: Stores generated scenario configurations")
    print("  - episode_scenarios: Links episodes to scenarios")
    print("  - scenario_metrics: Aggregated performance metrics by scenario type")
    
    # Initialize some default scenario metrics bins
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        if session.query(ScenarioMetrics).count() == 0:
            print("\n✓ Initializing default scenario metric bins...")
            
            weather_types = ['clear', 'windy', 'rainy', 'foggy', 'snowy', 'stormy']
            traffic_densities = ['low', 'medium', 'high', 'rush_hour']
            difficulty_bins = ['easy', 'medium', 'hard', 'extreme']
            
            default_metrics = []
            for weather in weather_types:
                for traffic in traffic_densities:
                    for difficulty in difficulty_bins:
                        metric = ScenarioMetrics(
                            weather_type=weather,
                            traffic_density=traffic,
                            difficulty_bin=difficulty,
                            has_failures=False,
                            has_edge_cases=False,
                            total_episodes=0,
                            success_count=0,
                            success_rate=0.0,
                            avg_reward=0.0,
                            avg_duration=0.0,
                            avg_collisions=0.0,
                            avg_altitude_violations=0.0
                        )
                        default_metrics.append(metric)
            
            session.add_all(default_metrics)
            session.commit()
            print(f"  Added {len(default_metrics)} default metric bins")
    
    except Exception as e:
        print(f"Warning: Could not initialize default metrics: {e}")
        session.rollback()
    finally:
        session.close()
    
    return engine


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Migrate database to add scenario tracking support'
    )
    parser.add_argument(
        '--db-type',
        choices=['sqlite', 'mysql'],
        help='Database type (defaults to .env DB_TYPE or sqlite)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without making changes'
    )
    
    args = parser.parse_args()
    
    try:
        migrate_database(db_type=args.db_type, dry_run=args.dry_run)
        print("\n🚁 Database is ready for scenario tracking!")
    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

