#!/usr/bin/env python3
"""
Database initialization script for FlyingCarRL

This script creates the necessary database schema for logging simulation data,
training episodes, and vehicle performance metrics.

Usage:
    python scripts/init_db.py [--db-type sqlite|mysql]
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    from sqlalchemy import (
        create_engine, Column, Integer, Float, String, Text,
        DateTime, JSON, ForeignKey, Boolean, Index
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required packages: pip install sqlalchemy python-dotenv pymysql")
    sys.exit(1)

# Load environment variables
load_dotenv()

Base = declarative_base()


# ===========================
# Database Models
# ===========================

class Vehicle(Base):
    """Vehicle type definitions"""
    __tablename__ = 'vehicles'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    type = Column(String(50), nullable=False)  # small, medium, large
    mass_kg = Column(Float, nullable=False)
    max_thrust_n = Column(Float, nullable=False)
    battery_capacity_kwh = Column(Float, nullable=False)
    dimensions = Column(JSON)  # {length, width, height}
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    episodes = relationship("Episode", back_populates="vehicle")


class Arena(Base):
    """Simulation arena/environment configurations"""
    __tablename__ = 'arenas'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    location = Column(String(200))  # e.g., "New York City"
    bounds = Column(JSON)  # {lat_min, lat_max, lon_min, lon_max}
    altitude_min_ft = Column(Float, default=400)
    altitude_max_ft = Column(Float, default=500)
    weather_config = Column(JSON)  # wind, precipitation, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    episodes = relationship("Episode", back_populates="arena")


class Episode(Base):
    """Training episode records"""
    __tablename__ = 'episodes'

    id = Column(Integer, primary_key=True)
    episode_number = Column(Integer, nullable=False)
    vehicle_id = Column(Integer, ForeignKey('vehicles.id'), nullable=False)
    arena_id = Column(Integer, ForeignKey('arenas.id'), nullable=False)

    # Episode metadata
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    num_agents = Column(Integer, default=1)

    # Performance metrics
    total_reward = Column(Float)
    avg_reward = Column(Float)
    collision_count = Column(Integer, default=0)
    altitude_violations = Column(Integer, default=0)
    successful_completion = Column(Boolean, default=False)

    # Trajectory data (compressed JSON)
    trajectory_data = Column(JSON)  # positions, velocities over time

    # RL algorithm info
    algorithm = Column(String(50))  # ppo, ddpg, sac, etc.
    model_version = Column(String(50))

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    vehicle = relationship("Vehicle", back_populates="episodes")
    arena = relationship("Arena", back_populates="episodes")
    sensor_logs = relationship("SensorLog", back_populates="episode", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="episode", cascade="all, delete-orphan")

    # Indexes for performance
    __table_args__ = (
        Index('idx_episode_vehicle', 'vehicle_id'),
        Index('idx_episode_arena', 'arena_id'),
        Index('idx_episode_number', 'episode_number'),
    )


class SensorLog(Base):
    """Sensor data logs from episodes"""
    __tablename__ = 'sensor_logs'

    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer, ForeignKey('episodes.id'), nullable=False)
    agent_id = Column(Integer, nullable=False)  # which agent in multi-agent scenario
    timestamp = Column(Float, nullable=False)  # simulation time

    # Sensor types and data
    sensor_type = Column(String(50))  # camera, lidar, imu, gps
    data_blob = Column(JSON)  # sensor-specific data

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    episode = relationship("Episode", back_populates="sensor_logs")

    # Indexes
    __table_args__ = (
        Index('idx_sensor_episode', 'episode_id'),
        Index('idx_sensor_type', 'sensor_type'),
    )


class Metric(Base):
    """Per-timestep metrics for detailed analysis"""
    __tablename__ = 'metrics'

    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer, ForeignKey('episodes.id'), nullable=False)
    timestep = Column(Integer, nullable=False)

    # Position and state
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)
    velocity = Column(Float)
    altitude_ft = Column(Float)

    # Energy
    battery_remaining_kwh = Column(Float)
    energy_consumption_kw = Column(Float)

    # Rewards and penalties
    step_reward = Column(Float)
    collision = Column(Boolean, default=False)
    altitude_violation = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    episode = relationship("Episode", back_populates="metrics")

    # Indexes
    __table_args__ = (
        Index('idx_metric_episode', 'episode_id'),
        Index('idx_metric_timestep', 'timestep'),
    )


class TrainingRun(Base):
    """Training run metadata"""
    __tablename__ = 'training_runs'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), unique=True, nullable=False)
    description = Column(Text)

    # Configuration
    algorithm = Column(String(50))
    hyperparameters = Column(JSON)

    # Status
    status = Column(String(50), default='running')  # running, completed, failed
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)

    # Results
    total_episodes = Column(Integer, default=0)
    best_reward = Column(Float)
    convergence_episode = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)


# ===========================
# Database Initialization
# ===========================

def get_database_url(db_type=None):
    """Get database connection URL based on type"""
    if db_type is None:
        db_type = os.getenv('DB_TYPE', 'sqlite')

    if db_type == 'sqlite':
        db_path = os.getenv('SQLITE_DB_PATH', 'data/database/flyingcar_rl.db')
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return f'sqlite:///{db_path}'

    elif db_type == 'mysql':
        host = os.getenv('MYSQL_HOST', 'localhost')
        port = os.getenv('MYSQL_PORT', '3306')
        user = os.getenv('MYSQL_USER', 'flyingcar_user')
        password = os.getenv('MYSQL_PASSWORD', '')
        database = os.getenv('MYSQL_DATABASE', 'flyingcar_rl')

        return f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'

    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def init_database(db_type=None, reset=False):
    """Initialize the database schema"""

    db_url = get_database_url(db_type)
    print(f"Connecting to database: {db_url.split('@')[-1] if '@' in db_url else db_url}")

    engine = create_engine(db_url, echo=True)

    if reset:
        print("\n⚠️  WARNING: Dropping all existing tables...")
        Base.metadata.drop_all(engine)

    print("\n✓ Creating tables...")
    Base.metadata.create_all(engine)

    print("\n✓ Database initialized successfully!")
    print(f"\nCreated tables: {', '.join(Base.metadata.tables.keys())}")

    # Create session for initial data
    Session = sessionmaker(bind=engine)
    session = Session()

    # Add default vehicle types if not exist
    if session.query(Vehicle).count() == 0:
        print("\n✓ Adding default vehicle types...")
        default_vehicles = [
            Vehicle(
                name="Scout Small",
                type="small",
                mass_kg=100,
                max_thrust_n=1200,
                battery_capacity_kwh=5,
                dimensions={"length": 1.5, "width": 1.5, "height": 0.8}
            ),
            Vehicle(
                name="Passenger Medium",
                type="medium",
                mass_kg=500,
                max_thrust_n=6000,
                battery_capacity_kwh=25,
                dimensions={"length": 4.0, "width": 3.0, "height": 1.5}
            ),
            Vehicle(
                name="Cargo Large",
                type="large",
                mass_kg=1000,
                max_thrust_n=12000,
                battery_capacity_kwh=50,
                dimensions={"length": 6.0, "width": 4.0, "height": 2.0}
            )
        ]
        session.add_all(default_vehicles)
        session.commit()
        print(f"  Added {len(default_vehicles)} default vehicles")

    # Add default arena
    if session.query(Arena).count() == 0:
        print("\n✓ Adding default arena...")
        default_arena = Arena(
            name="NYC Manhattan",
            location="New York City, Manhattan",
            bounds={
                "lat_min": 40.7128,
                "lat_max": 40.8128,
                "lon_min": -74.0260,
                "lon_max": -73.9260
            },
            altitude_min_ft=400,
            altitude_max_ft=500,
            weather_config={
                "wind_speed_mph": 10,
                "wind_direction_deg": 270,
                "temperature_f": 70
            }
        )
        session.add(default_arena)
        session.commit()
        print("  Added default NYC arena")

    session.close()

    return engine


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Initialize FlyingCarRL database')
    parser.add_argument(
        '--db-type',
        choices=['sqlite', 'mysql'],
        help='Database type (defaults to .env DB_TYPE or sqlite)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Drop existing tables and recreate (WARNING: destructive!)'
    )

    args = parser.parse_args()

    if args.reset:
        confirm = input("⚠️  This will delete all existing data. Are you sure? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    try:
        init_database(db_type=args.db_type, reset=args.reset)
        print("\n🚁 Database is ready for FlyingCarRL!")
    except Exception as e:
        print(f"\n❌ Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
