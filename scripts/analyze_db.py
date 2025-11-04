#!/usr/bin/env python3
"""
Database analysis script for FlyingCarRL.

Provides command-line interface for querying and analyzing training data.

Usage:
    python scripts/analyze_db.py --stats
    python scripts/analyze_db.py --episodes --vehicle-type=medium --limit=10
    python scripts/analyze_db.py --performance --vehicle-type=large
    python scripts/analyze_db.py --training-progress --vehicle-type=small --output=progress.csv
    python scripts/analyze_db.py --export-csv --output=episodes.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_manager import DatabaseManager


def print_stats(db_manager: DatabaseManager):
    """Print database statistics."""
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)

    stats = db_manager.get_database_stats()

    print(f"Total Episodes:      {stats['total_episodes']:,}")
    print(f"Total Metrics:       {stats['total_metrics']:,}")
    print(f"Total Sensor Logs:   {stats['total_sensor_logs']:,}")
    print(f"Training Runs:       {stats['total_training_runs']}")
    print(f"Vehicles:            {stats['total_vehicles']}")
    print(f"Arenas:              {stats['total_arenas']}")

    if 'latest_episode_time' in stats:
        print(f"Latest Episode:      {stats['latest_episode_time']}")

    print("=" * 60)


def print_episodes(
    db_manager: DatabaseManager,
    vehicle_type=None,
    arena_name=None,
    min_reward=None,
    successful_only=False,
    limit=10
):
    """Print episode data."""
    print("\n" + "=" * 60)
    print("EPISODES")
    print("=" * 60)

    df = db_manager.query_episodes(
        vehicle_type=vehicle_type,
        arena_name=arena_name,
        min_reward=min_reward,
        successful_only=successful_only,
        limit=limit
    )

    if len(df) == 0:
        print("No episodes found matching criteria.")
        return

    # Format columns for display
    df['start_time'] = pd.to_datetime(df['start_time']).dt.strftime('%Y-%m-%d %H:%M')
    df['duration'] = df['duration_seconds'].apply(lambda x: f"{x:.1f}s")
    df['reward'] = df['total_reward'].apply(lambda x: f"{x:.2f}")

    # Select columns for display
    display_cols = [
        'episode_number', 'vehicle_type', 'arena_name',
        'reward', 'collision_count', 'altitude_violations',
        'successful_completion', 'algorithm'
    ]

    print(df[display_cols].to_string(index=False))
    print(f"\nShowing {len(df)} episodes")
    print("=" * 60)


def print_performance(db_manager: DatabaseManager, vehicle_type: str):
    """Print vehicle performance statistics."""
    print("\n" + "=" * 60)
    print(f"PERFORMANCE: {vehicle_type.upper()}")
    print("=" * 60)

    perf = db_manager.get_vehicle_performance(vehicle_type)

    print(f"Total Episodes:      {perf['total_episodes']:,}")
    print(f"Average Reward:      {perf['avg_reward']:.2f}")
    print(f"Max Reward:          {perf['max_reward']:.2f}")
    print(f"Average Duration:    {perf['avg_duration']:.1f}s")
    print(f"Total Collisions:    {perf['total_collisions']:,}")
    print(f"Total Violations:    {perf['total_violations']:,}")
    print(f"Successful Episodes: {perf['successful_episodes']:,}")
    print(f"Success Rate:        {perf['success_rate']:.1%}")

    print("=" * 60)


def print_training_progress(
    db_manager: DatabaseManager,
    vehicle_type=None,
    window_size=100,
    output=None
):
    """Print and optionally plot training progress."""
    print("\n" + "=" * 60)
    print("TRAINING PROGRESS")
    print("=" * 60)

    df = db_manager.get_training_progress(
        vehicle_type=vehicle_type,
        window_size=window_size
    )

    if len(df) == 0:
        print("No training data found.")
        return

    print(f"Episodes: {len(df):,}")
    print(f"Latest Reward: {df['reward'].iloc[-1]:.2f}")
    print(f"Latest Rolling Avg: {df['reward_rolling_avg'].iloc[-1]:.2f}")
    print(f"Success Rate: {df['success_rate'].iloc[-1]:.1%}")

    if output:
        df.to_csv(output, index=False)
        print(f"\nExported to: {output}")

    print("=" * 60)


def print_training_runs(db_manager: DatabaseManager, status=None):
    """Print training runs."""
    print("\n" + "=" * 60)
    print("TRAINING RUNS")
    print("=" * 60)

    df = db_manager.get_training_runs(status=status)

    if len(df) == 0:
        print("No training runs found.")
        return

    # Format columns
    df['start_time'] = pd.to_datetime(df['start_time']).dt.strftime('%Y-%m-%d %H:%M')
    df['end_time'] = pd.to_datetime(df['end_time']).dt.strftime('%Y-%m-%d %H:%M')

    display_cols = [
        'id', 'name', 'algorithm', 'status',
        'total_episodes', 'best_reward', 'convergence_episode'
    ]

    print(df[display_cols].to_string(index=False))
    print(f"\nShowing {len(df)} training runs")
    print("=" * 60)


def export_to_csv(db_manager: DatabaseManager, output: str, vehicle_type=None, limit=None):
    """Export episodes to CSV."""
    print(f"\nExporting episodes to {output}...")

    db_manager.export_episodes_to_csv(
        output_path=output,
        vehicle_type=vehicle_type,
        limit=limit
    )

    print(f"✓ Exported successfully!")


def custom_query(db_manager: DatabaseManager, query: str):
    """Execute custom SQL query."""
    print("\n" + "=" * 60)
    print("CUSTOM QUERY RESULTS")
    print("=" * 60)

    try:
        df = pd.read_sql_query(query, db_manager.engine)
        print(df.to_string(index=False))
        print(f"\n{len(df)} rows returned")

    except Exception as e:
        print(f"Error executing query: {e}")

    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze FlyingCarRL training data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Actions
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--episodes', action='store_true', help='Show episodes')
    parser.add_argument('--performance', action='store_true', help='Show vehicle performance')
    parser.add_argument('--training-progress', action='store_true', help='Show training progress')
    parser.add_argument('--training-runs', action='store_true', help='Show training runs')
    parser.add_argument('--export-csv', action='store_true', help='Export episodes to CSV')

    # Filters
    parser.add_argument('--vehicle-type', type=str, help='Filter by vehicle type')
    parser.add_argument('--arena', type=str, help='Filter by arena name')
    parser.add_argument('--min-reward', type=float, help='Minimum reward threshold')
    parser.add_argument('--successful-only', action='store_true', help='Only successful episodes')
    parser.add_argument('--status', type=str, help='Training run status')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of results')
    parser.add_argument('--window-size', type=int, default=100, help='Rolling average window')

    # Output
    parser.add_argument('--output', type=str, help='Output file path')

    # Custom query
    parser.add_argument('--query', type=str, help='Custom SQL query')

    # Database
    parser.add_argument('--db-type', type=str, choices=['sqlite', 'mysql'], help='Database type')

    args = parser.parse_args()

    # Initialize database manager
    db_manager = DatabaseManager(db_type=args.db_type)

    try:
        # Execute actions
        if args.stats:
            print_stats(db_manager)

        elif args.episodes:
            print_episodes(
                db_manager,
                vehicle_type=args.vehicle_type,
                arena_name=args.arena,
                min_reward=args.min_reward,
                successful_only=args.successful_only,
                limit=args.limit
            )

        elif args.performance:
            if not args.vehicle_type:
                print("Error: --vehicle-type required for --performance")
                return 1

            print_performance(db_manager, args.vehicle_type)

        elif args.training_progress:
            print_training_progress(
                db_manager,
                vehicle_type=args.vehicle_type,
                window_size=args.window_size,
                output=args.output
            )

        elif args.training_runs:
            print_training_runs(db_manager, status=args.status)

        elif args.export_csv:
            if not args.output:
                print("Error: --output required for --export-csv")
                return 1

            export_to_csv(
                db_manager,
                output=args.output,
                vehicle_type=args.vehicle_type,
                limit=args.limit
            )

        elif args.query:
            custom_query(db_manager, args.query)

        else:
            # Default: show stats
            print_stats(db_manager)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        db_manager.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
