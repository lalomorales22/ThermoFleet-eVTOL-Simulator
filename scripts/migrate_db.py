#!/usr/bin/env python3
"""
Database migration script for FlyingCarRL.

Migrates data from SQLite to MySQL for production deployments.

Usage:
    python scripts/migrate_db.py --from-sqlite --to-mysql --batch-size=1000
    python scripts/migrate_db.py --from-sqlite --to-mysql --dry-run
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.init_db import (
    Base, Vehicle, Arena, Episode, SensorLog, Metric, TrainingRun,
    get_database_url, init_database
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Handles database migration between SQLite and MySQL."""

    def __init__(self, source_db_type: str, target_db_type: str):
        """
        Initialize migrator.

        Args:
            source_db_type: Source database type (sqlite or mysql)
            target_db_type: Target database type (sqlite or mysql)
        """
        self.source_db_type = source_db_type
        self.target_db_type = target_db_type

        # Create engines
        source_url = get_database_url(source_db_type)
        target_url = get_database_url(target_db_type)

        logger.info(f"Source DB: {source_url.split('@')[-1] if '@' in source_url else source_url}")
        logger.info(f"Target DB: {target_url.split('@')[-1] if '@' in target_url else target_url}")

        self.source_engine = create_engine(source_url, echo=False)
        self.target_engine = create_engine(target_url, echo=False)

        self.SourceSession = sessionmaker(bind=self.source_engine)
        self.TargetSession = sessionmaker(bind=self.target_engine)

    def init_target_schema(self):
        """Initialize target database schema."""
        logger.info("Initializing target database schema...")
        Base.metadata.create_all(self.target_engine)
        logger.info("✓ Target schema created")

    def migrate_table(
        self,
        model_class,
        batch_size: int = 1000,
        dry_run: bool = False
    ) -> int:
        """
        Migrate a single table.

        Args:
            model_class: SQLAlchemy model class
            batch_size: Number of records per batch
            dry_run: If True, don't actually write to target

        Returns:
            Number of records migrated
        """
        table_name = model_class.__tablename__
        logger.info(f"Migrating table: {table_name}")

        source_session = self.SourceSession()
        target_session = self.TargetSession()

        try:
            # Count total records
            total = source_session.query(model_class).count()
            logger.info(f"  Total records: {total:,}")

            if total == 0:
                logger.info(f"  Skipping empty table")
                return 0

            if dry_run:
                logger.info(f"  [DRY RUN] Would migrate {total:,} records")
                return total

            # Migrate in batches
            migrated = 0
            offset = 0

            while offset < total:
                # Fetch batch from source
                records = source_session.query(model_class).limit(batch_size).offset(offset).all()

                if not records:
                    break

                # Insert into target
                for record in records:
                    # Create new instance with same attributes
                    record_dict = {
                        col.name: getattr(record, col.name)
                        for col in model_class.__table__.columns
                    }

                    new_record = model_class(**record_dict)
                    target_session.add(new_record)

                target_session.commit()

                migrated += len(records)
                offset += batch_size

                logger.info(f"  Progress: {migrated:,}/{total:,} ({migrated/total*100:.1f}%)")

            logger.info(f"✓ Migrated {migrated:,} records from {table_name}")
            return migrated

        except Exception as e:
            logger.error(f"Error migrating {table_name}: {e}")
            target_session.rollback()
            raise

        finally:
            source_session.close()
            target_session.close()

    def migrate_all(self, batch_size: int = 1000, dry_run: bool = False) -> dict:
        """
        Migrate all tables.

        Args:
            batch_size: Batch size for migration
            dry_run: If True, don't actually migrate

        Returns:
            Dictionary with migration statistics
        """
        logger.info("=" * 60)
        logger.info("STARTING DATABASE MIGRATION")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Migration order (respects foreign keys)
        migration_order = [
            Vehicle,
            Arena,
            TrainingRun,
            Episode,
            Metric,
            SensorLog
        ]

        stats = {}

        for model_class in migration_order:
            try:
                count = self.migrate_table(
                    model_class,
                    batch_size=batch_size,
                    dry_run=dry_run
                )
                stats[model_class.__tablename__] = count

            except Exception as e:
                logger.error(f"Failed to migrate {model_class.__tablename__}: {e}")
                stats[model_class.__tablename__] = -1
                raise

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info("MIGRATION COMPLETE")
        logger.info("=" * 60)

        total_records = sum(v for v in stats.values() if v > 0)
        logger.info(f"Total records migrated: {total_records:,}")
        logger.info(f"Duration: {duration:.1f}s")

        for table, count in stats.items():
            status = "✓" if count >= 0 else "✗"
            logger.info(f"  {status} {table}: {count:,}")

        logger.info("=" * 60)

        return stats

    def verify_migration(self) -> bool:
        """
        Verify migration by comparing record counts.

        Returns:
            True if verification passed
        """
        logger.info("Verifying migration...")

        source_session = self.SourceSession()
        target_session = self.TargetSession()

        all_match = True

        try:
            for model_class in [Vehicle, Arena, TrainingRun, Episode, Metric, SensorLog]:
                source_count = source_session.query(model_class).count()
                target_count = target_session.query(model_class).count()

                table_name = model_class.__tablename__
                match = source_count == target_count

                status = "✓" if match else "✗"
                logger.info(
                    f"  {status} {table_name}: source={source_count:,}, target={target_count:,}"
                )

                if not match:
                    all_match = False

            if all_match:
                logger.info("✓ Verification passed!")
            else:
                logger.warning("✗ Verification failed - record counts don't match")

            return all_match

        finally:
            source_session.close()
            target_session.close()

    def close(self):
        """Close database connections."""
        self.source_engine.dispose()
        self.target_engine.dispose()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Migrate FlyingCarRL database between SQLite and MySQL'
    )

    parser.add_argument(
        '--from-sqlite',
        action='store_true',
        help='Migrate from SQLite'
    )

    parser.add_argument(
        '--from-mysql',
        action='store_true',
        help='Migrate from MySQL'
    )

    parser.add_argument(
        '--to-sqlite',
        action='store_true',
        help='Migrate to SQLite'
    )

    parser.add_argument(
        '--to-mysql',
        action='store_true',
        help='Migrate to MySQL'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for migration (default: 1000)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run - show what would be migrated without actually migrating'
    )

    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing migration'
    )

    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip verification after migration'
    )

    args = parser.parse_args()

    # Determine source and target
    if args.from_sqlite:
        source_db = 'sqlite'
    elif args.from_mysql:
        source_db = 'mysql'
    else:
        print("Error: Must specify --from-sqlite or --from-mysql")
        return 1

    if args.to_sqlite:
        target_db = 'sqlite'
    elif args.to_mysql:
        target_db = 'mysql'
    else:
        print("Error: Must specify --to-sqlite or --to-mysql")
        return 1

    if source_db == target_db:
        print(f"Error: Source and target are both {source_db}")
        return 1

    try:
        # Initialize migrator
        migrator = DatabaseMigrator(source_db, target_db)

        if args.verify_only:
            # Verify only
            success = migrator.verify_migration()
            return 0 if success else 1

        # Initialize target schema
        if not args.dry_run:
            confirm = input(
                f"\n⚠️  This will migrate from {source_db} to {target_db}.\n"
                f"   Target database will be initialized (existing data may be lost).\n"
                f"   Continue? [y/N]: "
            )

            if confirm.lower() != 'y':
                print("Aborted.")
                return 0

            migrator.init_target_schema()

        # Run migration
        stats = migrator.migrate_all(
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )

        # Verify
        if not args.dry_run and not args.skip_verify:
            success = migrator.verify_migration()
            if not success:
                logger.warning("Migration verification failed!")
                return 1

        logger.info("\n🚁 Migration successful!")

        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if 'migrator' in locals():
            migrator.close()


if __name__ == '__main__':
    load_dotenv()
    sys.exit(main())
