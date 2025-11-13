"""
Scenario Logging Extension for DatabaseLogger

Adds capability to log:
- Scenario templates to database
- Episode-scenario associations
- Scenario performance metrics

Author: ThermoFleet Team
Date: November 13, 2025
"""

import os
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.init_db import get_database_url

# Import scenario models (will need migration first)
try:
    from scripts.migrate_db_scenarios import ScenarioTemplate, EpisodeScenario, ScenarioMetrics
    SCENARIOS_AVAILABLE = True
except ImportError:
    SCENARIOS_AVAILABLE = False
    logging.warning("Scenario tables not available. Run migrate_db_scenarios.py first.")

logger = logging.getLogger(__name__)
load_dotenv()


class ScenarioLogger:
    """
    Extension to DatabaseLogger for scenario tracking.
    
    Logs:
    - Scenario templates (generated scenarios)
    - Episode-scenario associations
    - Aggregated scenario performance metrics
    """
    
    def __init__(self, db_type: Optional[str] = None):
        """
        Initialize scenario logger.
        
        Args:
            db_type: Database type ('sqlite' or 'mysql'). Defaults to env variable.
        """
        if not SCENARIOS_AVAILABLE:
            raise ImportError(
                "Scenario tables not available. Please run: "
                "python scripts/migrate_db_scenarios.py"
            )
        
        self.db_type = db_type or os.getenv('DB_TYPE', 'sqlite')
        
        # Create engine and session
        db_url = get_database_url(self.db_type)
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Current scenario tracking
        self.current_scenario_id: Optional[int] = None
        self.current_episode_scenario_id: Optional[int] = None
        
        logger.info(f"ScenarioLogger initialized with {self.db_type}")
    
    def log_scenario_template(self, scenario_dict: Dict[str, Any]) -> int:
        """
        Log a scenario template to the database.
        
        Args:
            scenario_dict: Scenario dictionary from Scenario.to_dict()
            
        Returns:
            scenario_template_id
        """
        session = self.SessionLocal()
        try:
            # Check if scenario already exists
            existing = session.query(ScenarioTemplate).filter_by(
                scenario_id=scenario_dict['scenario_id']
            ).first()
            
            if existing:
                # Update usage count
                existing.times_used += 1
                session.commit()
                logger.debug(f"Reusing scenario template: {existing.scenario_id}")
                return existing.id
            
            # Create new scenario template
            template = ScenarioTemplate(
                scenario_id=scenario_dict['scenario_id'],
                difficulty=scenario_dict['difficulty'],
                weather_config=scenario_dict['weather'],
                traffic_config=scenario_dict['traffic'],
                failures_config=scenario_dict['failures'],
                edge_cases_config=scenario_dict['edge_cases'],
                arena_bounds=scenario_dict['arena_bounds'],
                max_timesteps=scenario_dict['max_timesteps'],
                generator_seed=scenario_dict.get('seed'),
                generator_version='v1.0',
                times_used=1
            )
            
            session.add(template)
            session.commit()
            
            self.current_scenario_id = template.id
            logger.info(f"Logged scenario template: {template.scenario_id} (ID: {template.id})")
            
            return template.id
        
        except Exception as e:
            logger.error(f"Error logging scenario template: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def associate_episode_with_scenario(
        self,
        episode_id: int,
        scenario_template_id: int,
        completion_time: float,
        success: bool,
        reward: float,
        challenge_rating: Optional[float] = None,
        learning_value: Optional[float] = None
    ) -> int:
        """
        Associate an episode with a scenario.
        
        Args:
            episode_id: Episode ID from database
            scenario_template_id: Scenario template ID
            completion_time: Episode duration in seconds
            success: Whether episode was successful
            reward: Total episode reward
            challenge_rating: How challenging was this scenario (0-1)
            learning_value: Estimated learning value (0-1)
            
        Returns:
            episode_scenario_id
        """
        session = self.SessionLocal()
        try:
            episode_scenario = EpisodeScenario(
                episode_id=episode_id,
                scenario_id=scenario_template_id,
                completion_time=completion_time,
                success=success,
                reward=reward,
                challenge_rating=challenge_rating or 0.5,
                learning_value=learning_value or 0.5
            )
            
            session.add(episode_scenario)
            session.commit()
            
            self.current_episode_scenario_id = episode_scenario.id
            logger.debug(f"Associated episode {episode_id} with scenario {scenario_template_id}")
            
            # Update scenario template statistics
            self._update_scenario_statistics(scenario_template_id)
            
            return episode_scenario.id
        
        except Exception as e:
            logger.error(f"Error associating episode with scenario: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def _update_scenario_statistics(self, scenario_template_id: int):
        """Update aggregated statistics for a scenario template."""
        session = self.SessionLocal()
        try:
            # Get all episodes using this scenario
            episodes = session.query(EpisodeScenario).filter_by(
                scenario_id=scenario_template_id
            ).all()
            
            if not episodes:
                return
            
            # Calculate statistics
            total_episodes = len(episodes)
            successful_episodes = sum(1 for e in episodes if e.success)
            success_rate = successful_episodes / total_episodes
            avg_reward = sum(e.reward for e in episodes) / total_episodes
            
            # Update scenario template
            template = session.query(ScenarioTemplate).filter_by(
                id=scenario_template_id
            ).first()
            
            if template:
                template.times_used = total_episodes
                template.avg_success_rate = success_rate
                template.avg_reward = avg_reward
                session.commit()
        
        except Exception as e:
            logger.error(f"Error updating scenario statistics: {e}")
            session.rollback()
        finally:
            session.close()
    
    def update_scenario_metrics(
        self,
        weather_type: str,
        traffic_density: str,
        difficulty_bin: str,
        has_failures: bool,
        has_edge_cases: bool,
        success: bool,
        reward: float,
        duration: float,
        collisions: int,
        altitude_violations: int
    ):
        """
        Update aggregated scenario metrics for analysis.
        
        Args:
            weather_type: Weather type (clear, windy, rainy, etc.)
            traffic_density: Traffic density (low, medium, high, etc.)
            difficulty_bin: Difficulty category (easy, medium, hard, extreme)
            has_failures: Whether scenario had failures
            has_edge_cases: Whether scenario had edge cases
            success: Episode success
            reward: Episode reward
            duration: Episode duration in seconds
            collisions: Number of collisions
            altitude_violations: Number of altitude violations
        """
        session = self.SessionLocal()
        try:
            # Find or create metric entry
            metric = session.query(ScenarioMetrics).filter_by(
                weather_type=weather_type,
                traffic_density=traffic_density,
                difficulty_bin=difficulty_bin,
                has_failures=has_failures,
                has_edge_cases=has_edge_cases
            ).first()
            
            if not metric:
                # Create new metric entry
                metric = ScenarioMetrics(
                    weather_type=weather_type,
                    traffic_density=traffic_density,
                    difficulty_bin=difficulty_bin,
                    has_failures=has_failures,
                    has_edge_cases=has_edge_cases,
                    total_episodes=0,
                    success_count=0
                )
                session.add(metric)
            
            # Update metrics (incremental averaging)
            n = metric.total_episodes
            metric.total_episodes += 1
            metric.success_count += 1 if success else 0
            metric.success_rate = metric.success_count / metric.total_episodes
            
            # Incremental average updates
            metric.avg_reward = ((metric.avg_reward or 0) * n + reward) / (n + 1)
            metric.avg_duration = ((metric.avg_duration or 0) * n + duration) / (n + 1)
            metric.avg_collisions = ((metric.avg_collisions or 0) * n + collisions) / (n + 1)
            metric.avg_altitude_violations = (
                ((metric.avg_altitude_violations or 0) * n + altitude_violations) / (n + 1)
            )
            
            metric.last_updated = datetime.utcnow()
            
            session.commit()
            logger.debug(f"Updated scenario metrics for {weather_type}/{traffic_density}/{difficulty_bin}")
        
        except Exception as e:
            logger.error(f"Error updating scenario metrics: {e}")
            session.rollback()
        finally:
            session.close()
    
    def get_scenario_performance_summary(
        self,
        weather_type: Optional[str] = None,
        traffic_density: Optional[str] = None,
        difficulty_bin: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance summary for scenarios matching criteria.
        
        Args:
            weather_type: Filter by weather type
            traffic_density: Filter by traffic density
            difficulty_bin: Filter by difficulty
            
        Returns:
            Dictionary with performance statistics
        """
        session = self.SessionLocal()
        try:
            query = session.query(ScenarioMetrics)
            
            if weather_type:
                query = query.filter(ScenarioMetrics.weather_type == weather_type)
            if traffic_density:
                query = query.filter(ScenarioMetrics.traffic_density == traffic_density)
            if difficulty_bin:
                query = query.filter(ScenarioMetrics.difficulty_bin == difficulty_bin)
            
            metrics = query.all()
            
            if not metrics:
                return {}
            
            # Aggregate statistics
            total_episodes = sum(m.total_episodes for m in metrics)
            total_successes = sum(m.success_count for m in metrics)
            
            summary = {
                'total_episodes': total_episodes,
                'total_successes': total_successes,
                'overall_success_rate': total_successes / total_episodes if total_episodes > 0 else 0,
                'avg_reward': sum(m.avg_reward * m.total_episodes for m in metrics) / total_episodes if total_episodes > 0 else 0,
                'avg_duration': sum(m.avg_duration * m.total_episodes for m in metrics) / total_episodes if total_episodes > 0 else 0,
                'scenarios_tested': len(metrics)
            }
            
            return summary
        
        finally:
            session.close()
    
    def get_best_learning_scenarios(self, limit: int = 10) -> list:
        """
        Get scenarios that provided the most learning value.
        
        Args:
            limit: Number of scenarios to return
            
        Returns:
            List of scenario templates with high learning value
        """
        session = self.SessionLocal()
        try:
            # Query scenarios by learning value
            episodes = session.query(EpisodeScenario).order_by(
                EpisodeScenario.learning_value.desc()
            ).limit(limit).all()
            
            scenarios = []
            for ep in episodes:
                template = session.query(ScenarioTemplate).filter_by(
                    id=ep.scenario_id
                ).first()
                
                if template:
                    scenarios.append({
                        'scenario_id': template.scenario_id,
                        'difficulty': template.difficulty,
                        'learning_value': ep.learning_value,
                        'challenge_rating': ep.challenge_rating,
                        'times_used': template.times_used
                    })
            
            return scenarios
        
        finally:
            session.close()
    
    def get_difficulty_progression(self) -> Dict[str, Any]:
        """
        Get difficulty progression statistics for curriculum learning analysis.
        
        Returns:
            Dictionary with progression statistics
        """
        session = self.SessionLocal()
        try:
            # Get average success rate by difficulty bin
            difficulty_bins = ['easy', 'medium', 'hard', 'extreme']
            progression = {}
            
            for bin_name in difficulty_bins:
                metrics = session.query(ScenarioMetrics).filter_by(
                    difficulty_bin=bin_name
                ).all()
                
                if metrics:
                    total_episodes = sum(m.total_episodes for m in metrics)
                    total_successes = sum(m.success_count for m in metrics)
                    
                    progression[bin_name] = {
                        'episodes': total_episodes,
                        'success_rate': total_successes / total_episodes if total_episodes > 0 else 0,
                        'avg_reward': sum(m.avg_reward * m.total_episodes for m in metrics) / total_episodes if total_episodes > 0 else 0
                    }
            
            return progression
        
        finally:
            session.close()
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("ScenarioLogger closed")

