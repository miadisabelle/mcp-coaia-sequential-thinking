"""
Data Persistence Layer for Polycentric Agentic Lattice

This module provides persistent storage capabilities for:
- Agent interactions and message history
- Consensus decision outcomes and reasoning
- Structural tension chart data
- Creative orientation validation results
- Human companion loop interactions

Storage follows creative orientation principles - preserving advancing patterns
while maintaining audit trails for constitutional compliance.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class StorageConfig:
    """Configuration for data persistence."""
    db_path: str = "data/polycentric_lattice.db"
    enable_memory_integration: bool = True
    max_retention_days: int = 365
    backup_interval_hours: int = 24
    compression_enabled: bool = False


class PolycentricDataStore:
    """
    Persistent data store for polycentric lattice operations.
    
    Designed around creative orientation principles:
    - Stores advancing patterns and outcomes
    - Maintains constitutional audit trails
    - Enables pattern analysis for continuous learning
    - Supports CoAiA-memory integration for structural tension charts
    """
    
    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        self.db_path = Path(self.config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema with creative orientation structure."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Agent interactions and message flow
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE NOT NULL,
                    sender_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    content TEXT NOT NULL,  -- JSON
                    timestamp TEXT NOT NULL,
                    requires_response BOOLEAN DEFAULT FALSE,
                    response_received BOOLEAN DEFAULT FALSE,
                    conversation_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Consensus decisions and outcomes
                CREATE TABLE IF NOT EXISTS consensus_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT UNIQUE NOT NULL,
                    request_content TEXT NOT NULL,
                    primary_purpose TEXT NOT NULL,
                    mia_perspective TEXT,
                    miette_perspective TEXT,
                    haiku_perspective TEXT,
                    consensus_level REAL NOT NULL,
                    final_decision TEXT NOT NULL,
                    constitutional_compliance REAL,
                    human_consultation_requested BOOLEAN DEFAULT FALSE,
                    human_input TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT
                );
                
                -- Structural tension charts and creative processes
                CREATE TABLE IF NOT EXISTS structural_tensions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chart_id TEXT UNIQUE NOT NULL,
                    desired_outcome TEXT NOT NULL,
                    current_reality TEXT NOT NULL,
                    natural_progression TEXT,
                    tension_strength REAL,
                    resolution_status TEXT DEFAULT 'active',
                    advancing_pattern BOOLEAN DEFAULT TRUE,
                    constitutional_validation TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Agent collaboration outcomes
                CREATE TABLE IF NOT EXISTS collaborations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collaboration_id TEXT UNIQUE NOT NULL,
                    task_description TEXT NOT NULL,
                    participating_agents TEXT NOT NULL,  -- JSON array
                    outcome_quality REAL,
                    advancing_pattern_achieved BOOLEAN,
                    constitutional_compliance REAL,
                    completion_time_seconds INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT
                );
                
                -- Human companion loop interactions
                CREATE TABLE IF NOT EXISTS human_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT UNIQUE NOT NULL,
                    context_type TEXT NOT NULL,  -- 'decision', 'validation', 'creative_guidance'
                    ai_request TEXT NOT NULL,
                    human_response TEXT,
                    influence_on_outcome TEXT,  -- How human input affected the result
                    satisfaction_rating INTEGER,  -- 1-5 scale
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    responded_at TEXT
                );
                
                -- Constitutional audit trail
                CREATE TABLE IF NOT EXISTS constitutional_audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT UNIQUE NOT NULL,
                    content_validated TEXT NOT NULL,
                    principles_applied TEXT NOT NULL,  -- JSON array
                    compliance_score REAL NOT NULL,
                    violations TEXT,  -- JSON array
                    recommendations TEXT,  -- JSON array
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Creative orientation validation results
                CREATE TABLE IF NOT EXISTS orientation_validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    validation_id TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    creative_orientation_score REAL NOT NULL,
                    reactive_patterns_detected TEXT,  -- JSON
                    advancing_indicators TEXT,  -- JSON 
                    co_lint_results TEXT,  -- JSON
                    recommendations TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Performance metrics and learning data
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    agent_id TEXT,
                    value REAL NOT NULL,
                    context TEXT,  -- JSON
                    timestamp TEXT NOT NULL
                );
                
                -- Create indexes for performance
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON agent_messages(timestamp);
                CREATE INDEX IF NOT EXISTS idx_decisions_created ON consensus_decisions(created_at);
                CREATE INDEX IF NOT EXISTS idx_tensions_status ON structural_tensions(resolution_status);
                CREATE INDEX IF NOT EXISTS idx_collaborations_completed ON collaborations(completed_at);
                CREATE INDEX IF NOT EXISTS idx_audits_created ON constitutional_audits(created_at);
                CREATE INDEX IF NOT EXISTS idx_validations_score ON orientation_validations(creative_orientation_score);
                CREATE INDEX IF NOT EXISTS idx_metrics_type_time ON system_metrics(metric_type, timestamp);
            """)
            logger.info("Database schema initialized successfully")
    
    def store_agent_message(self, message_data: Dict[str, Any]) -> str:
        """Store an agent message with full context."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO agent_messages 
                    (message_id, sender_id, recipient_id, message_type, priority, 
                     content, timestamp, requires_response, conversation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message_data['message_id'],
                    message_data['sender_id'], 
                    message_data['recipient_id'],
                    message_data['message_type'],
                    message_data['priority'],
                    json.dumps(message_data['content']),
                    message_data['timestamp'],
                    message_data.get('requires_response', False),
                    message_data.get('conversation_id')
                ))
                return message_data['message_id']
    
    def store_consensus_decision(self, decision_data: Dict[str, Any]) -> str:
        """Store a consensus decision with all perspectives and outcomes."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                decision_id = decision_data.get('decision_id', str(uuid.uuid4()))
                cursor.execute("""
                    INSERT INTO consensus_decisions 
                    (decision_id, request_content, primary_purpose, mia_perspective,
                     miette_perspective, haiku_perspective, consensus_level, 
                     final_decision, constitutional_compliance, human_consultation_requested,
                     human_input, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision_id,
                    decision_data['request_content'],
                    decision_data.get('primary_purpose', ''),
                    decision_data.get('mia_perspective'),
                    decision_data.get('miette_perspective'), 
                    decision_data.get('haiku_perspective'),
                    decision_data.get('consensus_level', 0.0),
                    decision_data.get('final_decision', ''),
                    decision_data.get('constitutional_compliance'),
                    decision_data.get('human_consultation_requested', False),
                    decision_data.get('human_input'),
                    datetime.now(timezone.utc).isoformat() if decision_data.get('completed') else None
                ))
                return decision_id
    
    def store_structural_tension(self, tension_data: Dict[str, Any]) -> str:
        """Store structural tension chart data."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                chart_id = tension_data.get('chart_id', str(uuid.uuid4()))
                cursor.execute("""
                    INSERT INTO structural_tensions 
                    (chart_id, desired_outcome, current_reality, natural_progression,
                     tension_strength, resolution_status, advancing_pattern, 
                     constitutional_validation, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chart_id,
                    tension_data['desired_outcome'],
                    tension_data['current_reality'],
                    tension_data.get('natural_progression'),
                    tension_data.get('tension_strength'),
                    tension_data.get('resolution_status', 'active'),
                    tension_data.get('advancing_pattern', True),
                    json.dumps(tension_data.get('constitutional_validation', {})),
                    datetime.now(timezone.utc).isoformat()
                ))
                return chart_id
    
    def store_collaboration_outcome(self, collaboration_data: Dict[str, Any]) -> str:
        """Store collaboration outcomes and success metrics."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                collaboration_id = collaboration_data.get('collaboration_id', str(uuid.uuid4()))
                cursor.execute("""
                    INSERT INTO collaborations 
                    (collaboration_id, task_description, participating_agents,
                     outcome_quality, advancing_pattern_achieved, constitutional_compliance,
                     completion_time_seconds, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    collaboration_id,
                    collaboration_data['task_description'],
                    json.dumps(collaboration_data.get('participating_agents', [])),
                    collaboration_data.get('outcome_quality'),
                    collaboration_data.get('advancing_pattern_achieved'),
                    collaboration_data.get('constitutional_compliance'),
                    collaboration_data.get('completion_time_seconds'),
                    datetime.now(timezone.utc).isoformat() if collaboration_data.get('completed') else None
                ))
                return collaboration_id
    
    def store_human_interaction(self, interaction_data: Dict[str, Any]) -> str:
        """Store human companion loop interactions."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                interaction_id = interaction_data.get('interaction_id', str(uuid.uuid4()))
                cursor.execute("""
                    INSERT INTO human_interactions 
                    (interaction_id, context_type, ai_request, human_response,
                     influence_on_outcome, satisfaction_rating, responded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction_id,
                    interaction_data['context_type'],
                    interaction_data['ai_request'],
                    interaction_data.get('human_response'),
                    interaction_data.get('influence_on_outcome'),
                    interaction_data.get('satisfaction_rating'),
                    datetime.now(timezone.utc).isoformat() if interaction_data.get('human_response') else None
                ))
                return interaction_id
    
    def store_constitutional_audit(self, audit_data: Dict[str, Any]) -> str:
        """Store constitutional compliance audit results."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                audit_id = audit_data.get('audit_id', str(uuid.uuid4()))
                cursor.execute("""
                    INSERT INTO constitutional_audits 
                    (audit_id, content_validated, principles_applied, compliance_score,
                     violations, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    audit_id,
                    audit_data['content_validated'],
                    json.dumps(audit_data.get('principles_applied', [])),
                    audit_data['compliance_score'],
                    json.dumps(audit_data.get('violations', [])),
                    json.dumps(audit_data.get('recommendations', []))
                ))
                return audit_id
    
    def store_orientation_validation(self, validation_data: Dict[str, Any]) -> str:
        """Store creative orientation validation results."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                validation_id = validation_data.get('validation_id', str(uuid.uuid4()))
                cursor.execute("""
                    INSERT INTO orientation_validations 
                    (validation_id, content, creative_orientation_score, reactive_patterns_detected,
                     advancing_indicators, co_lint_results, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    validation_id,
                    validation_data['content'],
                    validation_data['creative_orientation_score'],
                    json.dumps(validation_data.get('reactive_patterns_detected', {})),
                    json.dumps(validation_data.get('advancing_indicators', {})),
                    json.dumps(validation_data.get('co_lint_results', {})),
                    json.dumps(validation_data.get('recommendations', []))
                ))
                return validation_id
    
    def get_advancing_patterns(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve patterns that demonstrate advancing behavior."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM structural_tensions 
                WHERE advancing_pattern = TRUE 
                  AND resolution_status IN ('active', 'advancing')
                ORDER BY last_updated DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_consensus_patterns(self, min_consensus: float = 0.7, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve high-consensus decisions for pattern analysis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM consensus_decisions 
                WHERE consensus_level >= ? 
                  AND completed_at IS NOT NULL
                ORDER BY created_at DESC 
                LIMIT ?
            """, (min_consensus, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_human_influence_analytics(self) -> Dict[str, Any]:
        """Analyze how human input influences outcomes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Response rate
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    COUNT(CASE WHEN human_response IS NOT NULL THEN 1 END) as responses,
                    AVG(CASE WHEN satisfaction_rating IS NOT NULL THEN satisfaction_rating END) as avg_satisfaction
                FROM human_interactions
            """)
            response_stats = cursor.fetchone()
            
            # Context type distribution
            cursor.execute("""
                SELECT context_type, COUNT(*) as count 
                FROM human_interactions 
                GROUP BY context_type
            """)
            context_distribution = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "response_rate": response_stats[1] / response_stats[0] if response_stats[0] > 0 else 0,
                "average_satisfaction": response_stats[2] or 0,
                "total_interactions": response_stats[0],
                "context_distribution": context_distribution
            }
    
    def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health and performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Recent activity counts
            cursor.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM agent_messages WHERE timestamp > datetime('now', '-1 day')) as messages_24h,
                    (SELECT COUNT(*) FROM consensus_decisions WHERE created_at > datetime('now', '-1 day')) as decisions_24h,
                    (SELECT COUNT(*) FROM structural_tensions WHERE resolution_status = 'active') as active_tensions,
                    (SELECT AVG(consensus_level) FROM consensus_decisions WHERE completed_at > datetime('now', '-7 days')) as avg_consensus_7d,
                    (SELECT COUNT(*) FROM structural_tensions WHERE advancing_pattern = TRUE) as advancing_patterns
            """)
            health_stats = cursor.fetchone()
            
            return {
                "messages_last_24h": health_stats[0] or 0,
                "decisions_last_24h": health_stats[1] or 0, 
                "active_structural_tensions": health_stats[2] or 0,
                "average_consensus_7d": health_stats[3] or 0,
                "advancing_patterns_count": health_stats[4] or 0,
                "system_status": "healthy" if health_stats[0] > 0 else "idle"
            }
    
    def get_orientation_patterns(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve creative orientation validation patterns for analysis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM orientation_validations 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                # Parse JSON fields
                try:
                    result['reactive_patterns_detected'] = json.loads(result['reactive_patterns_detected']) if result['reactive_patterns_detected'] else {}
                    result['advancing_indicators'] = json.loads(result['advancing_indicators']) if result['advancing_indicators'] else {}
                    result['co_lint_results'] = json.loads(result['co_lint_results']) if result['co_lint_results'] else {}
                    result['recommendations'] = json.loads(result['recommendations']) if result['recommendations'] else []
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON in orientation validation {result['validation_id']}")
                results.append(result)
            
            return results

    def export_coaia_memory_format(self, entity_type: str = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Export data in CoAiA-memory format for knowledge graph integration."""
        entities = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if entity_type in [None, 'structural_tension']:
                cursor.execute("""
                    SELECT * FROM structural_tensions 
                    ORDER BY last_updated DESC LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    entities.append({
                        "type": "entity",
                        "name": f"tension_{row['chart_id']}", 
                        "entityType": "structural_tension_chart",
                        "observations": [
                            f"Desired Outcome: {row['desired_outcome']}",
                            f"Current Reality: {row['current_reality']}",
                            f"Tension Strength: {row['tension_strength']}"
                        ],
                        "metadata": {
                            "chartId": row['chart_id'],
                            "resolution_status": row['resolution_status'],
                            "advancing_pattern": row['advancing_pattern'],
                            "created_at": row['created_at']
                        }
                    })
            
            if entity_type in [None, 'consensus_decision']:
                cursor.execute("""
                    SELECT * FROM consensus_decisions 
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    entities.append({
                        "type": "entity", 
                        "name": f"decision_{row['decision_id']}",
                        "entityType": "consensus_decision",
                        "observations": [
                            f"Request: {row['request_content']}",
                            f"Consensus Level: {row['consensus_level']:.2f}",
                            f"Decision: {row['final_decision']}"
                        ],
                        "metadata": {
                            "decisionId": row['decision_id'],
                            "consensus_level": row['consensus_level'],
                            "constitutional_compliance": row['constitutional_compliance'],
                            "created_at": row['created_at']
                        }
                    })
        
        return entities


# Global data store instance
data_store = PolycentricDataStore()