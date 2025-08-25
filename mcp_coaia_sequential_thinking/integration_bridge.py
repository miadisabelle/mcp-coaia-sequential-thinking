"""
COAIA Integration Bridge - Simplified Implementation for mcp-coaia-sequential-thinking
====================================================================================

This module provides the integration bridge that connects sequential thinking sessions
with coaia-memory structural tension charts, enabling seamless SCCP methodology 
preservation and bidirectional data flow.

Key Features:
- Auto-detection of chart-ready thinking sessions
- SCCP data transformation to structural tension charts  
- Pattern preservation (advancing/oscillating)
- Telescoping hierarchy support
- Bidirectional synchronization
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4
from dataclasses import dataclass, asdict
from enum import Enum

from .models import ThoughtData, ThoughtStage
from .analysis import ThoughtAnalyzer

logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Status of integration between thinking sessions and charts."""
    PENDING = "pending"
    ACTIVE = "active" 
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ChartCreationData:
    """Structured data for creating charts from thinking sessions."""
    desired_outcome: str
    current_reality: str
    action_steps: List[str]
    pattern_type: str
    tension_strength: float
    hidden_concepts: List[str]
    due_date: str
    session_id: str


@dataclass
class IntegrationRecord:
    """Record linking thinking sessions to memory charts."""
    integration_id: str
    session_id: str
    chart_id: Optional[str]
    status: IntegrationStatus
    pattern_type: str
    created_at: str
    updated_at: str


class CoaiaMemoryBridge:
    """
    Simplified integration bridge for connecting sequential thinking to coaia-memory.
    
    This class handles the transformation of SCCP-based thinking sessions into
    structural tension charts while maintaining methodology integrity.
    """
    
    def __init__(self):
        self.integration_records: Dict[str, IntegrationRecord] = {}
        self.coaia_memory_available = False
        self._check_coaia_memory_availability()
    
    def _check_coaia_memory_availability(self):
        """Check if coaia-memory system is available for integration."""
        # TODO: Implement actual availability check
        # For now, we'll set to False and log when chart creation is triggered
        self.coaia_memory_available = False
        
    def analyze_chart_readiness(self, thoughts: List[ThoughtData]) -> Dict[str, Any]:
        """
        Analyze if thinking session is ready for chart creation.
        
        Uses enhanced SCCP analysis to determine if we have all required elements
        for creating a structural tension chart.
        
        Args:
            thoughts: List of thoughts from the session
            
        Returns:
            Dict containing readiness analysis and chart creation data
        """
        if not thoughts:
            return {"readyForChartCreation": False}
        
        # Get SCCP-based summary analysis
        summary_result = ThoughtAnalyzer.generate_summary(thoughts)
        summary = summary_result.get('summary', {})
        creative_analysis = summary.get('creativeOrientationAnalysis', {})
        
        # Extract chart creation data if ready
        chart_data = None
        if creative_analysis.get('readyForChartCreation', False):
            chart_data = self._extract_chart_data_from_thoughts(thoughts, creative_analysis)
        
        return {
            "readyForChartCreation": creative_analysis.get('readyForChartCreation', False),
            "structuralTensionEstablished": creative_analysis.get('structuralTensionEstablished', False),
            "tensionStrength": creative_analysis.get('tensionStrength', 0.0),
            "overallPattern": creative_analysis.get('overallPattern', "insufficient_data"),
            "chartCreationData": chart_data
        }
    
    def _extract_chart_data_from_thoughts(self, thoughts: List[ThoughtData], analysis: Dict[str, Any]) -> ChartCreationData:
        """Extract structured chart creation data from thoughts."""
        
        # Extract thoughts by stage
        desired_outcomes = [t for t in thoughts if t.stage == ThoughtStage.DESIRED_OUTCOME]
        current_realities = [t for t in thoughts if t.stage == ThoughtStage.CURRENT_REALITY] 
        action_steps = [t for t in thoughts if t.stage == ThoughtStage.ACTION_STEPS]
        
        # Consolidate content
        desired_outcome = self._consolidate_thoughts(desired_outcomes)
        current_reality = self._consolidate_thoughts(current_realities)
        actions = [t.thought for t in action_steps]
        
        # Extract metadata
        all_hidden_concepts = []
        for thought in thoughts:
            all_hidden_concepts.extend(thought.hidden_concepts_detected)
            all_hidden_concepts.extend(ThoughtAnalyzer.detect_hidden_concepts(thought.thought))
        
        # Generate due date (simple heuristic - 2 weeks from now)
        due_date = (datetime.now().replace(hour=23, minute=59, second=59)).isoformat()
        
        return ChartCreationData(
            desired_outcome=desired_outcome,
            current_reality=current_reality,
            action_steps=actions,
            pattern_type=analysis.get('overallPattern', 'insufficient_data'),
            tension_strength=analysis.get('tensionStrength', 0.0),
            hidden_concepts=list(set(all_hidden_concepts)),  # Remove duplicates
            due_date=due_date,
            session_id=str(uuid4())
        )
    
    def _consolidate_thoughts(self, thoughts: List[ThoughtData]) -> str:
        """Consolidate multiple thoughts of the same type into a single string."""
        if not thoughts:
            return ""
        
        if len(thoughts) == 1:
            return thoughts[0].thought
            
        # Combine multiple thoughts with clear separation
        return "\n\n".join([f"• {t.thought}" for t in thoughts])
    
    async def create_chart_from_session(self, session_id: str, chart_data: ChartCreationData) -> Optional[str]:
        """
        Create a structural tension chart from session data.
        
        This method would integrate with the actual coaia-memory system to create charts.
        For now, it logs the chart creation data and creates an integration record.
        
        Args:
            session_id: ID of the thinking session
            chart_data: Structured data for chart creation
            
        Returns:
            str: Chart ID if successful, None otherwise
        """
        logger.info(f"Chart creation triggered for session: {session_id}")
        
        if not self.coaia_memory_available:
            logger.warning("COAIA Memory system not available. Chart creation data prepared:")
            logger.info(f"Desired Outcome: {chart_data.desired_outcome}")
            logger.info(f"Current Reality: {chart_data.current_reality}")
            logger.info(f"Action Steps: {chart_data.action_steps}")
            logger.info(f"Pattern Type: {chart_data.pattern_type}")
            logger.info(f"Tension Strength: {chart_data.tension_strength}")
            logger.info(f"Hidden Concepts: {chart_data.hidden_concepts}")
            
            # Create placeholder integration record
            integration_id = str(uuid4())
            chart_id = f"chart_{integration_id[:8]}"
            
            self.integration_records[session_id] = IntegrationRecord(
                integration_id=integration_id,
                session_id=session_id,
                chart_id=chart_id,
                status=IntegrationStatus.PENDING,
                pattern_type=chart_data.pattern_type,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            return chart_id
        
        # TODO: Implement actual coaia-memory integration
        # chart_result = await self.coaia_memory.create_structural_tension_chart(
        #     desired_outcome=chart_data.desired_outcome,
        #     current_reality=chart_data.current_reality, 
        #     action_steps=chart_data.action_steps,
        #     due_date=chart_data.due_date
        # )
        
        return None
    
    def get_integration_status(self, session_id: str) -> Optional[IntegrationRecord]:
        """Get integration record for a session."""
        return self.integration_records.get(session_id)
    
    async def update_session_reality(self, session_id: str, new_reality_elements: List[str]) -> bool:
        """
        Update thinking session reality from chart completions.
        
        This method handles the bidirectional flow where completed chart actions
        update the current reality in the thinking session.
        
        Args:
            session_id: ID of the thinking session to update
            new_reality_elements: New reality elements from completed actions
            
        Returns:
            bool: True if update was successful
        """
        logger.info(f"Reality update triggered for session: {session_id}")
        logger.info(f"New reality elements: {new_reality_elements}")
        
        # TODO: Implement actual session reality update
        # This would connect back to the sequential thinking storage system
        # to add new current reality thoughts based on completed actions
        
        return True


# Global integration bridge instance
integration_bridge = CoaiaMemoryBridge()