"""
Multi-Agent Consensus Decision Engine with Human Companion Loop

Implements the multi-agent consensus-based decision making system as described in PR #9 feedback.
Key features:
- Multi-AI-agents consensus based decision making
- Primary purpose focus for adequate decision iteration
- Human companion consultation loops for thought element clarification
- Integration with MMOR techniques (Design vs Execution elements)
- Delayed resolution principle implementation
"""

import json
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import asyncio
import logging

# Simple dataclasses instead of Pydantic models
from .constitutional_core import ConstitutionalCore

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions requiring consensus"""
    PRIMARY_CHOICE = "primary_choice"      # Fundamental/strategic decisions
    SECONDARY_CHOICE = "secondary_choice"  # Tactical/implementation decisions
    DESIGN_ELEMENT = "design_element"      # MMOR Design category
    EXECUTION_ELEMENT = "execution_element" # MMOR Execution category

class ConsensusStatus(Enum):
    """Status of consensus decision process"""
    TENSION_HOLDING = "tension_holding"    # Delayed resolution active
    CONSENSUS_EMERGING = "consensus_emerging"  # Agreement developing
    CONSENSUS_ACHIEVED = "consensus_achieved"  # Decision finalized
    HUMAN_CONSULTATION_REQUIRED = "human_consultation_required"
    ITERATION_REQUIRED = "iteration_required"

@dataclass
class DecisionTension:
    """Represents structural tension in decision-making process"""
    decision_id: str
    primary_purpose: str
    current_reality: str
    desired_outcome: str
    tension_level: float  # 0.0 to 1.0
    resolution_pressure: float  # Natural pressure to resolve
    delay_justification: str  # Why we're delaying resolution
    created_at: datetime
    
class DelayedResolutionPrinciple:
    """
    Implementation of Fritz's delayed resolution principle
    "Tolerate discrepancy, tension, and delayed resolution"
    """
    
    def __init__(self, constitutional_core: ConstitutionalCore):
        self.constitutional_core = constitutional_core
        self.active_tensions: Dict[str, DecisionTension] = {}
        self.resolution_threshold = 0.8  # Minimum consensus before resolution
        
    def create_decision_tension(
        self, 
        decision_id: str,
        primary_purpose: str,
        current_reality: str,
        desired_outcome: str,
        delay_justification: str = "Insufficient information for quality decision"
    ) -> DecisionTension:
        """Create structural tension for decision, avoiding premature resolution"""
        
        tension = DecisionTension(
            decision_id=decision_id,
            primary_purpose=primary_purpose,
            current_reality=current_reality,
            desired_outcome=desired_outcome,
            tension_level=self._calculate_tension_level(current_reality, desired_outcome),
            resolution_pressure=0.0,  # Start with no pressure
            delay_justification=delay_justification,
            created_at=datetime.utcnow()
        )
        
        self.active_tensions[decision_id] = tension
        logger.info(f"Created decision tension for {decision_id}: {tension.tension_level}")
        
        return tension
    
    def _calculate_tension_level(self, current_reality: str, desired_outcome: str) -> float:
        """Calculate structural tension level between current and desired states"""
        # Simple heuristic - in real implementation would use semantic analysis
        if not current_reality or current_reality.startswith("Ready to"):
            # Avoid premature resolution defaults
            return 0.1  # Low tension indicates premature resolution
        
        # Higher tension for greater discrepancy
        return min(1.0, len(desired_outcome.split()) / 10.0 + 0.3)
    
    def should_delay_resolution(self, decision_id: str, consensus_level: float) -> bool:
        """Determine if resolution should be delayed per Fritz's principle"""
        if decision_id not in self.active_tensions:
            return False
            
        tension = self.active_tensions[decision_id]
        
        # Delay resolution if:
        # 1. Consensus level below threshold
        # 2. Tension level indicates premature closure
        # 3. Constitutional principles suggest more exploration needed
        
        if consensus_level < self.resolution_threshold:
            return True
            
        if tension.tension_level < 0.3:  # Suspiciously low tension
            logger.warning(f"Suspiciously low tension for {decision_id} - may indicate premature resolution")
            return True
            
        return False
    
    def update_resolution_pressure(self, decision_id: str, pressure_increase: float):
        """Update natural pressure to resolve tension"""
        if decision_id in self.active_tensions:
            self.active_tensions[decision_id].resolution_pressure += pressure_increase

@dataclass
class MMORElement:
    """Managerial Moment of Truth element (Design vs Execution)"""
    element_id: str
    element_type: DecisionType  # DESIGN_ELEMENT or EXECUTION_ELEMENT
    description: str
    strategic_level: bool  # True for strategic, False for tactical
    current_assessment: Optional[str] = None
    desired_state: Optional[str] = None

@dataclass
class ConsensusVote:
    """Agent's vote on a decision proposal"""
    agent_id: str
    decision_id: str
    vote: str  # "approve", "reject", "abstain", "needs_clarification"
    reasoning: str
    confidence: float  # 0.0 to 1.0
    conditions: List[str] = field(default_factory=list)  # Conditions for approval
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ConsensusDecision:
    """Multi-agent consensus decision with human loop integration"""
    decision_id: str
    decision_type: DecisionType
    primary_purpose: str  # The entity's primary purpose driving this decision
    proposal: str
    current_reality: str
    desired_outcome: str
    
    # MMOR categorization
    mmor_elements: List[MMORElement] = field(default_factory=list)
    
    # Consensus tracking
    participating_agents: List[str] = field(default_factory=list)
    votes: List[ConsensusVote] = field(default_factory=list)
    consensus_status: ConsensusStatus = ConsensusStatus.TENSION_HOLDING
    consensus_level: float = 0.0  # 0.0 to 1.0
    
    # Human consultation
    human_consultation_needed: bool = False
    human_clarification_requests: List[str] = field(default_factory=list)
    human_response: Optional[str] = None
    
    # Delayed resolution
    tension: Optional[DecisionTension] = None
    resolution_delayed: bool = False
    delay_reason: Optional[str] = None
    
    # Iteration tracking
    iteration_count: int = 0
    max_iterations: int = 5
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

class ConsensusDecisionEngine:
    """
    Multi-agent consensus decision engine with human companion loop
    
    Implements the feedback from PR #9:
    - Multi-AI-agents consensus based decision making
    - Primary purpose focus for decision iteration
    - Human companion consultation for clarification
    - MMOR techniques integration
    - Delayed resolution principle
    """
    
    def __init__(self, constitutional_core: ConstitutionalCore):
        self.constitutional_core = constitutional_core
        self.delayed_resolution = DelayedResolutionPrinciple(constitutional_core)
        self.active_decisions: Dict[str, ConsensusDecision] = {}
        self.decision_history: List[ConsensusDecision] = []
        
    def initiate_consensus_decision(
        self,
        decision_id: str,
        decision_type: DecisionType,
        primary_purpose: str,
        proposal: str,
        current_reality: str,
        desired_outcome: str,
        participating_agents: List[str],
        mmor_elements: Optional[List[MMORElement]] = None
    ) -> ConsensusDecision:
        """Initiate a new consensus decision process"""
        
        # Create structural tension first (delayed resolution principle)
        tension = self.delayed_resolution.create_decision_tension(
            decision_id=decision_id,
            primary_purpose=primary_purpose,
            current_reality=current_reality,
            desired_outcome=desired_outcome,
            delay_justification=f"Multi-agent consensus required for {decision_type.value}"
        )
        
        decision = ConsensusDecision(
            decision_id=decision_id,
            decision_type=decision_type,
            primary_purpose=primary_purpose,
            proposal=proposal,
            current_reality=current_reality,
            desired_outcome=desired_outcome,
            mmor_elements=mmor_elements or [],
            participating_agents=participating_agents,
            tension=tension,
            resolution_delayed=True,
            delay_reason="Awaiting agent consensus and potential human consultation"
        )
        
        self.active_decisions[decision_id] = decision
        logger.info(f"Initiated consensus decision: {decision_id}")
        
        return decision
    
    def add_agent_vote(
        self,
        decision_id: str,
        agent_id: str,
        vote: str,
        reasoning: str,
        confidence: float,
        conditions: Optional[List[str]] = None
    ) -> bool:
        """Add an agent's vote to the consensus decision"""
        
        if decision_id not in self.active_decisions:
            return False
            
        decision = self.active_decisions[decision_id]
        
        vote_obj = ConsensusVote(
            agent_id=agent_id,
            decision_id=decision_id,
            vote=vote,
            reasoning=reasoning,
            confidence=confidence,
            conditions=conditions or []
        )
        
        # Remove any previous vote from same agent
        decision.votes = [v for v in decision.votes if v.agent_id != agent_id]
        decision.votes.append(vote_obj)
        
        # Update consensus level
        self._update_consensus_level(decision_id)
        
        # Check if human consultation needed
        if vote == "needs_clarification" or confidence < 0.5:
            decision.human_consultation_needed = True
            if reasoning not in decision.human_clarification_requests:
                decision.human_clarification_requests.append(reasoning)
        
        decision.updated_at = datetime.utcnow()
        
        logger.info(f"Added vote from {agent_id} for decision {decision_id}: {vote} (confidence: {confidence})")
        
        return True
    
    def _update_consensus_level(self, decision_id: str):
        """Update consensus level based on current votes"""
        decision = self.active_decisions[decision_id]
        
        if not decision.votes:
            decision.consensus_level = 0.0
            return
        
        # Calculate weighted consensus
        total_weight = 0.0
        approval_weight = 0.0
        
        for vote in decision.votes:
            weight = vote.confidence
            total_weight += weight
            
            if vote.vote == "approve":
                approval_weight += weight
            elif vote.vote == "reject":
                approval_weight -= weight * 0.5  # Rejection reduces consensus
        
        decision.consensus_level = max(0.0, approval_weight / total_weight if total_weight > 0 else 0.0)
        
        # Update consensus status
        if decision.consensus_level >= 0.8:
            decision.consensus_status = ConsensusStatus.CONSENSUS_ACHIEVED
        elif decision.consensus_level >= 0.6:
            decision.consensus_status = ConsensusStatus.CONSENSUS_EMERGING
        elif decision.human_consultation_needed:
            decision.consensus_status = ConsensusStatus.HUMAN_CONSULTATION_REQUIRED
        else:
            decision.consensus_status = ConsensusStatus.TENSION_HOLDING
    
    def check_resolution_readiness(self, decision_id: str) -> Tuple[bool, str]:
        """Check if decision is ready for resolution per delayed resolution principle"""
        
        if decision_id not in self.active_decisions:
            return False, "Decision not found"
            
        decision = self.active_decisions[decision_id]
        
        # Check if delayed resolution should continue
        should_delay = self.delayed_resolution.should_delay_resolution(
            decision_id, decision.consensus_level
        )
        
        if should_delay:
            return False, f"Resolution delayed: {decision.delay_reason}"
        
        # Check consensus requirements
        if decision.consensus_level < 0.8:
            return False, f"Insufficient consensus: {decision.consensus_level:.2f}"
        
        # Check human consultation if needed
        if decision.human_consultation_needed and not decision.human_response:
            return False, "Human consultation pending"
        
        # Check iteration limits
        if decision.iteration_count >= decision.max_iterations:
            return False, "Maximum iterations reached - escalation required"
        
        return True, "Ready for resolution"
    
    def request_human_consultation(
        self,
        decision_id: str,
        clarification_request: str
    ) -> Dict[str, Any]:
        """Request human companion consultation for decision clarification"""
        
        if decision_id not in self.active_decisions:
            return {"error": "Decision not found"}
            
        decision = self.active_decisions[decision_id]
        decision.human_consultation_needed = True
        
        if clarification_request not in decision.human_clarification_requests:
            decision.human_clarification_requests.append(clarification_request)
        
        decision.consensus_status = ConsensusStatus.HUMAN_CONSULTATION_REQUIRED
        decision.updated_at = datetime.utcnow()
        
        return {
            "decision_id": decision_id,
            "primary_purpose": decision.primary_purpose,
            "proposal": decision.proposal,
            "clarification_requests": decision.human_clarification_requests,
            "agent_perspectives": [
                {
                    "agent_id": vote.agent_id,
                    "vote": vote.vote,
                    "reasoning": vote.reasoning,
                    "confidence": vote.confidence
                }
                for vote in decision.votes
            ]
        }
    
    def provide_human_response(
        self,
        decision_id: str,
        human_response: str
    ) -> bool:
        """Provide human response to consultation request"""
        
        if decision_id not in self.active_decisions:
            return False
            
        decision = self.active_decisions[decision_id]
        decision.human_response = human_response
        decision.human_consultation_needed = False
        decision.consensus_status = ConsensusStatus.CONSENSUS_EMERGING
        decision.updated_at = datetime.utcnow()
        
        logger.info(f"Human response provided for decision {decision_id}")
        
        return True
    
    def iterate_decision(
        self,
        decision_id: str,
        updated_proposal: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> bool:
        """Iterate on decision based on agent reservations or human feedback"""
        
        if decision_id not in self.active_decisions:
            return False
            
        decision = self.active_decisions[decision_id]
        decision.iteration_count += 1
        
        if updated_proposal:
            decision.proposal = updated_proposal
        
        # Reset votes for new iteration
        decision.votes = []
        decision.consensus_level = 0.0
        decision.consensus_status = ConsensusStatus.TENSION_HOLDING
        decision.updated_at = datetime.utcnow()
        
        # Update tension with new information
        if additional_context:
            decision.current_reality = f"{decision.current_reality}. {additional_context}"
        
        logger.info(f"Decision {decision_id} iterated (iteration {decision.iteration_count})")
        
        return True
    
    def resolve_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Resolve consensus decision if conditions are met"""
        
        ready, reason = self.check_resolution_readiness(decision_id)
        
        if not ready:
            logger.warning(f"Decision {decision_id} not ready for resolution: {reason}")
            return None
            
        decision = self.active_decisions[decision_id]
        decision.resolved_at = datetime.utcnow()
        decision.consensus_status = ConsensusStatus.CONSENSUS_ACHIEVED
        decision.resolution_delayed = False
        
        # Move to history
        self.decision_history.append(decision)
        del self.active_decisions[decision_id]
        
        # Remove from active tensions
        if decision_id in self.delayed_resolution.active_tensions:
            del self.delayed_resolution.active_tensions[decision_id]
        
        resolution = {
            "decision_id": decision_id,
            "resolved_proposal": decision.proposal,
            "consensus_level": decision.consensus_level,
            "final_votes": [asdict(vote) for vote in decision.votes],
            "human_input": decision.human_response,
            "iterations": decision.iteration_count,
            "resolution_time": decision.resolved_at.isoformat()
        }
        
        logger.info(f"Decision {decision_id} resolved with {decision.consensus_level:.2f} consensus")
        
        return resolution
    
    def get_decision_status(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a decision"""
        
        if decision_id in self.active_decisions:
            decision = self.active_decisions[decision_id]
        else:
            # Check history
            history_decision = next(
                (d for d in self.decision_history if d.decision_id == decision_id),
                None
            )
            if not history_decision:
                return None
            decision = history_decision
        
        return {
            "decision_id": decision.decision_id,
            "decision_type": decision.decision_type.value,
            "primary_purpose": decision.primary_purpose,
            "consensus_status": decision.consensus_status.value,
            "consensus_level": decision.consensus_level,
            "votes_count": len(decision.votes),
            "human_consultation_needed": decision.human_consultation_needed,
            "resolution_delayed": decision.resolution_delayed,
            "iteration_count": decision.iteration_count,
            "created_at": decision.created_at.isoformat(),
            "updated_at": decision.updated_at.isoformat(),
            "resolved_at": decision.resolved_at.isoformat() if decision.resolved_at else None
        }
    
    def get_active_decisions(self) -> List[Dict[str, Any]]:
        """Get all active decisions requiring attention"""
        
        return [
            self.get_decision_status(decision_id)
            for decision_id in self.active_decisions.keys()
        ]