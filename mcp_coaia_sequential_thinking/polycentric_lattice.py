"""
Polycentric Agentic Lattice: Multi-agent architecture for generative agentic systems.

This module implements the polycentric architecture described in the survey, featuring
multiple semi-autonomous agents that engage in structured competition, cooperation,
and conflict resolution.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import time

from .constitutional_core import constitutional_core, ConstitutionalPrinciple

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles for different agents in the polycentric lattice."""
    CONSTITUTIONAL = "constitutional_guardian"
    ANALYSIS = "structural_analysis" 
    CREATIVE = "outcome_generation"
    COORDINATION = "inter_agent_coordination"
    INTEGRATION = "external_system_integration"
    CONFLICT_RESOLUTION = "conflict_mediation"


class MessageType(Enum):
    """Types of messages that can be exchanged between agents."""
    REQUEST = "request"
    RESPONSE = "response" 
    NOTIFICATION = "notification"
    COLLABORATION_INVITE = "collaboration_invite"
    COMPETITION_CHALLENGE = "competition_challenge"
    CONFLICT_REPORT = "conflict_report"
    RESOURCE_SHARE = "resource_share"


class MessagePriority(Enum):
    """Priority levels for inter-agent messages."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: datetime
    requires_response: bool = False
    response_deadline: Optional[datetime] = None
    conversation_id: Optional[str] = None


@dataclass
class AgentCapability:
    """Describes a capability that an agent possesses."""
    capability_id: str
    name: str
    description: str
    competency_score: float  # 0.0 to 1.0
    resource_cost: float
    execution_time_estimate: float


@dataclass
class CollaborationProposal:
    """Proposal for collaboration between agents."""
    proposal_id: str
    proposer_id: str
    target_agents: List[str]
    task_description: str
    required_capabilities: List[str]
    resource_allocation: Dict[str, float]
    expected_outcome: str
    deadline: datetime


@dataclass
class CompetitionChallenge:
    """Challenge for competition between agents."""
    challenge_id: str
    challenger_id: str
    target_agents: List[str]
    task_description: str
    evaluation_criteria: Dict[str, float]
    deadline: datetime
    winner_benefits: Dict[str, Any]


class BaseAgent(ABC):
    """Abstract base class for all agents in the polycentric lattice."""
    
    def __init__(self, agent_id: str, role: AgentRole, name: str):
        self.agent_id = agent_id
        self.role = role
        self.name = name
        self.active = True
        self.capabilities: Dict[str, AgentCapability] = {}
        self.message_queue = Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self.shared_resources: Dict[str, Any] = {}
        self.collaboration_history: List[str] = []
        self.competition_history: List[str] = []
        self.performance_metrics: Dict[str, float] = {
            "task_completion_rate": 0.0,
            "collaboration_success_rate": 0.0,
            "constitutional_compliance_score": 0.0,
            "innovation_score": 0.0
        }
        self.knowledge_base: Dict[str, Any] = {}
        self.last_activity: datetime = datetime.now()
        
        # Start message processing thread
        self.start_processing()
    
    @abstractmethod
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request and return a response."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of agent capabilities."""
        pass
    
    @abstractmethod
    def evaluate_collaboration_proposal(self, proposal: CollaborationProposal) -> Tuple[bool, str]:
        """Evaluate whether to accept a collaboration proposal."""
        pass
    
    @abstractmethod
    def evaluate_competition_challenge(self, challenge: CompetitionChallenge) -> Tuple[bool, str]:
        """Evaluate whether to accept a competition challenge."""
        pass
    
    def start_processing(self):
        """Start the agent's message processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._message_processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info(f"Agent {self.name} started message processing")
    
    def stop_processing(self):
        """Stop the agent's message processing."""
        self.active = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            logger.info(f"Agent {self.name} stopped message processing")
    
    def _message_processing_loop(self):
        """Main message processing loop for the agent."""
        while self.active:
            try:
                message = self.message_queue.get(timeout=1.0)
                self._process_message(message)
                self.last_activity = datetime.now()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing message in agent {self.name}: {e}")
    
    def _process_message(self, message: AgentMessage):
        """Process an individual message."""
        try:
            logger.debug(f"Agent {self.name} processing message {message.message_id} from {message.sender_id}")
            
            if message.message_type == MessageType.REQUEST:
                response = self.process_request(message.content)
                self._send_response(message, response)
            elif message.message_type == MessageType.COLLABORATION_INVITE:
                self._handle_collaboration_invite(message)
            elif message.message_type == MessageType.COMPETITION_CHALLENGE:
                self._handle_competition_challenge(message)
            elif message.message_type == MessageType.RESOURCE_SHARE:
                self._handle_resource_share(message)
            elif message.message_type == MessageType.CONFLICT_REPORT:
                self._handle_conflict_report(message)
                
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
    
    def send_message(self, recipient_id: str, message_type: MessageType, 
                    content: Dict[str, Any], priority: MessagePriority = MessagePriority.MEDIUM,
                    requires_response: bool = False) -> str:
        """Send a message to another agent."""
        message_id = str(uuid.uuid4())
        message = AgentMessage(
            message_id=message_id,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            priority=priority,
            content=content,
            timestamp=datetime.now(),
            requires_response=requires_response
        )
        
        # This would normally go through the coordination system
        # For now, just log the message (avoiding circular import issues)
        logger.debug(f"Message {message_id} sent from {self.agent_id} to {recipient_id}")
        # TODO: Implement proper message routing through coordination system
        
        return message_id
    
    def receive_message(self, message: AgentMessage):
        """Receive a message from another agent."""
        self.message_queue.put(message)
    
    def _send_response(self, original_message: AgentMessage, response_content: Dict[str, Any]):
        """Send a response to a request message."""
        if original_message.requires_response:
            self.send_message(
                original_message.sender_id,
                MessageType.RESPONSE,
                {
                    "original_message_id": original_message.message_id,
                    "response": response_content
                },
                priority=original_message.priority
            )
    
    def _handle_collaboration_invite(self, message: AgentMessage):
        """Handle a collaboration invitation."""
        proposal = CollaborationProposal(**message.content["proposal"])
        accept, reason = self.evaluate_collaboration_proposal(proposal)
        
        response = {
            "proposal_id": proposal.proposal_id,
            "accepted": accept,
            "reason": reason,
            "agent_capabilities": [cap.name for cap in self.get_capabilities()]
        }
        
        self._send_response(message, response)
        
        if accept:
            self.collaboration_history.append(proposal.proposal_id)
            logger.info(f"Agent {self.name} accepted collaboration {proposal.proposal_id}")
    
    def _handle_competition_challenge(self, message: AgentMessage):
        """Handle a competition challenge."""
        challenge = CompetitionChallenge(**message.content["challenge"])
        accept, reason = self.evaluate_competition_challenge(challenge)
        
        response = {
            "challenge_id": challenge.challenge_id,
            "accepted": accept,
            "reason": reason,
            "estimated_performance": self._estimate_challenge_performance(challenge)
        }
        
        self._send_response(message, response)
        
        if accept:
            self.competition_history.append(challenge.challenge_id)
            logger.info(f"Agent {self.name} accepted competition {challenge.challenge_id}")
    
    def _handle_resource_share(self, message: AgentMessage):
        """Handle resource sharing from another agent."""
        shared_resources = message.content.get("resources", {})
        for resource_id, resource_data in shared_resources.items():
            if self._should_accept_resource(resource_id, resource_data):
                self.shared_resources[resource_id] = resource_data
                logger.info(f"Agent {self.name} accepted shared resource {resource_id}")
    
    def _handle_conflict_report(self, message: AgentMessage):
        """Handle a conflict report."""
        # Default implementation - specialized agents may override
        logger.warning(f"Agent {self.name} received conflict report: {message.content}")
    
    def _should_accept_resource(self, resource_id: str, resource_data: Dict[str, Any]) -> bool:
        """Determine whether to accept a shared resource."""
        # Basic implementation - can be overridden by specific agents
        return True
    
    def _estimate_challenge_performance(self, challenge: CompetitionChallenge) -> float:
        """Estimate performance on a competition challenge."""
        # Basic implementation based on capability scores
        relevant_capabilities = []
        for cap_name in challenge.evaluation_criteria.keys():
            if cap_name in self.capabilities:
                relevant_capabilities.append(self.capabilities[cap_name].competency_score)
        
        if relevant_capabilities:
            return sum(relevant_capabilities) / len(relevant_capabilities)
        return 0.5  # Neutral estimate if no relevant capabilities
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update agent performance metrics."""
        self.performance_metrics.update(metrics)
        self.last_activity = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "active": self.active,
            "last_activity": self.last_activity.isoformat(),
            "capabilities_count": len(self.capabilities),
            "collaboration_history_count": len(self.collaboration_history),
            "competition_history_count": len(self.competition_history),
            "performance_metrics": self.performance_metrics,
            "message_queue_size": self.message_queue.qsize()
        }


class ConstitutionalAgent(BaseAgent):
    """Agent responsible for maintaining constitutional compliance."""
    
    def __init__(self, agent_id: str = None):
        if agent_id is None:
            agent_id = f"constitutional_{uuid.uuid4().hex[:8]}"
        super().__init__(agent_id, AgentRole.CONSTITUTIONAL, "Constitutional Guardian")
        
        # Initialize capabilities
        self.capabilities = {
            "constitutional_validation": AgentCapability(
                "constitutional_validation",
                "Constitutional Validation",
                "Validate content and decisions against constitutional principles",
                competency_score=0.95,
                resource_cost=0.2,
                execution_time_estimate=0.5
            ),
            "principle_conflict_resolution": AgentCapability(
                "principle_conflict_resolution", 
                "Principle Conflict Resolution",
                "Resolve conflicts between constitutional principles",
                competency_score=0.9,
                resource_cost=0.3,
                execution_time_estimate=1.0
            ),
            "decision_audit": AgentCapability(
                "decision_audit",
                "Decision Audit Trail",
                "Maintain complete audit trails for all decisions",
                competency_score=0.98,
                resource_cost=0.1,
                execution_time_estimate=0.3
            )
        }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process constitutional validation requests."""
        request_type = request.get("type")
        
        if request_type == "validate_content":
            content = request.get("content", "")
            context = request.get("context", {})
            result = constitutional_core.validate_content(content, context)
            
            return {
                "validation_result": result,
                "agent_assessment": self._assess_constitutional_compliance(result),
                "recommendations": self._generate_recommendations(result)
            }
        
        elif request_type == "resolve_conflict":
            conflicting_principles = request.get("principles", [])
            context = request.get("context", {})
            
            # Convert string principle names back to enum
            principle_enums = []
            for principle_name in conflicting_principles:
                for principle in ConstitutionalPrinciple:
                    if principle.value == principle_name:
                        principle_enums.append(principle)
                        break
            
            if principle_enums:
                resolved = constitutional_core.resolve_principle_conflict(principle_enums, context)
                return {
                    "resolved_principle": resolved.value,
                    "resolution_reasoning": f"Based on principle hierarchy, {resolved.value} takes priority",
                    "hierarchy_level": constitutional_core.principle_hierarchy.get(resolved, 999)
                }
            else:
                return {"error": "Could not find matching principles"}
        
        elif request_type == "audit_decision":
            decision_id = request.get("decision_id")
            audit_trail = constitutional_core.get_decision_audit_trail(decision_id)
            
            if audit_trail:
                return {
                    "audit_trail": {
                        "decision_id": audit_trail.decision_id,
                        "timestamp": audit_trail.timestamp.isoformat(),
                        "context": audit_trail.decision_context,
                        "outcome": audit_trail.decision_outcome,
                        "principles_applied": [p.value for p in audit_trail.applicable_principles]
                    }
                }
            else:
                return {"error": "Audit trail not found"}
        
        return {"error": "Unknown request type"}
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return constitutional agent capabilities."""
        return list(self.capabilities.values())
    
    def evaluate_collaboration_proposal(self, proposal: CollaborationProposal) -> Tuple[bool, str]:
        """Evaluate collaboration proposals based on constitutional principles."""
        # Constitutional agent collaborates on tasks requiring principle validation
        constitutional_keywords = [
            "validate", "compliance", "principle", "constitutional", "audit", "decision"
        ]
        
        task_desc = proposal.task_description.lower()
        has_constitutional_relevance = any(keyword in task_desc for keyword in constitutional_keywords)
        
        if has_constitutional_relevance:
            return True, "Task requires constitutional oversight"
        else:
            return False, "Task does not require constitutional validation"
    
    def evaluate_competition_challenge(self, challenge: CompetitionChallenge) -> Tuple[bool, str]:
        """Evaluate competition challenges - constitutional agent generally doesn't compete."""
        # Constitutional agent focuses on validation rather than competition
        return False, "Constitutional agent maintains neutral oversight role"
    
    def _assess_constitutional_compliance(self, validation_result: Dict[str, Any]) -> str:
        """Assess overall constitutional compliance."""
        score = validation_result.get("constitutional_compliance_score", 0.0)
        
        if score >= 0.9:
            return "Excellent constitutional compliance"
        elif score >= 0.7:
            return "Good constitutional compliance with minor improvements needed"
        elif score >= 0.5:
            return "Moderate constitutional compliance requiring attention"
        else:
            return "Poor constitutional compliance requiring significant revision"
    
    def _generate_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate constitutional recommendations."""
        recommendations = []
        violated_principles = validation_result.get("violated_principles", [])
        
        principle_recommendations = {
            "acknowledge_uncertainty_rather_than_invent_facts": 
                "Acknowledge uncertainty where facts are not clearly established",
            "prioritize_creating_desired_outcomes_over_eliminating_problems": 
                "Reframe from problem-solving to outcome-creation language",
            "establish_clear_tension_between_current_reality_and_desired_outcome": 
                "Clarify both current reality and desired outcome",
            "begin_inquiry_without_preconceptions_or_hypotheses": 
                "Start from direct observation rather than assumptions"
        }
        
        for principle in violated_principles:
            if principle in principle_recommendations:
                recommendations.append(principle_recommendations[principle])
        
        return recommendations


class AnalysisAgent(BaseAgent):
    """Agent specialized in structural analysis and pattern recognition."""
    
    def __init__(self, agent_id: str = None):
        if agent_id is None:
            agent_id = f"analysis_{uuid.uuid4().hex[:8]}"
        super().__init__(agent_id, AgentRole.ANALYSIS, "Structural Analyst")
        
        self.capabilities = {
            "pattern_recognition": AgentCapability(
                "pattern_recognition",
                "Pattern Recognition",
                "Identify advancing vs oscillating patterns in data",
                competency_score=0.88,
                resource_cost=0.4,
                execution_time_estimate=2.0
            ),
            "structural_analysis": AgentCapability(
                "structural_analysis",
                "Structural Analysis", 
                "Analyze underlying structures that determine behavior",
                competency_score=0.92,
                resource_cost=0.5,
                execution_time_estimate=3.0
            ),
            "tension_visualization": AgentCapability(
                "tension_visualization",
                "Tension Visualization",
                "Create mathematical models of structural tension",
                competency_score=0.85,
                resource_cost=0.3,
                execution_time_estimate=1.5
            )
        }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process analysis requests."""
        request_type = request.get("type")
        
        if request_type == "analyze_patterns":
            data = request.get("data", [])
            patterns = self._analyze_patterns(data)
            return {"patterns": patterns, "analysis_confidence": 0.85}
        
        elif request_type == "structural_analysis":
            context = request.get("context", {})
            structure = self._analyze_structure(context)
            return {"structural_analysis": structure}
        
        elif request_type == "tension_visualization":
            current_reality = request.get("current_reality", "")
            desired_outcome = request.get("desired_outcome", "")
            visualization = self._create_tension_visualization(current_reality, desired_outcome)
            return {"tension_visualization": visualization}
        
        return {"error": "Unknown analysis request type"}
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return analysis agent capabilities."""
        return list(self.capabilities.values())
    
    def evaluate_collaboration_proposal(self, proposal: CollaborationProposal) -> Tuple[bool, str]:
        """Evaluate collaboration proposals for analysis tasks."""
        analysis_keywords = [
            "analyze", "pattern", "structure", "understand", "examine", "study"
        ]
        
        task_desc = proposal.task_description.lower()
        has_analysis_relevance = any(keyword in task_desc for keyword in analysis_keywords)
        
        if has_analysis_relevance:
            return True, "Task requires structural analysis expertise"
        else:
            return False, "Task does not require analysis capabilities"
    
    def evaluate_competition_challenge(self, challenge: CompetitionChallenge) -> Tuple[bool, str]:
        """Evaluate competition challenges for analysis tasks."""
        # Analysis agent competes on analytical accuracy and insight quality
        analysis_criteria = [
            "accuracy", "insight", "pattern_recognition", "structural_understanding"
        ]
        
        has_analysis_competition = any(criteria in challenge.evaluation_criteria 
                                     for criteria in analysis_criteria)
        
        if has_analysis_competition:
            return True, "Competition matches analysis capabilities"
        else:
            return False, "Competition criteria do not match analysis strengths"
    
    def _analyze_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in the provided data."""
        # Simplified pattern analysis
        if len(data) < 2:
            return {"pattern_type": "insufficient_data", "confidence": 0.0}
        
        # Look for advancing vs oscillating patterns
        values = []
        for item in data:
            if "score" in item:
                values.append(item["score"])
            elif "value" in item:
                values.append(item["value"])
        
        if len(values) >= 2:
            trend = "advancing" if values[-1] > values[0] else "declining"
            volatility = sum(abs(values[i] - values[i-1]) for i in range(1, len(values))) / len(values)
            
            if volatility > 0.3:
                pattern_type = "oscillating"
            else:
                pattern_type = "advancing" if trend == "advancing" else "declining"
            
            return {
                "pattern_type": pattern_type,
                "trend": trend,
                "volatility": volatility,
                "confidence": min(0.9, len(values) / 10.0)
            }
        
        return {"pattern_type": "unknown", "confidence": 0.2}
    
    def _analyze_structure(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze underlying structural dynamics."""
        structure_indicators = {
            "feedback_loops": self._detect_feedback_loops(context),
            "tension_points": self._identify_tension_points(context),
            "leverage_points": self._find_leverage_points(context),
            "structural_relationships": self._map_relationships(context)
        }
        
        return structure_indicators
    
    def _create_tension_visualization(self, current_reality: str, desired_outcome: str) -> Dict[str, Any]:
        """Create a visualization of structural tension."""
        if not current_reality or not desired_outcome:
            return {"error": "Both current reality and desired outcome required"}
        
        # Simplified tension analysis
        tension_strength = len(set(current_reality.split()) & set(desired_outcome.split())) / max(len(current_reality.split()), len(desired_outcome.split()))
        
        return {
            "tension_strength": 1.0 - tension_strength,  # Higher when less overlap
            "current_reality_clarity": len(current_reality.split()) / 20.0,
            "desired_outcome_clarity": len(desired_outcome.split()) / 20.0,
            "visualization_type": "mathematical_model"
        }
    
    def _detect_feedback_loops(self, context: Dict[str, Any]) -> List[str]:
        """Detect feedback loops in the context."""
        # Simplified implementation
        return ["positive_feedback", "reinforcing_pattern"] if context else []
    
    def _identify_tension_points(self, context: Dict[str, Any]) -> List[str]:
        """Identify points of structural tension."""
        return ["resource_constraints", "competing_priorities"] if context else []
    
    def _find_leverage_points(self, context: Dict[str, Any]) -> List[str]:
        """Find points where small changes can have large impacts."""
        return ["system_paradigm", "decision_rules"] if context else []
    
    def _map_relationships(self, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Map structural relationships."""
        return {"causal_relationships": ["cause_1", "cause_2"], "correlations": ["corr_1"]} if context else {}


# We'll implement CreativeAgent, CoordinationAgent, and IntegrationAgent in subsequent phases
# For now, let's create the agent registry and coordination system

class AgentRegistry:
    """Central registry for all agents in the polycentric lattice."""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}  # agent_id -> capability_names
        self.active_collaborations: Dict[str, List[str]] = {}  # collaboration_id -> agent_ids
        self.active_competitions: Dict[str, List[str]] = {}  # competition_id -> agent_ids
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent in the lattice."""
        self.agents[agent.agent_id] = agent
        self.agent_capabilities[agent.agent_id] = [cap.name for cap in agent.get_capabilities()]
        logger.info(f"Registered agent {agent.name} with ID {agent.agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the lattice."""
        if agent_id in self.agents:
            self.agents[agent_id].stop_processing()
            del self.agents[agent_id]
            if agent_id in self.agent_capabilities:
                del self.agent_capabilities[agent_id]
            logger.info(f"Unregistered agent {agent_id}")
    
    def find_agents_with_capability(self, capability_name: str) -> List[str]:
        """Find all agents with a specific capability."""
        matching_agents = []
        for agent_id, capabilities in self.agent_capabilities.items():
            if capability_name in capabilities:
                matching_agents.append(agent_id)
        return matching_agents
    
    def get_agent_status_summary(self) -> Dict[str, Any]:
        """Get summary status of all agents."""
        return {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for agent in self.agents.values() if agent.active),
            "agent_roles": {role.value: 0 for role in AgentRole},
            "agents": {agent_id: agent.get_status() for agent_id, agent in self.agents.items()}
        }
    
    def broadcast_message(self, sender_id: str, message_type: MessageType, 
                         content: Dict[str, Any], exclude_sender: bool = True) -> List[str]:
        """Broadcast a message to all agents."""
        message_ids = []
        for agent_id, agent in self.agents.items():
            if exclude_sender and agent_id == sender_id:
                continue
            if agent.active:
                message_id = self.agents[sender_id].send_message(agent_id, message_type, content)
                message_ids.append(message_id)
        return message_ids


# Global agent registry
agent_registry = AgentRegistry()