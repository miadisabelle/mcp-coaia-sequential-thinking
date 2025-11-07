"""
Resilient Connection: Dynamic balance between goal-directed action and emergent exploration.

This module implements the final architectural paradigm from the survey, providing
mechanisms to maintain goals while enabling discovery of emergent possibilities.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import logging
import threading
import time
import random
import math
from collections import deque

from .constitutional_core import constitutional_core, ConstitutionalPrinciple
from .polycentric_lattice import BaseAgent, AgentRole, MessageType, MessagePriority, agent_registry

logger = logging.getLogger(__name__)


class ExplorationMode(Enum):
    """Modes of exploration in the resilient connection system."""
    EXPLOITATION = "goal_directed_exploitation"
    EXPLORATION = "novelty_driven_exploration"
    BALANCED = "dynamic_balance"
    ADAPTIVE = "context_adaptive"


class DiscoveryType(Enum):
    """Types of discoveries that can emerge from exploration."""
    NOVEL_APPROACH = "novel_approach"
    UNEXPECTED_CONNECTION = "unexpected_connection"
    ALTERNATIVE_GOAL = "alternative_goal"
    PROCESS_IMPROVEMENT = "process_improvement"
    EMERGENT_OPPORTUNITY = "emergent_opportunity"


@dataclass
class Goal:
    """Represents a goal in the resilient connection system."""
    goal_id: str
    description: str
    priority: float  # 0.0 to 1.0
    created_at: datetime
    target_completion: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    constitutional_alignment: float = 1.0  # 0.0 to 1.0
    metrics: Dict[str, float] = field(default_factory=dict)
    sub_goals: List[str] = field(default_factory=list)
    achieved: bool = False


@dataclass
class EmergentPossibility:
    """Represents an emergent possibility discovered through exploration."""
    possibility_id: str
    discovery_type: DiscoveryType
    description: str
    potential_value: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    discovered_at: datetime
    discovered_by: str  # agent_id
    related_goals: List[str] = field(default_factory=list)
    exploration_context: Dict[str, Any] = field(default_factory=dict)
    integration_feasibility: float = 0.5  # 0.0 to 1.0


@dataclass
class ExplorationPath:
    """Represents a path of exploration."""
    path_id: str
    starting_goal: str
    exploration_steps: List[Dict[str, Any]] = field(default_factory=list)
    discoveries: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    novelty_score: float = 0.0
    constitutional_compliance: float = 1.0


@dataclass
class BalanceMetrics:
    """Metrics for measuring the exploration-exploitation balance."""
    exploitation_ratio: float
    exploration_ratio: float
    goal_achievement_rate: float
    discovery_rate: float
    integration_success_rate: float
    constitutional_adherence: float
    overall_effectiveness: float
    timestamp: datetime


class NoveltySearchAgent(BaseAgent):
    """Agent specialized in novelty search and exploration."""
    
    def __init__(self, agent_id: str = None):
        if agent_id is None:
            agent_id = f"novelty_{uuid.uuid4().hex[:8]}"
        super().__init__(agent_id, AgentRole.CREATIVE, "Novelty Explorer")
        
        self.exploration_history: List[Dict[str, Any]] = []
        self.discovery_archive: Dict[str, EmergentPossibility] = {}
        self.novelty_threshold = 0.7  # Minimum novelty score for discoveries
        
        # Initialize capabilities
        from .polycentric_lattice import AgentCapability
        self.capabilities = {
            "novelty_search": AgentCapability(
                "novelty_search",
                "Novelty Search",
                "Search for novel solutions that differ from known approaches",
                competency_score=0.9,
                resource_cost=0.6,
                execution_time_estimate=3.0
            ),
            "pattern_disruption": AgentCapability(
                "pattern_disruption",
                "Pattern Disruption",
                "Disrupt existing patterns to discover new possibilities",
                competency_score=0.85,
                resource_cost=0.4,
                execution_time_estimate=2.0
            ),
            "emergent_opportunity_detection": AgentCapability(
                "emergent_opportunity_detection",
                "Emergent Opportunity Detection",
                "Detect and evaluate emergent opportunities",
                competency_score=0.88,
                resource_cost=0.3,
                execution_time_estimate=1.5
            )
        }
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process novelty search requests."""
        request_type = request.get("type")
        
        if request_type == "novelty_search":
            context = request.get("context", {})
            current_solutions = request.get("current_solutions", [])
            target_novelty = request.get("target_novelty", 0.7)
            
            novel_solutions = self._conduct_novelty_search(context, current_solutions, target_novelty)
            return {"novel_solutions": novel_solutions, "search_context": context}
        
        elif request_type == "evaluate_novelty":
            solution = request.get("solution", "")
            known_solutions = request.get("known_solutions", [])
            
            novelty_score = self._evaluate_novelty(solution, known_solutions)
            return {"novelty_score": novelty_score, "evaluation_details": self._get_novelty_details(solution)}
        
        elif request_type == "discover_opportunities":
            exploration_context = request.get("exploration_context", {})
            
            opportunities = self._discover_emergent_opportunities(exploration_context)
            return {"emergent_opportunities": opportunities}
        
        return {"error": "Unknown novelty search request type"}
    
    def get_capabilities(self) -> List:
        """Return novelty search agent capabilities."""
        return list(self.capabilities.values())
    
    def evaluate_collaboration_proposal(self, proposal) -> Tuple[bool, str]:
        """Evaluate collaboration proposals for novelty search."""
        novelty_keywords = [
            "novel", "creative", "innovative", "exploration", "discovery", "new", "different"
        ]
        
        task_desc = proposal.task_description.lower()
        has_novelty_relevance = any(keyword in task_desc for keyword in novelty_keywords)
        
        if has_novelty_relevance:
            return True, "Task requires novelty search and creative exploration"
        else:
            return False, "Task does not require novelty search capabilities"
    
    def evaluate_competition_challenge(self, challenge) -> Tuple[bool, str]:
        """Evaluate competition challenges for novelty."""
        # Novelty agent competes on creativity and innovation metrics
        novelty_criteria = [
            "creativity", "innovation", "novelty", "uniqueness", "originality"
        ]
        
        has_novelty_competition = any(criteria in challenge.evaluation_criteria 
                                    for criteria in novelty_criteria)
        
        if has_novelty_competition:
            return True, "Competition emphasizes novelty and creativity"
        else:
            return False, "Competition criteria do not emphasize novelty"
    
    def _conduct_novelty_search(self, context: Dict[str, Any], 
                              current_solutions: List[str], 
                              target_novelty: float) -> List[Dict[str, Any]]:
        """Conduct a novelty search to find solutions that differ from current ones."""
        novel_solutions = []
        
        # Generate variations and combinations
        for i in range(5):  # Generate 5 novel attempts
            novel_solution = self._generate_novel_solution(context, current_solutions)
            novelty_score = self._evaluate_novelty(novel_solution, current_solutions)
            
            if novelty_score >= target_novelty:
                constitutional_validation = constitutional_core.validate_content(novel_solution, context)
                
                solution_data = {
                    "solution": novel_solution,
                    "novelty_score": novelty_score,
                    "constitutional_compliance": constitutional_validation["constitutional_compliance_score"],
                    "generation_method": f"novelty_search_iteration_{i+1}",
                    "context_factors": list(context.keys())
                }
                
                novel_solutions.append(solution_data)
        
        # Sort by novelty score
        novel_solutions.sort(key=lambda x: x["novelty_score"], reverse=True)
        
        return novel_solutions[:3]  # Return top 3
    
    def _generate_novel_solution(self, context: Dict[str, Any], existing_solutions: List[str]) -> str:
        """Generate a novel solution that differs from existing ones."""
        # This is a simplified implementation - in practice would use more sophisticated generation
        novel_approaches = [
            "reverse the typical approach",
            "combine elements from different domains",
            "apply principles from nature",
            "use constraint removal strategy",
            "employ parallel processing approach",
            "integrate feedback loops",
            "leverage emergent properties",
            "apply systems thinking perspective"
        ]
        
        base_approach = random.choice(novel_approaches)
        context_elements = list(context.keys())
        
        if context_elements:
            context_element = random.choice(context_elements)
            novel_solution = f"Apply '{base_approach}' to the {context_element} aspect of the challenge, " \
                           f"creating a solution that fundamentally differs from existing approaches by " \
                           f"emphasizing emergent properties and structural dynamics."
        else:
            novel_solution = f"Create a solution using '{base_approach}' that emphasizes " \
                           f"structural innovation and emergent possibilities."
        
        return novel_solution
    
    def _evaluate_novelty(self, solution: str, known_solutions: List[str]) -> float:
        """Evaluate the novelty of a solution compared to known solutions."""
        if not known_solutions:
            return 0.8  # High novelty if no comparison solutions
        
        # Simple novelty calculation based on word overlap
        solution_words = set(solution.lower().split())
        
        max_similarity = 0.0
        for known in known_solutions:
            known_words = set(known.lower().split())
            if len(solution_words) > 0 and len(known_words) > 0:
                overlap = len(solution_words & known_words)
                total = len(solution_words | known_words)
                similarity = overlap / total if total > 0 else 0.0
                max_similarity = max(max_similarity, similarity)
        
        novelty = 1.0 - max_similarity
        return min(max(novelty, 0.0), 1.0)  # Clamp to [0,1]
    
    def _get_novelty_details(self, solution: str) -> Dict[str, Any]:
        """Get detailed analysis of a solution's novelty."""
        return {
            "unique_concepts": len(set(solution.lower().split())),
            "length_factor": min(len(solution) / 100, 1.0),
            "complexity_indicator": solution.count(',') + solution.count(';'),
            "innovation_keywords": self._count_innovation_keywords(solution)
        }
    
    def _count_innovation_keywords(self, text: str) -> int:
        """Count innovation-related keywords in text."""
        innovation_words = [
            "novel", "innovative", "creative", "unique", "original", "emergent",
            "breakthrough", "revolutionary", "groundbreaking", "paradigm"
        ]
        text_lower = text.lower()
        return sum(1 for word in innovation_words if word in text_lower)
    
    def _discover_emergent_opportunities(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover emergent opportunities in the exploration context."""
        opportunities = []
        
        # Look for patterns and connections in the context
        if context:
            for i in range(3):  # Generate 3 opportunities
                opportunity_type = random.choice(list(DiscoveryType))
                
                opportunity = {
                    "type": opportunity_type.value,
                    "description": self._generate_opportunity_description(opportunity_type, context),
                    "potential_value": random.uniform(0.6, 0.9),
                    "confidence": random.uniform(0.7, 0.85),
                    "context_relevance": len(context) / 10.0
                }
                
                opportunities.append(opportunity)
        
        return opportunities
    
    def _generate_opportunity_description(self, opportunity_type: DiscoveryType, 
                                        context: Dict[str, Any]) -> str:
        """Generate a description for an emergent opportunity."""
        context_keys = list(context.keys())
        
        if opportunity_type == DiscoveryType.NOVEL_APPROACH:
            return f"A novel approach that leverages {context_keys[0] if context_keys else 'available resources'} " \
                   f"in an unexpected way to achieve better outcomes."
        
        elif opportunity_type == DiscoveryType.UNEXPECTED_CONNECTION:
            if len(context_keys) >= 2:
                return f"An unexpected connection between {context_keys[0]} and {context_keys[1]} " \
                       f"that could unlock new possibilities."
            else:
                return "An unexpected connection that could unlock new possibilities."
        
        elif opportunity_type == DiscoveryType.EMERGENT_OPPORTUNITY:
            return f"An emergent opportunity arising from the intersection of current goals " \
                   f"and exploration activities."
        
        else:
            return f"A {opportunity_type.value.replace('_', ' ')} that emerged from the exploration process."


class ResilientConnectionEngine:
    """Core engine for managing the resilient connection between goals and exploration."""
    
    def __init__(self):
        self.active_goals: Dict[str, Goal] = {}
        self.emergent_possibilities: Dict[str, EmergentPossibility] = {}
        self.exploration_paths: Dict[str, ExplorationPath] = {}
        self.balance_history: deque = deque(maxlen=100)  # Keep last 100 balance measurements
        
        # Configuration
        self.exploration_ratio_target = 0.3  # 30% exploration, 70% exploitation
        self.novelty_threshold = 0.7
        self.integration_threshold = 0.6
        
        # State tracking
        self.current_mode = ExplorationMode.BALANCED
        self.last_balance_check = datetime.now()
        self.balance_check_interval = timedelta(minutes=5)
        
        # Performance tracking
        self.metrics = {
            "goals_achieved": 0,
            "possibilities_discovered": 0,
            "possibilities_integrated": 0,
            "exploration_success_rate": 0.0,
            "goal_achievement_rate": 0.0
        }
    
    def add_goal(self, description: str, priority: float = 0.5, 
                target_completion: Optional[datetime] = None) -> str:
        """Add a new goal to the system."""
        goal_id = str(uuid.uuid4())
        
        # Validate goal constitutionally
        validation = constitutional_core.validate_content(description, {"type": "goal"})
        
        goal = Goal(
            goal_id=goal_id,
            description=description,
            priority=priority,
            created_at=datetime.now(),
            target_completion=target_completion,
            constitutional_alignment=validation["constitutional_compliance_score"]
        )
        
        self.active_goals[goal_id] = goal
        logger.info(f"Added goal {goal_id}: {description}")
        
        return goal_id
    
    def discover_possibility(self, discovery_type: DiscoveryType, description: str,
                           discovered_by: str, potential_value: float = 0.5,
                           confidence: float = 0.5, 
                           exploration_context: Dict[str, Any] = None) -> str:
        """Register a newly discovered emergent possibility."""
        possibility_id = str(uuid.uuid4())
        
        possibility = EmergentPossibility(
            possibility_id=possibility_id,
            discovery_type=discovery_type,
            description=description,
            potential_value=potential_value,
            confidence=confidence,
            discovered_at=datetime.now(),
            discovered_by=discovered_by,
            exploration_context=exploration_context or {}
        )
        
        # Evaluate integration feasibility
        possibility.integration_feasibility = self._evaluate_integration_feasibility(possibility)
        
        # Find related goals
        possibility.related_goals = self._find_related_goals(possibility)
        
        self.emergent_possibilities[possibility_id] = possibility
        self.metrics["possibilities_discovered"] += 1
        
        logger.info(f"Discovered possibility {possibility_id}: {description}")
        
        return possibility_id
    
    def evaluate_resilient_connection_strength(self) -> float:
        """Evaluate the current strength of the resilient connection."""
        if not self.active_goals:
            return 0.0
        
        # Calculate goal alignment
        goal_alignment = sum(goal.constitutional_alignment * goal.priority 
                           for goal in self.active_goals.values()) / len(self.active_goals)
        
        # Calculate exploration effectiveness
        recent_discoveries = sum(1 for p in self.emergent_possibilities.values() 
                               if (datetime.now() - p.discovered_at).days <= 7)
        exploration_effectiveness = min(recent_discoveries / 5.0, 1.0)  # Normalize to 5 discoveries per week
        
        # Calculate integration success
        high_value_possibilities = sum(1 for p in self.emergent_possibilities.values() 
                                     if p.potential_value > 0.7)
        integration_potential = min(high_value_possibilities / 3.0, 1.0)  # Normalize to 3 high-value possibilities
        
        # Calculate balance stability
        if len(self.balance_history) > 0:
            recent_balance = self.balance_history[-1]
            balance_stability = 1.0 - abs(recent_balance.exploitation_ratio - (1.0 - self.exploration_ratio_target))
        else:
            balance_stability = 0.5
        
        # Weighted combination
        connection_strength = (
            goal_alignment * 0.3 +
            exploration_effectiveness * 0.25 +
            integration_potential * 0.25 +
            balance_stability * 0.2
        )
        
        return connection_strength
    
    def adjust_exploration_balance(self, current_context: Dict[str, Any]) -> ExplorationMode:
        """Dynamically adjust the exploration-exploitation balance."""
        connection_strength = self.evaluate_resilient_connection_strength()
        
        # Calculate current ratios
        total_activities = len(self.active_goals) + len(self.emergent_possibilities)
        if total_activities == 0:
            return ExplorationMode.BALANCED
        
        goal_activities = len([g for g in self.active_goals.values() if g.progress < 1.0])
        exploration_activities = len(self.emergent_possibilities)
        
        current_exploitation_ratio = goal_activities / total_activities
        current_exploration_ratio = exploration_activities / total_activities
        
        # Determine if adjustment is needed
        target_exploitation = 1.0 - self.exploration_ratio_target
        exploitation_deviation = abs(current_exploitation_ratio - target_exploitation)
        
        if exploitation_deviation > 0.2:  # Significant deviation
            if current_exploitation_ratio > target_exploitation:
                new_mode = ExplorationMode.EXPLORATION  # Need more exploration
            else:
                new_mode = ExplorationMode.EXPLOITATION  # Need more goal focus
        else:
            new_mode = ExplorationMode.BALANCED
        
        # Consider constitutional factors
        avg_constitutional_compliance = sum(g.constitutional_alignment for g in self.active_goals.values()) / len(self.active_goals) if self.active_goals else 1.0
        
        if avg_constitutional_compliance < 0.7:
            new_mode = ExplorationMode.EXPLOITATION  # Focus on constitutional alignment
        
        # Record balance metrics
        balance_metrics = BalanceMetrics(
            exploitation_ratio=current_exploitation_ratio,
            exploration_ratio=current_exploration_ratio,
            goal_achievement_rate=self._calculate_goal_achievement_rate(),
            discovery_rate=self._calculate_discovery_rate(),
            integration_success_rate=self._calculate_integration_success_rate(),
            constitutional_adherence=avg_constitutional_compliance,
            overall_effectiveness=connection_strength,
            timestamp=datetime.now()
        )
        
        self.balance_history.append(balance_metrics)
        self.current_mode = new_mode
        self.last_balance_check = datetime.now()
        
        logger.info(f"Adjusted exploration balance to {new_mode.value}, connection strength: {connection_strength:.3f}")
        
        return new_mode
    
    def integrate_emergent_possibility(self, possibility_id: str, 
                                     integration_strategy: str = "enhancement") -> Dict[str, Any]:
        """Integrate an emergent possibility with existing goals."""
        if possibility_id not in self.emergent_possibilities:
            return {"error": "Possibility not found"}
        
        possibility = self.emergent_possibilities[possibility_id]
        
        # Validate integration constitutionally
        integration_description = f"Integrate discovered possibility: {possibility.description}"
        validation = constitutional_core.validate_content(integration_description, {
            "type": "integration",
            "discovery_type": possibility.discovery_type.value
        })
        
        if not validation["overall_valid"]:
            return {
                "error": "Integration violates constitutional principles",
                "violations": validation["violated_principles"]
            }
        
        # Find best related goals for integration
        related_goals = [self.active_goals[goal_id] for goal_id in possibility.related_goals 
                        if goal_id in self.active_goals]
        
        integration_results = []
        
        for goal in related_goals:
            if integration_strategy == "enhancement":
                # Enhance existing goal with the possibility
                enhanced_description = f"{goal.description} Enhanced by: {possibility.description}"
                goal.description = enhanced_description
                goal.metrics[f"enhanced_by_{possibility_id}"] = possibility.potential_value
                
                integration_results.append({
                    "goal_id": goal.goal_id,
                    "integration_type": "enhancement",
                    "enhancement_value": possibility.potential_value
                })
            
            elif integration_strategy == "new_goal":
                # Create new goal based on the possibility
                new_goal_id = self.add_goal(
                    f"Pursue opportunity: {possibility.description}",
                    priority=possibility.potential_value * 0.8,  # Slightly lower priority than value
                )
                
                integration_results.append({
                    "goal_id": new_goal_id,
                    "integration_type": "new_goal",
                    "derived_from_possibility": possibility_id
                })
        
        # Mark possibility as integrated
        possibility.integration_feasibility = 1.0
        self.metrics["possibilities_integrated"] += 1
        
        logger.info(f"Integrated possibility {possibility_id} using {integration_strategy} strategy")
        
        return {
            "integration_successful": True,
            "integration_strategy": integration_strategy,
            "integration_results": integration_results,
            "constitutional_compliance": validation["constitutional_compliance_score"]
        }
    
    def generate_exploration_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for future exploration based on current state."""
        recommendations = []
        
        connection_strength = self.evaluate_resilient_connection_strength()
        current_mode = self.current_mode
        
        # Analyze gaps and opportunities
        if connection_strength < 0.5:
            recommendations.append({
                "priority": "high",
                "category": "connection_strengthening",
                "recommendation": "Focus on aligning goals with constitutional principles and improving exploration quality",
                "reasoning": f"Current connection strength ({connection_strength:.2f}) is below optimal threshold"
            })
        
        if len(self.emergent_possibilities) < 3:
            recommendations.append({
                "priority": "medium",
                "category": "exploration_expansion",
                "recommendation": "Increase exploration activities to discover more emergent possibilities",
                "reasoning": "Limited emergent possibilities may restrict future opportunities"
            })
        
        # Check for stagnant goals
        stagnant_goals = [g for g in self.active_goals.values() 
                         if g.progress < 0.1 and (datetime.now() - g.created_at).days > 7]
        
        if stagnant_goals:
            recommendations.append({
                "priority": "medium",
                "category": "goal_activation",
                "recommendation": f"Address {len(stagnant_goals)} stagnant goals through enhanced exploration or restructuring",
                "reasoning": "Stagnant goals may benefit from fresh perspectives or alternative approaches"
            })
        
        # Constitutional alignment recommendations
        low_alignment_goals = [g for g in self.active_goals.values() if g.constitutional_alignment < 0.7]
        
        if low_alignment_goals:
            recommendations.append({
                "priority": "high",
                "category": "constitutional_alignment",
                "recommendation": f"Review and realign {len(low_alignment_goals)} goals with constitutional principles",
                "reasoning": "Constitutional misalignment threatens system integrity"
            })
        
        return recommendations
    
    def _evaluate_integration_feasibility(self, possibility: EmergentPossibility) -> float:
        """Evaluate how feasible it is to integrate a possibility with existing goals."""
        if not self.active_goals:
            return 0.5  # Neutral if no goals
        
        # Consider potential value and confidence
        base_feasibility = (possibility.potential_value + possibility.confidence) / 2.0
        
        # Consider constitutional alignment
        validation = constitutional_core.validate_content(possibility.description, {
            "type": "possibility_integration"
        })
        constitutional_factor = validation["constitutional_compliance_score"]
        
        # Consider goal relevance
        related_goal_count = len(self._find_related_goals(possibility))
        relevance_factor = min(related_goal_count / 3.0, 1.0)  # Normalize to 3 related goals
        
        # Weighted combination
        feasibility = (
            base_feasibility * 0.4 +
            constitutional_factor * 0.4 +
            relevance_factor * 0.2
        )
        
        return feasibility
    
    def _find_related_goals(self, possibility: EmergentPossibility) -> List[str]:
        """Find goals that are related to an emergent possibility."""
        related_goals = []
        
        possibility_words = set(possibility.description.lower().split())
        
        for goal_id, goal in self.active_goals.items():
            goal_words = set(goal.description.lower().split())
            
            # Simple relevance calculation based on word overlap
            overlap = len(possibility_words & goal_words)
            total = len(possibility_words | goal_words)
            
            if total > 0:
                relevance = overlap / total
                if relevance > 0.2:  # 20% word overlap threshold
                    related_goals.append(goal_id)
        
        return related_goals
    
    def _calculate_goal_achievement_rate(self) -> float:
        """Calculate the rate of goal achievement."""
        if not self.active_goals:
            return 0.0
        
        achieved_goals = sum(1 for goal in self.active_goals.values() if goal.achieved)
        return achieved_goals / len(self.active_goals)
    
    def _calculate_discovery_rate(self) -> float:
        """Calculate the rate of discovery of emergent possibilities."""
        if not self.emergent_possibilities:
            return 0.0
        
        recent_discoveries = sum(1 for p in self.emergent_possibilities.values()
                               if (datetime.now() - p.discovered_at).days <= 7)
        
        return min(recent_discoveries / 5.0, 1.0)  # Normalize to 5 per week
    
    def _calculate_integration_success_rate(self) -> float:
        """Calculate the success rate of integrating possibilities."""
        if not self.emergent_possibilities:
            return 0.0
        
        integrated_possibilities = sum(1 for p in self.emergent_possibilities.values()
                                     if p.integration_feasibility > 0.8)
        
        return integrated_possibilities / len(self.emergent_possibilities)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the resilient connection system."""
        connection_strength = self.evaluate_resilient_connection_strength()
        
        return {
            "resilient_connection": {
                "connection_strength": connection_strength,
                "current_mode": self.current_mode.value,
                "exploration_ratio_target": self.exploration_ratio_target,
                "last_balance_check": self.last_balance_check.isoformat()
            },
            "goals": {
                "total_active": len(self.active_goals),
                "achieved": sum(1 for g in self.active_goals.values() if g.achieved),
                "high_priority": sum(1 for g in self.active_goals.values() if g.priority > 0.7),
                "constitutional_alignment_avg": sum(g.constitutional_alignment for g in self.active_goals.values()) / len(self.active_goals) if self.active_goals else 0.0
            },
            "emergent_possibilities": {
                "total_discovered": len(self.emergent_possibilities),
                "high_value": sum(1 for p in self.emergent_possibilities.values() if p.potential_value > 0.7),
                "highly_feasible": sum(1 for p in self.emergent_possibilities.values() if p.integration_feasibility > 0.8),
                "discovery_types": {dt.value: sum(1 for p in self.emergent_possibilities.values() if p.discovery_type == dt) for dt in DiscoveryType}
            },
            "performance_metrics": self.metrics,
            "balance_history_length": len(self.balance_history),
            "recommendations": self.generate_exploration_recommendations()
        }


# Global resilient connection engine
resilient_connection = ResilientConnectionEngine()