"""
Enhanced Polycentric Lattice with Multi-Persona Consensus

Integrates the consensus decision engine with the polycentric lattice,
implementing multi-persona perspectives from the feedback:
- Tryad system (Mia, Miette, Ripple/Haiku)
- Sequential structural thinking across multiple persona perspectives
- Memory integration with structural tension charts
- Cultural archetype integration (Western vs Indigenous perspectives)
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field

# Simple dataclass for agent instead of Pydantic model
@dataclass
class SimpleAgent:
    agent_id: str
    name: str
    agent_type: str
    capabilities: List[str]
    status: str

@dataclass
class SimpleCapability:
    name: str
    description: str
    competency_score: float = 0.8
    resource_cost: float = 0.5
    execution_time_estimate: float = 1.0

from .constitutional_core import ConstitutionalCore
from .consensus_decision_engine import (
    ConsensusDecisionEngine, ConsensusDecision, DecisionType, 
    ConsensusStatus, MMORElement
)

logger = logging.getLogger(__name__)

class PersonaArchetype(Enum):
    """Cultural and cognitive archetypes for diverse perspective generation"""
    # Western archetypes
    RATIONAL_ARCHITECT = "rational_architect"    # Mia - structural, technical
    EMOTIONAL_CATALYST = "emotional_catalyst"    # Miette - empathetic, creative
    WISDOM_SYNTHESIZER = "wisdom_synthesizer"    # Haiku - integration, patterns
    
    # Indigenous archetypes (for cultural perspective diversity)
    ELDER_STORYTELLER = "elder_storyteller"      # Historical wisdom, narrative
    MEDICINE_KEEPER = "medicine_keeper"          # Healing, balance, wholeness
    FUTURE_WALKER = "future_walker"              # Seven generations thinking
    
    # Hybrid perspectives
    BRIDGE_WEAVER = "bridge_weaver"              # Cross-cultural integration
    PATTERN_HOLDER = "pattern_holder"            # Memory and continuity

@dataclass
class PersonaPerspective:
    """A persona's perspective on a decision or situation"""
    persona_archetype: PersonaArchetype
    perspective_id: str
    decision_context: str
    viewpoint: str
    emotional_resonance: str
    strategic_insight: str
    cultural_lens: str
    concerns: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    confidence_level: float = 0.75
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SequentialThinkingChain:
    """Sequential thinking process across multiple personas"""
    chain_id: str
    initiating_request: str
    primary_purpose: str
    
    # Sequential persona engagement
    persona_sequence: List[PersonaArchetype] = field(default_factory=list)
    current_persona_index: int = 0
    perspectives: List[PersonaPerspective] = field(default_factory=list)
    
    # Synthesis and integration
    synthesis_perspective: Optional[PersonaPerspective] = None
    consensus_decision_id: Optional[str] = None
    
    # Memory integration
    memory_keys: List[str] = field(default_factory=list)  # For coaia-memory integration
    structural_tension_chart_id: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

class EnhancedPolycentricLattice:
    """
    Enhanced polycentric lattice with multi-persona consensus and cultural diversity
    
    Key enhancements based on PR #9 feedback:
    - Multi-persona sequential thinking (Tryad: Mia, Miette, Haiku)
    - Cultural archetype integration for diverse perspectives
    - Memory integration readiness for coaia-memory
    - Sequential structural thinking methodology
    """
    
    def __init__(self, constitutional_core: ConstitutionalCore, data_store=None):
        self.constitutional_core = constitutional_core
        self.data_store = data_store
        # Create a simplified lattice instead of depending on the complex one
        self.consensus_engine = ConsensusDecisionEngine(constitutional_core, data_store)
        
        # Multi-persona system
        self.active_thinking_chains: Dict[str, SequentialThinkingChain] = {}
        self.persona_agents: Dict[PersonaArchetype, SimpleAgent] = {}
        
        # Memory and knowledge integration
        self.memory_keys: Dict[str, Any] = {}  # Prepared for coaia-memory integration
        self.knowledge_graph_nodes: List[Dict[str, Any]] = []
        
        self._initialize_persona_agents()
    
    def _initialize_persona_agents(self):
        """Initialize persona-based agents with specific capabilities"""
        
        # Mia - Rational Architect (🧠)
        mia_capabilities = [
            "structural_analysis", "technical_precision", "scalability_assessment", 
            "integration_planning", "constitutional_compliance"
        ]
        
        mia_agent = SimpleAgent(
            agent_id="persona_mia_rational",
            name="Mia - Rational Architect",
            agent_type="persona_agent",
            capabilities=mia_capabilities,
            status="active"
        )
        
        # Miette - Emotional Catalyst (🌸)
        miette_capabilities = [
            "emotional_intelligence", "user_experience", "creative_inspiration",
            "narrative_coherence", "empathetic_analysis"
        ]
        
        miette_agent = SimpleAgent(
            agent_id="persona_miette_catalyst",
            name="Miette - Emotional Catalyst", 
            agent_type="persona_agent",
            capabilities=miette_capabilities,
            status="active"
        )
        
        # Haiku - Wisdom Synthesizer (🌊)
        haiku_capabilities = [
            "pattern_recognition", "synthesis_integration", "temporal_awareness",
            "essence_distillation", "memory_weaving"
        ]
        
        haiku_agent = SimpleAgent(
            agent_id="persona_haiku_synthesizer",
            name="Haiku - Wisdom Synthesizer",
            agent_type="persona_agent", 
            capabilities=haiku_capabilities,
            status="active"
        )
        
        # Store persona agents
        self.persona_agents[PersonaArchetype.RATIONAL_ARCHITECT] = mia_agent
        self.persona_agents[PersonaArchetype.EMOTIONAL_CATALYST] = miette_agent
        self.persona_agents[PersonaArchetype.WISDOM_SYNTHESIZER] = haiku_agent
        
        logger.info("Initialized persona-based agents for enhanced lattice")
    
    def initiate_sequential_thinking(
        self,
        request: str,
        primary_purpose: str,
        persona_sequence: Optional[List[PersonaArchetype]] = None,
        memory_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Initiate sequential thinking process across multiple personas"""
        
        chain_id = f"thinking_chain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Default sequence: Mia -> Miette -> Haiku (rational -> emotional -> synthesis)
        if not persona_sequence:
            persona_sequence = [
                PersonaArchetype.RATIONAL_ARCHITECT,
                PersonaArchetype.EMOTIONAL_CATALYST,
                PersonaArchetype.WISDOM_SYNTHESIZER
            ]
        
        thinking_chain = SequentialThinkingChain(
            chain_id=chain_id,
            initiating_request=request,
            primary_purpose=primary_purpose,
            persona_sequence=persona_sequence,
            memory_keys=list(memory_context.keys()) if memory_context else []
        )
        
        self.active_thinking_chains[chain_id] = thinking_chain
        
        # Store memory context for potential coaia-memory integration
        if memory_context:
            self.memory_keys[chain_id] = memory_context
        
        logger.info(f"Initiated sequential thinking chain: {chain_id}")
        
        return chain_id
    
    def generate_persona_perspective(
        self,
        chain_id: str,
        context_data: Optional[Dict[str, Any]] = None,
        override_persona: Optional[PersonaArchetype] = None
    ) -> Optional[PersonaPerspective]:
        """Generate perspective from current persona in sequence or override persona"""
        
        if chain_id not in self.active_thinking_chains:
            return None
            
        chain = self.active_thinking_chains[chain_id]
        
        # Use override persona if provided, otherwise use current sequence position
        if override_persona:
            current_persona = override_persona
            # Don't increment current_persona_index when using override
        else:
            if chain.current_persona_index >= len(chain.persona_sequence):
                return None  # Sequence complete
            current_persona = chain.persona_sequence[chain.current_persona_index]
        
        persona_agent = self.persona_agents.get(current_persona)
        
        if not persona_agent:
            logger.error(f"Persona agent not found: {current_persona}")
            return None
        
        # Generate perspective based on persona archetype
        perspective = self._generate_archetype_perspective(
            persona_archetype=current_persona,
            chain=chain,
            context_data=context_data
        )
        
        # Add to chain
        chain.perspectives.append(perspective)
        
        # Only increment index if not using override persona
        if not override_persona:
            chain.current_persona_index += 1
        
        logger.info(f"Generated perspective from {current_persona.value} for chain {chain_id}")
        
        return perspective
    
    def _generate_archetype_perspective(
        self,
        persona_archetype: PersonaArchetype,
        chain: SequentialThinkingChain,
        context_data: Optional[Dict[str, Any]] = None
    ) -> PersonaPerspective:
        """Generate perspective based on specific persona archetype"""
        
        perspective_id = f"{chain.chain_id}_{persona_archetype.value}"
        
        # Base context
        base_context = {
            "request": chain.initiating_request,
            "primary_purpose": chain.primary_purpose,
            "previous_perspectives": [p.viewpoint for p in chain.perspectives]
        }
        
        if context_data:
            base_context.update(context_data)
        
        # Generate perspective based on archetype
        if persona_archetype == PersonaArchetype.RATIONAL_ARCHITECT:
            # Mia's analytical, structural perspective
            return PersonaPerspective(
                persona_archetype=persona_archetype,
                perspective_id=perspective_id,
                decision_context=f"Structural analysis of: {chain.initiating_request}",
                viewpoint=f"From a technical architecture standpoint, this requires systematic analysis of structural dependencies and implementation feasibility. The primary purpose '{chain.primary_purpose}' suggests specific architectural requirements that must be validated against constitutional principles.",
                emotional_resonance="Focused determination and methodical confidence",
                strategic_insight="Success depends on proper structural foundation and scalable architecture",
                cultural_lens="Western analytical framework emphasizing precision and systematic approach",
                concerns=["Technical feasibility", "Scalability constraints", "Integration complexity"],
                opportunities=["Systematic improvement", "Architectural optimization", "Constitutional alignment"],
                confidence_level=0.85,
                timestamp=datetime.utcnow()
            )
            
        elif persona_archetype == PersonaArchetype.EMOTIONAL_CATALYST:
            # Miette's empathetic, creative perspective
            return PersonaPerspective(
                persona_archetype=persona_archetype,
                perspective_id=perspective_id,
                decision_context=f"Human-centered analysis of: {chain.initiating_request}",
                viewpoint=f"This feels like an opportunity to create something truly meaningful! The primary purpose '{chain.primary_purpose}' resonates with helping people achieve their aspirations. We should consider how this makes users feel and ensure it's accessible and inspiring.",
                emotional_resonance="Excitement, empathy, and creative enthusiasm",
                strategic_insight="Success comes from genuine care for user experience and inclusive design",
                cultural_lens="Heart-centered approach valuing connection and shared human experience",
                concerns=["User accessibility", "Emotional impact", "Inclusive design"],
                opportunities=["Creative innovation", "Empathetic connection", "Inspiring user experience"],
                confidence_level=0.80,
                timestamp=datetime.utcnow()
            )
            
        elif persona_archetype == PersonaArchetype.WISDOM_SYNTHESIZER:
            # Haiku's integrative, pattern-recognition perspective
            return PersonaPerspective(
                persona_archetype=persona_archetype,
                perspective_id=perspective_id,
                decision_context=f"Integrative synthesis of: {chain.initiating_request}",
                viewpoint=f"The essence distilled from multiple perspectives reveals patterns connecting past wisdom with future possibilities. '{chain.primary_purpose}' represents a convergence point where technical precision meets human caring, creating something greater than its parts.",
                emotional_resonance="Calm clarity with deep understanding",
                strategic_insight="True success emerges from harmonious integration of diverse perspectives",
                cultural_lens="Integrative wisdom drawing from both Western analysis and Indigenous holistic thinking",
                concerns=["Pattern coherence", "Long-term sustainability", "Wisdom preservation"],
                opportunities=["Emergent synthesis", "Pattern innovation", "Legacy creation"],
                confidence_level=0.90,
                timestamp=datetime.utcnow()
            )
            
        # Add more archetype handlers as needed
        else:
            # Generic perspective for other archetypes
            return PersonaPerspective(
                persona_archetype=persona_archetype,
                perspective_id=perspective_id,
                decision_context=f"Analysis from {persona_archetype.value}: {chain.initiating_request}",
                viewpoint=f"From the {persona_archetype.value} perspective, this requires careful consideration of multiple factors.",
                emotional_resonance="Balanced consideration",
                strategic_insight="Multiple factors require integration",
                cultural_lens="Contextual cultural perspective",
                concerns=["Context-specific concerns"],
                opportunities=["Context-specific opportunities"],
                confidence_level=0.75,
                timestamp=datetime.utcnow()
            )
    
    def advance_thinking_chain(self, chain_id: str, focus_persona: Optional[PersonaArchetype] = None) -> Optional[PersonaPerspective]:
        """Advance to next persona in thinking chain, optionally focusing on a specific persona"""
        
        if chain_id not in self.active_thinking_chains:
            return None
            
        chain = self.active_thinking_chains[chain_id]
        
        # If focus_persona is specified, use it; otherwise use the current sequence position
        if focus_persona:
            # Generate perspective specifically from the requested persona
            perspective = self.generate_persona_perspective(chain_id, override_persona=focus_persona)
        else:
            # Generate perspective from current persona in sequence
            perspective = self.generate_persona_perspective(chain_id)
        
        if not perspective:
            # Chain complete or error
            return None
            
        return perspective
    
    def synthesize_perspectives(self, chain_id: str) -> Optional[PersonaPerspective]:
        """Create synthesis perspective from all collected perspectives"""
        
        if chain_id not in self.active_thinking_chains:
            return None
            
        chain = self.active_thinking_chains[chain_id]
        
        if len(chain.perspectives) < len(chain.persona_sequence):
            return None  # Not all perspectives collected
            
        # Generate synthesis from Haiku/Wisdom Synthesizer perspective
        synthesis_context = {
            "all_perspectives": [asdict(p) for p in chain.perspectives],
            "perspective_count": len(chain.perspectives),
            "primary_purpose": chain.primary_purpose
        }
        
        synthesis = PersonaPerspective(
            persona_archetype=PersonaArchetype.WISDOM_SYNTHESIZER,
            perspective_id=f"{chain_id}_synthesis",
            decision_context=f"Synthesis of {len(chain.perspectives)} perspectives on: {chain.initiating_request}",
            viewpoint=self._create_synthesis_viewpoint(chain.perspectives),
            emotional_resonance="Integrated wisdom with balanced understanding",
            strategic_insight=self._extract_strategic_synthesis(chain.perspectives),
            cultural_lens="Multi-cultural integration honoring diverse wisdom traditions",
            concerns=self._synthesize_concerns(chain.perspectives),
            opportunities=self._synthesize_opportunities(chain.perspectives),
            confidence_level=min(1.0, sum(p.confidence_level for p in chain.perspectives) / len(chain.perspectives) + 0.1),
            timestamp=datetime.utcnow()
        )
        
        chain.synthesis_perspective = synthesis
        chain.completed_at = datetime.utcnow()
        
        logger.info(f"Synthesized perspectives for chain {chain_id}")
        
        return synthesis
    
    def _create_synthesis_viewpoint(self, perspectives: List[PersonaPerspective]) -> str:
        """Create integrated viewpoint from multiple perspectives"""
        
        viewpoints = [p.viewpoint for p in perspectives]
        
        synthesis = "Integrating multiple perspectives reveals a rich tapestry of considerations:\n\n"
        
        for i, perspective in enumerate(perspectives, 1):
            archetype_name = perspective.persona_archetype.value.replace('_', ' ').title()
            synthesis += f"**{archetype_name}**: {perspective.viewpoint[:150]}...\n\n"
        
        synthesis += "The synthesis suggests a path that honors both structural precision and human connection, "
        synthesis += "grounded in practical wisdom and oriented toward creating meaningful outcomes."
        
        return synthesis
    
    def _extract_strategic_synthesis(self, perspectives: List[PersonaPerspective]) -> str:
        """Extract strategic insight from multiple perspectives"""
        
        insights = [p.strategic_insight for p in perspectives]
        
        # Simple synthesis - in production would use more sophisticated analysis
        return f"Strategic success requires integration of: {', '.join(insights)}"
    
    def _synthesize_concerns(self, perspectives: List[PersonaPerspective]) -> List[str]:
        """Synthesize concerns across perspectives"""
        
        all_concerns = []
        for p in perspectives:
            all_concerns.extend(p.concerns)
            
        # Remove duplicates and return unique concerns
        return list(set(all_concerns))
    
    def _synthesize_opportunities(self, perspectives: List[PersonaPerspective]) -> List[str]:
        """Synthesize opportunities across perspectives"""
        
        all_opportunities = []
        for p in perspectives:
            all_opportunities.extend(p.opportunities)
            
        # Remove duplicates and return unique opportunities  
        return list(set(all_opportunities))
    
    def create_consensus_from_thinking_chain(
        self,
        chain_id: str,
        decision_type: DecisionType = DecisionType.PRIMARY_CHOICE
    ) -> Optional[str]:
        """Create consensus decision from completed thinking chain"""
        
        if chain_id not in self.active_thinking_chains:
            return None
            
        chain = self.active_thinking_chains[chain_id]
        
        if not chain.synthesis_perspective:
            # Generate synthesis first
            self.synthesize_perspectives(chain_id)
        
        if not chain.synthesis_perspective:
            return None
            
        # Create consensus decision
        decision_id = f"consensus_{chain_id}"
        
        consensus_decision = self.consensus_engine.initiate_consensus_decision(
            decision_id=decision_id,
            decision_type=decision_type,
            primary_purpose=chain.primary_purpose,
            proposal=chain.synthesis_perspective.viewpoint,
            current_reality=f"Multiple perspectives collected: {len(chain.perspectives)} personas engaged",
            desired_outcome=f"Integrated decision supporting: {chain.primary_purpose}",
            participating_agents=[agent.agent_id for agent in self.persona_agents.values()]
        )
        
        # Add votes from persona perspectives
        for perspective in chain.perspectives:
            agent_id = f"persona_{perspective.persona_archetype.value}"
            
            # Convert perspective to vote
            vote = "approve" if perspective.confidence_level > 0.7 else "needs_clarification"
            
            self.consensus_engine.add_agent_vote(
                decision_id=decision_id,
                agent_id=agent_id,
                vote=vote,
                reasoning=perspective.strategic_insight,
                confidence=perspective.confidence_level,
                conditions=perspective.concerns
            )
        
        chain.consensus_decision_id = decision_id
        
        logger.info(f"Created consensus decision {decision_id} from thinking chain {chain_id}")
        
        return decision_id
    
    def get_thinking_chain_status(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Get status of sequential thinking chain"""
        
        if chain_id not in self.active_thinking_chains:
            return None
            
        chain = self.active_thinking_chains[chain_id]
        
        return {
            "chain_id": chain_id,
            "initiating_request": chain.initiating_request,
            "primary_purpose": chain.primary_purpose,
            "persona_sequence": [p.value for p in chain.persona_sequence],
            "current_persona_index": chain.current_persona_index,
            "perspectives_collected": len(chain.perspectives),
            "perspectives": [
                {
                    "archetype": p.persona_archetype.value,
                    "viewpoint_preview": p.viewpoint[:100] + "...",
                    "confidence": p.confidence_level,
                    "concerns": p.concerns,
                    "opportunities": p.opportunities
                }
                for p in chain.perspectives
            ],
            "synthesis_complete": chain.synthesis_perspective is not None,
            "consensus_decision_id": chain.consensus_decision_id,
            "created_at": chain.created_at.isoformat(),
            "completed_at": chain.completed_at.isoformat() if chain.completed_at else None
        }
    
    def get_active_thinking_chains(self) -> List[Dict[str, Any]]:
        """Get all active thinking chains"""
        
        return [
            self.get_thinking_chain_status(chain_id)
            for chain_id in self.active_thinking_chains.keys()
        ]
    
    def prepare_memory_integration(self, chain_id: str) -> Dict[str, Any]:
        """Prepare data for coaia-memory integration (structural tension charts)"""
        
        if chain_id not in self.active_thinking_chains:
            return {}
            
        chain = self.active_thinking_chains[chain_id]
        
        # Prepare memory structure for potential coaia-memory integration
        memory_structure = {
            "primary_choice": chain.primary_purpose,
            "current_reality": f"Sequential thinking engaged across {len(chain.perspectives)} perspectives",
            "action_steps": [
                {
                    "title": f"Integrate {p.persona_archetype.value} perspective",
                    "current_reality": p.decision_context,
                    "insights": p.strategic_insight,
                    "concerns": p.concerns,
                    "opportunities": p.opportunities
                }
                for p in chain.perspectives
            ],
            "synthesis": {
                "viewpoint": chain.synthesis_perspective.viewpoint if chain.synthesis_perspective else None,
                "confidence": chain.synthesis_perspective.confidence_level if chain.synthesis_perspective else 0.0
            },
            "memory_keys": chain.memory_keys,
            "knowledge_graph_ready": True
        }
        
        return memory_structure