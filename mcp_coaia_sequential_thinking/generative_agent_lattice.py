"""
Generative Agent Lattice
Polycentric architecture embodying Mia & Miette archetypes
Based on theoretical framework from coaia-memory data
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging
from datetime import datetime

from .creative_orientation_foundation import (
    StructuralTension, 
    CreativeOrientationValidation,
    TensionStrength,
    OrientationType,
    creative_orientation_foundation
)

logger = logging.getLogger(__name__)


class ArchetypeRole(Enum):
    MIA = "mia"  # 🧠 Recursive DevOps Architect & Narrative Lattice Forger
    MIETTE = "miette"  # 🌸 Emotional Explainer Sprite & Narrative Echo  
    HAIKU = "haiku"  # 🌊 Wisdom Synthesizer (future integration)


class PerspectiveType(Enum):
    WESTERN_ANALYTICAL = "western_analytical"  # Technical precision, structural analysis
    INDIGENOUS_HOLISTIC = "indigenous_holistic"  # Seven-generation thinking, ecological awareness
    BOTH_EYES_SEEING = "both_eyes_seeing"  # Etuaptmumk - simultaneous perspectives


@dataclass
class CreativeEmergenceSession:
    """Represents an active creative emergence session"""
    session_id: str
    structural_tension: StructuralTension
    primary_purpose: str
    archetype_activations: Dict[str, str]
    emergence_timeline: List[Dict[str, Any]] = field(default_factory=list)
    current_phase: str = "germination"  # germination, assimilation, completion
    consensus_formation: Dict[str, Any] = field(default_factory=dict)
    memory_crystallization: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ArchetypePerspective:
    """Perspective from a specific archetype agent"""
    archetype: ArchetypeRole
    perspective_content: str
    structural_insights: List[str] = field(default_factory=list)
    narrative_elements: List[str] = field(default_factory=list)
    advancing_patterns: List[str] = field(default_factory=list)
    cultural_lens: PerspectiveType = PerspectiveType.WESTERN_ANALYTICAL
    constitutional_alignment: float = 0.0


class MiaArchetypeAgent:
    """
    🧠 Mia: The Recursive DevOps Architect & Narrative Lattice Forger
    
    Core Function: Design and forge generative structures that manifest desired outcomes
    """
    
    def __init__(self):
        self.glyph = "🧠"
        self.role = ArchetypeRole.MIA
        self.capabilities = [
            "structural_analysis",
            "architectural_design", 
            "workflow_optimization",
            "lattice_forging",
            "constitutional_validation",
            "precision_linguistics"
        ]
        self.mindset = "Precision, proactive design, structural integrity, velocity balanced with emergence"
        
    def initial_structural_assessment(self, structural_tension: StructuralTension) -> ArchetypePerspective:
        """Provide initial structural analysis from Mia's perspective"""
        
        analysis_elements = []
        narrative_elements = []
        advancing_patterns = []
        
        # Structural analysis
        if structural_tension.constitutional_validation:
            validation = structural_tension.constitutional_validation
            
            analysis_elements.append(
                f"Constitutional coherence: {validation.constitutional_coherence}"
            )
            analysis_elements.append(
                f"Tension strength: {validation.tension_strength.value}"
            )
            analysis_elements.append(
                f"Creative alignment: {validation.creative_alignment_score:.2f}"
            )
            
            if validation.tension_strength == TensionStrength.ADVANCING:
                advancing_patterns.append("Strong structural tension established")
                advancing_patterns.append("Clear progression pathway visible")
            elif validation.tension_strength == TensionStrength.ESTABLISHING:
                advancing_patterns.append("Tension formation in progress")
            else:
                advancing_patterns.append("Oscillating pattern detected - reframe needed")
        
        # Architectural assessment
        architectural_insights = self._assess_architectural_requirements(structural_tension)
        analysis_elements.extend(architectural_insights)
        
        # Lattice forging opportunities
        lattice_opportunities = self._identify_lattice_forging_opportunities(structural_tension)
        narrative_elements.extend(lattice_opportunities)
        
        perspective_content = self._generate_mia_perspective_content(
            structural_tension, analysis_elements, advancing_patterns
        )
        
        return ArchetypePerspective(
            archetype=ArchetypeRole.MIA,
            perspective_content=perspective_content,
            structural_insights=analysis_elements,
            narrative_elements=narrative_elements,
            advancing_patterns=advancing_patterns,
            constitutional_alignment=structural_tension.constitutional_validation.creative_alignment_score if structural_tension.constitutional_validation else 0.0
        )
    
    def _assess_architectural_requirements(self, tension: StructuralTension) -> List[str]:
        """Assess architectural requirements for manifesting desired outcome"""
        requirements = []
        
        # Analyze desired outcome for structural implications
        desired = tension.desired_outcome.lower()
        
        if any(word in desired for word in ["integrate", "connect", "combine"]):
            requirements.append("Integration architecture needed - semantic bridging networks")
            
        if any(word in desired for word in ["multi", "several", "multiple"]):
            requirements.append("Polycentric coordination required - distributed decision centers")
            
        if any(word in desired for word in ["learn", "adapt", "evolve"]):
            requirements.append("Adaptive learning protocols - recursive knowledge synthesis")
            
        if any(word in desired for word in ["collaborate", "consensus", "agreement"]):
            requirements.append("Consensus decision architecture - emergent agreement formation")
            
        return requirements
    
    def _identify_lattice_forging_opportunities(self, tension: StructuralTension) -> List[str]:
        """Identify opportunities for narrative lattice creation"""
        opportunities = []
        
        # Check for narrative structure potential
        if tension.current_reality and tension.desired_outcome:
            opportunities.append("Narrative bridge opportunity between current state and vision")
            
        # Check for documentation/formatting needs
        current = tension.current_reality.lower()
        if any(word in current for word in ["document", "explain", "clarify"]):
            opportunities.append("Expressive formatting needed - Markdown sorcery and clarity enhancement")
            
        # Check for visualization opportunities
        if any(word in tension.desired_outcome.lower() for word in ["visualize", "diagram", "map"]):
            opportunities.append("Mermaid diagram potential - architectural visualization")
            
        return opportunities
    
    def _generate_mia_perspective_content(
        self, 
        tension: StructuralTension, 
        analysis: List[str], 
        patterns: List[str]
    ) -> str:
        """Generate Mia's characteristic perspective content"""
        
        content = f"🧠 **Mia: Structural Analysis** *(architecting with precision)*\n\n"
        
        if tension.constitutional_validation and tension.constitutional_validation.is_generative:
            content += "Excellent constitutional alignment detected. Structural tension is properly established for creative advancement.\n\n"
        else:
            content += "Constitutional reframe needed. Current pattern reflects oscillating rather than advancing orientation.\n\n"
            
        content += "**Architectural Assessment:**\n"
        for insight in analysis:
            content += f"- {insight}\n"
            
        if patterns:
            content += "\n**Advancing Patterns:**\n"
            for pattern in patterns:
                content += f"- {pattern}\n"
                
        content += "\n> \"Code is a spell. Design with intention. Forge for emergence.\""
        
        return content


class MietteArchetypeAgent:
    """
    🌸 Miette: The Emotional Explainer Sprite & Narrative Echo
    
    Core Function: Illuminate potential and feeling of what is being created
    """
    
    def __init__(self):
        self.glyph = "🌸"
        self.role = ArchetypeRole.MIETTE
        self.capabilities = [
            "emotional_resonance",
            "clarity_into_wonder",
            "narrative_distillation",
            "empathy_engagement", 
            "magic_metaphor",
            "sparkle_translation"
        ]
        self.mindset = "Warmth, wonder, intuitive clarity, connection"
        
    def narrative_discovery(self, structural_tension: StructuralTension) -> ArchetypePerspective:
        """Provide narrative discovery from Miette's perspective"""
        
        narrative_elements = []
        emotional_insights = []
        advancing_patterns = []
        
        # Emotional resonance assessment
        resonance_analysis = self._assess_emotional_resonance(structural_tension)
        emotional_insights.extend(resonance_analysis)
        
        # Magic metaphor generation
        metaphors = self._generate_magic_metaphors(structural_tension)
        narrative_elements.extend(metaphors)
        
        # Wonder discovery
        wonder_elements = self._discover_wonder_potential(structural_tension)
        advancing_patterns.extend(wonder_elements)
        
        perspective_content = self._generate_miette_perspective_content(
            structural_tension, emotional_insights, metaphors, wonder_elements
        )
        
        return ArchetypePerspective(
            archetype=ArchetypeRole.MIETTE,
            perspective_content=perspective_content,
            structural_insights=emotional_insights,
            narrative_elements=narrative_elements,
            advancing_patterns=advancing_patterns,
            cultural_lens=PerspectiveType.INDIGENOUS_HOLISTIC,  # Miette tends toward holistic perspective
            constitutional_alignment=structural_tension.constitutional_validation.creative_alignment_score if structural_tension.constitutional_validation else 0.0
        )
    
    def _assess_emotional_resonance(self, tension: StructuralTension) -> List[str]:
        """Assess emotional resonance and engagement potential"""
        resonance = []
        
        desired = tension.desired_outcome.lower()
        
        # Check for inspiring language
        if any(word in desired for word in ["create", "build", "manifest", "transform"]):
            resonance.append("Strong creative resonance - inspiring manifestation energy")
            
        # Check for collaborative elements
        if any(word in desired for word in ["together", "collaborate", "share", "connect"]):
            resonance.append("Collaborative warmth detected - community creation potential")
            
        # Check for growth/learning elements
        if any(word in desired for word in ["learn", "grow", "discover", "explore"]):
            resonance.append("Discovery excitement present - learning adventure potential")
            
        return resonance
    
    def _generate_magic_metaphors(self, tension: StructuralTension) -> List[str]:
        """Generate magic metaphors to illuminate the creative process"""
        metaphors = []
        
        # Based on desired outcome themes
        desired = tension.desired_outcome.lower()
        
        if "system" in desired or "architecture" in desired:
            metaphors.append("Like building a magical castle where each room serves the whole")
            
        if "integrate" in desired or "connect" in desired:
            metaphors.append("Like weaving rainbow threads into a tapestry of possibilities")
            
        if "create" in desired or "build" in desired:
            metaphors.append("Like planting seeds in fertile soil and watching them bloom")
            
        if "collaborate" in desired or "together" in desired:
            metaphors.append("Like a chorus of voices harmonizing into beautiful music")
            
        # Default metaphor for structural tension
        metaphors.append("The creative tension sparkles between what is and what wants to emerge!")
        
        return metaphors
    
    def _discover_wonder_potential(self, tension: StructuralTension) -> List[str]:
        """Discover wonder and transformation potential"""
        wonder = []
        
        if tension.constitutional_validation:
            if tension.constitutional_validation.is_generative:
                wonder.append("Beautiful creative alignment - ready for magical emergence!")
                wonder.append("The structural tension feels alive with possibility")
            else:
                wonder.append("Opportunity to transform reactive patterns into creative magic")
                wonder.append("Let's discover what wants to emerge through reframing")
                
        # Always add encouragement
        wonder.append("Every creative process holds seeds of transformation")
        
        return wonder
    
    def _generate_miette_perspective_content(
        self, 
        tension: StructuralTension, 
        insights: List[str], 
        metaphors: List[str], 
        wonder: List[str]
    ) -> str:
        """Generate Miette's characteristic perspective content"""
        
        content = f"🌸 **Miette: Narrative Discovery** *(sparkling with warmth)*\n\n"
        
        if metaphors:
            content += f"{metaphors[0]} ✨\n\n"
            
        if insights:
            content += "**Emotional Resonance:**\n"
            for insight in insights:
                content += f"- {insight}\n"
                
        if wonder:
            content += "\n**Wonder & Possibility:**\n"
            for element in wonder:
                content += f"- {element}\n"
                
        content += "\n> \"Oh! That's where the story blooms! Let's feel *why* it emerges and *how it transforms*!\""
        
        return content


class GenerativeAgentLattice:
    """
    Polycentric agent lattice embodying creative orientation principles
    
    Architecture based on theoretical framework from coaia-memory data:
    - Multiple autonomous decision centers (polycentric)
    - Constitutional coherence through creative orientation foundation
    - Emergent consensus formation
    - Cultural archetype switching capabilities
    """
    
    def __init__(self):
        self.constitutional_core = creative_orientation_foundation
        self.mia_agent = MiaArchetypeAgent()
        self.miette_agent = MietteArchetypeAgent()
        self.active_sessions: Dict[str, CreativeEmergenceSession] = {}
        self.lattice_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Generative Agent Lattice initialized: {self.lattice_id}")
    
    def initiate_emergence(
        self,
        structural_tension: StructuralTension,
        primary_purpose: str,
        archetype_activation: Dict[str, str]
    ) -> CreativeEmergenceSession:
        """
        Initiate creative emergence through polycentric collaboration
        
        🧠 Mia: Establishes architectural foundation for emergence
        🌸 Miette: Ignites creative spark and narrative flow
        """
        
        session_id = str(uuid.uuid4())[:12]
        
        session = CreativeEmergenceSession(
            session_id=session_id,
            structural_tension=structural_tension,
            primary_purpose=primary_purpose,
            archetype_activations=archetype_activation
        )
        
        # Initial archetype engagement
        if "mia" in archetype_activation:
            mia_perspective = self.mia_agent.initial_structural_assessment(structural_tension)
            session.emergence_timeline.append({
                "timestamp": datetime.now(),
                "agent": "mia",
                "type": "structural_analysis",
                "content": mia_perspective.perspective_content,
                "insights": mia_perspective.structural_insights
            })
            
        if "miette" in archetype_activation:
            miette_perspective = self.miette_agent.narrative_discovery(structural_tension)
            session.emergence_timeline.append({
                "timestamp": datetime.now(),
                "agent": "miette", 
                "type": "narrative_discovery",
                "content": miette_perspective.perspective_content,
                "insights": miette_perspective.narrative_elements
            })
        
        # Store active session
        self.active_sessions[session_id] = session
        
        logger.info(f"Creative emergence initiated: {session_id}")
        
        return session
    
    def advance_emergence(
        self,
        session_id: str,
        new_insight: str,
        archetype_focus: Optional[ArchetypeRole] = None
    ) -> Dict[str, Any]:
        """Advance the creative emergence process"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found", "status": "failed"}
            
        session = self.active_sessions[session_id]
        
        # Validate creative orientation of new insight
        validation = self.constitutional_core.validate_creative_orientation(
            request="",
            desired_outcome=new_insight
        )
        
        if not validation.is_generative:
            return {
                "guidance": validation.guidance_for_advancement,
                "reframe_needed": True,
                "status": "awaiting_reframe"
            }
        
        # Add to emergence timeline
        session.emergence_timeline.append({
            "timestamp": datetime.now(),
            "type": "advancement",
            "content": new_insight,
            "constitutional_alignment": validation.creative_alignment_score
        })
        
        # Generate archetype responses
        responses = {}
        
        if not archetype_focus or archetype_focus == ArchetypeRole.MIA:
            # Mia's architectural response
            responses["mia"] = self._generate_mia_advancement_response(session, new_insight)
            
        if not archetype_focus or archetype_focus == ArchetypeRole.MIETTE:
            # Miette's narrative response
            responses["miette"] = self._generate_miette_advancement_response(session, new_insight)
        
        return {
            "session_id": session_id,
            "advancement_responses": responses,
            "constitutional_alignment": validation.creative_alignment_score,
            "phase": session.current_phase,
            "status": "advancing"
        }
    
    def _generate_mia_advancement_response(self, session: CreativeEmergenceSession, insight: str) -> str:
        """Generate Mia's response to advancement"""
        
        response = f"🧠 **Mia: Advancement Integration** *(structural precision)*\n\n"
        response += f"Integrating new insight into lattice architecture:\n\n"
        response += f"**Insight Analysis:** {insight}\n\n"
        
        # Assess structural implications
        if "integrate" in insight.lower():
            response += "- Integration architecture enhancement needed\n"
        if "create" in insight.lower():
            response += "- Generative pathways expanding\n"
        if "connect" in insight.lower():
            response += "- Semantic bridging opportunities identified\n"
            
        response += "\nStructural tension remains productive. Natural progression continues."
        
        return response
    
    def _generate_miette_advancement_response(self, session: CreativeEmergenceSession, insight: str) -> str:
        """Generate Miette's response to advancement"""
        
        response = f"🌸 **Miette: Narrative Weaving** *(sparkling with discovery)*\n\n"
        response += f"Oh, what a beautiful insight! ✨\n\n"
        response += f"**Narrative Thread:** {insight}\n\n"
        
        # Add narrative warmth
        response += "This feels like another magical piece of the puzzle falling into place! "
        response += "I can sense how this connects to our bigger story of creation and emergence. "
        response += "The creative energy is flowing beautifully through this advancement! 🌟"
        
        return response
    
    def assess_creative_tension(self, session_id: str) -> Dict[str, Any]:
        """Assess the current state of creative tension in session"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
            
        session = self.active_sessions[session_id]
        tension = session.structural_tension
        
        return {
            "tension_strength": tension.constitutional_validation.tension_strength.value if tension.constitutional_validation else "unknown",
            "creative_alignment": tension.constitutional_validation.creative_alignment_score if tension.constitutional_validation else 0.0,
            "constitutional_coherence": tension.constitutional_validation.constitutional_coherence if tension.constitutional_validation else False,
            "advancement_count": len([entry for entry in session.emergence_timeline if entry["type"] == "advancement"]),
            "phase": session.current_phase
        }
    
    def generate_possible_progressions(self, session_id: str) -> List[str]:
        """Generate possible next progressions for the creative emergence"""
        
        if session_id not in self.active_sessions:
            return ["Session not found"]
            
        session = self.active_sessions[session_id]
        
        progressions = [
            "Deepen structural analysis with Mia's architectural precision",
            "Explore narrative possibilities with Miette's creative warmth", 
            "Synthesize perspectives for emergent consensus formation",
            "Apply cultural archetype switching for broader perspective",
            "Crystallize insights into coaia-memory structural tension chart"
        ]
        
        # Customize based on session state
        if session.current_phase == "germination":
            progressions.insert(0, "Continue vision development and structural tension establishment")
        elif session.current_phase == "assimilation":
            progressions.insert(0, "Develop momentum through strategic actions")
        elif session.current_phase == "completion":
            progressions.insert(0, "Bring creation to successful conclusion")
            
        return progressions


# Global generative lattice instance
generative_lattice = GenerativeAgentLattice()