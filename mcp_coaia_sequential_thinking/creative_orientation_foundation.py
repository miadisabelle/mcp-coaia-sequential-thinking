"""
Creative Orientation Foundation
Establishes constitutional core for generative architecture
Based on creative orientation principles from __llms/* guidance
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class OrientationType(Enum):
    CREATIVE = "creative"  # Generative, manifestation-focused
    REACTIVE = "reactive"  # Problem-solving, elimination-focused


class TensionStrength(Enum):
    ADVANCING = "advancing"  # Clear progression toward desired outcome
    OSCILLATING = "oscillating"  # Stuck in reactive cycles
    ESTABLISHING = "establishing"  # Beginning to form structural tension


@dataclass
class CreativeOrientationValidation:
    """Validation result for creative orientation alignment"""
    is_generative: bool
    orientation_type: OrientationType
    tension_strength: TensionStrength
    creative_alignment_score: float  # 0.0 to 1.0
    guidance_for_advancement: List[str]
    constitutional_coherence: bool
    delayed_resolution_applied: bool
    

@dataclass 
class StructuralTension:
    """Core structural tension following Robert Fritz's methodology"""
    desired_outcome: str
    current_reality: str
    natural_progression: Optional[List[str]] = None
    tension_id: Optional[str] = None
    constitutional_validation: Optional[CreativeOrientationValidation] = None


class ConstitutionalCore:
    """
    Constitutional foundation implementing creative orientation principles
    
    🧠 Mia: This is the immutable governance layer that ensures all system
    operations align with creative orientation rather than reactive patterns.
    
    🌸 Miette: Like a warm, wise guardian that gently guides everything
    toward creation and growth! ✨
    """
    
    def __init__(self):
        self.creative_orientation_principles = self._load_constitutional_principles()
        self.problem_solving_patterns = self._load_reactive_patterns_to_avoid()
        self.advancing_language_patterns = self._load_creative_language_patterns()
        
    def _load_constitutional_principles(self) -> List[str]:
        """Load core creative orientation principles"""
        return [
            # From coaia-memory theoretical framework
            "Creative Priority: Prioritize bringing desired outcomes into being over eliminating unwanted conditions",
            "Non-Fabrication: Acknowledge uncertainty rather than inventing facts when knowledge insufficient",
            "Error as Compass: Treat failures as navigational cues for improvement rather than problems to hide",
            "Structural Awareness: Recognize that underlying structure determines behavior patterns",
            "Tension Establishment: Establish clear structural tension between current reality and desired outcomes",
            "Polycentric Coherence: Foster emergent behaviors through multiple autonomous decision centers",
            "Resilient Connection: Maintain core goals while engaging in open exploration",
            "Dialogue as Structure: Quality of inquiry depends on conversational structure over cognitive sophistication",
            
            # From Fritz's creative orientation framework
            "Delayed Resolution Principle: Tolerate tension and delayed resolution rather than premature closure",
            "Current Reality Clarity: Assess what is objectively without exaggeration or assumptions",
            "Desired Outcome Primacy: Focus on what you want to create independent of circumstances",
            "Natural Progression Discovery: Allow structural tension to reveal next steps organically",
            "Three-Phase Respect: Honor germination, assimilation, and completion phases of creation",
        ]
    
    def _load_reactive_patterns_to_avoid(self) -> List[str]:
        """Patterns that indicate problem-solving rather than creative orientation"""
        return [
            # Problem-solving language
            r"\b(fix|solve|eliminate|remove|get rid of|stop|prevent|avoid|troubleshoot)\b",
            r"\b(issue|problem|bug|error|failure|broken|wrong)\b",
            r"\b(gap.*(bridge|fill|close))\b",
            
            # Premature resolution patterns  
            r"\b(ready to|prepared to|all set|good to go)\b",
            r"\b(just need to|simply|easily|quick)\b",
            
            # Reactive assumptions
            r"\b(should|must|have to|need to)\b.*\b(immediately|now|first)\b",
        ]
    
    def _load_creative_language_patterns(self) -> List[str]:
        """Patterns that indicate creative/generative orientation"""
        return [
            # Creative manifestation language
            r"\b(create|generate|build|manifest|bring forth|establish|develop)\b",
            r"\b(desired outcome|vision|intention|purpose|goal)\b",
            r"\b(current reality|where.*now|present state)\b",
            r"\b(structural tension|creative tension)\b",
            r"\b(natural progression|organic development|emerging path)\b",
            
            # Advancing patterns
            r"\b(advance|progress|move toward|evolve|grow|emerge)\b",
            r"\b(possibilities|potential|opportunities)\b",
        ]
    
    def validate_creative_orientation(
        self, 
        request: str, 
        desired_outcome: str,
        context: Optional[str] = None
    ) -> CreativeOrientationValidation:
        """
        Validate alignment with creative orientation principles
        
        🧠 Mia: Performs constitutional validation to ensure generative patterns
        🌸 Miette: Gently guides toward creative reframing when needed
        """
        
        combined_text = f"{request} {desired_outcome} {context or ''}"
        
        # Check for reactive patterns
        reactive_score = self._calculate_reactive_patterns(combined_text)
        creative_score = self._calculate_creative_patterns(combined_text)
        
        # Apply delayed resolution principle
        delayed_resolution_applied = self._check_delayed_resolution_compliance(combined_text)
        
        # Determine orientation type
        if creative_score > reactive_score and creative_score > 0.2:  # Lowered threshold
            orientation_type = OrientationType.CREATIVE
            is_generative = True
        else:
            orientation_type = OrientationType.REACTIVE  
            is_generative = False
        
        # Assess structural tension strength
        tension_strength = self._assess_tension_strength(desired_outcome, request)
        
        # Generate guidance for advancement
        guidance = self._generate_advancement_guidance(
            orientation_type, 
            tension_strength, 
            reactive_score, 
            creative_score
        )
        
        return CreativeOrientationValidation(
            is_generative=is_generative,
            orientation_type=orientation_type,
            tension_strength=tension_strength,
            creative_alignment_score=creative_score,
            guidance_for_advancement=guidance,
            constitutional_coherence=creative_score > 0.4,  # Lowered threshold
            delayed_resolution_applied=delayed_resolution_applied
        )
    
    def _calculate_reactive_patterns(self, text: str) -> float:
        """Calculate reactive/problem-solving pattern density"""
        total_matches = 0
        for pattern in self.problem_solving_patterns:
            matches = len(re.findall(pattern, text.lower()))
            total_matches += matches
            
        # Normalize by text length
        words = len(text.split())
        return min(total_matches / max(words, 1), 1.0)
    
    def _calculate_creative_patterns(self, text: str) -> float:
        """Calculate creative/generative pattern density"""
        total_matches = 0
        for pattern in self.advancing_language_patterns:
            matches = len(re.findall(pattern, text.lower()))
            total_matches += matches * 2  # Weight creative patterns more heavily
            
        # Normalize by text length with boost for creative content
        words = len(text.split())
        base_score = total_matches / max(words, 1)
        
        # Boost for explicit creative verbs
        creative_verbs = ["create", "manifest", "generate", "build", "establish", "develop"]
        for verb in creative_verbs:
            if verb in text.lower():
                base_score += 0.15
                
        return min(base_score, 1.0)
    
    def _check_delayed_resolution_compliance(self, text: str) -> bool:
        """Check if delayed resolution principle is being applied"""
        # Look for premature resolution patterns
        premature_patterns = [
            r"\b(ready|prepared|all set)\b",
            r"\b(just need to|simply|easily)\b",
        ]
        
        for pattern in premature_patterns:
            if re.search(pattern, text.lower()):
                return False
                
        return True
    
    def _assess_tension_strength(self, desired_outcome: str, current_reality: str) -> TensionStrength:
        """Assess strength of structural tension"""
        
        # Check if both elements are present and substantial
        if not desired_outcome or not current_reality:
            return TensionStrength.ESTABLISHING
        
        # Check for creative language in desired outcome
        creative_in_outcome = any(
            re.search(pattern, desired_outcome.lower()) 
            for pattern in self.advancing_language_patterns
        )
        
        # Check for objective current reality (not reactive)
        objective_reality = not any(
            re.search(pattern, current_reality.lower())
            for pattern in self.problem_solving_patterns[:3]  # Core problem patterns
        )
        
        if creative_in_outcome and objective_reality:
            return TensionStrength.ADVANCING
        elif creative_in_outcome or objective_reality:
            return TensionStrength.ESTABLISHING
        else:
            return TensionStrength.OSCILLATING
    
    def _generate_advancement_guidance(
        self, 
        orientation_type: OrientationType,
        tension_strength: TensionStrength,
        reactive_score: float,
        creative_score: float
    ) -> List[str]:
        """Generate guidance for advancing toward creative orientation"""
        
        guidance = []
        
        if orientation_type == OrientationType.REACTIVE:
            guidance.append("🧠 Mia: Reframe from elimination orientation to creative manifestation")
            guidance.append("🌸 Miette: What do you want to CREATE instead of what you want to solve?")
            
        if tension_strength == TensionStrength.OSCILLATING:
            guidance.append("🧠 Mia: Establish clear structural tension between desired outcome and current reality")
            guidance.append("🌸 Miette: Let's discover what you truly want to bring into being!")
            
        if reactive_score > 0.3:
            guidance.append("Consider shifting language from problem-solving to creative manifestation")
            
        if creative_score < 0.2:
            guidance.append("Add more generative language: create, manifest, establish, bring forth")
            
        return guidance
    
    def guide_toward_creative_reframe(self, validation: CreativeOrientationValidation) -> Dict[str, Any]:
        """Provide guidance for reframing toward creative orientation"""
        
        return {
            "reframe_needed": True,
            "current_orientation": validation.orientation_type.value,
            "guidance": validation.guidance_for_advancement,
            "constitutional_principles": self.creative_orientation_principles[:5],  # Top 5
            "creative_reframe_examples": [
                {
                    "reactive": "Fix the broken authentication system",
                    "creative": "Create a secure, reliable authentication experience for users"
                },
                {
                    "reactive": "Solve the performance issues in the app", 
                    "creative": "Manifest optimal performance and responsive user experience"
                },
                {
                    "reactive": "Get rid of the confusing interface",
                    "creative": "Design an intuitive, delightful user interface"
                }
            ],
            "next_steps": [
                "Clarify your desired outcome (what you want to create)",
                "Assess current reality objectively (where you are now)",
                "Establish structural tension between the two",
                "Allow natural progression to emerge"
            ]
        }


# Global constitutional foundation instance
creative_orientation_foundation = ConstitutionalCore()


def establish_structural_tension(
    desired_outcome: str,
    current_reality: str,
    context: Optional[str] = None
) -> StructuralTension:
    """
    Establish structural tension following constitutional principles
    
    🧠 Mia: Creates the foundational structure for creative advancement
    🌸 Miette: Sets up the magical tension that draws new realities into being!
    """
    
    validation = creative_orientation_foundation.validate_creative_orientation(
        request=current_reality,
        desired_outcome=desired_outcome,
        context=context
    )
    
    tension = StructuralTension(
        desired_outcome=desired_outcome,
        current_reality=current_reality,
        constitutional_validation=validation
    )
    
    logger.info(f"Structural tension established: {validation.tension_strength.value}")
    
    return tension


def validate_creative_action(action_description: str) -> bool:
    """
    Validate that an action aligns with creative orientation
    
    Returns True if action is generative, False if reactive
    """
    validation = creative_orientation_foundation.validate_creative_orientation(
        request="",
        desired_outcome=action_description
    )
    
    return validation.is_generative