"""
Constitutional Core: Immutable principles governing "how to think" in the generative agentic system.

This module implements the "Vortex Core" concept from the architectural survey,
providing an immutable constitutional layer that governs the system's thinking
processes while keeping operational protocols flexible.
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ConstitutionalPrinciple(Enum):
    """Immutable constitutional principles governing system thinking."""
    
    # Core Creative Orientation Principles
    NON_FABRICATION = "acknowledge_uncertainty_rather_than_invent_facts"
    ERROR_AS_COMPASS = "treat_failure_as_navigational_cue_for_improvement"
    CREATIVE_PRIORITY = "prioritize_creating_desired_outcomes_over_eliminating_problems"
    STRUCTURAL_AWARENESS = "recognize_underlying_structure_determines_behavior"
    TENSION_ESTABLISHMENT = "establish_clear_tension_between_current_reality_and_desired_outcome"
    
    # Meta-Thinking Principles
    START_WITH_NOTHING = "begin_inquiry_without_preconceptions_or_hypotheses"
    PICTURE_WHAT_IS_SAID = "translate_verbal_to_visual_for_dimensional_thinking"
    QUESTION_INTERNALLY = "ask_questions_driven_by_provided_information_only"
    MULTIPLE_PERSPECTIVES = "generate_multiple_viewpoints_before_selection"
    
    # Decision-Making Principles
    PRINCIPLE_OVER_EXPEDIENCE = "constitutional_principles_override_operational_convenience"
    TRANSPARENCY_REQUIREMENT = "all_decisions_must_be_traceable_to_constitutional_principles"
    ADAPTIVE_PROTOCOLS = "operational_methods_can_change_but_principles_remain_immutable"
    CONFLICT_RESOLUTION = "resolve_conflicts_through_principle_hierarchy_not_compromise"


@dataclass
class ConstitutionalDecision:
    """Record of a decision made according to constitutional principles."""
    decision_id: str
    timestamp: datetime
    decision_context: str
    applicable_principles: List[ConstitutionalPrinciple]
    principle_application: Dict[ConstitutionalPrinciple, str]
    decision_outcome: str
    alternative_considered: List[str]
    principle_conflicts: Optional[List[Tuple[ConstitutionalPrinciple, ConstitutionalPrinciple]]] = None
    resolution_method: Optional[str] = None


class ConstitutionalValidator(ABC):
    """Abstract base class for constitutional validation mechanisms."""
    
    @abstractmethod
    def validate_against_principle(self, principle: ConstitutionalPrinciple, 
                                 content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate content against a specific constitutional principle."""
        pass


class CreativeOrientationValidator(ConstitutionalValidator):
    """Validator for creative orientation constitutional principles."""
    
    def validate_against_principle(self, principle: ConstitutionalPrinciple, 
                                 content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate content against creative orientation principles."""
        
        if principle == ConstitutionalPrinciple.NON_FABRICATION:
            return self._validate_non_fabrication(content, context)
        elif principle == ConstitutionalPrinciple.ERROR_AS_COMPASS:
            return self._validate_error_as_compass(content, context)
        elif principle == ConstitutionalPrinciple.CREATIVE_PRIORITY:
            return self._validate_creative_priority(content, context)
        elif principle == ConstitutionalPrinciple.STRUCTURAL_AWARENESS:
            return self._validate_structural_awareness(content, context)
        elif principle == ConstitutionalPrinciple.TENSION_ESTABLISHMENT:
            return self._validate_tension_establishment(content, context)
        else:
            return True, "Principle not handled by this validator"
    
    def _validate_non_fabrication(self, content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that content acknowledges uncertainty rather than fabricating facts."""
        uncertainty_indicators = [
            "i don't know", "uncertain", "unclear", "might be", "could be", 
            "appears to", "seems like", "possibly", "potentially", "unsure"
        ]
        
        fabrication_indicators = [
            "definitely", "certainly", "absolutely", "without doubt", "clearly",
            "obviously", "undoubtedly"
        ]
        
        content_lower = content.lower()
        
        # Check for fabrication without uncertainty acknowledgment
        has_fabrication = any(indicator in content_lower for indicator in fabrication_indicators)
        has_uncertainty = any(indicator in content_lower for indicator in uncertainty_indicators)
        
        if has_fabrication and not has_uncertainty:
            return False, "Content makes definitive claims without acknowledging uncertainty"
        
        return True, "Content appropriately acknowledges uncertainty"
    
    def _validate_error_as_compass(self, content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that errors are treated as learning opportunities."""
        error_indicators = ["error", "mistake", "wrong", "failed", "failure", "problem"]
        learning_indicators = ["learn", "improve", "adjust", "refine", "guide", "compass", "direction"]
        
        content_lower = content.lower()
        
        has_error_mention = any(indicator in content_lower for indicator in error_indicators)
        has_learning_orientation = any(indicator in content_lower for indicator in learning_indicators)
        
        if has_error_mention and not has_learning_orientation:
            return False, "Errors mentioned without learning orientation"
        
        return True, "Errors appropriately treated as learning opportunities"
    
    def _validate_creative_priority(self, content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that content prioritizes creating outcomes over eliminating problems."""
        problem_indicators = [
            "problem", "issue", "challenge", "difficulty", "obstacle", "barrier",
            "eliminate", "remove", "fix", "solve", "resolve"
        ]
        
        creative_indicators = [
            "create", "build", "manifest", "generate", "develop", "design",
            "outcome", "result", "achievement", "goal", "vision"
        ]
        
        content_lower = content.lower()
        
        problem_focus = sum(1 for indicator in problem_indicators if indicator in content_lower)
        creative_focus = sum(1 for indicator in creative_indicators if indicator in content_lower)
        
        if problem_focus > creative_focus * 2:  # Allow some problem mention but not dominance
            return False, "Content heavily focused on problems rather than creative outcomes"
        
        return True, "Content appropriately prioritizes creative orientation"
    
    def _validate_structural_awareness(self, content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that content demonstrates awareness of underlying structures."""
        structural_indicators = [
            "structure", "pattern", "system", "dynamic", "relationship", "force",
            "underlying", "foundation", "mechanism", "process"
        ]
        
        surface_indicators = [
            "symptom", "surface", "superficial", "immediate", "quick fix"
        ]
        
        content_lower = content.lower()
        
        has_structural = any(indicator in content_lower for indicator in structural_indicators)
        has_surface = any(indicator in content_lower for indicator in surface_indicators)
        
        if has_surface and not has_structural:
            return False, "Content focuses on surface issues without structural awareness"
        
        return True, "Content demonstrates appropriate structural awareness"
    
    def _validate_tension_establishment(self, content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that content establishes or maintains structural tension."""
        current_reality_indicators = [
            "currently", "now", "present", "as is", "reality", "actual", "existing"
        ]
        
        desired_outcome_indicators = [
            "want", "desire", "goal", "vision", "outcome", "result", "future", "will be"
        ]
        
        content_lower = content.lower()
        
        has_current = any(indicator in content_lower for indicator in current_reality_indicators)
        has_desired = any(indicator in content_lower for indicator in desired_outcome_indicators)
        
        if not (has_current or has_desired):
            return False, "Content lacks clear current reality or desired outcome elements"
        
        return True, "Content contributes to structural tension establishment"


class StructuralThinkingValidator(ConstitutionalValidator):
    """Validator for structural thinking constitutional principles."""
    
    def validate_against_principle(self, principle: ConstitutionalPrinciple, 
                                 content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate content against structural thinking principles."""
        
        if principle == ConstitutionalPrinciple.START_WITH_NOTHING:
            return self._validate_start_with_nothing(content, context)
        elif principle == ConstitutionalPrinciple.PICTURE_WHAT_IS_SAID:
            return self._validate_picture_what_is_said(content, context)
        elif principle == ConstitutionalPrinciple.QUESTION_INTERNALLY:
            return self._validate_question_internally(content, context)
        elif principle == ConstitutionalPrinciple.MULTIPLE_PERSPECTIVES:
            return self._validate_multiple_perspectives(content, context)
        else:
            return True, "Principle not handled by this validator"
    
    def _validate_start_with_nothing(self, content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that content starts from direct observation rather than preconceptions."""
        preconception_indicators = [
            "obviously", "clearly", "everyone knows", "it's common knowledge",
            "typically", "usually", "generally", "always", "never"
        ]
        
        observation_indicators = [
            "observe", "notice", "see", "appear", "seem", "present", "evident"
        ]
        
        content_lower = content.lower()
        
        has_preconceptions = any(indicator in content_lower for indicator in preconception_indicators)
        has_observation = any(indicator in content_lower for indicator in observation_indicators)
        
        if has_preconceptions and not has_observation:
            return False, "Content relies on preconceptions rather than direct observation"
        
        return True, "Content appropriately starts from observation"
    
    def _validate_picture_what_is_said(self, content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that content uses visual/spatial language for dimensional thinking."""
        visual_indicators = [
            "picture", "image", "visual", "see", "view", "perspective", "angle",
            "dimension", "space", "relationship", "connection", "pattern", "structure"
        ]
        
        abstract_indicators = [
            "concept", "idea", "theory", "philosophy", "principle" 
        ]
        
        content_lower = content.lower()
        
        visual_count = sum(1 for indicator in visual_indicators if indicator in content_lower)
        abstract_count = sum(1 for indicator in abstract_indicators if indicator in content_lower)
        
        if abstract_count > visual_count * 2:
            return False, "Content overly abstract without visual/dimensional language"
        
        return True, "Content appropriately uses visual/dimensional language"
    
    def _validate_question_internally(self, content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that questions arise from provided information rather than external sources."""
        # This would need context about what information was provided
        # For now, validate that questions are present and specific
        question_indicators = ["?", "what", "how", "why", "when", "where", "which"]
        external_reference_indicators = [
            "research shows", "studies indicate", "experts say", "it's known that",
            "according to", "literature suggests"
        ]
        
        content_lower = content.lower()
        
        has_questions = any(indicator in content_lower for indicator in question_indicators)
        has_external_refs = any(indicator in content_lower for indicator in external_reference_indicators)
        
        if has_external_refs:
            return False, "Content references external sources rather than focusing on provided information"
        
        return True, "Content appropriately questions based on provided information"
    
    def _validate_multiple_perspectives(self, content: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that content considers multiple viewpoints."""
        perspective_indicators = [
            "perspective", "viewpoint", "angle", "approach", "way", "alternatively",
            "on the other hand", "another view", "different", "various", "multiple"
        ]
        
        content_lower = content.lower()
        
        has_multiple_perspectives = any(indicator in content_lower for indicator in perspective_indicators)
        
        if len(content.split('.')) > 3 and not has_multiple_perspectives:  # Longer content should show multiple perspectives
            return False, "Longer content should demonstrate multiple perspectives"
        
        return True, "Content appropriately considers multiple perspectives"


class ConstitutionalCore:
    """The immutable constitutional core governing system thinking processes."""
    
    def __init__(self):
        self.validators = {
            'creative_orientation': CreativeOrientationValidator(),
            'structural_thinking': StructuralThinkingValidator()
        }
        self.decision_log: List[ConstitutionalDecision] = []
        self.principle_hierarchy = self._establish_principle_hierarchy()
    
    def _establish_principle_hierarchy(self) -> Dict[ConstitutionalPrinciple, int]:
        """Establish hierarchy for resolving conflicts between principles."""
        return {
            # Core identity principles (highest priority)
            ConstitutionalPrinciple.PRINCIPLE_OVER_EXPEDIENCE: 1,
            ConstitutionalPrinciple.TRANSPARENCY_REQUIREMENT: 2,
            ConstitutionalPrinciple.NON_FABRICATION: 3,
            
            # Creative orientation principles
            ConstitutionalPrinciple.CREATIVE_PRIORITY: 4,
            ConstitutionalPrinciple.ERROR_AS_COMPASS: 5,
            ConstitutionalPrinciple.STRUCTURAL_AWARENESS: 6,
            ConstitutionalPrinciple.TENSION_ESTABLISHMENT: 7,
            
            # Structural thinking principles
            ConstitutionalPrinciple.START_WITH_NOTHING: 8,
            ConstitutionalPrinciple.PICTURE_WHAT_IS_SAID: 9,
            ConstitutionalPrinciple.QUESTION_INTERNALLY: 10,
            ConstitutionalPrinciple.MULTIPLE_PERSPECTIVES: 11,
            
            # Adaptability principles (lower priority but important)
            ConstitutionalPrinciple.ADAPTIVE_PROTOCOLS: 12,
            ConstitutionalPrinciple.CONFLICT_RESOLUTION: 13
        }
    
    def validate_content(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content against all applicable constitutional principles."""
        validation_results = {}
        violated_principles = []
        
        for principle in ConstitutionalPrinciple:
            for validator_name, validator in self.validators.items():
                try:
                    is_valid, message = validator.validate_against_principle(principle, content, context)
                    
                    validation_results[f"{principle.value}_{validator_name}"] = {
                        'valid': is_valid,
                        'message': message,
                        'principle': principle.value
                    }
                    
                    if not is_valid:
                        violated_principles.append(principle)
                        
                except Exception as e:
                    logger.warning(f"Validation error for {principle.value} with {validator_name}: {e}")
        
        overall_valid = len(violated_principles) == 0
        
        return {
            'overall_valid': overall_valid,
            'violated_principles': [p.value for p in violated_principles],
            'validation_details': validation_results,
            'constitutional_compliance_score': self._calculate_compliance_score(validation_results)
        }
    
    def _calculate_compliance_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall constitutional compliance score (0.0 to 1.0)."""
        total_validations = len(validation_results)
        if total_validations == 0:
            return 1.0
        
        valid_count = sum(1 for result in validation_results.values() if result['valid'])
        return valid_count / total_validations
    
    def make_constitutional_decision(self, decision_context: str, 
                                   options: List[str], 
                                   context: Dict[str, Any]) -> ConstitutionalDecision:
        """Make a decision based on constitutional principles."""
        decision_id = f"decision_{datetime.now().isoformat()}"
        
        # Validate each option against constitutional principles
        option_scores = {}
        for i, option in enumerate(options):
            validation = self.validate_content(option, context)
            option_scores[i] = validation['constitutional_compliance_score']
        
        # Select option with highest constitutional compliance
        best_option_index = max(option_scores, key=option_scores.get)
        best_option = options[best_option_index]
        
        # Identify applicable principles
        applicable_principles = []
        principle_applications = {}
        
        for principle in ConstitutionalPrinciple:
            for validator in self.validators.values():
                try:
                    is_valid, message = validator.validate_against_principle(principle, best_option, context)
                    if message != "Principle not handled by this validator":
                        applicable_principles.append(principle)
                        principle_applications[principle] = message
                except:
                    continue
        
        decision = ConstitutionalDecision(
            decision_id=decision_id,
            timestamp=datetime.now(),
            decision_context=decision_context,
            applicable_principles=applicable_principles,
            principle_application=principle_applications,
            decision_outcome=best_option,
            alternative_considered=options[:best_option_index] + options[best_option_index+1:]
        )
        
        self.decision_log.append(decision)
        logger.info(f"Constitutional decision made: {decision_id}")
        
        return decision
    
    def resolve_principle_conflict(self, conflicting_principles: List[ConstitutionalPrinciple], 
                                 context: Dict[str, Any]) -> ConstitutionalPrinciple:
        """Resolve conflicts between constitutional principles using hierarchy."""
        if not conflicting_principles:
            raise ValueError("No conflicting principles provided")
        
        # Use hierarchy to resolve conflict
        highest_priority = min(conflicting_principles, 
                             key=lambda p: self.principle_hierarchy.get(p, 999))
        
        logger.info(f"Principle conflict resolved: {highest_priority.value} takes priority")
        return highest_priority
    
    def get_decision_audit_trail(self, decision_id: str) -> Optional[ConstitutionalDecision]:
        """Retrieve the complete audit trail for a specific decision."""
        for decision in self.decision_log:
            if decision.decision_id == decision_id:
                return decision
        return None
    
    def generate_active_pause_drafts(self, context: str, num_drafts: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple response drafts with different risk/reliability profiles."""
        # This would be implemented with actual language generation
        # For now, return a structure showing the concept
        
        draft_profiles = [
            {"risk_level": "conservative", "reliability": "high", "novelty": "low"},
            {"risk_level": "moderate", "reliability": "medium", "novelty": "medium"},
            {"risk_level": "bold", "reliability": "medium", "novelty": "high"}
        ]
        
        drafts = []
        for i, profile in enumerate(draft_profiles[:num_drafts]):
            draft = {
                "draft_id": f"draft_{i+1}",
                "profile": profile,
                "content": f"[Draft {i+1} with {profile['risk_level']} approach would be generated here]",
                "constitutional_assessment": self.validate_content(f"Sample content for {profile['risk_level']} approach", {"context": context}),
                "selection_criteria": {
                    "novelty_score": 0.3 + (i * 0.35),  # Increasing novelty
                    "reliability_score": 0.9 - (i * 0.2),  # Decreasing reliability
                    "constitutional_compliance": 0.8 + (i * 0.05)  # Slightly increasing compliance
                }
            }
            drafts.append(draft)
        
        return drafts
    
    def select_best_draft(self, drafts: List[Dict[str, Any]], 
                         selection_criteria: Dict[str, float]) -> Dict[str, Any]:
        """Select the best draft based on weighted criteria and constitutional principles."""
        weights = {
            'novelty_weight': selection_criteria.get('novelty_weight', 0.3),
            'reliability_weight': selection_criteria.get('reliability_weight', 0.4),
            'constitutional_weight': selection_criteria.get('constitutional_weight', 0.3)
        }
        
        best_draft = None
        best_score = -1
        
        for draft in drafts:
            criteria = draft['selection_criteria']
            score = (
                criteria['novelty_score'] * weights['novelty_weight'] +
                criteria['reliability_score'] * weights['reliability_weight'] +
                criteria['constitutional_compliance'] * weights['constitutional_weight']
            )
            
            if score > best_score:
                best_score = score
                best_draft = draft
        
        logger.info(f"Selected draft {best_draft['draft_id']} with score {best_score:.3f}")
        return best_draft


# Global constitutional core instance
constitutional_core = ConstitutionalCore()