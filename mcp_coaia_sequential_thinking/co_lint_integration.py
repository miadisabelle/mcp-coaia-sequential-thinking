"""
Co-Lint SCCP Integration Module

This module integrates CO-Lint Creative Orientation Linter with SCCP-based
structural tension methodology for real-time thought validation and guidance.

Connects to Issue #136 (CO-Lint) and #133 (AI consistency checker for structural tension methodology compliance)
"""

import logging
import os
import sys
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass

# Add co_lint to path if available
co_lint_path = os.path.join(os.path.dirname(__file__), '../../../co_lint')
if os.path.exists(co_lint_path):
    sys.path.insert(0, co_lint_path)

try:
    from co_lint.realtime_filter import lint_text
    from co_lint.rules import ALL_RULES
    CO_LINT_AVAILABLE = True
except ImportError:
    CO_LINT_AVAILABLE = False
    lint_text = None
    ALL_RULES = {}

from .models import ThoughtData, ThoughtStage

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels for SCCP-enhanced co-lint feedback."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    GUIDANCE = "guidance"


class StructuralTensionStrength(Enum):
    """Structural tension strength assessment levels."""
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    ADVANCING = "advancing"


@dataclass
class ValidationResult:
    """Enhanced validation result with SCCP-specific insights."""
    rule_id: str
    severity: ValidationSeverity
    message: str
    line_number: int
    suggestion: Optional[str] = None
    structural_insight: Optional[str] = None
    tension_impact: Optional[StructuralTensionStrength] = None


@dataclass 
class SCCPValidationSummary:
    """Summary of SCCP-based validation for a thought session."""
    has_desired_outcome: bool = False
    has_current_reality: bool = False 
    has_natural_progression: bool = False
    structural_tension_established: bool = False
    tension_strength: StructuralTensionStrength = StructuralTensionStrength.NONE
    advancing_pattern_detected: bool = False
    oscillating_patterns_count: int = 0
    creative_orientation_score: float = 0.0
    validation_results: List[ValidationResult] = None
    
    def __post_init__(self):
        if self.validation_results is None:
            self.validation_results = []


class CoLintSCCPFilter:
    """Enhanced Co-Lint filter with SCCP structural tension methodology integration."""
    
    def __init__(self, enable_guidance: bool = True):
        self.enable_guidance = enable_guidance
        self.co_lint_available = CO_LINT_AVAILABLE
        
        if not self.co_lint_available:
            logger.warning("CO-Lint not available - validation will use SCCP-only rules")
    
    def validate_thought_content(self, content: str, thought_data: Optional[ThoughtData] = None) -> SCCPValidationSummary:
        """
        Validate thought content using both CO-Lint rules and SCCP methodology.
        
        Args:
            content: The thought content to validate
            thought_data: Optional ThoughtData for context-aware validation
            
        Returns:
            SCCPValidationSummary with comprehensive validation results
        """
        summary = SCCPValidationSummary()
        
        # Run CO-Lint validation if available
        if self.co_lint_available:
            co_lint_results = self._run_co_lint_validation(content)
            summary.validation_results.extend(co_lint_results)
        
        # Run SCCP-specific validation
        sccp_results = self._run_sccp_validation(content, thought_data)
        summary.validation_results.extend(sccp_results)
        
        # Analyze structural tension components
        self._analyze_structural_tension(content, summary)
        
        # Assess creative orientation vs problem-solving language
        self._assess_creative_orientation(content, summary)
        
        # Calculate overall scores and insights
        self._calculate_summary_metrics(summary)
        
        return summary
    
    def _run_co_lint_validation(self, content: str) -> List[ValidationResult]:
        """Run standard CO-Lint validation rules."""
        results = []
        
        try:
            co_lint_findings = lint_text(content)
            
            for finding in co_lint_findings:
                severity = self._map_severity(finding.get('severity', 'warning'))
                
                result = ValidationResult(
                    rule_id=finding['rule'],
                    severity=severity,
                    message=finding['message'],
                    line_number=finding['line'],
                    suggestion=self._generate_sccp_suggestion(finding),
                    structural_insight=self._generate_structural_insight(finding)
                )
                
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error running CO-Lint validation: {e}")
            
        return results
    
    def _run_sccp_validation(self, content: str, thought_data: Optional[ThoughtData]) -> List[ValidationResult]:
        """Run SCCP-specific validation beyond standard CO-Lint rules."""
        results = []
        
        # Check for structural tension language patterns
        if thought_data and thought_data.stage == ThoughtStage.DESIRED_OUTCOME:
            if not self._contains_outcome_language(content):
                results.append(ValidationResult(
                    rule_id="SCCP001",
                    severity=ValidationSeverity.GUIDANCE,
                    message="Desired Outcome stage could benefit from clearer outcome statement",
                    line_number=1,
                    suggestion="Consider expressing what you want to create, not what you want to avoid or solve",
                    structural_insight="Strong desired outcomes create advancing structural tension"
                ))
        
        # Check for problem-solving vs creative orientation language
        problem_words = self._detect_problem_solving_language(content)
        if problem_words:
            results.append(ValidationResult(
                rule_id="SCCP002", 
                severity=ValidationSeverity.WARNING,
                message=f"Detected problem-solving language: {', '.join(problem_words)}",
                line_number=1,
                suggestion="Consider reframing from what you want to create rather than what you want to fix",
                structural_insight="Problem-solving language can weaken structural tension by focusing on what you don't want",
                tension_impact=StructuralTensionStrength.WEAK
            ))
        
        # Check for advancing vs oscillating pattern indicators
        if self._detect_oscillating_patterns(content):
            results.append(ValidationResult(
                rule_id="SCCP003",
                severity=ValidationSeverity.INFO,
                message="Content suggests oscillating pattern - alternating between approaches without clear direction",
                line_number=1,
                suggestion="Consider establishing clearer structural tension between desired outcome and current reality",
                structural_insight="Oscillating patterns often indicate weak or unclear structural tension"
            ))
        
        return results
    
    def _analyze_structural_tension(self, content: str, summary: SCCPValidationSummary):
        """Analyze content for structural tension components."""
        content_lower = content.lower()
        
        # Check for structural tension elements
        summary.has_desired_outcome = any(phrase in content_lower for phrase in [
            'desired outcome', 'want to create', 'goal is', 'outcome:', 'creating', 'building'
        ])
        
        summary.has_current_reality = any(phrase in content_lower for phrase in [
            'current reality', 'current state', 'where i am', 'currently', 'right now', 'present situation'
        ])
        
        summary.has_natural_progression = any(phrase in content_lower for phrase in [
            'natural progression', 'next steps', 'moving toward', 'pathway', 'progression', 'advancing'
        ])
        
        # Assess if structural tension is established
        summary.structural_tension_established = (
            summary.has_desired_outcome and summary.has_current_reality
        )
        
        # Determine tension strength
        if summary.structural_tension_established:
            if summary.has_natural_progression:
                summary.tension_strength = StructuralTensionStrength.STRONG
            else:
                summary.tension_strength = StructuralTensionStrength.MODERATE
        elif summary.has_desired_outcome or summary.has_current_reality:
            summary.tension_strength = StructuralTensionStrength.WEAK
        else:
            summary.tension_strength = StructuralTensionStrength.NONE
    
    def _assess_creative_orientation(self, content: str, summary: SCCPValidationSummary):
        """Assess creative orientation vs problem-solving language."""
        content_lower = content.lower()
        
        # Count creative orientation indicators
        creative_words = ['create', 'build', 'develop', 'design', 'establish', 'generate', 'bring forth']
        creative_count = sum(1 for word in creative_words if word in content_lower)
        
        # Count problem-solving indicators  
        problem_words = ['fix', 'solve', 'eliminate', 'avoid', 'prevent', 'stop', 'reduce', 'minimize']
        problem_count = sum(1 for word in problem_words if word in content_lower)
        
        # Calculate creative orientation score (0.0 to 1.0)
        total_words = creative_count + problem_count
        if total_words > 0:
            summary.creative_orientation_score = creative_count / total_words
        else:
            summary.creative_orientation_score = 0.5  # Neutral when no indicators found
        
        # Check for advancing patterns
        advancing_indicators = ['advancing', 'progressing', 'moving toward', 'developing', 'growing']
        summary.advancing_pattern_detected = any(indicator in content_lower for indicator in advancing_indicators)
        
        # Count oscillating patterns
        oscillating_words = ['back and forth', 'trying different', 'switching between', 'alternating', 'going in circles']
        summary.oscillating_patterns_count = sum(1 for phrase in oscillating_words if phrase in content_lower)
    
    def _calculate_summary_metrics(self, summary: SCCPValidationSummary):
        """Calculate overall summary metrics and insights."""
        # Adjust tension strength based on advancing patterns
        if summary.advancing_pattern_detected and summary.tension_strength in [StructuralTensionStrength.MODERATE, StructuralTensionStrength.STRONG]:
            summary.tension_strength = StructuralTensionStrength.ADVANCING
        
        # Add guidance for weak structural tension
        if summary.tension_strength == StructuralTensionStrength.WEAK:
            summary.validation_results.append(ValidationResult(
                rule_id="SCCP004",
                severity=ValidationSeverity.GUIDANCE,
                message="Structural tension could be strengthened",
                line_number=1,
                suggestion="Consider clearly stating both your desired outcome and current reality to create advancing tension",
                structural_insight="Strong structural tension between desired outcome and current reality creates natural forward movement"
            ))
    
    def _contains_outcome_language(self, content: str) -> bool:
        """Check if content contains clear outcome-oriented language."""
        outcome_patterns = ['want to create', 'goal is to', 'outcome is', 'creating', 'building', 'establishing']
        return any(pattern in content.lower() for pattern in outcome_patterns)
    
    def _detect_problem_solving_language(self, content: str) -> List[str]:
        """Detect problem-solving language patterns."""
        problem_words = ['fix', 'solve', 'eliminate', 'avoid', 'prevent', 'stop', 'reduce', 'minimize', 'overcome', 'handle']
        content_lower = content.lower()
        return [word for word in problem_words if word in content_lower]
    
    def _detect_oscillating_patterns(self, content: str) -> bool:
        """Detect language indicating oscillating rather than advancing patterns."""
        oscillating_indicators = [
            'back and forth', 'trying different', 'switching between', 
            'alternating', 'going in circles', 'keep trying', 'various approaches'
        ]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in oscillating_indicators)
    
    def _map_severity(self, co_lint_severity: str) -> ValidationSeverity:
        """Map CO-Lint severity levels to our enhanced validation severity."""
        mapping = {
            'error': ValidationSeverity.ERROR,
            'warn': ValidationSeverity.WARNING,
            'warning': ValidationSeverity.WARNING,
            'info': ValidationSeverity.INFO
        }
        return mapping.get(co_lint_severity.lower(), ValidationSeverity.WARNING)
    
    def _generate_sccp_suggestion(self, finding: Dict[str, Any]) -> str:
        """Generate SCCP-aware suggestions for CO-Lint findings."""
        rule_id = finding.get('rule', '')
        
        if rule_id == 'COL001':
            return "Establish structural tension by clearly stating your Desired Outcome, Current Reality, and Natural Progression"
        elif rule_id == 'COL002':
            return "Keep observations neutral and factual - let structural tension create the movement toward your outcome"
        elif rule_id == 'COL003':
            return "Frame assessment in terms of advancing or oscillating patterns rather than problems to solve"
        elif rule_id == 'COL004':
            return "Focus on what you want to create rather than what you want to eliminate or fix"
        elif rule_id == 'COL005':
            return "Use creative orientation language that supports advancing structural tension"
        else:
            return "Consider how this aligns with creating structural tension toward your desired outcome"
    
    def _generate_structural_insight(self, finding: Dict[str, Any]) -> str:
        """Generate structural insights for CO-Lint findings."""
        rule_id = finding.get('rule', '')
        
        insights = {
            'COL001': "Structural tension requires clear desired outcome and current reality to create advancing patterns",
            'COL002': "Neutral observations support clear structural tension without emotional interference", 
            'COL003': "Advancing/oscillating language helps recognize patterns that support or weaken structural tension",
            'COL004': "Create-language maintains focus on desired outcomes rather than avoiding unwanted states",
            'COL005': "Creative orientation language supports the natural pull of structural tension toward manifestation"
        }
        
        return insights.get(rule_id, "This finding relates to establishing and maintaining strong structural tension")


# Global filter instance
co_lint_filter = CoLintSCCPFilter(enable_guidance=True)


def validate_thought(content: str, thought_data: Optional[ThoughtData] = None) -> SCCPValidationSummary:
    """
    Public API for validating thought content using SCCP-enhanced CO-Lint filtering.
    
    Args:
        content: The thought content to validate
        thought_data: Optional ThoughtData for context-aware validation
        
    Returns:
        SCCPValidationSummary with comprehensive validation results
    """
    return co_lint_filter.validate_thought_content(content, thought_data)