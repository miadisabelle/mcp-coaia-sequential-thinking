"""
Co-Lint SCCP Integration Module

This module integrates CO-Lint Creative Orientation Linter with SCCP-based
structural tension methodology for real-time thought validation and guidance.

Enhanced with data persistence for pattern learning and constitutional compliance tracking.
Follows non-intrusive feedback principle: creative work remains primary deliverable,
with analytical data provided as structured metadata.

Connects to Issue #136 (CO-Lint) and #133 (AI consistency checker for structural tension methodology compliance)
"""

import logging
import os
import sys
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass

# Add co_lint to path if available - corrected path
co_lint_path = os.path.join(os.path.dirname(__file__), '..', 'co_lint')
co_lint_module_path = os.path.join(co_lint_path, 'co_lint')
if os.path.exists(co_lint_module_path):
    sys.path.insert(0, co_lint_path)

try:
    from co_lint.realtime_filter import lint_text
    from co_lint.rules import ALL_RULES
    CO_LINT_AVAILABLE = True
    # logger will be defined later
except ImportError as e:
    CO_LINT_AVAILABLE = False
    lint_text = None
    ALL_RULES = {}
    # Store import error for debugging
    import_error = str(e)

from .models import ThoughtData, ThoughtStage

# Import data persistence for pattern learning
try:
    from .data_persistence import data_store
    DATA_PERSISTENCE_AVAILABLE = True
except ImportError:
    DATA_PERSISTENCE_AVAILABLE = False
    data_store = None

logger = logging.getLogger(__name__)

# Log CO-Lint status after logger is defined
if CO_LINT_AVAILABLE:
    logger.info("CO-Lint successfully integrated with data persistence")
else:
    error_msg = import_error if 'import_error' in locals() else "Unknown import error"
    logger.warning(f"CO-Lint not available - using SCCP-only rules with data persistence. Error: {error_msg}")
    logger.debug(f"CO-Lint search paths: co_lint_path={co_lint_path}, co_lint_module_path={co_lint_module_path if 'co_lint_module_path' in locals() else 'undefined'}")


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
    validator = CoLintSCCPFilter()
    summary = validator.validate_thought_content(content, thought_data)
    
    # Store validation results for pattern learning if data persistence is available
    if DATA_PERSISTENCE_AVAILABLE and data_store:
        try:
            validation_data = {
                'content': content,
                'creative_orientation_score': summary.creative_orientation_score,
                'reactive_patterns_detected': {
                    'patterns': [finding.message for finding in summary.validation_results if finding.severity == ValidationSeverity.ERROR],
                    'count': len([f for f in summary.validation_results if f.severity == ValidationSeverity.ERROR])
                },
                'advancing_indicators': {
                    'patterns': [finding.message for finding in summary.validation_results if finding.severity == ValidationSeverity.INFO],
                    'structural_tension_strength': summary.tension_strength.value,
                    'advancing_pattern_detected': summary.advancing_pattern_detected
                },
                'co_lint_results': {
                    'findings': [{'rule': f.rule_id, 'message': f.message, 'severity': f.severity.value} 
                               for f in summary.validation_results],
                    'total_findings': len(summary.validation_results),
                    'structural_tension_established': summary.structural_tension_established
                },
                'recommendations': [f.recommendation for f in summary.validation_results if hasattr(f, 'recommendation') and f.recommendation]
            }
            
            data_store.store_orientation_validation(validation_data)
            logger.debug(f"Stored validation results for pattern learning")
            
        except Exception as e:
            logger.warning(f"Failed to store validation data: {e}")
    
    return summary


# Enhanced pattern analysis functions for creative orientation learning and agent self-awareness

def get_user_creative_patterns(limit: int = 100) -> Dict[str, Any]:
    """
    Analyze user's creative orientation patterns from stored validation data.
    
    Returns insights about creative vs reactive tendencies, common patterns,
    and recommendations for improving creative orientation.
    """
    if not DATA_PERSISTENCE_AVAILABLE or not data_store:
        return {"error": "Data persistence not available"}
    
    try:
        # Get recent validation data for pattern analysis
        patterns = data_store.get_orientation_patterns(limit)
        
        if not patterns:
            return {"message": "No validation data available yet"}
        
        # Analyze patterns
        total_validations = len(patterns)
        avg_creative_score = sum(p.get('creative_orientation_score', 0) for p in patterns) / total_validations
        
        # Identify most common reactive patterns
        reactive_patterns = {}
        advancing_indicators = {}
        
        for pattern in patterns:
            # Count reactive patterns
            reactive_data = pattern.get('reactive_patterns_detected', {})
            if isinstance(reactive_data, dict):
                for p in reactive_data.get('patterns', []):
                    reactive_patterns[p] = reactive_patterns.get(p, 0) + 1
            
            # Count advancing indicators  
            advancing_data = pattern.get('advancing_indicators', {})
            if isinstance(advancing_data, dict):
                for p in advancing_data.get('patterns', []):
                    advancing_indicators[p] = advancing_indicators.get(p, 0) + 1
        
        # Generate insights with agent self-awareness
        insights = {
            "total_validations": total_validations,
            "average_creative_orientation_score": round(avg_creative_score, 2),
            "orientation_trend": "creative" if avg_creative_score > 0.7 else "reactive" if avg_creative_score < 0.4 else "mixed",
            "most_common_reactive_patterns": sorted(reactive_patterns.items(), key=lambda x: x[1], reverse=True)[:5],
            "strongest_advancing_indicators": sorted(advancing_indicators.items(), key=lambda x: x[1], reverse=True)[:5],
            "recommendations": _generate_personalized_recommendations(avg_creative_score, reactive_patterns, advancing_indicators),
            # Enhanced agent self-awareness
            "agent_orientation_awareness": {
                "current_orientation_status": _determine_orientation_status(avg_creative_score),
                "tool_usage_guidance": _generate_tool_usage_guidance(avg_creative_score, reactive_patterns),
                "mcp_interaction_recommendations": _generate_mcp_interaction_recommendations(avg_creative_score)
            },
            # CoAiA-memory integration ready data
            "coaia_memory_integration": {
                "structural_tension_ready": True,
                "pattern_entities": _prepare_coaia_memory_entities(patterns[:20]),  # Latest 20 patterns
                "knowledge_graph_nodes": _generate_knowledge_graph_nodes(reactive_patterns, advancing_indicators)
            }
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"Error analyzing creative patterns: {e}")
        return {"error": str(e)}


def _determine_orientation_status(avg_score: float) -> Dict[str, Any]:
    """Determine the agent's current orientation status for self-awareness."""
    if avg_score >= 0.8:
        return {
            "status": "strongly_creative",
            "confidence": "high",
            "description": "Agent consistently demonstrates creative orientation - ideal for structural tension work",
            "mcp_tool_readiness": "optimal"
        }
    elif avg_score >= 0.6:
        return {
            "status": "moderately_creative", 
            "confidence": "good",
            "description": "Agent shows good creative orientation with occasional reactive tendencies",
            "mcp_tool_readiness": "ready"
        }
    elif avg_score >= 0.4:
        return {
            "status": "mixed_orientation",
            "confidence": "moderate",
            "description": "Agent exhibits mixed creative/reactive patterns - needs guidance",
            "mcp_tool_readiness": "needs_guidance"
        }
    else:
        return {
            "status": "reactive_dominant",
            "confidence": "low", 
            "description": "Agent demonstrates strong reactive patterns - requires orientation training",
            "mcp_tool_readiness": "requires_training"
        }


def _generate_tool_usage_guidance(avg_score: float, reactive_patterns: Dict) -> List[str]:
    """Generate specific guidance for how agents should use MCP tools based on their orientation."""
    guidance = []
    
    if avg_score < 0.4:
        guidance.append("CAUTION: Before using initiate_sequential_thinking, establish clear desired outcome first")
        guidance.append("AVOID: Using advance_thinking_chain when focused on problem-solving - reframe as outcome creation")
        guidance.append("PRACTICE: Use create_consensus_decision only after establishing structural tension")
    
    elif avg_score < 0.6:
        guidance.append("RECOMMENDED: Begin each MCP tool interaction with desired outcome clarification")
        guidance.append("MONITOR: Watch for reactive language patterns during sequential thinking chains")
        guidance.append("STRENGTHEN: Use synthesize_thinking_chain to integrate creative perspectives")
    
    else:
        guidance.append("OPTIMAL: Agent demonstrates strong creative orientation - all MCP tools available")
        guidance.append("LEADERSHIP: Can guide other agents in creative orientation practices")
        guidance.append("ADVANCED: Ready for complex structural tension chart creation")
    
    # Specific reactive pattern guidance
    common_reactive = list(reactive_patterns.keys())[:3]
    if any('solve' in pattern.lower() for pattern in common_reactive):
        guidance.append("ALERT: 'Solve' language detected - transform to 'create' before using MCP tools")
    
    if any('problem' in pattern.lower() for pattern in common_reactive):
        guidance.append("REDIRECT: Problem-focus detected - establish desired outcome first")
    
    return guidance


def _generate_mcp_interaction_recommendations(avg_score: float) -> Dict[str, Any]:
    """Generate specific recommendations for MCP tool interactions."""
    if avg_score >= 0.7:
        return {
            "initiate_sequential_thinking": "Ready - agent can establish clear structural tensions",
            "advance_thinking_chain": "Optimal - agent maintains creative flow between personas",
            "create_consensus_decision": "Excellent - agent facilitates advancing pattern decisions",
            "run_full_analysis_chain": "Advanced - agent handles complex multi-perspective integration",
            "overall_recommendation": "Agent demonstrates creative mastery - all tools available"
        }
    elif avg_score >= 0.5:
        return {
            "initiate_sequential_thinking": "Good - verify desired outcome clarity before proceeding",
            "advance_thinking_chain": "Ready - monitor for reactive pattern emergence",
            "create_consensus_decision": "Suitable - ensure advancing pattern focus",
            "run_full_analysis_chain": "Recommended - with orientation awareness",
            "overall_recommendation": "Agent shows creative capacity - proceed with awareness"
        }
    else:
        return {
            "initiate_sequential_thinking": "CAUTION - establish orientation training first",
            "advance_thinking_chain": "NOT RECOMMENDED - reactive patterns may disrupt flow",
            "create_consensus_decision": "REQUIRES GUIDANCE - risk of problem-solving orientation",
            "run_full_analysis_chain": "DELAY - complete creative orientation training first",
            "overall_recommendation": "Agent needs creative orientation development before using MCP tools"
        }


def _prepare_coaia_memory_entities(patterns: List[Dict]) -> List[Dict[str, Any]]:
    """Prepare pattern data for coaia-memory knowledge graph integration."""
    entities = []
    
    for i, pattern in enumerate(patterns):
        entity = {
            "type": "creative_orientation_pattern",
            "id": f"pattern_{i}",
            "creative_score": pattern.get('creative_orientation_score', 0),
            "timestamp": pattern.get('created_at'),
            "reactive_indicators": pattern.get('reactive_patterns_detected', {}),
            "advancing_indicators": pattern.get('advancing_indicators', {}),
            "structural_tension_data": {
                "established": pattern.get('co_lint_results', {}).get('structural_tension_established', False),
                "advancing_pattern": pattern.get('advancing_indicators', {}).get('advancing_pattern_detected', False)
            }
        }
        entities.append(entity)
    
    return entities


def _generate_knowledge_graph_nodes(reactive_patterns: Dict, advancing_indicators: Dict) -> Dict[str, Any]:
    """Generate knowledge graph nodes for structural tension charting integration."""
    return {
        "reactive_pattern_nodes": [
            {
                "id": f"reactive_{i}",
                "pattern": pattern,
                "frequency": count,
                "type": "reactive_indicator",
                "intervention_needed": count > 3  # High frequency patterns need intervention
            }
            for i, (pattern, count) in enumerate(reactive_patterns.items())
        ],
        "advancing_pattern_nodes": [
            {
                "id": f"advancing_{i}",
                "pattern": pattern,
                "frequency": count,
                "type": "advancing_indicator", 
                "strength": "high" if count > 5 else "moderate" if count > 2 else "developing"
            }
            for i, (pattern, count) in enumerate(advancing_indicators.items())
        ],
        "structural_tension_nodes": {
            "current_reality_node": {
                "reactive_dominance": len(reactive_patterns) / max(len(advancing_indicators), 1),
                "pattern_complexity": len(reactive_patterns) + len(advancing_indicators)
            },
            "desired_outcome_node": {
                "creative_orientation_target": 0.8,
                "advancing_pattern_goal": "dominant_creative_orientation"
            }
        }
    }


def _generate_personalized_recommendations(avg_score: float, reactive_patterns: Dict, advancing_indicators: Dict) -> List[str]:
    """Generate personalized recommendations based on user patterns."""
    recommendations = []
    
    if avg_score < 0.4:
        recommendations.append("Focus on establishing clear desired outcomes before analyzing current reality")
        recommendations.append("Practice framing challenges as 'What do I want to create?' instead of 'What problem needs fixing?'")
    
    if avg_score < 0.6:
        recommendations.append("Strengthen structural tension by clarifying both current reality and desired outcomes")
        recommendations.append("Use advancing language: 'create', 'build', 'develop' instead of 'fix', 'solve', 'eliminate'")
    
    # Recommendations based on most common reactive patterns
    common_reactive = list(reactive_patterns.keys())[:3]
    if any('problem' in pattern.lower() for pattern in common_reactive):
        recommendations.append("Reduce problem-focused language - reframe as outcome creation opportunities")
    
    if any('fix' in pattern.lower() or 'solve' in pattern.lower() for pattern in common_reactive):
        recommendations.append("Transform fix/solve language into build/create language for advancing patterns")
    
    # Leverage existing strengths
    strong_advancing = list(advancing_indicators.keys())[:2] 
    if strong_advancing:
        recommendations.append(f"Continue leveraging your strengths in: {', '.join(strong_advancing)}")
    
    return recommendations[:5]  # Limit to top 5 recommendations


# Export enhanced co_lint validator for integration with other modules
enhanced_co_lint_validator = CoLintSCCPFilter()