"""
Creative Orientation Validation Engine

Advanced SCCP-based pattern recognition engine that goes beyond basic CO-Lint rules
to provide sophisticated structural tension analysis and creative orientation guidance.

This engine implements Robert Fritz's structural tension methodology for AI systems,
providing real-time validation and guidance for creative orientation vs problem-solving approaches.

Connected to Issues:
- #133 (AI consistency checker for structural tension methodology compliance)
- #130 (Creative Observer System development)
- #136 (CO-Lint integration)
"""

import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

from .models import ThoughtData, ThoughtStage
from .co_lint_integration import StructuralTensionStrength, ValidationSeverity

logger = logging.getLogger(__name__)


class PatternSignature(Enum):
    """Pattern signatures for advanced SCCP analysis."""
    ADVANCING_STRONG = "advancing_strong"
    ADVANCING_MODERATE = "advancing_moderate"
    OSCILLATING_HIGH_ENERGY = "oscillating_high_energy"
    OSCILLATING_LOW_ENERGY = "oscillating_low_energy"
    CIRCULAR_REASONING = "circular_reasoning"
    CREATIVE_BREAKTHROUGH = "creative_breakthrough"
    PROBLEM_SOLVING_TRAP = "problem_solving_trap"
    STRUCTURAL_TENSION_EMERGING = "structural_tension_emerging"


class CreativeOrientationMetric(Enum):
    """Advanced metrics for creative orientation assessment."""
    OUTCOME_CLARITY = "outcome_clarity"
    REALITY_GROUNDING = "reality_grounding"
    NATURAL_PROGRESSION = "natural_progression"
    ENERGY_DIRECTION = "energy_direction"
    LANGUAGE_CONSISTENCY = "language_consistency"
    TENSION_SUSTAINABILITY = "tension_sustainability"


@dataclass
class PatternAnalysis:
    """Advanced pattern analysis results."""
    signature: PatternSignature
    confidence: float
    contributing_factors: List[str]
    energy_level: float
    direction_vector: Tuple[float, float]  # (advancement, oscillation)
    sustainability_score: float
    
    
@dataclass 
class CreativeOrientationProfile:
    """Comprehensive creative orientation profile for a session."""
    overall_pattern: PatternSignature
    tension_strength: StructuralTensionStrength
    creative_metrics: Dict[CreativeOrientationMetric, float] = field(default_factory=dict)
    pattern_evolution: List[PatternAnalysis] = field(default_factory=list)
    language_consistency_score: float = 0.0
    energy_sustainability_index: float = 0.0
    breakthrough_indicators: List[str] = field(default_factory=list)
    structural_recommendations: List[str] = field(default_factory=list)


class AdvancedPatternRecognizer:
    """Advanced pattern recognition using SCCP methodology and mathematical models."""
    
    def __init__(self):
        self.outcome_keywords = {
            'strong': ['create', 'build', 'establish', 'develop', 'generate', 'bring forth', 'manifest'],
            'moderate': ['make', 'do', 'get', 'have', 'achieve', 'reach', 'attain'],
            'weak': ['try to', 'attempt to', 'hope to', 'want to', 'would like to']
        }
        
        self.reality_keywords = {
            'strong': ['currently', 'right now', 'present state', 'where I am', 'actual situation'],
            'moderate': ['today', 'at this point', 'so far', 'until now', 'presently'],
            'weak': ['sometimes', 'often', 'usually', 'generally', 'typically']
        }
        
        self.progression_keywords = {
            'strong': ['natural progression', 'advancing toward', 'moving in direction of', 'pathway to'],
            'moderate': ['next step', 'moving toward', 'progressing to', 'developing'],
            'weak': ['might', 'could', 'perhaps', 'maybe', 'possibly']
        }
        
        self.oscillating_indicators = [
            r'\b(?:back\s+and\s+forth|going\s+in\s+circles|trying\s+different|switching\s+between)\b',
            r'\b(?:sometimes\s+\w+,?\s+sometimes\s+\w+|either\s+\w+\s+or\s+\w+)\b',
            r'\b(?:keep\s+trying|various\s+approaches|different\s+methods)\b',
            r'\b(?:on\s+one\s+hand.*on\s+the\s+other\s+hand|however|but\s+then)\b'
        ]
        
        self.breakthrough_indicators = [
            r'\b(?:suddenly\s+realized|breakthrough|aha\s+moment|clicked|became\s+clear)\b',
            r'\b(?:now\s+I\s+see|makes\s+sense\s+now|understand\s+now|clarity)\b',
            r'\b(?:insight|revelation|realization|understanding\s+emerged)\b'
        ]

    def analyze_session_patterns(self, thoughts: List[ThoughtData]) -> CreativeOrientationProfile:
        """Analyze entire session for advanced creative orientation patterns."""
        profile = CreativeOrientationProfile(
            overall_pattern=PatternSignature.STRUCTURAL_TENSION_EMERGING,
            tension_strength=StructuralTensionStrength.NONE
        )
        
        if not thoughts:
            return profile
        
        # Analyze each thought for patterns
        for thought in thoughts:
            pattern = self._analyze_thought_pattern(thought)
            profile.pattern_evolution.append(pattern)
        
        # Calculate overall metrics
        profile.creative_metrics = self._calculate_creative_metrics(thoughts)
        profile.overall_pattern = self._determine_overall_pattern(profile.pattern_evolution)
        profile.tension_strength = self._assess_tension_strength(thoughts, profile.creative_metrics)
        profile.language_consistency_score = self._calculate_language_consistency(thoughts)
        profile.energy_sustainability_index = self._calculate_energy_sustainability(profile.pattern_evolution)
        
        # Generate insights and recommendations
        profile.breakthrough_indicators = self._detect_breakthrough_moments(thoughts)
        profile.structural_recommendations = self._generate_structural_recommendations(profile)
        
        return profile
    
    def _analyze_thought_pattern(self, thought: ThoughtData) -> PatternAnalysis:
        """Analyze individual thought for pattern characteristics."""
        content = thought.thought.lower()
        
        # Calculate energy and direction vectors
        outcome_strength = self._calculate_outcome_strength(content)
        reality_strength = self._calculate_reality_strength(content)
        progression_strength = self._calculate_progression_strength(content)
        oscillation_score = self._calculate_oscillation_score(content)
        
        # Energy level calculation
        energy_level = (outcome_strength + reality_strength + progression_strength) / 3
        
        # Direction vector (advancement vs oscillation)
        advancement = max(0, (outcome_strength + progression_strength) / 2 - oscillation_score)
        oscillation = oscillation_score
        direction_vector = (advancement, oscillation)
        
        # Determine pattern signature
        signature = self._classify_pattern_signature(
            energy_level, advancement, oscillation, outcome_strength, reality_strength
        )
        
        # Calculate confidence and sustainability
        confidence = self._calculate_pattern_confidence(outcome_strength, reality_strength, progression_strength)
        sustainability = self._calculate_sustainability_score(energy_level, advancement, oscillation)
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(
            content, outcome_strength, reality_strength, progression_strength, oscillation_score
        )
        
        return PatternAnalysis(
            signature=signature,
            confidence=confidence,
            contributing_factors=contributing_factors,
            energy_level=energy_level,
            direction_vector=direction_vector,
            sustainability_score=sustainability
        )
    
    def _calculate_outcome_strength(self, content: str) -> float:
        """Calculate strength of outcome-oriented language."""
        strong_count = sum(1 for word in self.outcome_keywords['strong'] if word in content)
        moderate_count = sum(1 for word in self.outcome_keywords['moderate'] if word in content)
        weak_count = sum(1 for word in self.outcome_keywords['weak'] if word in content)
        
        # Weighted scoring
        score = (strong_count * 1.0 + moderate_count * 0.6 + weak_count * 0.3)
        
        # Normalize to 0-1 range (assuming max 5 occurrences)
        return min(1.0, score / 5.0)
    
    def _calculate_reality_strength(self, content: str) -> float:
        """Calculate strength of current reality grounding."""
        strong_count = sum(1 for phrase in self.reality_keywords['strong'] if phrase in content)
        moderate_count = sum(1 for phrase in self.reality_keywords['moderate'] if phrase in content)
        weak_count = sum(1 for phrase in self.reality_keywords['weak'] if phrase in content)
        
        score = (strong_count * 1.0 + moderate_count * 0.6 + weak_count * 0.3)
        return min(1.0, score / 3.0)
    
    def _calculate_progression_strength(self, content: str) -> float:
        """Calculate strength of natural progression language."""
        strong_count = sum(1 for phrase in self.progression_keywords['strong'] if phrase in content)
        moderate_count = sum(1 for phrase in self.progression_keywords['moderate'] if phrase in content)
        weak_count = sum(1 for phrase in self.progression_keywords['weak'] if phrase in content)
        
        score = (strong_count * 1.0 + moderate_count * 0.6 + weak_count * 0.3)
        return min(1.0, score / 3.0)
    
    def _calculate_oscillation_score(self, content: str) -> float:
        """Calculate oscillation pattern indicators."""
        oscillation_matches = 0
        for pattern in self.oscillating_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                oscillation_matches += 1
        
        return min(1.0, oscillation_matches / len(self.oscillating_indicators))
    
    def _classify_pattern_signature(self, energy: float, advancement: float, oscillation: float,
                                    outcome_strength: float, reality_strength: float) -> PatternSignature:
        """Classify the pattern signature based on calculated metrics."""
        
        if advancement > 0.7 and energy > 0.6:
            return PatternSignature.ADVANCING_STRONG
        elif advancement > 0.4 and energy > 0.3:
            return PatternSignature.ADVANCING_MODERATE
        elif oscillation > 0.6 and energy > 0.4:
            return PatternSignature.OSCILLATING_HIGH_ENERGY
        elif oscillation > 0.3 and energy < 0.4:
            return PatternSignature.OSCILLATING_LOW_ENERGY
        elif oscillation > advancement and advancement < 0.2:
            return PatternSignature.CIRCULAR_REASONING
        elif outcome_strength > 0.7 and reality_strength > 0.7:
            return PatternSignature.STRUCTURAL_TENSION_EMERGING
        elif advancement < 0.2 and oscillation < 0.2:
            return PatternSignature.PROBLEM_SOLVING_TRAP
        else:
            return PatternSignature.STRUCTURAL_TENSION_EMERGING
    
    def _calculate_pattern_confidence(self, outcome: float, reality: float, progression: float) -> float:
        """Calculate confidence in pattern classification."""
        # Higher confidence when all elements are clearly present or clearly absent
        variance = np.var([outcome, reality, progression])
        mean_strength = np.mean([outcome, reality, progression])
        
        # Confidence is higher when variance is low and mean is either high or low (clear signal)
        if mean_strength > 0.7 or mean_strength < 0.3:
            return 1.0 - variance
        else:
            return 0.5 - variance  # Medium confidence for ambiguous cases
    
    def _calculate_sustainability_score(self, energy: float, advancement: float, oscillation: float) -> float:
        """Calculate how sustainable this pattern is over time."""
        # Advancing patterns with good energy are most sustainable
        # High oscillation reduces sustainability
        sustainability = energy * (advancement / (advancement + oscillation + 0.1))
        return min(1.0, max(0.0, sustainability))
    
    def _identify_contributing_factors(self, content: str, outcome: float, reality: float,
                                     progression: float, oscillation: float) -> List[str]:
        """Identify what factors are contributing to the current pattern."""
        factors = []
        
        if outcome > 0.5:
            factors.append("Clear desired outcome expressed")
        elif outcome < 0.2:
            factors.append("Outcome clarity needs strengthening")
        
        if reality > 0.5:
            factors.append("Current reality well grounded")
        elif reality < 0.2:
            factors.append("Current reality assessment weak")
        
        if progression > 0.5:
            factors.append("Natural progression pathway identified")
        elif progression < 0.2:
            factors.append("Progression pathway unclear")
        
        if oscillation > 0.4:
            factors.append("Oscillating pattern language detected")
        
        # Check for breakthrough indicators
        for pattern in self.breakthrough_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                factors.append("Breakthrough moment indicated")
                break
        
        return factors
    
    def _calculate_creative_metrics(self, thoughts: List[ThoughtData]) -> Dict[CreativeOrientationMetric, float]:
        """Calculate comprehensive creative orientation metrics."""
        if not thoughts:
            return {}
        
        all_content = " ".join([thought.thought for thought in thoughts]).lower()
        
        metrics = {}
        metrics[CreativeOrientationMetric.OUTCOME_CLARITY] = self._calculate_outcome_strength(all_content)
        metrics[CreativeOrientationMetric.REALITY_GROUNDING] = self._calculate_reality_strength(all_content)
        metrics[CreativeOrientationMetric.NATURAL_PROGRESSION] = self._calculate_progression_strength(all_content)
        metrics[CreativeOrientationMetric.ENERGY_DIRECTION] = self._calculate_energy_direction(thoughts)
        metrics[CreativeOrientationMetric.LANGUAGE_CONSISTENCY] = self._calculate_language_consistency(thoughts)
        metrics[CreativeOrientationMetric.TENSION_SUSTAINABILITY] = self._calculate_session_sustainability(thoughts)
        
        return metrics
    
    def _calculate_energy_direction(self, thoughts: List[ThoughtData]) -> float:
        """Calculate overall energy direction across session."""
        if not thoughts:
            return 0.0
        
        advancement_scores = []
        for thought in thoughts:
            content = thought.thought.lower()
            outcome_strength = self._calculate_outcome_strength(content)
            progression_strength = self._calculate_progression_strength(content)
            oscillation = self._calculate_oscillation_score(content)
            
            advancement = max(0, (outcome_strength + progression_strength) / 2 - oscillation)
            advancement_scores.append(advancement)
        
        return np.mean(advancement_scores)
    
    def _calculate_language_consistency(self, thoughts: List[ThoughtData]) -> float:
        """Calculate consistency of creative orientation language."""
        if len(thoughts) < 2:
            return 1.0
        
        create_words = ['create', 'build', 'develop', 'establish', 'generate']
        problem_words = ['fix', 'solve', 'eliminate', 'avoid', 'prevent']
        
        create_scores = []
        problem_scores = []
        
        for thought in thoughts:
            content = thought.thought.lower()
            create_count = sum(1 for word in create_words if word in content)
            problem_count = sum(1 for word in problem_words if word in content)
            
            total = create_count + problem_count
            if total > 0:
                create_ratio = create_count / total
            else:
                create_ratio = 0.5  # Neutral when no indicators
            
            create_scores.append(create_ratio)
        
        # Consistency is 1 - variance in create ratios
        consistency = 1.0 - np.var(create_scores)
        return max(0.0, min(1.0, consistency))
    
    def _calculate_session_sustainability(self, thoughts: List[ThoughtData]) -> float:
        """Calculate sustainability of tension across the session."""
        if not thoughts:
            return 0.0
        
        sustainability_scores = []
        for thought in thoughts:
            pattern = self._analyze_thought_pattern(thought)
            sustainability_scores.append(pattern.sustainability_score)
        
        # Trend analysis - is sustainability improving, declining, or stable?
        if len(sustainability_scores) > 1:
            trend = np.polyfit(range(len(sustainability_scores)), sustainability_scores, 1)[0]
            avg_sustainability = np.mean(sustainability_scores)
            
            # Bonus for improving trend, penalty for declining
            return min(1.0, max(0.0, avg_sustainability + trend * 0.1))
        
        return sustainability_scores[0] if sustainability_scores else 0.0
    
    def _determine_overall_pattern(self, pattern_evolution: List[PatternAnalysis]) -> PatternSignature:
        """Determine overall session pattern from evolution."""
        if not pattern_evolution:
            return PatternSignature.STRUCTURAL_TENSION_EMERGING
        
        # Weight recent patterns more heavily
        weighted_signatures = []
        for i, pattern in enumerate(pattern_evolution):
            weight = (i + 1) / len(pattern_evolution)  # Later patterns get higher weight
            weighted_signatures.extend([pattern.signature] * int(weight * 10))
        
        # Most common weighted signature
        if weighted_signatures:
            return Counter(weighted_signatures).most_common(1)[0][0]
        else:
            return pattern_evolution[-1].signature  # Latest pattern if weighting fails
    
    def _assess_tension_strength(self, thoughts: List[ThoughtData], 
                                metrics: Dict[CreativeOrientationMetric, float]) -> StructuralTensionStrength:
        """Assess overall structural tension strength."""
        if not metrics:
            return StructuralTensionStrength.NONE
        
        outcome_score = metrics.get(CreativeOrientationMetric.OUTCOME_CLARITY, 0)
        reality_score = metrics.get(CreativeOrientationMetric.REALITY_GROUNDING, 0) 
        progression_score = metrics.get(CreativeOrientationMetric.NATURAL_PROGRESSION, 0)
        energy_direction = metrics.get(CreativeOrientationMetric.ENERGY_DIRECTION, 0)
        
        # Structural tension requires both outcome and reality
        has_tension = outcome_score > 0.3 and reality_score > 0.3
        
        if not has_tension:
            return StructuralTensionStrength.NONE
        elif energy_direction > 0.7 and progression_score > 0.5:
            return StructuralTensionStrength.ADVANCING
        elif outcome_score > 0.6 and reality_score > 0.6:
            return StructuralTensionStrength.STRONG
        elif outcome_score > 0.4 and reality_score > 0.4:
            return StructuralTensionStrength.MODERATE
        else:
            return StructuralTensionStrength.WEAK
    
    def _detect_breakthrough_moments(self, thoughts: List[ThoughtData]) -> List[str]:
        """Detect breakthrough moments in the session."""
        breakthroughs = []
        
        for i, thought in enumerate(thoughts):
            content = thought.thought.lower()
            for pattern in self.breakthrough_indicators:
                if re.search(pattern, content, re.IGNORECASE):
                    breakthroughs.append(f"Thought #{i+1}: Breakthrough pattern detected")
                    break
        
        return breakthroughs
    
    def _generate_structural_recommendations(self, profile: CreativeOrientationProfile) -> List[str]:
        """Generate structural recommendations based on profile analysis."""
        recommendations = []
        
        # Analyze metrics for specific recommendations
        metrics = profile.creative_metrics
        
        if metrics.get(CreativeOrientationMetric.OUTCOME_CLARITY, 0) < 0.4:
            recommendations.append("Strengthen desired outcome clarity with specific, measurable language")
        
        if metrics.get(CreativeOrientationMetric.REALITY_GROUNDING, 0) < 0.4:
            recommendations.append("Ground current reality assessment with concrete, observable facts")
        
        if metrics.get(CreativeOrientationMetric.NATURAL_PROGRESSION, 0) < 0.4:
            recommendations.append("Clarify natural progression pathway from current reality to desired outcome")
        
        if metrics.get(CreativeOrientationMetric.LANGUAGE_CONSISTENCY, 0) < 0.6:
            recommendations.append("Maintain consistent creative orientation language throughout session")
        
        if profile.energy_sustainability_index < 0.5:
            recommendations.append("Focus on sustainable structural tension rather than forced effort")
        
        if profile.overall_pattern == PatternSignature.OSCILLATING_HIGH_ENERGY:
            recommendations.append("Channel high energy into advancing structural tension rather than oscillating patterns")
        elif profile.overall_pattern == PatternSignature.PROBLEM_SOLVING_TRAP:
            recommendations.append("Shift from problem-solving to outcome-creating language and focus")
        
        return recommendations


# Global engine instance
pattern_engine = AdvancedPatternRecognizer()


def analyze_creative_orientation(thoughts: List[ThoughtData]) -> CreativeOrientationProfile:
    """
    Public API for advanced creative orientation analysis.
    
    Args:
        thoughts: List of ThoughtData to analyze
        
    Returns:
        CreativeOrientationProfile with comprehensive analysis
    """
    return pattern_engine.analyze_session_patterns(thoughts)