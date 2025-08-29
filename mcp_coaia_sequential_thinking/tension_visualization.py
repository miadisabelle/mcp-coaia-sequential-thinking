"""
Mathematical Framework for Structural Tension Visualization

This module implements geometric and mathematical models for visualizing structural tension
as described in Robert Fritz's creative orientation methodology. It provides quantitative
measures and visual representations of the tension between desired outcomes and current reality.

Key Features:
- Vector field models for structural tension forces
- Spring physics simulations for tension dynamics
- Geometric progression analysis for advancing vs oscillating patterns
- Telescoping visualization framework for COAIA Memory integration

Connected to Issues:
- #130 (Creative Observer System development)
- #133 (AI consistency checker for structural tension methodology compliance)
- #128 (MCP Creative Orientation LLMS and Memory Graph)
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import json

from .models import ThoughtData
from .co_lint_integration import StructuralTensionStrength
from .creative_orientation_engine import CreativeOrientationProfile, PatternSignature

logger = logging.getLogger(__name__)


class VectorFieldType(Enum):
    """Types of vector fields for structural tension modeling."""
    ATTRACTIVE = "attractive"  # Pulling toward desired outcome
    REPULSIVE = "repulsive"    # Pushing away from current reality
    CIRCULAR = "circular"      # Oscillating patterns
    GRADIENT = "gradient"      # Smooth advancement field
    TURBULENT = "turbulent"    # Chaotic, weak tension


class GeometricModel(Enum):
    """Geometric models for visualizing structural tension."""
    SPRING_SYSTEM = "spring_system"
    VECTOR_FIELD = "vector_field" 
    POTENTIAL_WELL = "potential_well"
    PHASE_SPACE = "phase_space"
    TOPOLOGY_MAP = "topology_map"


@dataclass
class TensionVector:
    """3D vector representation of structural tension."""
    magnitude: float
    direction: Tuple[float, float, float]  # (x, y, z) unit vector
    source: str  # What creates this tension
    stability: float  # How stable this vector is over time
    

@dataclass
class GeometricTensionModel:
    """Complete geometric model of structural tension."""
    model_type: GeometricModel
    tension_vectors: List[TensionVector] = field(default_factory=list)
    field_equations: Dict[str, Any] = field(default_factory=dict)
    stability_matrix: np.ndarray = None
    energy_landscape: np.ndarray = None
    critical_points: List[Tuple[float, float, float]] = field(default_factory=list)
    advancement_trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    

@dataclass
class VisualizationMetrics:
    """Quantitative metrics for tension visualization."""
    tension_strength: float
    advancement_rate: float
    oscillation_amplitude: float
    energy_efficiency: float
    stability_index: float
    convergence_probability: float
    breakthrough_potential: float


class StructuralTensionMathematics:
    """Mathematical framework for structural tension analysis."""
    
    def __init__(self):
        self.golden_ratio = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618
        self.tension_constant = 1.0  # Base tension strength
        self.damping_coefficient = 0.1  # Energy loss factor
        self.noise_threshold = 0.05  # Minimum signal strength
    
    def calculate_tension_vector(self, desired_outcome_clarity: float, 
                                current_reality_strength: float,
                                natural_progression_clarity: float) -> TensionVector:
        """
        Calculate the primary structural tension vector.
        
        Based on Robert Fritz's model: structural tension = f(desired outcome, current reality)
        Natural progression provides the direction vector.
        """
        # Magnitude using geometric mean (balanced contribution)
        magnitude = math.sqrt(desired_outcome_clarity * current_reality_strength)
        
        # Direction influenced by natural progression clarity
        if natural_progression_clarity > 0.3:
            # Strong progression creates forward direction
            direction = self._normalize_vector((natural_progression_clarity, 0.5, 0.2))
        else:
            # Weak progression creates more vertical (aspirational) direction
            direction = self._normalize_vector((0.2, 0.8, 0.3))
        
        # Stability based on consistency of components
        component_variance = np.var([desired_outcome_clarity, current_reality_strength, natural_progression_clarity])
        stability = max(0.0, 1.0 - component_variance)
        
        return TensionVector(
            magnitude=magnitude,
            direction=direction,
            source="primary_structural_tension",
            stability=stability
        )
    
    def model_spring_dynamics(self, tension_vector: TensionVector, 
                             oscillation_amplitude: float,
                             time_steps: int = 100) -> GeometricTensionModel:
        """
        Model structural tension as a spring system.
        
        Uses Hooke's Law with modifications for creative dynamics:
        F = -k(x - x₀) + creative_force + damping
        """
        model = GeometricTensionModel(model_type=GeometricModel.SPRING_SYSTEM)
        
        # Spring constant based on tension strength
        k = tension_vector.magnitude * self.tension_constant
        
        # Equilibrium position (desired outcome)
        x0 = np.array([1.0, 1.0, 1.0])  # Normalized desired outcome position
        
        # Current position (current reality)
        x_current = np.array([0.0, 0.0, 0.0])  # Starting from current reality
        
        # Initialize trajectory
        trajectory = []
        velocity = np.zeros(3)
        
        for t in range(time_steps):
            # Spring force toward equilibrium
            spring_force = -k * (x_current - x0)
            
            # Creative force (aligned with direction vector)
            creative_force = tension_vector.magnitude * np.array(tension_vector.direction)
            
            # Damping force
            damping_force = -self.damping_coefficient * velocity
            
            # Oscillation component (if present)
            if oscillation_amplitude > 0.1:
                oscillation_force = oscillation_amplitude * np.sin(2 * math.pi * t / 20) * np.array([1, 0, 0])
            else:
                oscillation_force = np.zeros(3)
            
            # Total force
            total_force = spring_force + creative_force + damping_force + oscillation_force
            
            # Update velocity and position (Euler integration)
            dt = 0.1
            velocity += total_force * dt
            x_current += velocity * dt
            
            trajectory.append(tuple(x_current))
        
        model.advancement_trajectory = trajectory
        model.tension_vectors = [tension_vector]
        
        # Calculate stability matrix (Jacobian at equilibrium)
        model.stability_matrix = np.array([
            [-k, 0, 0],
            [0, -k, 0],
            [0, 0, -k]
        ])
        
        return model
    
    def create_vector_field(self, creative_profile: CreativeOrientationProfile,
                           field_resolution: int = 20) -> GeometricTensionModel:
        """
        Create a vector field representation of structural tension.
        
        The field shows the direction and strength of creative forces at each point
        in the outcome-reality-progression space.
        """
        model = GeometricTensionModel(model_type=GeometricModel.VECTOR_FIELD)
        
        # Create 3D grid
        x = np.linspace(0, 1, field_resolution)  # Current reality axis
        y = np.linspace(0, 1, field_resolution)  # Desired outcome axis  
        z = np.linspace(0, 1, field_resolution)  # Natural progression axis
        
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Calculate vector field based on creative profile
        field_strength = self._get_field_strength_from_profile(creative_profile)
        
        if creative_profile.overall_pattern == PatternSignature.ADVANCING_STRONG:
            field_type = VectorFieldType.ATTRACTIVE
        elif creative_profile.overall_pattern in [PatternSignature.OSCILLATING_HIGH_ENERGY, 
                                                PatternSignature.OSCILLATING_LOW_ENERGY]:
            field_type = VectorFieldType.CIRCULAR
        elif creative_profile.overall_pattern == PatternSignature.PROBLEM_SOLVING_TRAP:
            field_type = VectorFieldType.REPULSIVE
        else:
            field_type = VectorFieldType.GRADIENT
        
        # Generate vector field equations
        U, V, W = self._generate_field_vectors(X, Y, Z, field_type, field_strength)
        
        model.field_equations = {
            'X': X.tolist(),
            'Y': Y.tolist(), 
            'Z': Z.tolist(),
            'U': U.tolist(),
            'V': V.tolist(),
            'W': W.tolist(),
            'field_type': field_type.value,
            'field_strength': field_strength
        }
        
        # Find critical points (equilibria)
        model.critical_points = self._find_critical_points(U, V, W, X, Y, Z)
        
        return model
    
    def analyze_energy_landscape(self, model: GeometricTensionModel) -> np.ndarray:
        """
        Analyze the energy landscape of structural tension.
        
        Lower energy corresponds to stronger structural tension.
        """
        if model.model_type == GeometricModel.SPRING_SYSTEM:
            return self._calculate_spring_energy_landscape(model)
        elif model.model_type == GeometricModel.VECTOR_FIELD:
            return self._calculate_field_energy_landscape(model)
        else:
            logger.warning(f"Energy landscape not implemented for {model.model_type}")
            return np.array([[]])
    
    def calculate_visualization_metrics(self, model: GeometricTensionModel,
                                      creative_profile: CreativeOrientationProfile) -> VisualizationMetrics:
        """Calculate quantitative metrics for visualization."""
        
        # Tension strength from primary vector or field strength
        if model.tension_vectors:
            tension_strength = model.tension_vectors[0].magnitude
        else:
            tension_strength = model.field_equations.get('field_strength', 0.0)
        
        # Advancement rate from trajectory analysis
        advancement_rate = self._calculate_advancement_rate(model)
        
        # Oscillation amplitude from trajectory variance
        oscillation_amplitude = self._calculate_oscillation_amplitude(model)
        
        # Energy efficiency from profile metrics
        energy_efficiency = creative_profile.energy_sustainability_index
        
        # Stability from tension vector stability or eigenvalue analysis
        stability_index = self._calculate_stability_index(model)
        
        # Convergence probability based on pattern and stability
        convergence_probability = self._calculate_convergence_probability(
            creative_profile, stability_index
        )
        
        # Breakthrough potential from profile indicators
        breakthrough_potential = len(creative_profile.breakthrough_indicators) / 5.0
        breakthrough_potential = min(1.0, breakthrough_potential)
        
        return VisualizationMetrics(
            tension_strength=tension_strength,
            advancement_rate=advancement_rate,
            oscillation_amplitude=oscillation_amplitude,
            energy_efficiency=energy_efficiency,
            stability_index=stability_index,
            convergence_probability=convergence_probability,
            breakthrough_potential=breakthrough_potential
        )
    
    def generate_telescoping_data(self, models: List[GeometricTensionModel]) -> Dict[str, Any]:
        """
        Generate data for telescoping visualization in COAIA Memory charts.
        
        Creates hierarchical zoom levels from session-level down to thought-level detail.
        """
        telescoping_data = {
            'zoom_levels': [],
            'transition_matrices': [],
            'detail_hierarchies': {}
        }
        
        # Level 0: Overall session pattern
        if models:
            primary_model = models[0]
            telescoping_data['zoom_levels'].append({
                'level': 0,
                'scope': 'session_overview',
                'model_type': primary_model.model_type.value,
                'critical_points': primary_model.critical_points,
                'summary_metrics': self._summarize_model_metrics(primary_model)
            })
            
            # Level 1: Stage-based breakdown
            telescoping_data['zoom_levels'].append({
                'level': 1,
                'scope': 'stage_breakdown', 
                'models_count': len(models),
                'stage_transitions': self._calculate_stage_transitions(models)
            })
            
            # Level 2: Individual thought analysis
            telescoping_data['zoom_levels'].append({
                'level': 2,
                'scope': 'thought_detail',
                'thought_vectors': [
                    {
                        'model_index': i,
                        'tension_vectors': [
                            {
                                'magnitude': tv.magnitude,
                                'direction': tv.direction,
                                'stability': tv.stability
                            }
                            for tv in model.tension_vectors
                        ]
                    }
                    for i, model in enumerate(models)
                ]
            })
        
        return telescoping_data
    
    def _normalize_vector(self, vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Normalize a 3D vector to unit length."""
        magnitude = math.sqrt(sum(x**2 for x in vector))
        if magnitude == 0:
            return (0.0, 0.0, 0.0)
        return tuple(x / magnitude for x in vector)
    
    def _get_field_strength_from_profile(self, profile: CreativeOrientationProfile) -> float:
        """Extract field strength from creative orientation profile."""
        if profile.tension_strength == StructuralTensionStrength.ADVANCING:
            return 1.0
        elif profile.tension_strength == StructuralTensionStrength.STRONG:
            return 0.8
        elif profile.tension_strength == StructuralTensionStrength.MODERATE:
            return 0.6
        elif profile.tension_strength == StructuralTensionStrength.WEAK:
            return 0.3
        else:
            return 0.1
    
    def _generate_field_vectors(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                               field_type: VectorFieldType, strength: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate vector field components based on field type."""
        
        if field_type == VectorFieldType.ATTRACTIVE:
            # Point toward (1, 1, 1) - the desired outcome
            U = strength * (1.0 - X)
            V = strength * (1.0 - Y) 
            W = strength * (1.0 - Z)
            
        elif field_type == VectorFieldType.GRADIENT:
            # Smooth gradient field toward desired outcome
            U = strength * np.exp(-(X - 1)**2) * (1.0 - X)
            V = strength * np.exp(-(Y - 1)**2) * (1.0 - Y)
            W = strength * np.exp(-(Z - 1)**2) * (1.0 - Z)
            
        elif field_type == VectorFieldType.CIRCULAR:
            # Oscillating/circular patterns
            U = strength * np.sin(2 * np.pi * Y) * np.cos(2 * np.pi * Z)
            V = strength * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Z)
            W = strength * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
            
        elif field_type == VectorFieldType.REPULSIVE:
            # Push away from current state (problem-solving trap)
            U = -strength * X / (X**2 + Y**2 + Z**2 + 0.01)
            V = -strength * Y / (X**2 + Y**2 + Z**2 + 0.01)
            W = -strength * Z / (X**2 + Y**2 + Z**2 + 0.01)
            
        else:  # TURBULENT
            # Chaotic, weak field
            U = strength * 0.1 * np.random.normal(size=X.shape)
            V = strength * 0.1 * np.random.normal(size=Y.shape)
            W = strength * 0.1 * np.random.normal(size=Z.shape)
        
        return U, V, W
    
    def _find_critical_points(self, U: np.ndarray, V: np.ndarray, W: np.ndarray,
                             X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> List[Tuple[float, float, float]]:
        """Find critical points (equilibria) in the vector field."""
        critical_points = []
        
        # Simple approach: find points where field magnitude is minimal
        field_magnitude = np.sqrt(U**2 + V**2 + W**2)
        
        # Find local minima
        for i in range(1, field_magnitude.shape[0] - 1):
            for j in range(1, field_magnitude.shape[1] - 1):
                for k in range(1, field_magnitude.shape[2] - 1):
                    current = field_magnitude[i, j, k]
                    
                    # Check if this is a local minimum
                    neighbors = [
                        field_magnitude[i-1, j, k], field_magnitude[i+1, j, k],
                        field_magnitude[i, j-1, k], field_magnitude[i, j+1, k],
                        field_magnitude[i, j, k-1], field_magnitude[i, j, k+1]
                    ]
                    
                    if current < min(neighbors) and current < 0.1:
                        critical_points.append((X[i, j, k], Y[i, j, k], Z[i, j, k]))
        
        return critical_points
    
    def _calculate_spring_energy_landscape(self, model: GeometricTensionModel) -> np.ndarray:
        """Calculate energy landscape for spring system."""
        if not model.advancement_trajectory:
            return np.array([[]])
        
        trajectory = np.array(model.advancement_trajectory)
        
        # Calculate potential energy at each point
        # E = 0.5 * k * r² where r is distance from equilibrium
        equilibrium = np.array([1.0, 1.0, 1.0])
        
        energy = []
        for point in trajectory:
            distance = np.linalg.norm(point - equilibrium)
            potential_energy = 0.5 * self.tension_constant * distance**2
            energy.append(potential_energy)
        
        return np.array(energy)
    
    def _calculate_field_energy_landscape(self, model: GeometricTensionModel) -> np.ndarray:
        """Calculate energy landscape for vector field."""
        # Energy is inverse of field strength (stronger field = lower energy)
        field_strength = model.field_equations.get('field_strength', 0.0)
        
        if field_strength == 0:
            return np.array([[1.0]])  # High energy for no field
        else:
            return np.array([[1.0 / field_strength]])  # Lower energy for stronger field
    
    def _calculate_advancement_rate(self, model: GeometricTensionModel) -> float:
        """Calculate rate of advancement toward desired outcome."""
        if not model.advancement_trajectory:
            return 0.0
        
        trajectory = np.array(model.advancement_trajectory)
        desired_outcome = np.array([1.0, 1.0, 1.0])
        
        # Calculate distance to desired outcome over time
        distances = [np.linalg.norm(point - desired_outcome) for point in trajectory]
        
        # Rate is negative slope of distance (advancing = decreasing distance)
        if len(distances) > 1:
            # Simple linear regression for trend
            x = np.arange(len(distances))
            slope = np.polyfit(x, distances, 1)[0]
            return max(0.0, -slope)  # Positive advancement rate
        
        return 0.0
    
    def _calculate_oscillation_amplitude(self, model: GeometricTensionModel) -> float:
        """Calculate amplitude of oscillating patterns."""
        if not model.advancement_trajectory:
            return 0.0
        
        trajectory = np.array(model.advancement_trajectory)
        
        # Calculate variance in each dimension
        variances = np.var(trajectory, axis=0)
        
        # Amplitude is geometric mean of variances
        return float(np.sqrt(np.prod(variances)))
    
    def _calculate_stability_index(self, model: GeometricTensionModel) -> float:
        """Calculate stability index from model analysis."""
        if model.stability_matrix is not None:
            # Eigenvalue analysis for stability
            eigenvalues = np.linalg.eigvals(model.stability_matrix)
            real_parts = np.real(eigenvalues)
            
            # Stable if all real parts are negative
            if all(part < 0 for part in real_parts):
                return 1.0 - abs(max(real_parts))  # More negative = more stable
            else:
                return 0.0  # Unstable
        
        # Fallback: use tension vector stability
        if model.tension_vectors:
            return np.mean([tv.stability for tv in model.tension_vectors])
        
        return 0.5  # Neutral stability
    
    def _calculate_convergence_probability(self, profile: CreativeOrientationProfile, 
                                         stability: float) -> float:
        """Calculate probability of converging to desired outcome."""
        base_probability = stability
        
        # Boost for advancing patterns
        if profile.overall_pattern in [PatternSignature.ADVANCING_STRONG, PatternSignature.ADVANCING_MODERATE]:
            base_probability += 0.2
        
        # Penalty for oscillating patterns
        if profile.overall_pattern in [PatternSignature.OSCILLATING_HIGH_ENERGY, PatternSignature.OSCILLATING_LOW_ENERGY]:
            base_probability -= 0.3
        
        # Boost for high energy sustainability
        base_probability += profile.energy_sustainability_index * 0.2
        
        return max(0.0, min(1.0, base_probability))
    
    def _summarize_model_metrics(self, model: GeometricTensionModel) -> Dict[str, float]:
        """Summarize key metrics for telescoping visualization."""
        metrics = {}
        
        if model.tension_vectors:
            metrics['avg_tension_magnitude'] = np.mean([tv.magnitude for tv in model.tension_vectors])
            metrics['avg_stability'] = np.mean([tv.stability for tv in model.tension_vectors])
        
        if model.critical_points:
            metrics['critical_points_count'] = len(model.critical_points)
        
        if model.advancement_trajectory:
            trajectory = np.array(model.advancement_trajectory)
            metrics['trajectory_length'] = len(trajectory)
            metrics['final_distance_to_goal'] = float(np.linalg.norm(trajectory[-1] - np.array([1.0, 1.0, 1.0])))
        
        return metrics
    
    def _calculate_stage_transitions(self, models: List[GeometricTensionModel]) -> List[Dict[str, float]]:
        """Calculate transition metrics between stages."""
        transitions = []
        
        for i in range(len(models) - 1):
            current_model = models[i]
            next_model = models[i + 1]
            
            # Compare tension strengths
            current_strength = current_model.tension_vectors[0].magnitude if current_model.tension_vectors else 0.0
            next_strength = next_model.tension_vectors[0].magnitude if next_model.tension_vectors else 0.0
            
            transition = {
                'from_stage': i,
                'to_stage': i + 1,
                'tension_change': next_strength - current_strength,
                'continuity_score': min(1.0, 1.0 - abs(next_strength - current_strength))
            }
            
            transitions.append(transition)
        
        return transitions


# Global mathematics engine
tension_math = StructuralTensionMathematics()


def create_tension_visualization(creative_profile: CreativeOrientationProfile,
                                thoughts: List[ThoughtData]) -> Dict[str, Any]:
    """
    Create comprehensive mathematical visualization of structural tension.
    
    Args:
        creative_profile: Creative orientation analysis results
        thoughts: List of thoughts for detailed analysis
        
    Returns:
        Complete visualization data with geometric models and metrics
    """
    visualization_data = {
        'models': [],
        'metrics': None,
        'telescoping_data': None,
        'mathematical_summary': {}
    }
    
    try:
        # Create primary tension vector from overall profile metrics
        outcome_clarity = creative_profile.creative_metrics.get('outcome_clarity', 0.0)
        reality_grounding = creative_profile.creative_metrics.get('reality_grounding', 0.0) 
        natural_progression = creative_profile.creative_metrics.get('natural_progression', 0.0)
        
        primary_vector = tension_math.calculate_tension_vector(
            outcome_clarity, reality_grounding, natural_progression
        )
        
        # Create spring system model
        spring_model = tension_math.model_spring_dynamics(
            primary_vector, 
            oscillation_amplitude=float(len([p for p in creative_profile.pattern_evolution 
                                           if 'oscillating' in p.signature.value])) / max(1, len(creative_profile.pattern_evolution))
        )
        visualization_data['models'].append(spring_model)
        
        # Create vector field model
        field_model = tension_math.create_vector_field(creative_profile)
        visualization_data['models'].append(field_model)
        
        # Calculate visualization metrics
        visualization_data['metrics'] = tension_math.calculate_visualization_metrics(
            spring_model, creative_profile
        )
        
        # Generate telescoping data for COAIA Memory integration
        visualization_data['telescoping_data'] = tension_math.generate_telescoping_data(
            visualization_data['models']
        )
        
        # Mathematical summary
        visualization_data['mathematical_summary'] = {
            'primary_tension_magnitude': primary_vector.magnitude,
            'primary_tension_direction': primary_vector.direction,
            'stability_eigenvalues': spring_model.stability_matrix.diagonal().tolist() if spring_model.stability_matrix is not None else [],
            'field_strength': field_model.field_equations.get('field_strength', 0.0),
            'critical_points_count': len(field_model.critical_points),
            'energy_minimum': float(np.min(tension_math.analyze_energy_landscape(spring_model))) if spring_model.advancement_trajectory else 0.0
        }
        
        logger.info(f"Created tension visualization with {len(visualization_data['models'])} models")
        
    except Exception as e:
        logger.error(f"Error creating tension visualization: {e}")
        visualization_data['error'] = str(e)
    
    return visualization_data