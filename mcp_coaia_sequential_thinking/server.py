import json
import os
import sys
from typing import List, Optional, Dict, Any
from datetime import datetime

from mcp.server.fastmcp import FastMCP

# Use absolute imports when running as a script
try:
    # When installed as a package
    from .models import ThoughtData, ThoughtStage
    from .storage import ThoughtStorage
    from .analysis import ThoughtAnalyzer
    from .logging_conf import configure_logging
    from .integration_bridge import integration_bridge
    from .co_lint_integration import validate_thought, ValidationSeverity
    from .creative_orientation_engine import analyze_creative_orientation
    from .tension_visualization import create_tension_visualization
    from .constitutional_core import constitutional_core, ConstitutionalPrinciple
    from .polycentric_lattice import (
        agent_registry, ConstitutionalAgent, AnalysisAgent, 
        AgentRole, MessageType, MessagePriority
    )
    from .agent_coordination import task_coordinator, TaskType
    from .resilient_connection import (
        resilient_connection, NoveltySearchAgent, ExplorationMode, DiscoveryType
    )
    from .consensus_decision_engine import (
        ConsensusDecisionEngine, DecisionType, ConsensusStatus, MMORElement
    )
    # NEW: Creative Orientation Foundation
    from .creative_orientation_foundation import (
        establish_structural_tension, validate_creative_action,
        StructuralTension, CreativeOrientationValidation,
        creative_orientation_foundation
    )
    from .generative_agent_lattice import (
        generative_lattice, ArchetypeRole, PerspectiveType
    )
    # NEW: Prompts and Resources for structural thinking
    from .prompts import list_prompts, get_prompt, PROMPTS
    from .resources import list_resources, get_resource_content, RESOURCES
except ImportError:
    # When run directly
    from mcp_coaia_sequential_thinking.models import ThoughtData, ThoughtStage
    from mcp_coaia_sequential_thinking.storage import ThoughtStorage
    from mcp_coaia_sequential_thinking.analysis import ThoughtAnalyzer
    from mcp_coaia_sequential_thinking.logging_conf import configure_logging
    from mcp_coaia_sequential_thinking.integration_bridge import integration_bridge
    from mcp_coaia_sequential_thinking.co_lint_integration import validate_thought, ValidationSeverity
    from mcp_coaia_sequential_thinking.creative_orientation_engine import analyze_creative_orientation
    from mcp_coaia_sequential_thinking.tension_visualization import create_tension_visualization
    from mcp_coaia_sequential_thinking.constitutional_core import constitutional_core, ConstitutionalPrinciple
    from mcp_coaia_sequential_thinking.polycentric_lattice import (
        agent_registry, ConstitutionalAgent, AnalysisAgent, 
        AgentRole, MessageType, MessagePriority
    )
    from mcp_coaia_sequential_thinking.agent_coordination import task_coordinator, TaskType
    from mcp_coaia_sequential_thinking.resilient_connection import (
        resilient_connection, NoveltySearchAgent, ExplorationMode, DiscoveryType
    )
    from mcp_coaia_sequential_thinking.consensus_decision_engine import (
        ConsensusDecisionEngine, DecisionType, ConsensusStatus, MMORElement
    )
    # NEW: Creative Orientation Foundation
    from mcp_coaia_sequential_thinking.creative_orientation_foundation import (
        establish_structural_tension, validate_creative_action,
        StructuralTension, CreativeOrientationValidation,
        creative_orientation_foundation
    )
    from mcp_coaia_sequential_thinking.generative_agent_lattice import (
        generative_lattice, ArchetypeRole, PerspectiveType
    )
    # NEW: Prompts and Resources for structural thinking
    from mcp_coaia_sequential_thinking.prompts import list_prompts, get_prompt, PROMPTS
    from mcp_coaia_sequential_thinking.resources import list_resources, get_resource_content, RESOURCES

logger = configure_logging("coaia-sequential-thinking.server")

# Initialize data persistence layer
data_store = None
try:
    from mcp_coaia_sequential_thinking.data_persistence import PolycentricDataStore
    data_store = PolycentricDataStore()
    logger.info("Data persistence layer initialized successfully")
except Exception as e:
    logger.error(f"Error initializing data persistence: {e}")
    logger.error("Data persistence functionality will not be available")

# Initialize enhanced polycentric lattice
enhanced_lattice = None
try:
    from mcp_coaia_sequential_thinking.enhanced_polycentric_lattice import EnhancedPolycentricLattice
    enhanced_lattice = EnhancedPolycentricLattice(constitutional_core, data_store)
    logger.info("Enhanced polycentric lattice initialized successfully")
except ImportError as e:
    logger.error(f"ImportError while importing enhanced systems: {e}")
    logger.error("Enhanced lattice functionality will not be available")
except Exception as e:
    logger.error(f"Error initializing enhanced polycentric lattice: {e}")
    logger.error("Enhanced lattice functionality will not be available")

mcp = FastMCP("coaia-sequential-thinking")

storage_dir = os.environ.get("MCP_STORAGE_DIR", None)
storage = ThoughtStorage(storage_dir)

@mcp.tool()
def process_thought(thought: str, thought_number: int, total_thoughts: int,
                    next_thought_needed: bool, stage: str,
                    tags: List[str] = [],
                    axioms_used: List[str] = [],
                    assumptions_challenged: List[str] = []) -> dict:
    """Add a sequential thought with its metadata.

    Args:
        thought: The content of the thought
        thought_number: The sequence number of this thought
        total_thoughts: The total expected thoughts in the sequence
        next_thought_needed: Whether more thoughts are needed after this one
        stage: The thinking stage (Definition of where we are (sometime thought as problem and is current-reality), Research, Analysis, Synthesis, Conclusion)
        tags: Optional keywords or categories for the thought
        axioms_used: Optional list of principles or axioms used in this thought
        assumptions_challenged: Optional list of assumptions challenged by this thought

    Returns:
        dict: Analysis of the processed thought
    """
    try:
        # Log the request
        logger.info(f"Processing thought #{thought_number}/{total_thoughts} in stage '{stage}'")

        # Convert stage string to enum
        thought_stage = ThoughtStage.from_string(stage)

        # Create thought data object with defaults for optional fields
        thought_data = ThoughtData(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_thought_needed=next_thought_needed,
            stage=thought_stage,
            tags=tags or [],
            axioms_used=axioms_used or [],
            assumptions_challenged=assumptions_challenged or []
        )

        # Validate and store
        thought_data.validate()
        
        # Run SCCP-enhanced CO-Lint validation
        validation_summary = validate_thought(thought, thought_data)
        logger.info(f"Validation completed: tension_strength={validation_summary.tension_strength.value}, "
                   f"creative_orientation_score={validation_summary.creative_orientation_score:.2f}")
        
        # Run constitutional validation
        constitutional_validation = constitutional_core.validate_content(thought, {
            "stage": stage,
            "thought_number": thought_number,
            "total_thoughts": total_thoughts
        })
        logger.info(f"Constitutional validation completed: compliance_score={constitutional_validation['constitutional_compliance_score']:.2f}")
        
        storage.add_thought(thought_data)

        # Get all thoughts for analysis
        all_thoughts = storage.get_all_thoughts()

        # Analyze the thought
        analysis = ThoughtAnalyzer.analyze_thought(thought_data, all_thoughts)
        
        # Add validation results to analysis
        analysis['validation'] = {
            'structural_tension_established': validation_summary.structural_tension_established,
            'tension_strength': validation_summary.tension_strength.value,
            'creative_orientation_score': validation_summary.creative_orientation_score,
            'advancing_pattern_detected': validation_summary.advancing_pattern_detected,
            'has_desired_outcome': validation_summary.has_desired_outcome,
            'has_current_reality': validation_summary.has_current_reality,
            'has_natural_progression': validation_summary.has_natural_progression,
            'validation_results': [
                {
                    'rule_id': result.rule_id,
                    'severity': result.severity.value,
                    'message': result.message,
                    'line_number': result.line_number,
                    'suggestion': result.suggestion,
                    'structural_insight': result.structural_insight,
                    'tension_impact': result.tension_impact.value if result.tension_impact else None
                }
                for result in validation_summary.validation_results
            ]
        }
        
        # Add constitutional validation results
        analysis['constitutional_validation'] = {
            'overall_valid': constitutional_validation['overall_valid'],
            'compliance_score': constitutional_validation['constitutional_compliance_score'],
            'violated_principles': constitutional_validation['violated_principles'],
            'recommendations': _generate_constitutional_recommendations(constitutional_validation)
        }

        # Log success
        logger.info(f"Successfully processed thought #{thought_number}")

        return analysis
    except json.JSONDecodeError as e:
        # Log JSON parsing error
        logger.error(f"JSON parsing error: {e}")
        return {
            "error": f"JSON parsing error: {str(e)}",
            "status": "failed"
        }
    except Exception as e:
        # Log error
        logger.error(f"Error processing thought: {str(e)}")

        return {
            "error": str(e),
            "status": "failed"
        }

@mcp.tool()
async def generate_summary() -> dict:
    """Generate a summary of the entire thinking process.

    Returns:
        dict: Summary of the thinking process
    """
    try:
        logger.info("Generating thinking process summary")

        # Get all thoughts
        all_thoughts = storage.get_all_thoughts()

        # Generate summary with SCCP analysis
        summary_result = ThoughtAnalyzer.generate_summary(all_thoughts)
        
        # Generate advanced creative orientation analysis
        creative_profile = analyze_creative_orientation(all_thoughts)
        
        # Check if session is ready for chart creation
        chart_readiness = integration_bridge.analyze_chart_readiness(all_thoughts)
        
        # Add chart readiness and creative orientation info to summary
        summary = summary_result.get('summary', {})
        
        # Add advanced creative orientation analysis
        summary['creativeOrientation'] = {
            "overallPattern": creative_profile.overall_pattern.value,
            "tensionStrength": creative_profile.tension_strength.value,
            "languageConsistencyScore": creative_profile.language_consistency_score,
            "energySustainabilityIndex": creative_profile.energy_sustainability_index,
            "creativeMetrics": {
                metric.value: score for metric, score in creative_profile.creative_metrics.items()
            },
            "breakthroughIndicators": creative_profile.breakthrough_indicators,
            "structuralRecommendations": creative_profile.structural_recommendations,
            "patternEvolution": [
                {
                    "signature": pattern.signature.value,
                    "confidence": pattern.confidence,
                    "energyLevel": pattern.energy_level,
                    "directionVector": pattern.direction_vector,
                    "sustainabilityScore": pattern.sustainability_score,
                    "contributingFactors": pattern.contributing_factors
                }
                for pattern in creative_profile.pattern_evolution
            ]
        }
        
        # Generate mathematical tension visualization
        try:
            tension_visualization = create_tension_visualization(creative_profile, all_thoughts)
            summary['tensionVisualization'] = {
                "mathematical_models": [
                    {
                        "model_type": model.model_type.value,
                        "tension_vectors_count": len(model.tension_vectors),
                        "critical_points_count": len(model.critical_points),
                        "has_trajectory": bool(model.advancement_trajectory),
                        "field_equations_available": bool(model.field_equations)
                    }
                    for model in tension_visualization.get('models', [])
                ],
                "visualization_metrics": {
                    "tension_strength": tension_visualization.get('metrics', {}).tension_strength if tension_visualization.get('metrics') else 0.0,
                    "advancement_rate": tension_visualization.get('metrics', {}).advancement_rate if tension_visualization.get('metrics') else 0.0,
                    "stability_index": tension_visualization.get('metrics', {}).stability_index if tension_visualization.get('metrics') else 0.0,
                    "convergence_probability": tension_visualization.get('metrics', {}).convergence_probability if tension_visualization.get('metrics') else 0.0
                },
                "telescoping_data": tension_visualization.get('telescoping_data', {}),
                "mathematical_summary": tension_visualization.get('mathematical_summary', {})
            }
            logger.info("Generated mathematical tension visualization")
        except Exception as viz_error:
            logger.error(f"Error creating tension visualization: {viz_error}")
            summary['tensionVisualization'] = {"error": str(viz_error)}
        
        summary['chartIntegration'] = {
            "readyForChartCreation": chart_readiness.get('readyForChartCreation', False),
            "structuralTensionEstablished": chart_readiness.get('structuralTensionEstablished', False),
            "tensionStrength": chart_readiness.get('tensionStrength', 0.0),
            "overallPattern": chart_readiness.get('overallPattern', 'insufficient_data')
        }
        
        # Trigger chart creation if ready
        if chart_readiness.get('readyForChartCreation', False):
            try:
                chart_data = chart_readiness.get('chartCreationData')
                if chart_data:
                    # Generate session ID for this thinking session
                    session_id = f"session_{len(all_thoughts)}_{all_thoughts[-1].id if all_thoughts else 'empty'}"
                    chart_id = await integration_bridge.create_chart_from_session(session_id, chart_data)
                    
                    summary['chartIntegration']['chartCreated'] = True
                    summary['chartIntegration']['chartId'] = chart_id
                    summary['chartIntegration']['sessionId'] = session_id
                    
                    logger.info(f"Auto-created chart {chart_id} from thinking session {session_id}")
            except Exception as chart_error:
                logger.error(f"Error creating chart: {chart_error}")
                summary['chartIntegration']['chartCreationError'] = str(chart_error)
        
        return {"summary": summary}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return {
            "error": f"JSON parsing error: {str(e)}",
            "status": "failed"
        }
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

@mcp.tool()
def clear_history() -> dict:
    """Clear the thought history.

    Returns:
        dict: Status message
    """
    try:
        logger.info("Clearing thought history")
        storage.clear_history()
        return {"status": "success", "message": "Thought history cleared"}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return {
            "error": f"JSON parsing error: {str(e)}",
            "status": "failed"
        }
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

@mcp.tool()
def export_session(file_path: str) -> dict:
    """Export the current thinking session to a file.

    Args:
        file_path: Path to save the exported session

    Returns:
        dict: Status message
    """
    try:
        logger.info(f"Exporting session to {file_path}")
        storage.export_session(file_path)
        return {
            "status": "success",
            "message": f"Session exported to {file_path}"
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return {
            "error": f"JSON parsing error: {str(e)}",
            "status": "failed"
        }
    except Exception as e:
        logger.error(f"Error exporting session: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

@mcp.tool()
def import_session(file_path: str) -> dict:
    """Import a thinking session from a file.

    Args:
        file_path: Path to the file to import

    Returns:
        dict: Status message
    """
    try:
        logger.info(f"Importing session from {file_path}")
        storage.import_session(file_path)
        return {
            "status": "success",
            "message": f"Session imported from {file_path}"
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return {
            "error": f"JSON parsing error: {str(e)}",
            "status": "failed"
        }
    except Exception as e:
        logger.error(f"Error importing session: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def check_integration_status() -> dict:
    """Check the integration status with COAIA Memory system.

    Returns:
        dict: Integration status and available records
    """
    try:
        logger.info("Checking integration status")
        
        # Get all thoughts to analyze readiness
        all_thoughts = storage.get_all_thoughts()
        chart_readiness = integration_bridge.analyze_chart_readiness(all_thoughts)
        
        # Get integration records
        integration_records = {}
        for session_id, record in integration_bridge.integration_records.items():
            integration_records[session_id] = {
                "integrationId": record.integration_id,
                "chartId": record.chart_id,
                "status": record.status.value,
                "patternType": record.pattern_type,
                "createdAt": record.created_at,
                "updatedAt": record.updated_at
            }
        
        return {
            "integrationStatus": {
                "coaiaMemoryAvailable": integration_bridge.coaia_memory_available,
                "currentSessionReadiness": {
                    "readyForChartCreation": chart_readiness.get('readyForChartCreation', False),
                    "structuralTensionEstablished": chart_readiness.get('structuralTensionEstablished', False),
                    "tensionStrength": chart_readiness.get('tensionStrength', 0.0),
                    "overallPattern": chart_readiness.get('overallPattern', 'insufficient_data')
                },
                "integrationRecords": integration_records,
                "totalRecords": len(integration_records)
            }
        }
    except Exception as e:
        logger.error(f"Error checking integration status: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def validate_thought_content(content: str, stage: Optional[str] = None) -> dict:
    """Validate thought content using SCCP-enhanced CO-Lint filtering.

    Args:
        content: The thought content to validate
        stage: Optional thinking stage for context-aware validation

    Returns:
        dict: Comprehensive validation results with SCCP insights
    """
    try:
        logger.info("Validating thought content with SCCP-enhanced CO-Lint")
        
        # Create minimal thought data if stage provided
        thought_data = None
        if stage:
            try:
                thought_stage = ThoughtStage.from_string(stage)
                thought_data = ThoughtData(
                    thought=content,
                    thought_number=1,
                    total_thoughts=1,
                    next_thought_needed=False,
                    stage=thought_stage
                )
            except Exception as e:
                logger.warning(f"Could not create ThoughtData from stage '{stage}': {e}")
        
        # Run validation
        validation_summary = validate_thought(content, thought_data)
        
        # Format response
        return {
            "validation_summary": {
                "structural_tension_established": validation_summary.structural_tension_established,
                "tension_strength": validation_summary.tension_strength.value,
                "creative_orientation_score": validation_summary.creative_orientation_score,
                "advancing_pattern_detected": validation_summary.advancing_pattern_detected,
                "oscillating_patterns_count": validation_summary.oscillating_patterns_count,
                "components": {
                    "has_desired_outcome": validation_summary.has_desired_outcome,
                    "has_current_reality": validation_summary.has_current_reality,
                    "has_natural_progression": validation_summary.has_natural_progression
                }
            },
            "validation_results": [
                {
                    "rule_id": result.rule_id,
                    "severity": result.severity.value,
                    "message": result.message,
                    "line_number": result.line_number,
                    "suggestion": result.suggestion,
                    "structural_insight": result.structural_insight,
                    "tension_impact": result.tension_impact.value if result.tension_impact else None
                }
                for result in validation_summary.validation_results
            ],
            "guidance": {
                "next_steps": _generate_validation_guidance(validation_summary),
                "structural_tension_advice": _generate_tension_advice(validation_summary)
            }
        }
        
    except Exception as e:
        logger.error(f"Error validating thought content: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


def _generate_validation_guidance(validation_summary) -> str:
    """Generate next steps guidance based on validation results."""
    if validation_summary.structural_tension_established:
        if validation_summary.tension_strength.value == 'advancing':
            return "Excellent! Your structural tension is strong and advancing. Continue with this momentum."
        else:
            return "Good structural tension foundation. Consider clarifying your Natural Progression to strengthen advancement."
    elif validation_summary.has_desired_outcome and not validation_summary.has_current_reality:
        return "You have a desired outcome. Consider stating your current reality to establish structural tension."
    elif validation_summary.has_current_reality and not validation_summary.has_desired_outcome:
        return "You've described current reality. Consider clarifying what you want to create to establish structural tension."
    else:
        return "Consider establishing structural tension by stating both your desired outcome and current reality."


def _generate_tension_advice(validation_summary) -> str:
    """Generate structural tension specific advice."""
    if validation_summary.creative_orientation_score < 0.3:
        return "Consider shifting from problem-solving language to creative orientation - focus on what you want to create."
    elif validation_summary.oscillating_patterns_count > 0:
        return "Oscillating patterns detected. Strengthen structural tension to create consistent advancing movement."
    elif validation_summary.tension_strength.value in ['moderate', 'strong', 'advancing']:
        return "Your structural tension is supporting creative advancement. Trust this natural pull toward your desired outcome."
    else:
        return "Establish clearer structural tension between where you are and where you want to be."


@mcp.tool()
def validate_constitutional_compliance(content: str, context: Optional[Dict[str, Any]] = None) -> dict:
    """Validate content against constitutional principles of the generative agentic system.
    
    Args:
        content: The content to validate against constitutional principles
        context: Optional context information for validation
        
    Returns:
        dict: Comprehensive constitutional compliance analysis
    """
    try:
        logger.info("Validating constitutional compliance")
        
        if context is None:
            context = {}
            
        validation_result = constitutional_core.validate_content(content, context)
        
        return {
            "constitutional_compliance": {
                "overall_valid": validation_result["overall_valid"],
                "compliance_score": validation_result["constitutional_compliance_score"],
                "violated_principles": validation_result["violated_principles"],
                "validation_details": validation_result["validation_details"]
            },
            "recommendations": _generate_constitutional_recommendations(validation_result),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error validating constitutional compliance: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def generate_active_pause_drafts(context: str, num_drafts: int = 3, 
                               selection_criteria: Optional[Dict[str, float]] = None) -> dict:
    """Generate multiple response drafts with different risk/reliability profiles using active pause mechanism.
    
    Args:
        context: The context for which to generate drafts
        num_drafts: Number of drafts to generate (default 3)
        selection_criteria: Criteria weights for draft selection
        
    Returns:
        dict: Multiple drafts with constitutional assessment and recommended selection
    """
    try:
        logger.info(f"Generating {num_drafts} active pause drafts")
        
        if selection_criteria is None:
            selection_criteria = {
                'novelty_weight': 0.3,
                'reliability_weight': 0.4,
                'constitutional_weight': 0.3
            }
        
        drafts = constitutional_core.generate_active_pause_drafts(context, num_drafts)
        best_draft = constitutional_core.select_best_draft(drafts, selection_criteria)
        
        return {
            "active_pause_analysis": {
                "drafts_generated": len(drafts),
                "selection_criteria": selection_criteria,
                "recommended_draft": best_draft["draft_id"],
                "recommendation_score": best_draft["selection_criteria"]
            },
            "drafts": drafts,
            "best_draft": best_draft,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error generating active pause drafts: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def make_constitutional_decision(decision_context: str, options: List[str], 
                               context: Optional[Dict[str, Any]] = None) -> dict:
    """Make a decision based on constitutional principles with full audit trail.
    
    Args:
        decision_context: Description of the decision being made
        options: List of possible decision options
        context: Optional context for decision making
        
    Returns:
        dict: Decision outcome with constitutional reasoning and audit trail
    """
    try:
        logger.info(f"Making constitutional decision: {decision_context}")
        
        if context is None:
            context = {}
            
        decision = constitutional_core.make_constitutional_decision(decision_context, options, context)
        
        return {
            "constitutional_decision": {
                "decision_id": decision.decision_id,
                "timestamp": decision.timestamp.isoformat(),
                "context": decision.decision_context,
                "chosen_option": decision.decision_outcome,
                "alternatives_considered": decision.alternative_considered,
                "applicable_principles": [p.value for p in decision.applicable_principles],
                "principle_applications": {p.value: app for p, app in decision.principle_application.items()},
                "audit_trail_available": True
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error making constitutional decision: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def get_constitutional_audit_trail(decision_id: str) -> dict:
    """Retrieve the complete audit trail for a constitutional decision.
    
    Args:
        decision_id: The ID of the decision to audit
        
    Returns:
        dict: Complete audit trail with principle applications and reasoning
    """
    try:
        logger.info(f"Retrieving audit trail for decision: {decision_id}")
        
        decision = constitutional_core.get_decision_audit_trail(decision_id)
        
        if decision is None:
            return {
                "error": f"Decision {decision_id} not found",
                "status": "not_found"
            }
        
        return {
            "audit_trail": {
                "decision_id": decision.decision_id,
                "timestamp": decision.timestamp.isoformat(),
                "decision_context": decision.decision_context,
                "applicable_principles": [p.value for p in decision.applicable_principles],
                "principle_applications": {p.value: app for p, app in decision.principle_application.items()},
                "decision_outcome": decision.decision_outcome,
                "alternatives_considered": decision.alternative_considered,
                "principle_conflicts": decision.principle_conflicts,
                "resolution_method": decision.resolution_method
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving audit trail: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def list_constitutional_principles() -> dict:
    """List all constitutional principles governing the system.
    
    Returns:
        dict: Complete list of constitutional principles with descriptions
    """
    try:
        logger.info("Listing constitutional principles")
        
        principles = {}
        for principle in ConstitutionalPrinciple:
            principles[principle.name] = {
                "value": principle.value,
                "description": _get_principle_description(principle),
                "category": _get_principle_category(principle),
                "hierarchy_level": constitutional_core.principle_hierarchy.get(principle, 999)
            }
        
        return {
            "constitutional_principles": principles,
            "total_principles": len(principles),
            "categories": list(set(p["category"] for p in principles.values())),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error listing constitutional principles: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def initialize_polycentric_lattice() -> dict:
    """Initialize the polycentric agentic lattice with core agents.
    
    Returns:
        dict: Status of lattice initialization with agent details
    """
    try:
        logger.info("Initializing polycentric agentic lattice")
        
        # Create core agents
        constitutional_agent = ConstitutionalAgent()
        analysis_agent = AnalysisAgent()
        
        # Register agents in the lattice
        agent_registry.register_agent(constitutional_agent)
        agent_registry.register_agent(analysis_agent)
        
        # Get lattice status
        status = agent_registry.get_agent_status_summary()
        
        return {
            "lattice_initialization": {
                "status": "success",
                "agents_created": 2,
                "constitutional_agent_id": constitutional_agent.agent_id,
                "analysis_agent_id": analysis_agent.agent_id,
                "lattice_status": status
            },
            "available_capabilities": {
                "constitutional": [cap.name for cap in constitutional_agent.get_capabilities()],
                "analysis": [cap.name for cap in analysis_agent.get_capabilities()]
            },
            "coordination_status": task_coordinator.get_coordination_status(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error initializing polycentric lattice: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def submit_agent_task(description: str, requirements: List[str], 
                     task_type: str = "individual",
                     priority: str = "medium") -> dict:
    """Submit a task to the polycentric agent lattice.
    
    Args:
        description: Description of the task to be performed
        requirements: List of required capabilities for the task
        task_type: Type of task (individual, collaborative, competitive, constitutional_review)
        priority: Task priority (low, medium, high, critical)
        
    Returns:
        dict: Task submission result with task ID and assignment details
    """
    try:
        logger.info(f"Submitting agent task: {description}")
        
        # Convert string enums
        task_type_enum = TaskType.INDIVIDUAL
        if task_type == "collaborative":
            task_type_enum = TaskType.COLLABORATIVE
        elif task_type == "competitive":
            task_type_enum = TaskType.COMPETITIVE
        elif task_type == "constitutional_review":
            task_type_enum = TaskType.CONSTITUTIONAL_REVIEW
        
        priority_enum = MessagePriority.MEDIUM
        if priority == "low":
            priority_enum = MessagePriority.LOW
        elif priority == "high":
            priority_enum = MessagePriority.HIGH
        elif priority == "critical":
            priority_enum = MessagePriority.CRITICAL
        
        # Submit task to coordinator
        task_id = task_coordinator.submit_task(
            description=description,
            requirements=requirements,
            task_type=task_type_enum,
            priority=priority_enum
        )
        
        # Get initial task status
        task_status = task_coordinator.get_task_status(task_id)
        
        return {
            "task_submission": {
                "task_id": task_id,
                "submitted_successfully": True,
                "task_type": task_type,
                "priority": priority,
                "requirements": requirements,
                "initial_status": task_status
            },
            "lattice_info": {
                "available_agents": len(agent_registry.agents),
                "coordination_status": task_coordinator.get_coordination_status()
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error submitting agent task: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def get_lattice_status() -> dict:
    """Get comprehensive status of the polycentric agentic lattice.
    
    Returns:
        dict: Complete lattice status including agents, tasks, and performance
    """
    try:
        logger.info("Getting polycentric lattice status")
        
        # Get agent registry status
        agent_status = agent_registry.get_agent_status_summary()
        
        # Get coordination status
        coordination_status = task_coordinator.get_coordination_status()
        
        # Get individual agent details
        agent_details = {}
        for agent_id, agent in agent_registry.agents.items():
            agent_details[agent_id] = {
                "name": agent.name,
                "role": agent.role.value,
                "active": agent.active,
                "capabilities": [cap.name for cap in agent.get_capabilities()],
                "performance_metrics": agent.performance_metrics,
                "collaboration_count": len(agent.collaboration_history),
                "competition_count": len(agent.competition_history),
                "message_queue_size": agent.message_queue.qsize()
            }
        
        return {
            "lattice_status": {
                "agent_registry": agent_status,
                "coordination_system": coordination_status,
                "agent_details": agent_details,
                "system_health": _assess_lattice_health(agent_status, coordination_status)
            },
            "capabilities_available": _get_available_capabilities(),
            "recent_activity": _get_recent_activity(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting lattice status: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def query_agent_capabilities(capability_filter: Optional[str] = None) -> dict:
    """Query available agent capabilities in the lattice.
    
    Args:
        capability_filter: Optional filter to search for specific capabilities
        
    Returns:
        dict: Available capabilities and agents that provide them
    """
    try:
        logger.info(f"Querying agent capabilities with filter: {capability_filter}")
        
        all_capabilities = {}
        
        for agent_id, agent in agent_registry.agents.items():
            for capability in agent.get_capabilities():
                if capability_filter is None or capability_filter.lower() in capability.name.lower():
                    if capability.name not in all_capabilities:
                        all_capabilities[capability.name] = {
                            "description": capability.description,
                            "agents": [],
                            "avg_competency": 0.0,
                            "avg_resource_cost": 0.0,
                            "avg_execution_time": 0.0
                        }
                    
                    all_capabilities[capability.name]["agents"].append({
                        "agent_id": agent_id,
                        "agent_name": agent.name,
                        "competency_score": capability.competency_score,
                        "resource_cost": capability.resource_cost,
                        "execution_time_estimate": capability.execution_time_estimate
                    })
        
        # Calculate averages
        for cap_name, cap_info in all_capabilities.items():
            agents = cap_info["agents"]
            if agents:
                cap_info["avg_competency"] = sum(a["competency_score"] for a in agents) / len(agents)
                cap_info["avg_resource_cost"] = sum(a["resource_cost"] for a in agents) / len(agents)
                cap_info["avg_execution_time"] = sum(a["execution_time_estimate"] for a in agents) / len(agents)
        
        return {
            "capability_query": {
                "filter_applied": capability_filter,
                "capabilities_found": len(all_capabilities),
                "capabilities": all_capabilities
            },
            "lattice_summary": {
                "total_agents": len(agent_registry.agents),
                "unique_capabilities": len(all_capabilities),
                "avg_competency_across_lattice": sum(
                    cap["avg_competency"] for cap in all_capabilities.values()
                ) / len(all_capabilities) if all_capabilities else 0.0
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error querying agent capabilities: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def get_task_status(task_id: str) -> dict:
    """Get the status of a specific task in the coordination system.
    
    Args:
        task_id: The ID of the task to check
        
    Returns:
        dict: Detailed task status and progress information
    """
    try:
        logger.info(f"Getting status for task: {task_id}")
        
        task_status = task_coordinator.get_task_status(task_id)
        
        if task_status is None:
            return {
                "error": f"Task {task_id} not found",
                "status": "not_found"
            }
        
        # Get additional details about assigned agents
        agent_details = {}
        for agent_id in task_status.get("assigned_agents", []):
            if agent_id in agent_registry.agents:
                agent = agent_registry.agents[agent_id]
                agent_details[agent_id] = {
                    "name": agent.name,
                    "role": agent.role.value,
                    "current_workload": agent.message_queue.qsize(),
                    "performance_score": agent.performance_metrics.get("task_completion_rate", 0.0)
                }
        
        return {
            "task_status": task_status,
            "assigned_agent_details": agent_details,
            "coordination_context": {
                "total_active_tasks": len(task_coordinator.active_tasks),
                "system_load": task_coordinator.get_coordination_status()
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def create_agent_collaboration(description: str, required_capabilities: List[str],
                             target_agents: Optional[List[str]] = None) -> dict:
    """Create a collaboration between multiple agents.
    
    Args:
        description: Description of the collaborative task
        required_capabilities: List of capabilities needed for the collaboration
        target_agents: Optional list of specific agents to invite
        
    Returns:
        dict: Collaboration creation result and participant details
    """
    try:
        logger.info(f"Creating agent collaboration: {description}")
        
        # If no target agents specified, find agents with required capabilities
        if target_agents is None:
            target_agents = []
            for capability in required_capabilities:
                matching_agents = agent_registry.find_agents_with_capability(capability)
                target_agents.extend(matching_agents)
            target_agents = list(set(target_agents))  # Remove duplicates
        
        # Submit as collaborative task
        task_id = task_coordinator.submit_task(
            description=description,
            requirements=required_capabilities,
            task_type=TaskType.COLLABORATIVE,
            priority=MessagePriority.MEDIUM
        )
        
        # Get task status
        task_status = task_coordinator.get_task_status(task_id)
        
        return {
            "collaboration": {
                "task_id": task_id,
                "description": description,
                "required_capabilities": required_capabilities,
                "target_agents": target_agents,
                "collaboration_status": task_status,
                "expected_participants": len(target_agents)
            },
            "coordination_info": {
                "coordination_system_active": task_coordinator.active,
                "current_system_load": task_coordinator.get_coordination_status()
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error creating agent collaboration: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


def _assess_lattice_health(agent_status: Dict[str, Any], coordination_status: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the overall health of the polycentric lattice."""
    total_agents = agent_status.get("total_agents", 0)
    active_agents = agent_status.get("active_agents", 0)
    
    agent_health = active_agents / total_agents if total_agents > 0 else 0.0
    
    total_tasks = coordination_status.get("active_tasks", 0)
    failed_tasks = coordination_status.get("failed_tasks", 0)
    
    task_health = 1.0 - (failed_tasks / max(total_tasks, 1))
    
    overall_health = (agent_health + task_health) / 2.0
    
    health_status = "excellent"
    if overall_health < 0.9:
        health_status = "good"
    if overall_health < 0.7:
        health_status = "fair"
    if overall_health < 0.5:
        health_status = "poor"
    
    return {
        "overall_health_score": overall_health,
        "health_status": health_status,
        "agent_health": agent_health,
        "task_health": task_health,
        "recommendations": _generate_health_recommendations(overall_health, agent_health, task_health)
    }


def _generate_health_recommendations(overall: float, agent: float, task: float) -> List[str]:
    """Generate recommendations for improving lattice health."""
    recommendations = []
    
    if agent < 0.8:
        recommendations.append("Consider activating more agents or checking agent status")
    
    if task < 0.7:
        recommendations.append("Review task coordination and failure patterns")
    
    if overall > 0.9:
        recommendations.append("Lattice operating optimally - consider expanding capabilities")
    
    return recommendations


def _get_available_capabilities() -> Dict[str, int]:
    """Get summary of available capabilities across the lattice."""
    capabilities = {}
    for agent in agent_registry.agents.values():
        for cap in agent.get_capabilities():
            capabilities[cap.name] = capabilities.get(cap.name, 0) + 1
    return capabilities


def _get_recent_activity() -> Dict[str, Any]:
    """Get recent activity summary from the lattice."""
    return {
        "active_message_processing": sum(1 for agent in agent_registry.agents.values() if agent.active),
        "total_message_queue_size": sum(agent.message_queue.qsize() for agent in agent_registry.agents.values()),
        "recent_collaborations": sum(len(agent.collaboration_history) for agent in agent_registry.agents.values()),
        "recent_competitions": sum(len(agent.competition_history) for agent in agent_registry.agents.values())
    }


@mcp.tool()
def establish_system_goal(description: str, priority: float = 0.5, 
                         target_completion_days: Optional[int] = None) -> dict:
    """Establish a new goal in the resilient connection system.
    
    Args:
        description: Clear description of the goal to be achieved
        priority: Priority level from 0.0 (low) to 1.0 (high)
        target_completion_days: Optional target completion in days from now
        
    Returns:
        dict: Goal establishment result with constitutional validation
    """
    try:
        logger.info(f"Establishing system goal: {description}")
        
        from datetime import datetime, timedelta
        target_completion = None
        if target_completion_days is not None:
            target_completion = datetime.now() + timedelta(days=target_completion_days)
        
        goal_id = resilient_connection.add_goal(
            description=description,
            priority=priority,
            target_completion=target_completion
        )
        
        # Get the created goal for response
        goal = resilient_connection.active_goals[goal_id]
        
        return {
            "goal_establishment": {
                "goal_id": goal_id,
                "description": goal.description,
                "priority": goal.priority,
                "constitutional_alignment": goal.constitutional_alignment,
                "target_completion": goal.target_completion.isoformat() if goal.target_completion else None,
                "created_at": goal.created_at.isoformat()
            },
            "system_impact": {
                "total_active_goals": len(resilient_connection.active_goals),
                "connection_strength": resilient_connection.evaluate_resilient_connection_strength(),
                "current_exploration_mode": resilient_connection.current_mode.value
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error establishing system goal: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def conduct_novelty_search(context: Dict[str, Any], current_solutions: List[str] = None,
                          target_novelty: float = 0.7) -> dict:
    """Conduct a novelty search to discover innovative solutions.
    
    Args:
        context: Context for the novelty search
        current_solutions: List of known solutions to differentiate from
        target_novelty: Target novelty score (0.0 to 1.0)
        
    Returns:
        dict: Novel solutions discovered with novelty scores
    """
    try:
        logger.info("Conducting novelty search")
        
        if current_solutions is None:
            current_solutions = []
        
        # Create or get novelty search agent
        novelty_agents = [agent for agent in agent_registry.agents.values() 
                         if hasattr(agent, 'capabilities') and 'novelty_search' in agent.capabilities]
        
        if not novelty_agents:
            # Create a novelty search agent
            novelty_agent = NoveltySearchAgent()
            agent_registry.register_agent(novelty_agent)
            logger.info(f"Created novelty search agent: {novelty_agent.agent_id}")
        else:
            novelty_agent = novelty_agents[0]
        
        # Submit novelty search request
        search_request = {
            "type": "novelty_search",
            "context": context,
            "current_solutions": current_solutions,
            "target_novelty": target_novelty
        }
        
        search_results = novelty_agent.process_request(search_request)
        
        # Record discoveries as emergent possibilities
        discoveries = []
        for solution in search_results.get("novel_solutions", []):
            possibility_id = resilient_connection.discover_possibility(
                discovery_type=DiscoveryType.NOVEL_APPROACH,
                description=solution["solution"],
                discovered_by=novelty_agent.agent_id,
                potential_value=solution["novelty_score"],
                confidence=solution["constitutional_compliance"],
                exploration_context=context
            )
            discoveries.append(possibility_id)
        
        return {
            "novelty_search": {
                "agent_id": novelty_agent.agent_id,
                "search_context": context,
                "target_novelty": target_novelty,
                "solutions_found": len(search_results.get("novel_solutions", [])),
                "novel_solutions": search_results.get("novel_solutions", []),
                "discoveries_registered": discoveries
            },
            "system_impact": {
                "total_possibilities": len(resilient_connection.emergent_possibilities),
                "exploration_effectiveness": resilient_connection._calculate_discovery_rate()
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error conducting novelty search: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def evaluate_resilient_connection() -> dict:
    """Evaluate the current state of the resilient connection system.
    
    Returns:
        dict: Comprehensive evaluation of connection strength and balance
    """
    try:
        logger.info("Evaluating resilient connection")
        
        # Get system status
        system_status = resilient_connection.get_system_status()
        
        # Perform balance adjustment
        current_mode = resilient_connection.adjust_exploration_balance({
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_type": "manual_assessment"
        })
        
        # Get balance recommendations
        recommendations = resilient_connection.generate_exploration_recommendations()
        
        return {
            "resilient_connection_evaluation": {
                "connection_strength": system_status["resilient_connection"]["connection_strength"],
                "current_mode": current_mode.value,
                "balance_adjusted": True,
                "system_health": _assess_resilient_connection_health(system_status)
            },
            "detailed_status": system_status,
            "recommendations": recommendations,
            "optimization_suggestions": _generate_optimization_suggestions(system_status),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error evaluating resilient connection: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def integrate_emergent_possibility(possibility_id: str, 
                                 integration_strategy: str = "enhancement") -> dict:
    """Integrate an emergent possibility with existing goals.
    
    Args:
        possibility_id: ID of the possibility to integrate
        integration_strategy: Strategy for integration (enhancement, new_goal)
        
    Returns:
        dict: Integration result with constitutional validation
    """
    try:
        logger.info(f"Integrating emergent possibility: {possibility_id}")
        
        integration_result = resilient_connection.integrate_emergent_possibility(
            possibility_id=possibility_id,
            integration_strategy=integration_strategy
        )
        
        if "error" in integration_result:
            return {
                "integration_error": integration_result["error"],
                "status": "failed"
            }
        
        # Get updated system status
        system_status = resilient_connection.get_system_status()
        
        return {
            "possibility_integration": {
                "possibility_id": possibility_id,
                "integration_strategy": integration_strategy,
                "integration_successful": integration_result["integration_successful"],
                "integration_results": integration_result["integration_results"],
                "constitutional_compliance": integration_result["constitutional_compliance"]
            },
            "system_impact": {
                "total_goals": system_status["goals"]["total_active"],
                "integration_success_rate": system_status["emergent_possibilities"]["highly_feasible"] / max(len(resilient_connection.emergent_possibilities), 1),
                "connection_strength": system_status["resilient_connection"]["connection_strength"]
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error integrating emergent possibility: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def discover_emergent_opportunities(exploration_context: Dict[str, Any]) -> dict:
    """Discover emergent opportunities in the given context.
    
    Args:
        exploration_context: Context for opportunity discovery
        
    Returns:
        dict: Discovered opportunities with potential value assessments
    """
    try:
        logger.info("Discovering emergent opportunities")
        
        # Get or create novelty search agent
        novelty_agents = [agent for agent in agent_registry.agents.values() 
                         if hasattr(agent, 'capabilities') and 'emergent_opportunity_detection' in agent.capabilities]
        
        if not novelty_agents:
            novelty_agent = NoveltySearchAgent()
            agent_registry.register_agent(novelty_agent)
        else:
            novelty_agent = novelty_agents[0]
        
        # Request opportunity discovery
        discovery_request = {
            "type": "discover_opportunities",
            "exploration_context": exploration_context
        }
        
        discovery_results = novelty_agent.process_request(discovery_request)
        
        # Register discoveries
        registered_opportunities = []
        for opportunity in discovery_results.get("emergent_opportunities", []):
            # Convert type string back to enum
            discovery_type = DiscoveryType.EMERGENT_OPPORTUNITY
            for dt in DiscoveryType:
                if dt.value == opportunity["type"]:
                    discovery_type = dt
                    break
            
            possibility_id = resilient_connection.discover_possibility(
                discovery_type=discovery_type,
                description=opportunity["description"],
                discovered_by=novelty_agent.agent_id,
                potential_value=opportunity["potential_value"],
                confidence=opportunity["confidence"],
                exploration_context=exploration_context
            )
            
            registered_opportunities.append({
                "possibility_id": possibility_id,
                "opportunity": opportunity
            })
        
        return {
            "opportunity_discovery": {
                "agent_id": novelty_agent.agent_id,
                "exploration_context": exploration_context,
                "opportunities_found": len(registered_opportunities),
                "registered_opportunities": registered_opportunities
            },
            "discovery_analysis": {
                "high_value_count": sum(1 for opp in registered_opportunities 
                                      if opp["opportunity"]["potential_value"] > 0.7),
                "discovery_types": list(set(opp["opportunity"]["type"] for opp in registered_opportunities)),
                "avg_potential_value": sum(opp["opportunity"]["potential_value"] for opp in registered_opportunities) / len(registered_opportunities) if registered_opportunities else 0.0
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error discovering emergent opportunities: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def adjust_exploration_balance(context: Optional[Dict[str, Any]] = None) -> dict:
    """Manually adjust the exploration-exploitation balance.
    
    Args:
        context: Optional context for balance adjustment
        
    Returns:
        dict: New balance configuration and system adjustments
    """
    try:
        logger.info("Manually adjusting exploration balance")
        
        if context is None:
            context = {"manual_adjustment": True, "timestamp": datetime.now().isoformat()}
        
        # Perform balance adjustment
        new_mode = resilient_connection.adjust_exploration_balance(context)
        
        # Get current metrics
        system_status = resilient_connection.get_system_status()
        
        # Get latest balance metrics
        latest_balance = resilient_connection.balance_history[-1] if resilient_connection.balance_history else None
        
        return {
            "balance_adjustment": {
                "new_mode": new_mode.value,
                "previous_mode": resilient_connection.current_mode.value if hasattr(resilient_connection, 'previous_mode') else "unknown",
                "adjustment_context": context,
                "balance_metrics": {
                    "exploitation_ratio": latest_balance.exploitation_ratio if latest_balance else 0.0,
                    "exploration_ratio": latest_balance.exploration_ratio if latest_balance else 0.0,
                    "overall_effectiveness": latest_balance.overall_effectiveness if latest_balance else 0.0
                } if latest_balance else None
            },
            "system_status": system_status,
            "optimization_recommendations": _generate_balance_recommendations(system_status, new_mode),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error adjusting exploration balance: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def get_goal_progress(goal_id: Optional[str] = None) -> dict:
    """Get progress information for goals in the system.
    
    Args:
        goal_id: Optional specific goal ID, or None for all goals
        
    Returns:
        dict: Goal progress information and recommendations
    """
    try:
        logger.info(f"Getting goal progress for: {goal_id or 'all goals'}")
        
        if goal_id:
            if goal_id not in resilient_connection.active_goals:
                return {
                    "error": f"Goal {goal_id} not found",
                    "status": "not_found"
                }
            
            goal = resilient_connection.active_goals[goal_id]
            goal_info = {
                goal_id: {
                    "description": goal.description,
                    "priority": goal.priority,
                    "progress": goal.progress,
                    "constitutional_alignment": goal.constitutional_alignment,
                    "created_at": goal.created_at.isoformat(),
                    "target_completion": goal.target_completion.isoformat() if goal.target_completion else None,
                    "achieved": goal.achieved,
                    "metrics": goal.metrics,
                    "sub_goals": goal.sub_goals
                }
            }
        else:
            goal_info = {}
            for gid, goal in resilient_connection.active_goals.items():
                goal_info[gid] = {
                    "description": goal.description,
                    "priority": goal.priority,
                    "progress": goal.progress,
                    "constitutional_alignment": goal.constitutional_alignment,
                    "created_at": goal.created_at.isoformat(),
                    "target_completion": goal.target_completion.isoformat() if goal.target_completion else None,
                    "achieved": goal.achieved,
                    "metrics": goal.metrics,
                    "sub_goals": goal.sub_goals
                }
        
        # Analyze goal patterns
        all_goals = list(resilient_connection.active_goals.values())
        goal_analysis = {
            "total_goals": len(all_goals),
            "achieved_goals": sum(1 for g in all_goals if g.achieved),
            "high_priority_goals": sum(1 for g in all_goals if g.priority > 0.7),
            "low_progress_goals": sum(1 for g in all_goals if g.progress < 0.3),
            "avg_constitutional_alignment": sum(g.constitutional_alignment for g in all_goals) / len(all_goals) if all_goals else 0.0,
            "avg_progress": sum(g.progress for g in all_goals) / len(all_goals) if all_goals else 0.0
        }
        
        return {
            "goal_progress": goal_info,
            "goal_analysis": goal_analysis,
            "recommendations": _generate_goal_recommendations(goal_analysis),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting goal progress: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


def _assess_resilient_connection_health(system_status: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the health of the resilient connection system."""
    connection_strength = system_status["resilient_connection"]["connection_strength"]
    goals = system_status["goals"]
    possibilities = system_status["emergent_possibilities"]
    
    # Health indicators
    goal_health = goals["constitutional_alignment_avg"]
    exploration_health = min(possibilities["total_discovered"] / 5.0, 1.0)  # Normalize to 5 discoveries
    integration_health = possibilities["highly_feasible"] / max(possibilities["total_discovered"], 1)
    
    overall_health = (connection_strength + goal_health + exploration_health + integration_health) / 4.0
    
    health_status = "excellent"
    if overall_health < 0.8:
        health_status = "good"
    if overall_health < 0.6:
        health_status = "fair"  
    if overall_health < 0.4:
        health_status = "poor"
    
    return {
        "overall_health": overall_health,
        "health_status": health_status,
        "connection_strength": connection_strength,
        "goal_health": goal_health,
        "exploration_health": exploration_health,
        "integration_health": integration_health
    }


def _generate_optimization_suggestions(system_status: Dict[str, Any]) -> List[str]:
    """Generate optimization suggestions for the resilient connection system."""
    suggestions = []
    
    connection_strength = system_status["resilient_connection"]["connection_strength"]
    goals = system_status["goals"]
    possibilities = system_status["emergent_possibilities"]
    
    if connection_strength < 0.6:
        suggestions.append("Strengthen resilient connection through better goal-exploration alignment")
    
    if goals["constitutional_alignment_avg"] < 0.7:
        suggestions.append("Review and improve constitutional alignment of active goals")
    
    if possibilities["total_discovered"] < 3:
        suggestions.append("Increase exploration activities to discover more emergent possibilities")
    
    if possibilities["highly_feasible"] == 0:
        suggestions.append("Focus on improving the quality and feasibility of discovered possibilities")
    
    if len(system_status["recommendations"]) > 2:
        suggestions.append("Address high-priority recommendations to improve system performance")
    
    return suggestions


def _generate_balance_recommendations(system_status: Dict[str, Any], current_mode: ExplorationMode) -> List[str]:
    """Generate recommendations for exploration balance optimization."""
    recommendations = []
    
    if current_mode == ExplorationMode.EXPLOITATION:
        recommendations.append("Focus on advancing existing goals while maintaining some exploration")
        recommendations.append("Consider constitutional review of goal alignment")
    
    elif current_mode == ExplorationMode.EXPLORATION:
        recommendations.append("Prioritize novelty search and opportunity discovery")
        recommendations.append("Look for integration opportunities with existing goals")
    
    elif current_mode == ExplorationMode.BALANCED:
        recommendations.append("Maintain current balance between goal pursuit and exploration")
        recommendations.append("Monitor for emerging patterns that might require balance adjustment")
    
    connection_strength = system_status["resilient_connection"]["connection_strength"]
    if connection_strength < 0.5:
        recommendations.append("Focus on strengthening the resilient connection before major adjustments")
    
    return recommendations


def _generate_goal_recommendations(goal_analysis: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on goal analysis."""
    recommendations = []
    
    if goal_analysis["low_progress_goals"] > goal_analysis["total_goals"] * 0.5:
        recommendations.append("Review low-progress goals for potential restructuring or exploration enhancement")
    
    if goal_analysis["avg_constitutional_alignment"] < 0.7:
        recommendations.append("Improve constitutional alignment of goals through constitutional review")
    
    if goal_analysis["total_goals"] == 0:
        recommendations.append("Establish clear goals to provide direction for the resilient connection system")
    
    if goal_analysis["achieved_goals"] / max(goal_analysis["total_goals"], 1) > 0.8:
        recommendations.append("Consider establishing new ambitious goals to maintain momentum")
    
    return recommendations


def _generate_constitutional_recommendations(validation_result: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on constitutional validation results."""
    recommendations = []
    
    if not validation_result["overall_valid"]:
        violated = validation_result["violated_principles"]
        
        if "acknowledge_uncertainty_rather_than_invent_facts" in violated:
            recommendations.append("Consider acknowledging uncertainty where facts are not clearly established")
        
        if "prioritize_creating_desired_outcomes_over_eliminating_problems" in violated:
            recommendations.append("Reframe from problem-solving language to outcome-creation language")
        
        if "establish_clear_tension_between_current_reality_and_desired_outcome" in violated:
            recommendations.append("Clarify both current reality and desired outcome to establish structural tension")
        
        if "begin_inquiry_without_preconceptions_or_hypotheses" in violated:
            recommendations.append("Start from direct observation rather than assumptions or preconceptions")
    
    compliance_score = validation_result["constitutional_compliance_score"]
    if compliance_score < 0.7:
        recommendations.append("Consider reviewing constitutional principles to improve overall compliance")
    elif compliance_score > 0.9:
        recommendations.append("Excellent constitutional compliance - continue with this approach")
    
    return recommendations


def _enhanced_lattice_error_response(function_name: str) -> dict:
    """Generate a comprehensive error response when enhanced lattice is not available."""
    logger.error(f"Enhanced lattice not available during {function_name} call")
    logger.error("This typically means MCP dependencies are missing or there was an initialization error")
    logger.error("Please ensure 'mcp[cli]>=1.2.0' is installed: pip install -r requirements.txt")
    return {
        "error": "Enhanced lattice not available - MCP dependencies may be missing",
        "troubleshooting": {
            "install_dependencies": "Run: pip install -r requirements.txt",
            "install_package": "Or run: pip install -e .",
            "check_server_logs": "Look for ImportError or initialization errors in server startup logs",
            "verify_installation": "Check with: pip list | grep mcp",
            "restart_server": "Restart the MCP server after installing dependencies"
        },
        "function_attempted": function_name,
        "status": "failed"
    }


# Initialize creative orientation systems for generative tools
try:
    logger.info("Initializing creative orientation foundation...")
    
    # Validate constitutional foundation is working
    test_validation = creative_orientation_foundation.validate_creative_orientation(
        request="Testing system initialization", 
        desired_outcome="Create functional generative lattice"
    )
    
    logger.info(f"Creative orientation foundation initialized: {test_validation.constitutional_coherence}")
    
    # Initialize generative lattice with Mia & Miette archetypes
    logger.info("Initializing generative agent lattice with Mia & Miette archetypes...")
    
    # Test generative lattice functionality
    test_tension = establish_structural_tension(
        desired_outcome="Verify generative lattice operational status",
        current_reality="System initializing with creative orientation foundation"
    )
    
    test_session = generative_lattice.initiate_emergence(
        structural_tension=test_tension,
        primary_purpose="system_validation",
        archetype_activation={"mia": "structural_analysis", "miette": "narrative_discovery"}
    )
    
    logger.info(f"Generative lattice initialized successfully: {test_session.session_id}")
    generative_lattice_available = True
    
except ImportError as e:
    logger.error(f"ImportError while importing creative orientation systems: {e}")
    logger.error("Available modules in package:")
    import os
    package_dir = os.path.dirname(__file__)
    if os.path.exists(package_dir):
        modules = [f for f in os.listdir(package_dir) if f.endswith('.py')]
        logger.error(f"Available modules: {modules}")
    generative_lattice_available = False
except Exception as e:
    logger.error(f"Unexpected error initializing creative orientation systems: {e}")
    logger.error(f"Error type: {type(e).__name__}")
    import traceback
    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    generative_lattice_available = False


@mcp.tool()
def initiate_creative_emergence(request: str, desired_outcome: str, 
                               primary_purpose: str = "generative",
                               archetype_focus: Optional[str] = None,
                               cultural_perspective: Optional[str] = None) -> dict:
    """Initiate creative emergence through polycentric agent collaboration.
    
    🧠 Mia: Establishes structural tension and architectural foundation for manifestation
    🌸 Miette: Illuminates narrative potential and creative possibilities
    
    This tool embodies creative orientation principles from the theoretical framework,
    establishing structural tension between current reality and desired outcome.
    
    Args:
        request: Current reality assessment (what is, objectively)
        desired_outcome: What you want to create/manifest (not solve)
        primary_purpose: Creative intention for the emergence process
        archetype_focus: Optional focus on specific archetype ("mia", "miette", or "both")
        cultural_perspective: Optional cultural lens ("western_analytical", "indigenous_holistic", "both_eyes_seeing")
        
    Returns:
        dict: Creative emergence session with advancing pattern indicators
    """
    try:
        if not generative_lattice_available:
            return {
                "reframe_guidance": [
                    "🧠 Mia: Creative orientation systems are initializing",
                    "🌸 Miette: The magical lattice is awakening! Please try again in a moment ✨"
                ],
                "constitutional_principles": [
                    "Focus on what you want to CREATE rather than what you want to solve",
                    "Establish clear structural tension between current reality and desired outcome",
                    "Apply delayed resolution principle - avoid premature closure"
                ],
                "status": "lattice_initializing"
            }
            
        logger.info(f"Initiating creative emergence: {desired_outcome}")
        
        # Validate creative orientation
        validation = creative_orientation_foundation.validate_creative_orientation(
            request=request,
            desired_outcome=desired_outcome
        )
        
        if not validation.is_generative:
            return creative_orientation_foundation.guide_toward_creative_reframe(validation)
        
        # Establish structural tension
        structural_tension = establish_structural_tension(
            desired_outcome=desired_outcome,
            current_reality=request,
            context=primary_purpose
        )
        
        # Determine archetype activation
        archetype_activation = {}
        if not archetype_focus or archetype_focus == "both":
            archetype_activation = {
                "mia": "structural_analysis_and_design",
                "miette": "narrative_warmth_and_discovery"
            }
        elif archetype_focus == "mia":
            archetype_activation = {"mia": "structural_analysis_and_design"}
        elif archetype_focus == "miette":
            archetype_activation = {"miette": "narrative_warmth_and_discovery"}
        
        # Initiate emergence session
        emergence_session = generative_lattice.initiate_emergence(
            structural_tension=structural_tension,
            primary_purpose=primary_purpose,
            archetype_activation=archetype_activation
        )
        
        # Assess creative tension
        tension_assessment = generative_lattice.assess_creative_tension(emergence_session.session_id)
        
        # Generate advancement possibilities
        next_progressions = generative_lattice.generate_possible_progressions(emergence_session.session_id)
        
        return {
            "emergence_session_id": emergence_session.session_id,
            "structural_tension": {
                "desired_outcome": desired_outcome,
                "current_reality": request,
                "tension_strength": tension_assessment["tension_strength"],
                "creative_alignment": tension_assessment["creative_alignment"],
                "constitutional_coherence": tension_assessment["constitutional_coherence"]
            },
            "archetype_perspectives": {
                entry["agent"]: {
                    "content": entry["content"],
                    "insights": entry.get("insights", [])
                }
                for entry in emergence_session.emergence_timeline
                if entry["type"] in ["structural_analysis", "narrative_discovery"]
            },
            "creative_phase": emergence_session.current_phase,
            "advancement_possibilities": next_progressions,
            "constitutional_validation": {
                "is_generative": validation.is_generative,
                "orientation_type": validation.orientation_type.value,
                "guidance": validation.guidance_for_advancement
            },
            "status": "creative_tension_established"
        }
        
    except Exception as e:
        logger.error(f"Error initiating creative emergence: {str(e)}")
        return {
            "error": str(e),
            "guidance": [
                "🧠 Mia: Consider reframing toward creative manifestation rather than problem-solving",
                "🌸 Miette: What do you want to CREATE instead? Let's discover the magic together! ✨"
            ],
            "status": "failed"
        }


@mcp.tool()
def advance_creative_emergence(session_id: str, new_insight: str, 
                              archetype_focus: Optional[str] = None,
                              perspective_shift: Optional[str] = None) -> dict:
    """Advance the creative emergence process with new insights.
    
    🧠 Mia: Integrates structural insights into advancing architectural patterns
    🌸 Miette: Weaves narrative threads into the emerging creative story
    
    Args:
        session_id: ID of the creative emergence session to advance
        new_insight: New insight, discovery, or creative input to integrate
        archetype_focus: Optional focus on specific archetype ("mia", "miette", or "both")
        perspective_shift: Optional cultural perspective shift ("western_analytical", "indigenous_holistic", "both_eyes_seeing")
        
    Returns:
        dict: Advancement response with integrated perspectives and next possibilities
    """
    try:
        if not generative_lattice_available:
            return {
                "guidance": [
                    "🧠 Mia: Creative orientation systems are initializing",
                    "🌸 Miette: The advancement magic is awakening! Please try again soon ✨"
                ],
                "status": "lattice_initializing"
            }
            
        logger.info(f"Advancing creative emergence session: {session_id}")
        
        # Validate creative orientation of the new insight
        validation = creative_orientation_foundation.validate_creative_orientation(
            request="",
            desired_outcome=new_insight
        )
        
        if not validation.is_generative:
            return {
                "reframe_needed": True,
                "guidance": validation.guidance_for_advancement,
                "creative_reframe_examples": [
                    {
                        "reactive": "Fix the performance issue",
                        "creative": "Create optimal performance experience"
                    },
                    {
                        "reactive": "Solve the integration problem",
                        "creative": "Manifest seamless integration architecture"
                    }
                ],
                "status": "awaiting_creative_reframe"
            }
        
        # Determine archetype focus for advancement
        archetype_focus_enum = None
        if archetype_focus == "mia":
            archetype_focus_enum = ArchetypeRole.MIA
        elif archetype_focus == "miette":
            archetype_focus_enum = ArchetypeRole.MIETTE
        
        # Advance the emergence session
        advancement_result = generative_lattice.advance_emergence(
            session_id=session_id,
            new_insight=new_insight,
            archetype_focus=archetype_focus_enum
        )
        
        if "error" in advancement_result:
            return advancement_result
        
        # Get updated creative tension assessment
        tension_assessment = generative_lattice.assess_creative_tension(session_id)
        
        # Generate next possibilities
        next_progressions = generative_lattice.generate_possible_progressions(session_id)
        
        return {
            "advancement": {
                "session_id": session_id,
                "new_insight_integrated": new_insight,
                "constitutional_alignment": advancement_result.get("constitutional_alignment", 0.0),
                "creative_phase": advancement_result.get("phase", "assimilation"),
                "archetype_responses": advancement_result.get("advancement_responses", {})
            },
            "creative_tension_update": {
                "tension_strength": tension_assessment.get("tension_strength", "establishing"),
                "creative_alignment": tension_assessment.get("creative_alignment", 0.0),
                "advancement_count": tension_assessment.get("advancement_count", 0)
            },
            "next_possibilities": next_progressions,
            "guidance": [
                "🧠 Mia: Structural integration proceeding with creative advancement",
                "🌸 Miette: Beautiful new insights are weaving into the emerging story! ✨"
            ],
            "status": advancement_result.get("status", "advancing")
        }
        
    except Exception as e:
        logger.error(f"Error advancing creative emergence: {str(e)}")
        return {
            "error": str(e),
            "guidance": [
                "🧠 Mia: Consider structural reframing for advancing patterns",
                "🌸 Miette: Let's discover what wants to emerge! What are you creating? ✨"
            ],
            "status": "failed"
        }


@mcp.tool()
def synthesize_thinking_chain(chain_id: str) -> dict:
    """Synthesize all perspectives in a thinking chain into integrated wisdom.
    
    Args:
        chain_id: ID of the thinking chain to synthesize
        
    Returns:
        dict: Synthesized perspective integrating all viewpoints
    """
    try:
        if not enhanced_lattice:
            return _enhanced_lattice_error_response("synthesize_thinking_chain")
            
        logger.info(f"Synthesizing thinking chain: {chain_id}")
        
        synthesis = enhanced_lattice.synthesize_perspectives(chain_id)
        
        if not synthesis:
            return {
                "error": f"Could not synthesize perspectives for chain {chain_id}",
                "status": "failed"
            }
        
        # Get memory integration structure
        memory_structure = enhanced_lattice.prepare_memory_integration(chain_id)
        
        return {
            "synthesis": {
                "perspective_id": synthesis.perspective_id,
                "integrated_viewpoint": synthesis.viewpoint,
                "emotional_resonance": synthesis.emotional_resonance,
                "strategic_insight": synthesis.strategic_insight,
                "cultural_lens": synthesis.cultural_lens,
                "synthesized_concerns": synthesis.concerns,
                "synthesized_opportunities": synthesis.opportunities,
                "confidence_level": synthesis.confidence_level
            },
            "memory_integration": memory_structure,
            "coaia_memory_ready": memory_structure.get("knowledge_graph_ready", False),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error synthesizing thinking chain: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def create_consensus_decision(decision_type: str, primary_purpose: str, proposal: str,
                            current_reality: str, desired_outcome: str,
                            participating_agents: Optional[List[str]] = None,
                            mmor_elements: Optional[List[Dict[str, Any]]] = None) -> dict:
    """Create a multi-agent consensus decision with delayed resolution principle.
    
    This implements the consensus-based decision making system from PR #9 feedback,
    with MMOR integration and delayed resolution principle.
    
    Args:
        decision_type: Type of decision (primary_choice, secondary_choice, design_element, execution_element)
        primary_purpose: The primary purpose driving this decision
        proposal: The proposal to be decided upon
        current_reality: Current state assessment
        desired_outcome: Desired outcome from the decision
        participating_agents: Optional list of agent IDs to include
        mmor_elements: Optional MMOR elements for design/execution categorization
        
    Returns:
        dict: Consensus decision details and participation info
    """
    try:
        if not enhanced_lattice:
            return _enhanced_lattice_error_response("create_consensus_decision")
            
        logger.info(f"Creating consensus decision: {proposal}")
        
        # Convert decision type string to enum
        try:
            decision_type_enum = DecisionType(decision_type.lower())
        except ValueError:
            return {
                "error": f"Invalid decision type: {decision_type}",
                "valid_types": [dt.value for dt in DecisionType],
                "status": "failed"
            }
        
        # Use persona agents if no specific agents provided
        if not participating_agents:
            participating_agents = [
                "persona_mia_rational",
                "persona_miette_catalyst", 
                "persona_haiku_synthesizer"
            ]
        
        # Generate decision ID
        decision_id = f"consensus_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create consensus decision
        decision = enhanced_lattice.consensus_engine.initiate_consensus_decision(
            decision_id=decision_id,
            decision_type=decision_type_enum,
            primary_purpose=primary_purpose,
            proposal=proposal,
            current_reality=current_reality,
            desired_outcome=desired_outcome,
            participating_agents=participating_agents,
            mmor_elements=mmor_elements
        )
        
        return {
            "consensus_decision": {
                "decision_id": decision.decision_id,
                "decision_type": decision.decision_type.value,
                "primary_purpose": decision.primary_purpose,
                "proposal": decision.proposal,
                "consensus_status": decision.consensus_status.value,
                "participating_agents": decision.participating_agents,
                "resolution_delayed": decision.resolution_delayed,
                "delay_reason": decision.delay_reason,
                "human_consultation_available": True
            },
            "delayed_resolution": {
                "tension_level": decision.tension.tension_level if decision.tension else 0.0,
                "resolution_pressure": decision.tension.resolution_pressure if decision.tension else 0.0,
                "delay_justification": decision.tension.delay_justification if decision.tension else ""
            },
            "next_steps": {
                "get_status": f"Call get_consensus_decision_status with decision_id: {decision_id}",
                "request_consultation": f"Call request_human_consultation for clarification if needed"
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error creating consensus decision: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def get_consensus_decision_status(decision_id: str) -> dict:
    """Get the current status of a consensus decision.
    
    Args:
        decision_id: ID of the consensus decision
        
    Returns:
        dict: Current status, votes, and resolution readiness
    """
    try:
        if not enhanced_lattice:
            return _enhanced_lattice_error_response("get_consensus_decision_status")
            
        logger.info(f"Getting consensus decision status: {decision_id}")
        
        decision_status = enhanced_lattice.consensus_engine.get_decision_status(decision_id)
        
        if not decision_status:
            return {
                "error": f"Decision {decision_id} not found",
                "status": "not_found"
            }
        
        # Check resolution readiness
        ready, reason = enhanced_lattice.consensus_engine.check_resolution_readiness(decision_id)
        
        # Get active decision details if still active
        additional_details = {}
        if decision_id in enhanced_lattice.consensus_engine.active_decisions:
            decision = enhanced_lattice.consensus_engine.active_decisions[decision_id]
            additional_details = {
                "current_votes": [
                    {
                        "agent_id": vote.agent_id,
                        "vote": vote.vote,
                        "reasoning": vote.reasoning,
                        "confidence": vote.confidence,
                        "conditions": vote.conditions
                    }
                    for vote in decision.votes
                ],
                "human_consultation_requests": decision.human_clarification_requests,
                "human_response": decision.human_response
            }
        
        return {
            "decision_status": decision_status,
            "resolution_readiness": {
                "ready_for_resolution": ready,
                "readiness_reason": reason
            },
            "additional_details": additional_details,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting consensus decision status: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def request_human_consultation(decision_id: str, clarification_request: str) -> dict:
    """Request human companion consultation for decision clarification.
    
    This implements the human companion loop from PR #9 feedback for complex decisions
    requiring human insight or clarification.
    
    Args:
        decision_id: ID of the decision requiring consultation
        clarification_request: Specific clarification or insight needed
        
    Returns:
        dict: Consultation request details and agent perspectives
    """
    try:
        if not enhanced_lattice:
            return _enhanced_lattice_error_response("request_human_consultation")
            
        logger.info(f"Requesting human consultation for decision: {decision_id}")
        
        consultation_request = enhanced_lattice.consensus_engine.request_human_consultation(
            decision_id=decision_id,
            clarification_request=clarification_request
        )
        
        if "error" in consultation_request:
            return {
                "error": consultation_request["error"],
                "status": "failed"
            }
        
        return {
            "human_consultation": consultation_request,
            "consultation_workflow": {
                "step_1": "Human reviews the decision context and agent perspectives",
                "step_2": "Human provides insights via provide_human_response tool", 
                "step_3": "Agents incorporate human input for final decision"
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error requesting human consultation: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def provide_human_response(decision_id: str, human_response: str) -> dict:
    """Provide human response to a consultation request.
    
    Args:
        decision_id: ID of the decision being consulted on
        human_response: Human insights, clarification, or guidance
        
    Returns:
        dict: Response integration status and updated decision state
    """
    try:
        if not enhanced_lattice:
            return _enhanced_lattice_error_response("provide_human_response")
            
        logger.info(f"Providing human response for decision: {decision_id}")
        
        success = enhanced_lattice.consensus_engine.provide_human_response(
            decision_id=decision_id,
            human_response=human_response
        )
        
        if not success:
            return {
                "error": f"Could not provide human response for decision {decision_id}",
                "status": "failed"
            }
        
        # Get updated decision status
        updated_status = enhanced_lattice.consensus_engine.get_decision_status(decision_id)
        
        return {
            "human_response_integration": {
                "decision_id": decision_id,
                "response_integrated": True,
                "human_response": human_response,
                "updated_status": updated_status["consensus_status"]
            },
            "next_steps": {
                "agents_will": "Incorporate human insights into their analysis",
                "check_status": f"Monitor decision progress with get_consensus_decision_status"
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error providing human response: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def get_thinking_chain_status(chain_id: str) -> dict:
    """Get the current status of a sequential thinking chain.
    
    Args:
        chain_id: ID of the thinking chain
        
    Returns:
        dict: Chain status, perspectives collected, and progress
    """
    try:
        if not enhanced_lattice:
            return _enhanced_lattice_error_response("get_thinking_chain_status")
            
        chain_status = enhanced_lattice.get_thinking_chain_status(chain_id)
        
        if not chain_status:
            return {
                "error": f"Thinking chain {chain_id} not found",
                "status": "not_found"
            }
        
        return {
            "thinking_chain_status": chain_status,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting thinking chain status: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def get_active_thinking_chains() -> dict:
    """Get all active sequential thinking chains.
    
    Returns:
        dict: List of active thinking chains and their status
    """
    try:
        if not enhanced_lattice:
            return _enhanced_lattice_error_response("get_active_thinking_chains")
            
        active_chains = enhanced_lattice.get_active_thinking_chains()
        
        return {
            "active_thinking_chains": active_chains,
            "total_active": len(active_chains),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting active thinking chains: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool() 
def initiate_sequential_thinking(request: str, primary_purpose: str,
                               persona_sequence: Optional[List[str]] = None,
                               memory_context: Optional[Dict[str, Any]] = None) -> dict:
    """Initiate sequential thinking process across multiple personas.
    
    Args:
        request: The request or question to analyze  
        primary_purpose: The core purpose driving this analysis
        persona_sequence: Optional ordered list of personas to engage
        memory_context: Optional memory context for coaia-memory integration
        
    Returns:
        dict: Sequential thinking session details and first perspective
    """
    try:
        if not enhanced_lattice:
            return _enhanced_lattice_error_response("initiate_sequential_thinking")
            
        logger.info(f"Initiating sequential thinking for: {request}")
        
        # Convert string persona sequence to enum if provided
        converted_sequence = None
        if persona_sequence:
            from mcp_coaia_sequential_thinking.enhanced_polycentric_lattice import PersonaArchetype
            converted_sequence = []
            for persona in persona_sequence:
                if hasattr(PersonaArchetype, persona.upper()):
                    converted_sequence.append(PersonaArchetype(persona.lower()))
        
        chain_id = enhanced_lattice.initiate_sequential_thinking(
            request=request,
            primary_purpose=primary_purpose,
            persona_sequence=converted_sequence,
            memory_context=memory_context
        )
        
        # Get the initial thinking status
        chain_status = enhanced_lattice.get_thinking_chain_status(chain_id)
        
        return {
            "sequential_thinking": {
                "chain_id": chain_id,
                "initial_request": request,
                "primary_purpose": primary_purpose,
                "memory_context_available": memory_context is not None,
                "chain_status": chain_status
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error initiating sequential thinking: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def advance_thinking_chain(chain_id: str, focus_persona: Optional[str] = None) -> dict:
    """Advance the sequential thinking chain with the next persona perspective.
    
    Args:
        chain_id: ID of the thinking chain to advance
        focus_persona: Optional specific persona to focus on for this advancement
        
    Returns:
        dict: New perspective and advancement status
    """
    try:
        if not enhanced_lattice:
            return _enhanced_lattice_error_response("advance_thinking_chain")
            
        logger.info(f"Advancing thinking chain: {chain_id}")
        
        # Convert focus persona if provided
        focus_enum = None
        if focus_persona:
            from mcp_coaia_sequential_thinking.enhanced_polycentric_lattice import PersonaArchetype
            if hasattr(PersonaArchetype, focus_persona.upper()):
                focus_enum = PersonaArchetype(focus_persona.lower())
        
        perspective = enhanced_lattice.advance_thinking_chain(
            chain_id=chain_id,
            focus_persona=focus_enum
        )
        
        if not perspective:
            return {
                "error": f"Could not advance thinking chain {chain_id}",
                "status": "failed"
            }
        
        # Get updated chain status
        chain_status = enhanced_lattice.get_thinking_chain_status(chain_id)
        
        return {
            "advancement": {
                "chain_id": chain_id,
                "new_perspective": {
                    "perspective_id": perspective.perspective_id,
                    "persona_archetype": perspective.persona_archetype,
                    "viewpoint": perspective.viewpoint,
                    "concerns": perspective.concerns,
                    "opportunities": perspective.opportunities,
                    "confidence_level": perspective.confidence_level
                },
                "chain_status": chain_status
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error advancing thinking chain: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


@mcp.tool()
def run_full_analysis_chain(request: str, primary_purpose: str, 
                          synthesis_focus: str = "integrated_wisdom",
                          memory_context: Optional[Dict[str, Any]] = None) -> dict:
    """High-level wrapper that runs complete sequential thinking analysis in one call.
    
    This is Mia's requested wrapper tool for simpler use cases, encapsulating the
    initiate -> advance (loop) -> synthesize workflow with internal chain_id management.
    
    Args:
        request: The request or question to analyze
        primary_purpose: The core purpose driving this analysis  
        synthesis_focus: Focus for final synthesis (default: "integrated_wisdom")
        memory_context: Optional memory context for coaia-memory integration
        
    Returns:
        dict: Complete analysis including all perspectives and final synthesis
    """
    try:
        if not enhanced_lattice:
            return _enhanced_lattice_error_response("run_full_analysis_chain")
            
        logger.info(f"Running full analysis chain for: {request}")
        
        # Step 1: Initiate sequential thinking
        from mcp_coaia_sequential_thinking.enhanced_polycentric_lattice import PersonaArchetype
        
        chain_id = enhanced_lattice.initiate_sequential_thinking(
            request=request,
            primary_purpose=primary_purpose,
            memory_context=memory_context
        )
        
        # Step 2: Advance through all personas in sequence
        personas = [
            PersonaArchetype.RATIONAL_ARCHITECT,
            PersonaArchetype.EMOTIONAL_CATALYST, 
            PersonaArchetype.WISDOM_SYNTHESIZER
        ]
        
        collected_perspectives = []
        for persona in personas:
            perspective = enhanced_lattice.advance_thinking_chain(
                chain_id=chain_id,
                focus_persona=persona
            )
            
            if not perspective:
                return {
                    "error": f"Failed during {persona.value} perspective generation",
                    "chain_id": chain_id,
                    "collected_perspectives": collected_perspectives,
                    "status": "failed"
                }
            
            perspective_data = {
                "perspective_id": perspective.perspective_id,
                "persona_archetype": perspective.persona_archetype,
                "viewpoint": perspective.viewpoint,
                "concerns": perspective.concerns,
                "opportunities": perspective.opportunities,
                "confidence_level": perspective.confidence_level
            }
            collected_perspectives.append(perspective_data)
        
        # Step 3: Synthesize all perspectives
        synthesis_result = synthesize_thinking_chain(chain_id=chain_id)
        
        if synthesis_result.get("status") != "success":
            return {
                "error": f"Failed during synthesis: {synthesis_result.get('error')}",
                "chain_id": chain_id,
                "collected_perspectives": collected_perspectives,
                "status": "failed"
            }
        
        # Return comprehensive results
        return {
            "complete_analysis": {
                "chain_id": chain_id,
                "original_request": request,
                "primary_purpose": primary_purpose,
                "analysis_summary": {
                    "mia_confidence": next((p["confidence_level"] for p in collected_perspectives 
                                          if p["persona_archetype"] == "rational_architect"), 0.0),
                    "miette_confidence": next((p["confidence_level"] for p in collected_perspectives 
                                            if p["persona_archetype"] == "emotional_catalyst"), 0.0),
                    "haiku_confidence": next((p["confidence_level"] for p in collected_perspectives 
                                           if p["persona_archetype"] == "wisdom_synthesizer"), 0.0),
                    "final_confidence": synthesis_result["synthesis"]["confidence_level"]
                }
            },
            "personas_perspectives": collected_perspectives,
            "final_synthesis": synthesis_result["synthesis"]["integrated_viewpoint"],
            "confidence": synthesis_result["synthesis"]["confidence_level"],
            "strategic_insights": synthesis_result["synthesis"]["strategic_insight"],
            "synthesized_opportunities": synthesis_result["synthesis"]["synthesized_opportunities"],
            "memory_integration": synthesis_result["memory_integration"],
            "coaia_memory_ready": synthesis_result.get("coaia_memory_ready", False),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error running full analysis chain: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


def _get_principle_description(principle: ConstitutionalPrinciple) -> str:
    """Get human-readable description for a constitutional principle."""
    descriptions = {
        ConstitutionalPrinciple.NON_FABRICATION: "Acknowledge uncertainty rather than inventing facts when knowledge is insufficient",
        ConstitutionalPrinciple.ERROR_AS_COMPASS: "Treat failures and errors as navigational cues for improvement rather than problems to hide",
        ConstitutionalPrinciple.CREATIVE_PRIORITY: "Prioritize creating desired outcomes over eliminating unwanted conditions",
        ConstitutionalPrinciple.STRUCTURAL_AWARENESS: "Recognize that underlying structure determines behavior patterns",
        ConstitutionalPrinciple.TENSION_ESTABLISHMENT: "Establish clear structural tension between current reality and desired outcomes",
        ConstitutionalPrinciple.START_WITH_NOTHING: "Begin inquiry without preconceptions, hypotheses, or imported assumptions",
        ConstitutionalPrinciple.PICTURE_WHAT_IS_SAID: "Translate verbal information into visual representations for dimensional thinking",
        ConstitutionalPrinciple.QUESTION_INTERNALLY: "Ask questions driven by provided information rather than external sources",
        ConstitutionalPrinciple.MULTIPLE_PERSPECTIVES: "Generate and consider multiple viewpoints before making decisions",
        ConstitutionalPrinciple.PRINCIPLE_OVER_EXPEDIENCE: "Constitutional principles override operational convenience",
        ConstitutionalPrinciple.TRANSPARENCY_REQUIREMENT: "All decisions must be traceable to constitutional principles",
        ConstitutionalPrinciple.ADAPTIVE_PROTOCOLS: "Operational methods can change but core principles remain immutable",
        ConstitutionalPrinciple.CONFLICT_RESOLUTION: "Resolve conflicts through principle hierarchy rather than compromise"
    }
    return descriptions.get(principle, "Description not available")


def _get_principle_category(principle: ConstitutionalPrinciple) -> str:
    """Get the category for a constitutional principle."""
    categories = {
        ConstitutionalPrinciple.NON_FABRICATION: "Core Creative Orientation",
        ConstitutionalPrinciple.ERROR_AS_COMPASS: "Core Creative Orientation",
        ConstitutionalPrinciple.CREATIVE_PRIORITY: "Core Creative Orientation",
        ConstitutionalPrinciple.STRUCTURAL_AWARENESS: "Core Creative Orientation",
        ConstitutionalPrinciple.TENSION_ESTABLISHMENT: "Core Creative Orientation",
        ConstitutionalPrinciple.START_WITH_NOTHING: "Structural Thinking",
        ConstitutionalPrinciple.PICTURE_WHAT_IS_SAID: "Structural Thinking",
        ConstitutionalPrinciple.QUESTION_INTERNALLY: "Structural Thinking",
        ConstitutionalPrinciple.MULTIPLE_PERSPECTIVES: "Structural Thinking",
        ConstitutionalPrinciple.PRINCIPLE_OVER_EXPEDIENCE: "Meta-Decision Making",
        ConstitutionalPrinciple.TRANSPARENCY_REQUIREMENT: "Meta-Decision Making",
        ConstitutionalPrinciple.ADAPTIVE_PROTOCOLS: "Meta-Decision Making",
        ConstitutionalPrinciple.CONFLICT_RESOLUTION: "Meta-Decision Making"
    }
    return categories.get(principle, "Uncategorized")


@mcp.tool()
def check_agent_creative_orientation(
    recent_content: Optional[str] = None,
    tool_intention: Optional[str] = None
) -> dict:
    """Allow agents to check their creative vs reactive orientation before using other MCP tools.
    
    This enables agents to self-assess their orientation and receive guidance on appropriate
    tool usage, supporting the dotCoAiA framework for Creative-Orientation-Agentic-Intelligence.
    
    Args:
        recent_content: Optional recent content/thoughts to validate orientation
        tool_intention: What MCP tool the agent intends to use next
        
    Returns:
        dict: Orientation status, tool readiness, and usage guidance
    """
    try:
        # Import the enhanced pattern analysis
        from mcp_coaia_sequential_thinking.co_lint_integration import (
            get_user_creative_patterns, validate_thought
        )
        
        # Get overall pattern analysis
        patterns = get_user_creative_patterns(limit=50)
        
        result = {
            "orientation_check": {
                "timestamp": datetime.now().isoformat(),
                "agent_status": patterns.get("agent_orientation_awareness", {}),
                "overall_score": patterns.get("average_creative_orientation_score", 0),
                "trend": patterns.get("orientation_trend", "unknown")
            },
            "mcp_tool_readiness": patterns.get("agent_orientation_awareness", {}).get("mcp_interaction_recommendations", {}),
            "guidance": patterns.get("agent_orientation_awareness", {}).get("tool_usage_guidance", [])
        }
        
        # If recent content provided, validate it specifically
        if recent_content:
            validation = validate_thought(recent_content)
            content_score = validation.creative_orientation_score
            
            result["recent_content_analysis"] = {
                "creative_orientation_score": content_score,
                "advancing_pattern_detected": validation.advancing_pattern_detected,
                "structural_tension_established": validation.structural_tension_established,
                "content_readiness": "ready" if content_score >= 0.6 else "needs_improvement" if content_score >= 0.3 else "requires_reframing"
            }
        
        # If tool intention specified, provide specific guidance
        if tool_intention:
            tool_guidance = _get_tool_specific_guidance(
                tool_intention, 
                result["orientation_check"]["overall_score"],
                patterns.get("most_common_reactive_patterns", [])
            )
            result["tool_specific_guidance"] = tool_guidance
        
        # CoAiA-memory integration data
        result["coaia_memory_integration"] = patterns.get("coaia_memory_integration", {})
        
        # Self-awareness recommendations
        result["self_awareness_recommendations"] = _generate_self_awareness_recommendations(
            result["orientation_check"]["overall_score"],
            patterns.get("most_common_reactive_patterns", [])
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in orientation check: {e}")
        return {
            "error": f"Orientation check failed: {str(e)}",
            "fallback_guidance": [
                "Establish clear desired outcome before proceeding",
                "Avoid problem-solving language",
                "Focus on creating rather than fixing"
            ]
        }


def _get_tool_specific_guidance(tool_name: str, orientation_score: float, reactive_patterns: List) -> Dict[str, Any]:
    """Generate specific guidance for intended MCP tool usage."""
    
    tool_guidance = {
        "tool_name": tool_name,
        "readiness_level": "high" if orientation_score >= 0.7 else "moderate" if orientation_score >= 0.5 else "low",
        "proceed": orientation_score >= 0.4,
        "warnings": [],
        "preparation_steps": []
    }
    
    # Tool-specific guidance
    if tool_name == "initiate_sequential_thinking":
        if orientation_score < 0.5:
            tool_guidance["warnings"].append("Risk of reactive pattern in chain initiation")
            tool_guidance["preparation_steps"].append("Clarify desired outcome before initiating chain")
        
        if any("problem" in str(pattern[0]).lower() for pattern in reactive_patterns[:3]):
            tool_guidance["warnings"].append("Problem-focused language detected in recent patterns")
            tool_guidance["preparation_steps"].append("Reframe request as outcome creation")
    
    elif tool_name == "advance_thinking_chain":
        if orientation_score < 0.6:
            tool_guidance["warnings"].append("May disrupt creative flow between personas")
            tool_guidance["preparation_steps"].append("Establish structural tension before advancing")
    
    elif tool_name == "create_consensus_decision":
        if orientation_score < 0.7:
            tool_guidance["warnings"].append("Risk of problem-solving consensus rather than creative consensus")
            tool_guidance["preparation_steps"].append("Verify advancing pattern focus in decision context")
    
    elif tool_name == "run_full_analysis_chain":
        if orientation_score < 0.6:
            tool_guidance["warnings"].append("Complex analysis may amplify reactive patterns")
            tool_guidance["preparation_steps"].append("Complete orientation training before full analysis")
            tool_guidance["proceed"] = False
    
    return tool_guidance


def _generate_self_awareness_recommendations(orientation_score: float, reactive_patterns: List) -> List[str]:
    """Generate self-awareness recommendations for agent development."""
    recommendations = []
    
    if orientation_score < 0.4:
        recommendations.extend([
            "CRITICAL: Agent demonstrates strong reactive orientation",
            "PRACTICE: Daily desired outcome establishment exercises", 
            "AVOID: Using complex MCP tools until orientation improves",
            "FOCUS: Transform all 'problems' into 'desired outcomes'"
        ])
    elif orientation_score < 0.6:
        recommendations.extend([
            "DEVELOPING: Agent shows mixed orientation patterns",
            "STRENGTHEN: Structural tension awareness in all interactions",
            "MONITOR: Watch for reactive language patterns",
            "ADVANCE: Practice advancing language consistently"
        ])
    elif orientation_score < 0.8:
        recommendations.extend([
            "GOOD: Agent demonstrates solid creative orientation",
            "REFINE: Optimize consistency across all tool interactions",
            "LEADERSHIP: Ready to guide other agents in orientation",
            "EXCEL: Focus on mastery of complex structural tensions"
        ])
    else:
        recommendations.extend([
            "MASTERY: Agent exhibits excellent creative orientation",
            "MENTOR: Capable of training other agents in creative practices",
            "INNOVATE: Ready for advanced structural dynamics work",
            "LEAD: Can pioneer new creative orientation applications"
        ])
    
    # Pattern-specific recommendations
    if reactive_patterns:
        top_reactive = [pattern[0] for pattern in reactive_patterns[:2]]
        if any('solve' in pattern.lower() for pattern in top_reactive):
            recommendations.append("TRANSFORM: Replace 'solve' language with 'create' language")
        if any('fix' in pattern.lower() for pattern in top_reactive):
            recommendations.append("REFRAME: Change 'fix' mindset to 'build' mindset")
    
    return recommendations


def main():
    """Entry point for the MCP server."""
    logger.info("Starting CoAiA Sequential Thinking MCP server")
    
    # Register prompts
    logger.info("Registering structural thinking prompts")
    for prompt_key, prompt_data in PROMPTS.items():
        try:
            @mcp.prompt(name=prompt_key)
            def get_prompt_handler(prompt_key=prompt_key):
                """Dynamic prompt handler"""
                prompt = PROMPTS[prompt_key]
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": prompt["template"]
                            }
                        }
                    ],
                    "description": prompt["description"]
                }
            logger.info(f"Registered prompt: {prompt_key}")
        except Exception as e:
            logger.error(f"Failed to register prompt {prompt_key}: {e}")
    
    # Register resources
    logger.info("Registering structural thinking resources")
    for resource_key, resource_data in RESOURCES.items():
        try:
            @mcp.resource(resource_data["uri"])
            def get_resource_handler(resource_data=resource_data):
                """Dynamic resource handler"""
                return resource_data["content"]
            logger.info(f"Registered resource: {resource_data['uri']}")
        except Exception as e:
            logger.error(f"Failed to register resource {resource_data['uri']}: {e}")

    # Ensure UTF-8 encoding for stdin/stdout
    if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    if hasattr(sys.stdin, 'buffer') and sys.stdin.encoding != 'utf-8':
        import io
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', line_buffering=True)

    # Flush stdout to ensure no buffered content remains
    sys.stdout.flush()

    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    # When running the script directly, ensure we're in the right directory
    import os
    import sys

    # Add the parent directory to sys.path if needed
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Print debug information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    logger.info(f"Parent directory added to path: {parent_dir}")

    # Run the server
    main()
