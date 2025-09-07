import json
import os
import sys
from typing import List, Optional, Dict, Any

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

logger = configure_logging("coaia-sequential-thinking.server")


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


def main():
    """Entry point for the MCP server."""
    logger.info("Starting CoAiA Sequential Thinking MCP server")

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
