import json
import os
import sys
from typing import List, Optional

from mcp.server.fastmcp import FastMCP, Context

# Use absolute imports when running as a script
try:
    # When installed as a package
    from .models import ThoughtData, ThoughtStage
    from .storage import ThoughtStorage
    from .analysis import ThoughtAnalyzer
    from .logging_conf import configure_logging
    from .integration_bridge import integration_bridge
except ImportError:
    # When run directly
    from mcp_coaia_sequential_thinking.models import ThoughtData, ThoughtStage
    from mcp_coaia_sequential_thinking.storage import ThoughtStorage
    from mcp_coaia_sequential_thinking.analysis import ThoughtAnalyzer
    from mcp_coaia_sequential_thinking.logging_conf import configure_logging
    from mcp_coaia_sequential_thinking.integration_bridge import integration_bridge

logger = configure_logging("coaia-sequential-thinking.server")


mcp = FastMCP("sequential-thinking")

storage_dir = os.environ.get("MCP_STORAGE_DIR", None)
storage = ThoughtStorage(storage_dir)

@mcp.tool()
def process_thought(thought: str, thought_number: int, total_thoughts: int,
                    next_thought_needed: bool, stage: str,
                    tags: List[str] = [],
                    axioms_used: List[str] = [],
                    assumptions_challenged: List[str] = [],
                    ctx: Optional[Context] = None) -> dict:
    """Add a sequential thought with its metadata.

    Args:
        thought: The content of the thought
        thought_number: The sequence number of this thought
        total_thoughts: The total expected thoughts in the sequence
        next_thought_needed: Whether more thoughts are needed after this one
        stage: The thinking stage (Problem Definition, Research, Analysis, Synthesis, Conclusion)
        tags: Optional keywords or categories for the thought
        axioms_used: Optional list of principles or axioms used in this thought
        assumptions_challenged: Optional list of assumptions challenged by this thought
        ctx: Optional MCP context object

    Returns:
        dict: Analysis of the processed thought
    """
    try:
        # Log the request
        logger.info(f"Processing thought #{thought_number}/{total_thoughts} in stage '{stage}'")

        # Report progress if context is available
        if ctx:
            ctx.report_progress(thought_number - 1, total_thoughts)

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
        storage.add_thought(thought_data)

        # Get all thoughts for analysis
        all_thoughts = storage.get_all_thoughts()

        # Analyze the thought
        analysis = ThoughtAnalyzer.analyze_thought(thought_data, all_thoughts)

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
        
        # Check if session is ready for chart creation
        chart_readiness = integration_bridge.analyze_chart_readiness(all_thoughts)
        
        # Add chart readiness info to summary
        summary = summary_result.get('summary', {})
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
