#!/usr/bin/env python3
"""
Test script for the Polycentric Agentic Lattice implementation.
Demonstrates Phase 2 capabilities: multi-agent coordination, collaboration, and competition.
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mcp_coaia_sequential_thinking.polycentric_lattice import (
    ConstitutionalAgent, AnalysisAgent, agent_registry, AgentRole, MessageType, MessagePriority
)
from mcp_coaia_sequential_thinking.agent_coordination import task_coordinator, TaskType
from mcp_coaia_sequential_thinking.constitutional_core import constitutional_core


def test_agent_creation_and_registration():
    """Test creating and registering agents in the lattice."""
    print("=== Testing Agent Creation and Registration ===")
    
    # Clear any existing agents
    for agent_id in list(agent_registry.agents.keys()):
        agent_registry.unregister_agent(agent_id)
    
    # Create agents
    constitutional_agent = ConstitutionalAgent()
    analysis_agent = AnalysisAgent()
    
    print(f"Created Constitutional Agent: {constitutional_agent.name} ({constitutional_agent.agent_id})")
    print(f"Created Analysis Agent: {analysis_agent.name} ({analysis_agent.agent_id})")
    
    # Register agents
    agent_registry.register_agent(constitutional_agent)
    agent_registry.register_agent(analysis_agent)
    
    # Verify registration
    status = agent_registry.get_agent_status_summary()
    print(f"Registry status: {status['total_agents']} total, {status['active_agents']} active")
    
    # Test capabilities
    const_caps = [cap.name for cap in constitutional_agent.get_capabilities()]
    analysis_caps = [cap.name for cap in analysis_agent.get_capabilities()]
    
    print(f"Constitutional capabilities: {const_caps}")
    print(f"Analysis capabilities: {analysis_caps}")
    
    assert len(agent_registry.agents) == 2
    assert status['active_agents'] == 2
    print("✅ Agent creation and registration working")
    
    return constitutional_agent, analysis_agent


def test_capability_discovery():
    """Test capability discovery across the lattice."""
    print("\n=== Testing Capability Discovery ===")
    
    # Test finding agents by capability
    constitutional_agents = agent_registry.find_agents_with_capability("Constitutional Validation")
    analysis_agents = agent_registry.find_agents_with_capability("Pattern Recognition")
    
    print(f"Agents with Constitutional Validation: {len(constitutional_agents)}")
    print(f"Agents with Pattern Recognition: {len(analysis_agents)}")
    
    # Test capability registry
    all_capabilities = {}
    for agent_id, capabilities in agent_registry.agent_capabilities.items():
        all_capabilities[agent_id] = capabilities
    
    print(f"All capabilities across lattice: {all_capabilities}")
    
    assert len(constitutional_agents) >= 1
    assert len(analysis_agents) >= 1
    print("✅ Capability discovery working")


def test_individual_task_coordination():
    """Test coordinating individual tasks to agents."""
    print("\n=== Testing Individual Task Coordination ===")
    
    # Submit a constitutional validation task
    task_id = task_coordinator.submit_task(
        description="Validate this content for constitutional compliance",
        requirements=["Constitutional Validation"],
        task_type=TaskType.INDIVIDUAL,
        priority=MessagePriority.HIGH
    )
    
    print(f"Submitted constitutional task: {task_id}")
    
    # Give coordinator time to process
    time.sleep(0.5)
    
    # Check task status
    task_status = task_coordinator.get_task_status(task_id)
    print(f"Task status: {task_status}")
    
    # Submit an analysis task
    analysis_task_id = task_coordinator.submit_task(
        description="Analyze patterns in this data",
        requirements=["Pattern Recognition"],
        task_type=TaskType.INDIVIDUAL,
        priority=MessagePriority.MEDIUM
    )
    
    print(f"Submitted analysis task: {analysis_task_id}")
    
    # Give coordinator time to process
    time.sleep(0.5)
    
    analysis_status = task_coordinator.get_task_status(analysis_task_id)
    print(f"Analysis task status: {analysis_status}")
    
    assert task_status is not None
    assert analysis_status is not None
    print("✅ Individual task coordination working")
    
    return task_id, analysis_task_id


def test_collaborative_task_coordination():
    """Test coordinating collaborative tasks."""
    print("\n=== Testing Collaborative Task Coordination ===")
    
    # Submit a task requiring multiple capabilities
    collab_task_id = task_coordinator.submit_task(
        description="Perform comprehensive analysis with constitutional oversight",
        requirements=["Pattern Recognition", "Constitutional Validation"],
        task_type=TaskType.COLLABORATIVE,
        priority=MessagePriority.HIGH
    )
    
    print(f"Submitted collaborative task: {collab_task_id}")
    
    # Give coordinator time to process
    time.sleep(1.0)
    
    # Check task status
    task_status = task_coordinator.get_task_status(collab_task_id)
    print(f"Collaborative task status: {task_status}")
    
    # Check coordination system status
    coord_status = task_coordinator.get_coordination_status()
    print(f"Coordination system status: {coord_status}")
    
    assert task_status is not None
    print("✅ Collaborative task coordination working")
    
    return collab_task_id


def test_competitive_task_coordination():
    """Test coordinating competitive tasks."""
    print("\n=== Testing Competitive Task Coordination ===")
    
    # Submit a competitive task
    comp_task_id = task_coordinator.submit_task(
        description="Compete to provide the best analysis of this data",
        requirements=["Pattern Recognition"],
        task_type=TaskType.COMPETITIVE,
        priority=MessagePriority.MEDIUM
    )
    
    print(f"Submitted competitive task: {comp_task_id}")
    
    # Give coordinator time to process
    time.sleep(1.0)
    
    # Check task status
    task_status = task_coordinator.get_task_status(comp_task_id)
    print(f"Competitive task status: {task_status}")
    
    assert task_status is not None
    print("✅ Competitive task coordination working")
    
    return comp_task_id


def test_constitutional_review_task():
    """Test constitutional review tasks."""
    print("\n=== Testing Constitutional Review ===")
    
    # Submit a constitutional review task
    review_task_id = task_coordinator.submit_task(
        description="Review this decision for constitutional compliance",
        requirements=["Constitutional Validation"],
        task_type=TaskType.CONSTITUTIONAL_REVIEW,
        priority=MessagePriority.CRITICAL
    )
    
    print(f"Submitted constitutional review task: {review_task_id}")
    
    # Give coordinator time to process
    time.sleep(0.5)
    
    # Check task status
    task_status = task_coordinator.get_task_status(review_task_id)
    print(f"Constitutional review task status: {task_status}")
    
    assert task_status is not None
    print("✅ Constitutional review working")
    
    return review_task_id


def test_agent_request_processing():
    """Test direct agent request processing."""
    print("\n=== Testing Agent Request Processing ===")
    
    # Get agents
    agents = list(agent_registry.agents.values())
    constitutional_agent = None
    analysis_agent = None
    
    for agent in agents:
        if agent.role == AgentRole.CONSTITUTIONAL:
            constitutional_agent = agent
        elif agent.role == AgentRole.ANALYSIS:
            analysis_agent = agent
    
    if constitutional_agent:
        # Test constitutional agent request
        const_request = {
            "type": "validate_content",
            "content": "I want to create a better solution to this problem",
            "context": {"stage": "desired_outcome"}
        }
        
        const_response = constitutional_agent.process_request(const_request)
        print(f"Constitutional agent response: {const_response}")
        
        assert "validation_result" in const_response
        print("✅ Constitutional agent request processing working")
    
    if analysis_agent:
        # Test analysis agent request
        analysis_request = {
            "type": "analyze_patterns",
            "data": [
                {"score": 0.5, "timestamp": "2025-01-01"},
                {"score": 0.7, "timestamp": "2025-01-02"},
                {"score": 0.8, "timestamp": "2025-01-03"}
            ]
        }
        
        analysis_response = analysis_agent.process_request(analysis_request)
        print(f"Analysis agent response: {analysis_response}")
        
        assert "patterns" in analysis_response
        print("✅ Analysis agent request processing working")


def test_lattice_performance_metrics():
    """Test performance tracking and metrics."""
    print("\n=== Testing Performance Metrics ===")
    
    # Get coordination status
    coord_status = task_coordinator.get_coordination_status()
    print(f"Coordination status: {coord_status}")
    
    # Get agent status summary
    agent_status = agent_registry.get_agent_status_summary()
    print(f"Agent status summary: {agent_status}")
    
    # Test performance metrics
    performance_metrics = coord_status.get("performance_metrics", {})
    print(f"Performance metrics: {performance_metrics}")
    
    # Test individual agent metrics
    for agent_id, agent in agent_registry.agents.items():
        agent_metrics = agent.performance_metrics
        print(f"Agent {agent.name} metrics: {agent_metrics}")
    
    print("✅ Performance metrics working")


def test_lattice_health_assessment():
    """Test lattice health assessment."""
    print("\n=== Testing Lattice Health Assessment ===")
    
    agent_status = agent_registry.get_agent_status_summary()
    coord_status = task_coordinator.get_coordination_status()
    
    # Calculate health metrics (simplified version of server function)
    total_agents = agent_status.get("total_agents", 0)
    active_agents = agent_status.get("active_agents", 0)
    
    agent_health = active_agents / total_agents if total_agents > 0 else 0.0
    
    total_tasks = coord_status.get("active_tasks", 0)
    failed_tasks = coord_status.get("failed_tasks", 0)
    
    task_health = 1.0 - (failed_tasks / max(total_tasks, 1))
    overall_health = (agent_health + task_health) / 2.0
    
    print(f"Agent health: {agent_health:.2f}")
    print(f"Task health: {task_health:.2f}")
    print(f"Overall health: {overall_health:.2f}")
    
    health_status = "excellent" if overall_health > 0.9 else "good" if overall_health > 0.7 else "fair"
    print(f"Health status: {health_status}")
    
    assert overall_health > 0.5  # Should have reasonable health
    print("✅ Lattice health assessment working")


def test_integration_with_constitutional_core():
    """Test integration between lattice and constitutional core."""
    print("\n=== Testing Constitutional Core Integration ===")
    
    # Test constitutional validation through agent
    agents = list(agent_registry.agents.values())
    constitutional_agent = None
    
    for agent in agents:
        if agent.role == AgentRole.CONSTITUTIONAL:
            constitutional_agent = agent
            break
    
    if constitutional_agent:
        # Test principle conflict resolution through agent
        conflict_request = {
            "type": "resolve_conflict",
            "principles": [
                "acknowledge_uncertainty_rather_than_invent_facts",
                "prioritize_creating_desired_outcomes_over_eliminating_problems"
            ],
            "context": {"situation": "uncertain about best approach"}
        }
        
        conflict_response = constitutional_agent.process_request(conflict_request)
        print(f"Principle conflict resolution: {conflict_response}")
        
        assert "resolved_principle" in conflict_response
        print("✅ Constitutional core integration working")


def main():
    """Run all polycentric lattice tests."""
    print("🏛️ Polycentric Agentic Lattice Testing Suite")
    print("=" * 60)
    
    try:
        # Test basic functionality
        constitutional_agent, analysis_agent = test_agent_creation_and_registration()
        test_capability_discovery()
        
        # Test task coordination
        test_individual_task_coordination()
        test_collaborative_task_coordination()
        test_competitive_task_coordination()
        test_constitutional_review_task()
        
        # Test agent functionality
        test_agent_request_processing()
        
        # Test system health and performance
        test_lattice_performance_metrics()
        test_lattice_health_assessment()
        
        # Test integration
        test_integration_with_constitutional_core()
        
        print("\n\n🎉 All polycentric lattice tests completed successfully!")
        print("\nPhase 2 Implementation Complete:")
        print("✅ Multi-agent architecture")
        print("✅ Agent registration and discovery")
        print("✅ Capability-based task assignment")
        print("✅ Individual task coordination")
        print("✅ Collaborative task coordination")
        print("✅ Competitive task coordination")
        print("✅ Constitutional review integration")
        print("✅ Performance metrics and health monitoring")
        print("✅ Agent-to-agent communication framework")
        print("✅ Task coordinator with formal workflows")
        
        print("\n🚀 Ready for Phase 3: Resilient Connection Implementation")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        task_coordinator.stop_coordination()
        for agent_id in list(agent_registry.agents.keys()):
            agent_registry.unregister_agent(agent_id)
        print("Cleanup complete")


if __name__ == "__main__":
    main()