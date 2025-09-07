"""
Agent Coordination System: Manages polycentric agent interactions.

This module provides the coordination layer for the polycentric lattice,
handling agent registration, task distribution, collaboration management,
and conflict resolution.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import threading
import time
from queue import Queue, Empty

from .polycentric_lattice import (
    BaseAgent, AgentRole, MessageType, MessagePriority, AgentMessage,
    CollaborationProposal, CompetitionChallenge, agent_registry
)
from .constitutional_core import constitutional_core

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that can be coordinated."""
    INDIVIDUAL = "individual_task"
    COLLABORATIVE = "collaborative_task"
    COMPETITIVE = "competitive_task"
    CONSTITUTIONAL_REVIEW = "constitutional_review"


class TaskStatus(Enum):
    """Status of coordinated tasks."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CoordinatedTask:
    """A task being coordinated across agents."""
    task_id: str
    task_type: TaskType
    description: str
    requirements: List[str]  # Required capabilities
    assigned_agents: List[str]
    status: TaskStatus
    created_at: datetime
    deadline: Optional[datetime] = None
    priority: MessagePriority = MessagePriority.MEDIUM
    context: Dict[str, Any] = None
    results: Dict[str, Any] = None
    constitutional_validated: bool = False


class TaskCoordinator:
    """Coordinates tasks across the polycentric agent lattice."""
    
    def __init__(self):
        self.active_tasks: Dict[str, CoordinatedTask] = {}
        self.task_queue = Queue()
        self.coordination_thread: Optional[threading.Thread] = None
        self.active = True
        self.performance_tracker = PerformanceTracker()
        
        # Start coordination loop
        self.start_coordination()
    
    def start_coordination(self):
        """Start the task coordination loop."""
        if self.coordination_thread is None or not self.coordination_thread.is_alive():
            self.coordination_thread = threading.Thread(target=self._coordination_loop)
            self.coordination_thread.daemon = True
            self.coordination_thread.start()
            logger.info("Task coordination started")
    
    def stop_coordination(self):
        """Stop task coordination."""
        self.active = False
        if self.coordination_thread and self.coordination_thread.is_alive():
            self.coordination_thread.join(timeout=5.0)
            logger.info("Task coordination stopped")
    
    def submit_task(self, description: str, requirements: List[str], 
                   task_type: TaskType = TaskType.INDIVIDUAL,
                   deadline: Optional[datetime] = None,
                   priority: MessagePriority = MessagePriority.MEDIUM,
                   context: Dict[str, Any] = None) -> str:
        """Submit a new task for coordination."""
        task_id = str(uuid.uuid4())
        
        task = CoordinatedTask(
            task_id=task_id,
            task_type=task_type,
            description=description,
            requirements=requirements,
            assigned_agents=[],
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            deadline=deadline,
            priority=priority,
            context=context or {}
        )
        
        self.active_tasks[task_id] = task
        self.task_queue.put(task_id)
        
        logger.info(f"Submitted task {task_id}: {description}")
        return task_id
    
    def _coordination_loop(self):
        """Main coordination loop."""
        while self.active:
            try:
                # Process pending tasks
                try:
                    task_id = self.task_queue.get(timeout=1.0)
                    self._process_task(task_id)
                except Empty:
                    pass
                
                # Check on active tasks
                self._monitor_active_tasks()
                
                # Update performance metrics
                self.performance_tracker.update_metrics()
                
                time.sleep(0.1)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
    
    def _process_task(self, task_id: str):
        """Process a pending task."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        
        try:
            if task.task_type == TaskType.INDIVIDUAL:
                self._assign_individual_task(task)
            elif task.task_type == TaskType.COLLABORATIVE:
                self._coordinate_collaboration(task)
            elif task.task_type == TaskType.COMPETITIVE:
                self._coordinate_competition(task)
            elif task.task_type == TaskType.CONSTITUTIONAL_REVIEW:
                self._coordinate_constitutional_review(task)
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            task.status = TaskStatus.FAILED
    
    def _assign_individual_task(self, task: CoordinatedTask):
        """Assign an individual task to the best suited agent."""
        # Find agents with required capabilities
        suitable_agents = []
        for requirement in task.requirements:
            matching_agents = agent_registry.find_agents_with_capability(requirement)
            suitable_agents.extend(matching_agents)
        
        if not suitable_agents:
            logger.warning(f"No suitable agents found for task {task.task_id}")
            task.status = TaskStatus.FAILED
            return
        
        # Select best agent based on performance and availability
        best_agent = self._select_best_agent(suitable_agents, task.requirements)
        
        if best_agent:
            task.assigned_agents = [best_agent]
            task.status = TaskStatus.ASSIGNED
            
            # Send task to agent
            self._send_task_to_agent(best_agent, task)
            logger.info(f"Assigned task {task.task_id} to agent {best_agent}")
        else:
            task.status = TaskStatus.FAILED
    
    def _coordinate_collaboration(self, task: CoordinatedTask):
        """Coordinate a collaborative task among multiple agents."""
        # Find agents for each required capability AND all active agents for broader evaluation
        agent_capabilities = {}
        all_target_agents = set()
        
        for requirement in task.requirements:
            matching_agents = agent_registry.find_agents_with_capability(requirement)
            agent_capabilities[requirement] = matching_agents
            all_target_agents.update(matching_agents)
        
        # If no exact capability matches, include all active agents for evaluation
        if not all_target_agents:
            all_target_agents = set(agent_registry.agents.keys())
        
        # Create collaboration proposal
        proposal_id = str(uuid.uuid4())
        proposal = CollaborationProposal(
            proposal_id=proposal_id,
            proposer_id="task_coordinator",
            target_agents=list(all_target_agents),
            task_description=task.description,
            required_capabilities=task.requirements,
            resource_allocation={agent: 1.0 for agent in all_target_agents},
            expected_outcome=f"Completion of task {task.task_id}",
            deadline=task.deadline or datetime.now() + timedelta(hours=24)
        )
        
        # Send collaboration invitations through message system
        accepted_agents = []
        for agent_id in proposal.target_agents:
            if agent_id in agent_registry.agents:
                agent = agent_registry.agents[agent_id]
                
                # Send collaboration invitation message
                message_content = {
                    "proposal": proposal.__dict__
                }
                # Note: In a full message-based system, we would send this message
                # and wait for responses. For now we do direct evaluation.
                # agent.send_message(
                #     agent_id,
                #     MessageType.COLLABORATION_INVITE,
                #     message_content,
                #     priority=MessagePriority.HIGH,
                #     requires_response=True
                # )
                
                # For now, also do direct evaluation to ensure functionality
                # In a full implementation, we'd wait for message responses
                accept, reason = agent.evaluate_collaboration_proposal(proposal)
                if accept:
                    accepted_agents.append(agent_id)
                    logger.info(f"Agent {agent_id} accepted collaboration: {reason}")
                else:
                    logger.debug(f"Agent {agent_id} declined collaboration: {reason}")
        
        # Accept collaboration if we have at least one agent or agents covering all requirements
        required_capabilities_covered = all(
            any(cap_id in [cap.capability_id for cap in agent_registry.agents[agent_id].get_capabilities()] 
                for agent_id in accepted_agents)
            for cap_id in task.requirements
        )
        
        if accepted_agents and (len(accepted_agents) >= 1 or required_capabilities_covered):
            task.assigned_agents = accepted_agents
            task.status = TaskStatus.ASSIGNED
            
            # Notify agents of collaboration start
            for agent_id in accepted_agents:
                self._send_collaboration_start(agent_id, task, proposal)
            
            logger.info(f"Collaboration started for task {task.task_id} with agents {accepted_agents}")
        else:
            logger.warning(f"No agents accepted collaboration for task {task.task_id}. Available agents: {list(all_target_agents)}")
            task.status = TaskStatus.FAILED
    
    def _coordinate_competition(self, task: CoordinatedTask):
        """Coordinate a competitive task among agents."""
        # Find agents with relevant capabilities
        competing_agents = []
        for requirement in task.requirements:
            matching_agents = agent_registry.find_agents_with_capability(requirement)
            competing_agents.extend(matching_agents)
        
        competing_agents = list(set(competing_agents))  # Remove duplicates
        
        if len(competing_agents) < 2:
            logger.warning(f"Need at least 2 agents for competition task {task.task_id}")
            task.status = TaskStatus.FAILED
            return
        
        # Create competition challenge
        challenge_id = str(uuid.uuid4())
        challenge = CompetitionChallenge(
            challenge_id=challenge_id,
            challenger_id="task_coordinator",
            target_agents=competing_agents,
            task_description=task.description,
            evaluation_criteria={req: 1.0 for req in task.requirements},
            deadline=task.deadline or datetime.now() + timedelta(hours=24),
            winner_benefits={"performance_bonus": 1.1, "reputation_boost": 0.1}
        )
        
        # Send competition invitations
        participating_agents = []
        for agent_id in competing_agents:
            if agent_id in agent_registry.agents:
                agent = agent_registry.agents[agent_id]
                accept, reason = agent.evaluate_competition_challenge(challenge)
                if accept:
                    participating_agents.append(agent_id)
        
        if len(participating_agents) >= 2:
            task.assigned_agents = participating_agents
            task.status = TaskStatus.ASSIGNED
            
            # Notify agents of competition start
            for agent_id in participating_agents:
                self._send_competition_start(agent_id, task, challenge)
            
            logger.info(f"Competition started for task {task.task_id} with agents {participating_agents}")
        else:
            logger.warning(f"Insufficient agents for competition task {task.task_id}")
            task.status = TaskStatus.FAILED
    
    def _coordinate_constitutional_review(self, task: CoordinatedTask):
        """Coordinate constitutional review of task or decision."""
        # Find constitutional agents
        constitutional_agents = agent_registry.find_agents_with_capability("constitutional_validation")
        
        if not constitutional_agents:
            logger.warning(f"No constitutional agents available for task {task.task_id}")
            task.status = TaskStatus.FAILED
            return
        
        # Assign to first available constitutional agent
        constitutional_agent = constitutional_agents[0]
        task.assigned_agents = [constitutional_agent]
        task.status = TaskStatus.ASSIGNED
        
        # Send constitutional review request
        self._send_constitutional_review(constitutional_agent, task)
        logger.info(f"Constitutional review started for task {task.task_id}")
    
    def _select_best_agent(self, candidate_agents: List[str], requirements: List[str]) -> Optional[str]:
        """Select the best agent for a task based on performance and capabilities."""
        if not candidate_agents:
            return None
        
        best_agent = None
        best_score = -1
        
        for agent_id in candidate_agents:
            if agent_id not in agent_registry.agents:
                continue
            
            agent = agent_registry.agents[agent_id]
            
            # Calculate agent score based on multiple factors
            score = self._calculate_agent_score(agent, requirements)
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    def _calculate_agent_score(self, agent: BaseAgent, requirements: List[str]) -> float:
        """Calculate agent suitability score for a task."""
        # Base score from performance metrics
        base_score = agent.performance_metrics.get("task_completion_rate", 0.5)
        
        # Capability match score
        agent_capabilities = [cap.name for cap in agent.get_capabilities()]
        capability_match = sum(1 for req in requirements if req in agent_capabilities) / len(requirements)
        
        # Constitutional compliance score
        constitutional_score = agent.performance_metrics.get("constitutional_compliance_score", 0.8)
        
        # Availability score (based on current workload)
        availability_score = 1.0 - (agent.message_queue.qsize() / 100.0)  # Normalize queue size
        
        # Weighted combination
        total_score = (
            base_score * 0.3 +
            capability_match * 0.4 +
            constitutional_score * 0.2 +
            availability_score * 0.1
        )
        
        return total_score
    
    def _send_task_to_agent(self, agent_id: str, task: CoordinatedTask):
        """Send a task assignment to an agent."""
        if agent_id in agent_registry.agents:
            agent = agent_registry.agents[agent_id]
            message_content = {
                "type": "task_assignment",
                "task_id": task.task_id,
                "description": task.description,
                "requirements": task.requirements,
                "deadline": task.deadline.isoformat() if task.deadline else None,
                "context": task.context
            }
            
            agent.send_message(
                "task_coordinator",
                MessageType.REQUEST,
                message_content,
                priority=task.priority,
                requires_response=True
            )
    
    def _send_collaboration_start(self, agent_id: str, task: CoordinatedTask, proposal: CollaborationProposal):
        """Send collaboration start notification to an agent."""
        if agent_id in agent_registry.agents:
            agent = agent_registry.agents[agent_id]
            message_content = {
                "type": "collaboration_start",
                "task_id": task.task_id,
                "proposal": proposal.__dict__,
                "collaborating_agents": task.assigned_agents
            }
            
            agent.send_message(
                "task_coordinator",
                MessageType.NOTIFICATION,
                message_content,
                priority=task.priority
            )
    
    def _send_competition_start(self, agent_id: str, task: CoordinatedTask, challenge: CompetitionChallenge):
        """Send competition start notification to an agent."""
        if agent_id in agent_registry.agents:
            agent = agent_registry.agents[agent_id]
            message_content = {
                "type": "competition_start",
                "task_id": task.task_id,
                "challenge": challenge.__dict__,
                "competing_agents": task.assigned_agents
            }
            
            agent.send_message(
                "task_coordinator",
                MessageType.NOTIFICATION,
                message_content,
                priority=task.priority
            )
    
    def _send_constitutional_review(self, agent_id: str, task: CoordinatedTask):
        """Send constitutional review request to an agent."""
        if agent_id in agent_registry.agents:
            agent = agent_registry.agents[agent_id]
            message_content = {
                "type": "constitutional_review",
                "task_id": task.task_id,
                "content_to_review": task.description,
                "context": task.context
            }
            
            agent.send_message(
                "task_coordinator",
                MessageType.REQUEST,
                message_content,
                priority=MessagePriority.HIGH,
                requires_response=True
            )
    
    def _monitor_active_tasks(self):
        """Monitor progress of active tasks."""
        current_time = datetime.now()
        
        for task_id, task in list(self.active_tasks.items()):
            # Check for deadline violations
            if task.deadline and current_time > task.deadline and task.status != TaskStatus.COMPLETED:
                logger.warning(f"Task {task_id} missed deadline")
                task.status = TaskStatus.FAILED
            
            # Clean up completed/failed tasks older than 1 hour
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if current_time - task.created_at > timedelta(hours=1):
                    del self.active_tasks[task_id]
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task."""
        if task_id not in self.active_tasks:
            return None
        
        task = self.active_tasks[task_id]
        return {
            "task_id": task.task_id,
            "description": task.description,
            "status": task.status.value,
            "assigned_agents": task.assigned_agents,
            "created_at": task.created_at.isoformat(),
            "deadline": task.deadline.isoformat() if task.deadline else None,
            "results": task.results
        }
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get overall coordination system status."""
        return {
            "active_tasks": len(self.active_tasks),
            "pending_tasks": sum(1 for task in self.active_tasks.values() if task.status == TaskStatus.PENDING),
            "in_progress_tasks": sum(1 for task in self.active_tasks.values() if task.status == TaskStatus.IN_PROGRESS),
            "completed_tasks": sum(1 for task in self.active_tasks.values() if task.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for task in self.active_tasks.values() if task.status == TaskStatus.FAILED),
            "task_queue_size": self.task_queue.qsize(),
            "coordination_active": self.active,
            "performance_metrics": self.performance_tracker.get_metrics()
        }


class PerformanceTracker:
    """Tracks performance metrics for the coordination system."""
    
    def __init__(self):
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_completion_time": 0.0,
            "collaboration_success_rate": 0.0,
            "competition_completion_rate": 0.0,
            "constitutional_compliance_rate": 1.0
        }
        self.task_history: List[Dict[str, Any]] = []
    
    def record_task_completion(self, task: CoordinatedTask, completion_time: float, success: bool):
        """Record task completion metrics."""
        self.task_history.append({
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "completion_time": completion_time,
            "success": success,
            "agents_involved": len(task.assigned_agents),
            "constitutional_validated": task.constitutional_validated
        })
        
        # Keep only recent history
        if len(self.task_history) > 1000:
            self.task_history = self.task_history[-1000:]
    
    def update_metrics(self):
        """Update performance metrics based on recent history."""
        if not self.task_history:
            return
        
        recent_tasks = self.task_history[-100:]  # Last 100 tasks
        
        self.metrics["tasks_completed"] = sum(1 for task in recent_tasks if task["success"])
        self.metrics["tasks_failed"] = sum(1 for task in recent_tasks if not task["success"])
        
        if recent_tasks:
            completion_times = [task["completion_time"] for task in recent_tasks if task["success"]]
            if completion_times:
                self.metrics["average_completion_time"] = sum(completion_times) / len(completion_times)
            
            collaborative_tasks = [task for task in recent_tasks if task["agents_involved"] > 1]
            if collaborative_tasks:
                successful_collaborations = sum(1 for task in collaborative_tasks if task["success"])
                self.metrics["collaboration_success_rate"] = successful_collaborations / len(collaborative_tasks)
            
            constitutional_tasks = [task for task in recent_tasks if task["constitutional_validated"]]
            if constitutional_tasks:
                self.metrics["constitutional_compliance_rate"] = sum(1 for task in constitutional_tasks if task["success"]) / len(constitutional_tasks)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.metrics.copy()


# Global task coordinator
task_coordinator = TaskCoordinator()