"""
Stateful Inquiry Engine - Core Component for State Persistence

This module implements Mia's architectural specification from rispecs/ to resolve
critical state persistence failures in the MCP Sequential Thinking toolset.

The StatefulInquiryEngine acts as the system's memory, ensuring that every step
of a complex inquiry is preserved, honored, and built upon.
"""

import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .data_persistence import get_data_store


@dataclass
class PersonaPerspective:
    """Structured object for capturing an AI persona's viewpoint within an inquiry"""
    persona_name: str
    perspective_content: str
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class Inquiry:
    """Master object holding the entire state of a reasoning process"""
    inquiry_id: str
    master_tension: Dict[str, str]  # desired_outcome, current_reality
    status: str  # GATHERING_PERSPECTIVES, SYNTHESIZING, AWAITING_CONSENSUS, COMPLETE
    perspectives: List[PersonaPerspective]
    decisions: List[str]  # decision_ids
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]


class StatefulInquiryEngine:
    """
    Central singleton engine for managing stateful reasoning processes.
    
    Responsibilities:
    1. Inquiry Lifecycle Management - create, load, modify inquiries
    2. State Persistence - all changes immediately written to database
    3. Encapsulation of Logic - business logic lives here, not in tools
    
    Quality Criteria:
    - State survives tool calls and server restarts
    - Atomic transactions (load → modify → save)
    - No long-term state in memory
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - only one instance exists"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize engine with database connection"""
        if self._initialized:
            return
        self.data_store = get_data_store()
        self._initialized = True
    
    def initiate_inquiry(
        self,
        initial_request: str,
        desired_outcome: str,
        current_reality: str,
        primary_purpose: str = "integrated_wisdom"
    ) -> Dict[str, Any]:
        """
        Create a new inquiry - the sacred container for reasoning.
        
        Replaces: initiate_creative_emergence, initiate_sequential_thinking
        
        Returns:
            inquiry_id and status for tracking the inquiry
        """
        inquiry_id = f"inquiry_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        inquiry = Inquiry(
            inquiry_id=inquiry_id,
            master_tension={
                "desired_outcome": desired_outcome,
                "current_reality": current_reality,
                "initial_request": initial_request
            },
            status="GATHERING_PERSPECTIVES",
            perspectives=[],
            decisions=[],
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            metadata={
                "primary_purpose": primary_purpose,
                "tool_version": "stateful_engine_v1.0"
            }
        )
        
        # Persist immediately to database
        self._save_inquiry(inquiry)
        
        return {
            "inquiry_id": inquiry_id,
            "status": "initiated",
            "master_tension": inquiry.master_tension,
            "message": "Inquiry container created. State persisted to database."
        }
    
    def advance_inquiry(
        self,
        inquiry_id: str,
        persona_name: Optional[str] = None,
        new_insight: Optional[str] = None,
        action: str = "add_perspective"
    ) -> Dict[str, Any]:
        """
        Advance an inquiry by adding a new perspective or insight.
        
        Replaces: advance_thinking_chain
        
        Args:
            inquiry_id: The persistent identifier for the inquiry
            persona_name: Which persona is providing this perspective (Mia, Miette, Haiku)
            new_insight: The content to add
            action: Type of advancement (add_perspective, synthesize, etc.)
        
        Returns:
            Updated inquiry status and new perspective
        """
        # Load inquiry from database
        inquiry = self._load_inquiry(inquiry_id)
        if not inquiry:
            return {
                "status": "error",
                "message": f"Inquiry {inquiry_id} not found in database"
            }
        
        if action == "add_perspective":
            # Create and add perspective
            perspective = PersonaPerspective(
                persona_name=persona_name or "default",
                perspective_content=new_insight or "",
                timestamp=datetime.utcnow().isoformat(),
                metadata={"action": action}
            )
            inquiry.perspectives.append(perspective)
            inquiry.updated_at = datetime.utcnow().isoformat()
            
            # Save updated state immediately
            self._save_inquiry(inquiry)
            
            return {
                "inquiry_id": inquiry_id,
                "status": "advanced",
                "perspective_count": len(inquiry.perspectives),
                "latest_persona": persona_name,
                "message": "Perspective added and state persisted"
            }
        
        elif action == "change_status":
            inquiry.status = new_insight or inquiry.status
            inquiry.updated_at = datetime.utcnow().isoformat()
            self._save_inquiry(inquiry)
            
            return {
                "inquiry_id": inquiry_id,
                "status": inquiry.status,
                "message": f"Status changed to {inquiry.status}"
            }
        
        return {
            "status": "error",
            "message": f"Unknown action: {action}"
        }
    
    def get_inquiry_status(self, inquiry_id: str) -> Dict[str, Any]:
        """
        Get comprehensive status of an inquiry from database.
        
        Replaces: get_thinking_chain_status, get_consensus_decision_status
        
        Returns:
            Complete inquiry state including all perspectives and decisions
        """
        inquiry = self._load_inquiry(inquiry_id)
        if not inquiry:
            return {
                "status": "error",
                "message": f"Inquiry {inquiry_id} not found"
            }
        
        return {
            "inquiry_id": inquiry.inquiry_id,
            "status": inquiry.status,
            "master_tension": inquiry.master_tension,
            "perspective_count": len(inquiry.perspectives),
            "perspectives": [
                {
                    "persona": p.persona_name,
                    "timestamp": p.timestamp,
                    "content": p.perspective_content[:200] + "..." if len(p.perspective_content) > 200 else p.perspective_content
                }
                for p in inquiry.perspectives
            ],
            "decision_count": len(inquiry.decisions),
            "created_at": inquiry.created_at,
            "updated_at": inquiry.updated_at,
            "message": "State loaded from database"
        }
    
    def _save_inquiry(self, inquiry: Inquiry) -> None:
        """Persist inquiry to database immediately"""
        # Use data_store to save inquiry state
        inquiry_data = {
            "inquiry_id": inquiry.inquiry_id,
            "data": json.dumps(asdict(inquiry)),
            "status": inquiry.status,
            "updated_at": inquiry.updated_at
        }
        
        # Save to database using existing data_store infrastructure
        self.data_store.save_agent_message(
            agent_name="StatefulInquiryEngine",
            message_type="inquiry_state",
            content=json.dumps(inquiry_data),
            metadata={"inquiry_id": inquiry.inquiry_id}
        )
    
    def _load_inquiry(self, inquiry_id: str) -> Optional[Inquiry]:
        """Load inquiry from database"""
        try:
            # Query database for inquiry state
            # Using agent_messages table with inquiry_state type
            messages = self.data_store.get_agent_messages(
                agent_name="StatefulInquiryEngine",
                message_type="inquiry_state"
            )
            
            # Find the most recent message for this inquiry_id
            for msg in reversed(messages):  # Most recent first
                metadata = json.loads(msg.get("metadata", "{}"))
                if metadata.get("inquiry_id") == inquiry_id:
                    inquiry_data = json.loads(msg["content"])
                    data = json.loads(inquiry_data["data"])
                    
                    # Reconstruct Inquiry object
                    perspectives = [
                        PersonaPerspective(**p) for p in data["perspectives"]
                    ]
                    data["perspectives"] = perspectives
                    return Inquiry(**data)
            
            return None
        except Exception as e:
            print(f"Error loading inquiry {inquiry_id}: {e}")
            return None


# Global singleton instance
_stateful_engine = None

def get_stateful_engine() -> StatefulInquiryEngine:
    """Get or create the global StatefulInquiryEngine instance"""
    global _stateful_engine
    if _stateful_engine is None:
        _stateful_engine = StatefulInquiryEngine()
    return _stateful_engine
