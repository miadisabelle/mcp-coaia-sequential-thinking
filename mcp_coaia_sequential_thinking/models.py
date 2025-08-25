from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, field_validator


class ThoughtStage(Enum):
    """SCCP-based creative orientation stages following Robert Fritz methodology."""
    DESIRED_OUTCOME = "Desired Outcome"
    CURRENT_REALITY = "Current Reality"
    ACTION_STEPS = "Action Steps"
    PATTERN_RECOGNITION = "Pattern Recognition"
    CONCEPT_DETECTION = "Concept Detection"

    @classmethod
    def from_string(cls, value: str) -> 'ThoughtStage':
        """Convert a string to a thinking stage.

        Args:
            value: The string representation of the thinking stage

        Returns:
            ThoughtStage: The corresponding ThoughtStage enum value

        Raises:
            ValueError: If the string does not match any valid thinking stage
        """
        # Case-insensitive comparison
        for stage in cls:
            if stage.value.casefold() == value.casefold():
                return stage

        # If no match found
        valid_stages = ", ".join(stage.value for stage in cls)
        raise ValueError(f"Invalid thinking stage: '{value}'. Valid stages are: {valid_stages}")


class ThoughtData(BaseModel):
    """Data structure for a single thought in the sequential thinking process with SCCP elements."""
    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool
    stage: ThoughtStage
    tags: List[str] = Field(default_factory=list)
    axioms_used: List[str] = Field(default_factory=list)
    assumptions_challenged: List[str] = Field(default_factory=list)
    # SCCP-based fields for pattern recognition
    pattern_type: Optional[str] = Field(default=None, description="advancing or oscillating pattern")
    structural_tension_strength: Optional[float] = Field(default=None, description="0.0-1.0 measure of tension clarity")
    hidden_concepts_detected: List[str] = Field(default_factory=list, description="limiting concepts identified")
    action_step_strategic: Optional[bool] = Field(default=None, description="whether action is strategic vs reactive")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    id: UUID = Field(default_factory=uuid4)

    def __hash__(self):
        """Make ThoughtData hashable based on its ID."""
        return hash(self.id)

    def __eq__(self, other):
        """Compare ThoughtData objects based on their ID."""
        if not isinstance(other, ThoughtData):
            return False
        return self.id == other.id

    @field_validator('thought')
    def thought_not_empty(cls, v: str) -> str:
        """Validate that thought content is not empty."""
        if not v or not v.strip():
            raise ValueError("Thought content cannot be empty")
        return v

    @field_validator('thought_number')
    def thought_number_positive(cls, v: int) -> int:
        """Validate that thought number is positive."""
        if v < 1:
            raise ValueError("Thought number must be positive")
        return v

    @field_validator('total_thoughts')
    def total_thoughts_valid(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate that total thoughts is valid."""
        thought_number = values.data.get('thought_number')
        if thought_number is not None and v < thought_number:
            raise ValueError("Total thoughts must be greater or equal to current thought number")
        return v
    
    @field_validator('pattern_type')
    def pattern_type_valid(cls, v: Optional[str]) -> Optional[str]:
        """Validate pattern type is either advancing or oscillating."""
        if v is not None and v not in ["advancing", "oscillating"]:
            raise ValueError("Pattern type must be either 'advancing' or 'oscillating'")
        return v
    
    @field_validator('structural_tension_strength')
    def structural_tension_strength_valid(cls, v: Optional[float]) -> Optional[float]:
        """Validate structural tension strength is between 0.0 and 1.0."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("Structural tension strength must be between 0.0 and 1.0")
        return v

    def validate(self) -> bool:
        """Legacy validation method for backward compatibility.

        Returns:
            bool: True if the thought data is valid

        Raises:
            ValueError: If any validation checks fail
        """
        # Validation is now handled by Pydantic automatically
        return True

    def to_dict(self, include_id: bool = False) -> dict:
        """Convert the thought data to a dictionary representation.

        Args:
            include_id: Whether to include the ID in the dictionary representation.
                        Default is False to maintain compatibility with tests.

        Returns:
            dict: Dictionary representation of the thought data
        """
        from .utils import to_camel_case

        # Get all model fields, excluding internal properties
        data = self.model_dump()
        
        # Handle special conversions
        data["stage"] = self.stage.value
        
        if not include_id:
            # Remove ID for external representations
            data.pop("id", None)
        else:
            # Convert ID to string for JSON serialization
            data["id"] = str(data["id"])
        
        # Convert snake_case keys to camelCase for API consistency
        result = {}
        for key, value in data.items():
            if key == "stage":
                # Stage is already handled above
                continue
                
            camel_key = to_camel_case(key)
            result[camel_key] = value
        
        # Ensure these fields are always present with camelCase naming
        result["thought"] = self.thought
        result["thoughtNumber"] = self.thought_number
        result["totalThoughts"] = self.total_thoughts
        result["nextThoughtNeeded"] = self.next_thought_needed
        result["stage"] = self.stage.value
        result["tags"] = self.tags
        result["axiomsUsed"] = self.axioms_used
        result["assumptionsChallenged"] = self.assumptions_challenged
        # SCCP fields with camelCase naming
        result["patternType"] = self.pattern_type
        result["structuralTensionStrength"] = self.structural_tension_strength
        result["hiddenConceptsDetected"] = self.hidden_concepts_detected
        result["actionStepStrategic"] = self.action_step_strategic
        result["timestamp"] = self.timestamp
        
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'ThoughtData':
        """Create a ThoughtData instance from a dictionary.

        Args:
            data: Dictionary containing thought data

        Returns:
            ThoughtData: A new ThoughtData instance
        """
        from .utils import to_snake_case
        
        # Convert any camelCase keys to snake_case
        snake_data = {}
        mappings = {
            "thoughtNumber": "thought_number",
            "totalThoughts": "total_thoughts",
            "nextThoughtNeeded": "next_thought_needed",
            "axiomsUsed": "axioms_used",
            "assumptionsChallenged": "assumptions_challenged",
            # SCCP field mappings
            "patternType": "pattern_type",
            "structuralTensionStrength": "structural_tension_strength",
            "hiddenConceptsDetected": "hidden_concepts_detected",
            "actionStepStrategic": "action_step_strategic"
        }
        
        # Process known direct mappings
        for camel_key, snake_key in mappings.items():
            if camel_key in data:
                snake_data[snake_key] = data[camel_key]
        
        # Copy fields that don't need conversion
        for key in ["thought", "tags", "timestamp"]:
            if key in data:
                snake_data[key] = data[key]
                
        # Handle special fields
        if "stage" in data:
            snake_data["stage"] = ThoughtStage.from_string(data["stage"])
            
        # Set default values for missing fields
        snake_data.setdefault("tags", [])
        snake_data.setdefault("axioms_used", data.get("axiomsUsed", []))
        snake_data.setdefault("assumptions_challenged", data.get("assumptionsChallenged", []))
        # SCCP field defaults
        snake_data.setdefault("pattern_type", data.get("patternType"))
        snake_data.setdefault("structural_tension_strength", data.get("structuralTensionStrength"))
        snake_data.setdefault("hidden_concepts_detected", data.get("hiddenConceptsDetected", []))
        snake_data.setdefault("action_step_strategic", data.get("actionStepStrategic"))
        snake_data.setdefault("timestamp", datetime.now().isoformat())

        # Add ID if present, otherwise generate a new one
        if "id" in data:
            try:
                snake_data["id"] = UUID(data["id"])
            except (ValueError, TypeError):
                snake_data["id"] = uuid4()

        return cls(**snake_data)

    model_config = {
        "arbitrary_types_allowed": True
    }
