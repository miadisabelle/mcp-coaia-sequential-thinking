# Specification: CoLintIntegration Module
> Applying the RISE Framework

This document provides the creative and technical specification for the `CoLintIntegration` module, designed to integrate real-time creative orientation validation into the Sequential Thinking System.

## Phase 1: Reverse-Engineering & Intent Extraction

### Core Creative Intent
The `CoLintIntegration` module is designed to enable the creation of a **structurally sound and compliance-aware thinking process**. Its purpose is not to "fix" or "prevent" violations, but to generate awareness, enabling the user and the system to consciously and naturally advance toward a state of linguistic alignment with creative orientation principles.

It enables the creation of:
- **Awareness:** Makes the user and system aware of linguistic patterns that may be oscillating or reactive.
- **Distinction:** Allows the system to distinguish between thoughts and summaries that are structurally aligned for chart creation and those that are not.
- **Advancement Potential:** Creates the foundation for a future coaching system that can provide targeted, context-aware guidance.

### Structural Tension Analysis
- **Desired Outcome:** A seamless, non-disruptive creative process where the user's thinking and the system's outputs consistently align with creative orientation, leading to more effective and powerful structural tension charts.
- **Current Reality:** A user's thinking process and the AI's outputs may contain ingrained reactive or problem-solving language, which undermines the effectiveness of the creative process. The `co-lint` tool exists but is separate and not integrated, providing no real-time awareness within the thinking flow.
- **Natural Progression:** The tension between the desire for a pure creative process and the reality of ingrained reactive language is resolved by introducing `CoLintIntegration`. This module provides real-time, non-disruptive awareness by attaching metadata to the thought process. This awareness naturally guides the user and system toward using more aligned language, advancing the quality of the entire creative sequence without interrupting the flow.

---

## Phase 2: Intent Refinement & Specification

### Creative Advancement Scenarios

#### Scenario 1: Awareness Creation During Thought Capture
- **Desired Outcome**: A user wants to capture their "Current Reality" with neutral, objective observation.
- **Current Reality**: The user, operating from habit, submits a thought containing problem-solving language: *"The problem is that my project is completely disorganized, and I need to fix this issue."*
- **Natural Progression**:
    1. The `process_thought` tool receives the text.
    2. `CoLintIntegration.validate_thought_content()` is invoked. It does not block the process.
    3. It analyzes the text and identifies violations of neutral observation (e.g., "problem," "fix this issue").
    4. This violation data is attached to the thought object as metadata.
- **Resolution**: The user's thought is successfully captured, preserving their creative flow. The system now possesses an awareness that this component of the structure is misaligned. This enables a future action (like targeted coaching or intelligent summary refinement) to be more effective, creating an advancing pattern toward structural integrity.

#### Scenario 2: Quality Gating for Chart Creation
- **Desired Outcome**: To create a valid and potent Structural Tension Chart from a completed thinking sequence.
- **Current Reality**: A thinking sequence is complete, and the `generate_summary` tool has produced a summary. This summary may contain subtle reactive language inherited from the user's thoughts or the AI's own structural bias.
- **Natural Progression**:
    1. `CoLintIntegration.validate_summary_content()` is invoked on the generated summary.
    2. A `compliance_score` is calculated based on the density and severity of any detected violations.
    3. The `Integration Bridge` component programmatically checks this score.
- **Resolution**: If the score is high, the system proceeds to create the chart, having validated its linguistic and structural integrity. If the score is low, the system chooses a different advancing pattern, such as initiating a "summary refinement" loop, thus preventing the creation of a structurally flawed chart that would likely lead to oscillation.

### Supporting Structures (Feature Inventory)
- **`validate_thought_content(thought: str, stage: str)`**: Enables the creation of a **compliance-aware thought stream** by attaching validation metadata to each thought.
- **`validate_summary_content(summary: str)`**: Enables the creation of **structurally-validated summaries** by assessing the final output of a thinking sequence.
- **`compliance_score` (Output)**: Enables the creation of programmatic **quality gates** for advancing the creative process to subsequent stages like chart creation.
- **`violations` (Output)**: Enables the creation of **targeted, context-aware coaching opportunities** in future system phases.

---

## Phase 3: Export Optimization (Technical Specification)

### Technical Documentation Export

#### **Module**: `mcp_coaia_sequential_thinking.colint_integration`
#### **Class**: `CoLintIntegration`

This class encapsulates the logic for integrating `co-lint` functionality into the Sequential Thinking pipeline.

##### **`__init__(self)`**
- **Creative Intent**: To prepare the integration by ensuring `co-lint` is accessible.
- **Structural Intent**: Establishes the connection to the `co_lint` module. Handles import errors gracefully to prevent system failure if `co-lint` is not present, ensuring resilience.

##### **`validate_thought_content(self, thought_content: str, stage: str) -> dict`**
- **Creative Intent**: To enable awareness of linguistic alignment for a single piece of thought content.
- **Parameters**:
    - `thought_content` (str): The raw text of the thought.
    - `stage` (str): The current SCCP stage (e.g., "Desired Outcome," "Current Reality") to enable future context-specific validation.
- **Returns** (dict): A dictionary containing:
    - `violations` (list): A list of violation dictionaries, each detailing the rule and message.
    - `has_violations` (bool): A boolean for quick programmatic checks.
    - `suggestions` (list): Placeholder for future coaching (Phase 3).
    - `coaching` (list): Placeholder for future reframing options (Phase 3).

##### **`validate_summary_content(self, summary_content: str) -> dict`**
- **Creative Intent**: To enable a holistic quality assessment of a completed thinking sequence's output.
- **Parameters**:
    - `summary_content` (str): The full text of the generated summary.
- **Returns** (dict): A dictionary containing:
    - `violations` (list): A list of violation dictionaries.
    - `has_violations` (bool): A boolean for quick programmatic checks.
    - `compliance_score` (float): A score from 0.0 to 1.0 representing the degree of alignment with creative orientation principles. Enables quality gating.
    - `suggestions` (list): Placeholder for future coaching (Phase 3).
