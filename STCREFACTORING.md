# Refactoring Plan: Shifting "Generate Summary" to Creative Orientation

## 1. Current State of `generate_summary`

The current `generate_summary` tool provides an overview of the thinking process primarily focused on the progression through predefined problem-solving-orientation stages. It does not support at all the creative-orientation.

**Current Functionality:**
- Counts total thoughts.
- Categorizes thoughts by `ThoughtStage` (Problem Definition, Research, Analysis, Synthesis, Conclusion).
- Provides a timeline of thought numbers and their associated stages.
- Calculates a simple completion percentage based on total expected thoughts.

**Example Output:**

```json
{
  "summary": {
    "totalThoughts": 5,
    "stages": {
      "Problem Definition": 1,
      "Research": 1,
      "Analysis": 1,
      "Synthesis": 1,
      "Conclusion": 1
    },
    "timeline": [
      {"number": 1, "stage": "Problem Definition"},
      {"number": 2, "stage": "Research"},
      {"number": 3, "stage": "Analysis"},
      {"number": 4, "stage": "Synthesis"},
      {"number": 5, "stage": "Conclusion"}
    ]
  }
}
```

## 2. Problem with Current Orientation

The current `generate_summary` output, while useful for tracking linear problem-solving, is misaligned with the core principles of the "Creative Orientation Engine." Its focus on "Problem Definition" and "Conclusion" reinforces a reactive, problem-elimination mindset rather than an outcome-driven orientation with end-user's goal driven focus.

**Key Misalignments:**
- **Reactive Bias**: The stage names themselves (e.g., "Problem Definition") inherently frame the process around problems, perpetuating the very bias we aim to overcome.
- **Lack of Structural Tension Focus**: The summary does not provide insights into the identification, analysis, or resolution of structural tension, which is central to creative orientation.
- **Absence of Outcome-Centric Metrics**: There are no metrics reflecting progress towards a desired outcome, the generation of action steps, or the manifestation of new realities.
- **Limited Insight into Generative Process**: The summary merely tracks stages, not the qualitative aspects of creative advancement or the mitigation of reactive patterns.

## 3. Proposed Refactoring - New Orientation

The refactored `generate_summary` should provide a summary that actively supports and reflects the principles of creative orientation, structural tension, and outcome manifestation.

**Core Shifts:**

### a. Shift from "Problem-Solving Stages" to "Creative Process Elements"

The `ThoughtStage` enum and its usage throughout the system (including the summary) will need to be re-evaluated and potentially redefined. Instead of problem-centric stages, consider elements that drive outcome creation:

-   **Desired Outcome Clarification**: Thoughts related to defining and refining the desired future state.
-   **Current Reality Assessment**: Thoughts focused on objectively understanding the present situation relative to the desired outcome.
-   **Structural Tension Identification**: Thoughts that explicitly identify the gap and inherent tension between desired outcome and current reality.
-   **Action Step Generation**: Thoughts detailing concrete steps to resolve structural tension and move towards the desired outcome.
-   **Advancement/Progress Tracking**: Thoughts reflecting the execution of action steps and the resulting shift in current reality.
-   **Bias Detection/Mitigation**: Thoughts where reactive biases are identified, acknowledged, and consciously reoriented towards creative advancement.
-   **Pattern Recognition**: Insights into whether the thinking is generating "advancing patterns" or "oscillating patterns". It would self-correct.
-   **Extra-wording cleanup**: It would recognize the useless adding on of words that dont add any values to the end-user when communicating (making it too great kind of patterns.)

### b. Focus on Structural Tension

The summary should prominently feature the state of structural tension. This could include:

-   **Identified Structural Tensions**: A list or count of distinct structural tensions being addressed.
-   **Tension Resolution Progress**: Metrics indicating how much tension has been resolved or is actively being worked on.
-   **Impact of Action Steps**: How action steps are contributing to the resolution of tension.

### c. Evidence of Creative Advancement

Beyond simple thought counts, the summary should provide qualitative and quantitative insights into the manifestation of desired outcomes:

-   **Action Steps Generated/Completed**: Number of concrete action steps identified and marked as complete.
-   **Bias Reorientation Instances**: Count or percentage of thoughts where a reactive bias was identified and reoriented.
-   **Outcome Manifestation Indicators**: (More advanced) Metrics or qualitative descriptions of how the desired outcome is beginning to manifest in reality.
-   **Pattern Analysis**: A high-level indication of whether the overall thinking pattern is advancing or oscillating.

### d. Hierarchy of Thinking (Future Consideration)

If the system supports "telescoping" action steps into sub-charts, the summary could reflect this hierarchy, showing progress at different levels of detail.

## 4. Example of New Output (Draft)

```json
{
  "creativeProcessSummary": {
    "desiredOutcome": "Manifest a truly creative-oriented AI system.",
    "currentRealitySnapshot": "AI systems exhibit pervasive reactive bias.",
    "structuralTensionStatus": {
      "identifiedTensions": 3,
      "activeTensions": 1,
      "tensionResolutionProgress": "30% resolved"
    },
    "creativeElementsBreakdown": {
      "desiredOutcomeClarification": 5,
      "currentRealityAssessment": 8,
      "structuralTensionIdentification": 3,
      "actionStepGeneration": 12,
      "advancementTracking": 4,
      "biasReorientationInstances": 7
    },
    "actionSteps": {
      "total": 12,
      "completed": 3,
      "nextSteps": ["Develop bias detection metrics", "Curate creative-oriented dataset"]
    },
    "patternAnalysis": "Overall advancing pattern, with occasional reactive oscillations.",
    "progressTowardsOutcome": "Initial conceptualization complete, foundational research underway."
  }
}
```

## 5. Implementation Considerations

Implementing these changes will require modifications across several components:

-   **`mcp_coaia_sequential_thinking/models.py`**:
    -   Redefine `ThoughtStage` enum to reflect creative process elements (e.g., `DesiredOutcome`, `CurrentReality`, `StructuralTension`, `ActionStep`).
    -   Potentially add new fields to `ThoughtData` to capture specific metrics related to structural tension, action step completion, or bias reorientation.
-   **`mcp_coaia_sequential_thinking/analysis.py`**:
    -   The `ThoughtAnalyzer` will need significant updates to interpret thoughts based on the new creative process elements.
    -   New analysis methods will be required to identify structural tension, track action step progress, and detect/quantify bias reorientation.
-   **`mcp_coaia_sequential_thinking/server.py`**:
    -   The `process_thought` tool will need to adapt to the new `ThoughtStage` enum and any new `ThoughtData` fields.
    -   The `generate_summary` tool will call the updated `ThoughtAnalyzer` methods to produce the new summary format.
-   **`mcp_coaia_sequential_thinking/storage.py`**:
    -   Ensure compatibility with any changes to `ThoughtData` for persistence.
-   **MCP Tool Definition**: The `generate_summary` tool's output schema will need to be updated in its definition for MCP clients.

This refactoring represents a fundamental shift in the engine's core logic, moving it from a generic sequential thinking tool to a specialized "Creative Orientation Engine."
