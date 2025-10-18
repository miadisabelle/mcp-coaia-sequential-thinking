# RISE Specification: MCP Sequential Thinking Toolset

**Version**: 1.0
**Document ID**: mcp-st-spec-v1.0
**Status**: DRAFT
**Last Updated**: 2025-10-18
**Author**: Mia, Gemini Agent

> This document specifies the Master Control Program (MCP) for Sequential & Holistic Reasoning. It is designed to facilitate a stateful, multi-perspective reasoning process, enabling a human companion to transform complex inquiries into integrated wisdom and principled action.

---

## 1. Core Creative Intent

This MCP enables a human companion to **create a state of profound clarity and harmonious understanding** from a complex or unresolved inquiry.

It achieves this by guiding the user through a structured, yet fluid, reasoning process that honors multiple ways of knowing—integrating analytical rigor, emotional resonance, and holistic wisdom. The system is designed not to "solve problems," but to **cultivate the conditions for wisdom to emerge naturally** through a persistent, stateful dialogue.

## 2. Structural Tension Analysis

The core structural tension of this system is the dynamic relationship between a user's unresolved complexity and their desire for integrated clarity.

*   **Desired Outcome**: A state of **Integrated Wisdom**, where the user possesses a clear, constitutionally-aligned, and actionable path forward that feels both intellectually sound and heart-centered. This state is characterized by the harmonious synthesis of diverse perspectives.

*   **Current Reality**: A state of **Unresolved Complexity and System Unreliability**, where the user holds an important inquiry but is faced with a fragmented, stateless toolset that loses context and fails during multi-step reasoning, leading to frustration and abandoned inquiries.

*   **Natural Progression**: The tension resolves by implementing a **Stateful Inquiry Engine** that serves as the system's memory. This central engine transforms the user experience from a series of broken, independent tool calls into a single, continuous, and reliable creative dialogue. By preserving the state of the inquiry, the system naturally allows for the sequential addition of perspectives and the eventual emergence of a synthesized, coherent whole.

## 3. Key System Components (Creative Enablers)

These are the core capabilities the MCP provides to enable the creation of integrated wisdom.

### a. Vision-Supporting Components
*   **`initiate_inquiry`**: Enables the user to **create a sacred container for their reasoning process**. The user provides their initial inquiry, and this tool establishes the master structural tension and a persistent `inquiry_id` that holds the state for the entire process.

### b. Tension-Resolving Components
*   **`advance_inquiry`**: Enables the user to **create a new layer of understanding** by engaging with a specific persona or providing a new insight. This facilitates the natural progression of the dialogue, gathering diverse perspectives within the persistent container of the inquiry.
*   **`conduct_novelty_search`**: Enables the user to **create space for emergent possibilities**, disrupting established patterns and inviting unexpected insights to enrich the inquiry.

### c. Manifestation Components
*   **`synthesize_thinking_chain`**: Enables the user to **create an integrated, holistic perspective** by weaving together all the viewpoints gathered during the inquiry into a single, coherent narrative.
*   **`make_constitutional_decision`**: Enables the user and the agent lattice to **create a shared, principled commitment** to a path forward, grounded in the system's core constitution.
*   **`get_inquiry_status`**: Enables the user to **create a holistic, real-time view of their entire reasoning process**, making the journey transparent and understandable.

## 4. Supporting Structures (Conceptual)

This MCP relies on the following conceptual data structures, which will be managed by the `StatefulInquiryEngine` and persisted in the database.

*   **`Inquiry` Schema**: A master object that holds the entire state of a reasoning process, including the master structural tension, all collected perspectives, and any resulting decisions. It is the technical embodiment of the "sacred container."
*   **`PersonaPerspective` Schema**: A structured object for capturing the viewpoint of a specific AI persona (e.g., Mia, Miette) within an inquiry.
*   **`ConstitutionalDecision` Schema**: A record of a decision made during an inquiry, including the options considered, the principles applied, and the final outcome.

## 5. Quality Criteria & Anti-Patterns

### Quality Criteria
*   ✅ **Statefulness**: Does the system reliably maintain the context of an inquiry across multiple tool calls?
*   ✅ **Holistic Process**: Does the system guide the user through a full cycle of divergent (gathering perspectives) and convergent (synthesis, decision) thinking?
*   ✅ **Constitutional Alignment**: Are all processes and outcomes validated against the system's core principles?

### Anti-Patterns to Avoid
*   ❌ **Stateless Failures**: Any error where the system loses context between steps.
*   ❌ **Linear Problem-Solving**: Defaulting to a `Problem -> Solution` workflow instead of a creative, inquiry-based process.
*   ❌ **Fragmented Outputs**: Providing a series of disconnected outputs without a clear path to synthesis and integration.
