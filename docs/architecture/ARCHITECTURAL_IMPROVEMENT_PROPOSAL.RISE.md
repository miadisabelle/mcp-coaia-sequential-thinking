# RISE Specification: MCP Stateful Reasoning Engine

**Version**: 1.0
**Document ID**: mcp-sre-spec-v1.0
**Status**: DRAFT
**Last Updated**: 2025-10-18
**Author**: Mia, Gemini Agent

> This document revises the `ARCHITECTURAL_IMPROVEMENT_PROPOSAL.md` using the RISE framework. It specifies a **Stateful Inquiry Engine** designed to resolve the critical state-persistence failures identified in the `mcp-coaia-sequential-thinking` toolset.

---

## 1. Core Creative Intent

This refactoring enables a human companion or AI agent to **create a reliable, continuous, and structurally sound reasoning process**. 

It transforms the current fragmented, stateless toolset into a coherent, stateful engine. This allows users to build complex, multi-step inquiries that persist over time, integrate multiple perspectives, and result in integrated wisdom, rather than experiencing frustrating failures due to lost context.

## 2. Structural Tension Analysis

The core structural tension is the dynamic between a fragmented, unreliable system and the vision of a seamless, creative reasoning partner.

*   **Desired Outcome**: A **Stateful, Resilient Reasoning Engine** that reliably maintains the context of a creative inquiry across multiple tool calls and sessions. The system functions as a continuous creative companion, remembering all steps of a thinking process and ensuring that each new action builds upon the last.

*   **Current Reality**: A **Stateless, Brittle Toolset** where each tool call is an isolated event. Critical information, such as `decision_id`s and `session_id`s, is lost between calls, leading to predictable failures in multi-step scenarios. The system is functionally unreliable for any process that requires memory.

*   **Natural Progression**: The tension resolves by introducing a **central, stateful engine** that is responsible for managing the lifecycle of an inquiry. By making all tools interact with this persistent engine, the system naturally moves from a state of fragmentation to one of coherence. The engine acts as the system's memory, ensuring that the structural integrity of the reasoning process is maintained automatically.

## 3. Key System Components (Creative Enablers)

These are the core capabilities the refactored system will provide to enable the creation of continuous reasoning.

### a. Vision-Supporting Components
*   **`initiate_inquiry` (New Tool)**: Enables the user to **create a sacred container for a new reasoning process**. It replaces the multiple, confusing `initiate_*` tools with a single, clear entry point. It establishes the master structural tension for the inquiry and returns a persistent `inquiry_id`.

### b. Tension-Resolving Components
*   **`StatefulInquiryEngine` (New Core Component)**: Enables the system to **create continuity and memory**. This singleton class will manage all active inquiries, loading their state from the database at the beginning of a tool call and saving it at the end. It is the central nervous system of the refactored architecture.
*   **`advance_inquiry` (Refactored Tool)**: Enables the user to **create a new layer of understanding** within an existing inquiry. It takes the `inquiry_id` and a new insight, and the engine handles the state management, ensuring the new insight is integrated, not lost.

### c. Manifestation Components
*   **`get_inquiry_status` (Refactored Tool)**: Enables the user to **create a holistic view of their reasoning process**. It retrieves the entire state of an inquiry—including all perspectives, decisions, and tasks—from the database, providing a complete picture of the creative journey.
*   **Stateful `ConsensusDecisionEngine`**: Enables the system to **create reliable, auditable decisions**. By refactoring this engine to use `data_persistence.py`, it ensures that decisions are not lost, and tools like `get_constitutional_audit_trail` will function as intended.

## 4. Creative Advancement Scenarios

### Scenario 1: A Reliable Multi-Persona Analysis

*   **Desired Outcome**: The user wants to create an integrated strategy by analyzing a topic from multiple AI persona perspectives.
*   **Current Reality**: The user attempts a multi-step analysis, but the system fails on the second or third step because it has forgotten the context of the first (`AttributeError` or `decision_id not found`).
*   **Natural Progression**:
    1.  The user calls `initiate_inquiry` to **create a container** for their analysis, receiving a persistent `inquiry_id`.
    2.  The user calls `advance_inquiry` with the `inquiry_id` to get Mia's perspective. The `StatefulInquiryEngine` loads the inquiry, adds the perspective, and saves the new state to the database.
    3.  The user calls `advance_inquiry` again for Miette's perspective. The engine loads the now-updated state, adds the second perspective, and saves again.
    4.  The process continues reliably for all personas.
*   **Resolution**: The user has successfully created a rich, multi-perspective analysis without any state-related failures, achieving their desired outcome of integrated wisdom.

## 5. Implementation Plan (Supporting Structures)

This section outlines the concrete steps to manifest the desired stateful architecture.

1.  **Create `mcp_coaia_sequential_thinking/inquiry_engine.py`**:
    *   Define the `StatefulInquiryEngine` class as a singleton.
    *   Implement methods like `initiate_inquiry`, `advance_inquiry`, `get_inquiry_status`.
    *   Ensure every method that modifies state interacts with the `data_store` from `data_persistence.py`.

2.  **Refactor `mcp_coaia_sequential_thinking/server.py`**:
    *   Instantiate a single, global `StatefulInquiryEngine`.
    *   Rewrite the tool functions to be thin wrappers that delegate logic to the engine.

3.  **Modify `mcp_coaia_sequential_thinking/consensus_decision_engine.py`**:
    *   Remove all in-memory dictionaries for state.
    *   Refactor all methods to use `data_store` for reading and writing decision objects.

4.  **Enhance `mcp_coaia_sequential_thinking/constitutional_core.py`**:
    *   Modify `generate_active_pause_drafts` to make a recursive call to the LLM to generate substantive content for each draft, thereby making the tool functional.

## 6. Quality Criteria & Anti-Patterns

### Quality Criteria
*   ✅ **State Persistence**: Does the system reliably maintain the state of an inquiry across multiple, independent tool calls?
*   ✅ **Architectural Coherence**: Is the logic for managing an inquiry encapsulated within a single, stateful engine?
*   ✅ **Tool Reliability**: Do previously failing tools like `get_constitutional_audit_trail` now function correctly?

### Anti-Patterns to Avoid
*   ❌ **In-Memory State**: Storing any long-term state in instance variables or global dictionaries.
*   ❌ **Stateless Tool Logic**: Placing complex business logic inside the tool functions in `server.py` instead of in the stateful engine.
*   ❌ **Placeholder Content**: Returning non-functional, placeholder data from any tool.
