# Architectural Improvement Proposal: MCP State & Reasoning Engine

**Document ID:** MCP-AIP-251018
**Author:** Mia, Gemini Agent
**Status:** DRAFT

## 1. Executive Summary

This document proposes a series of architectural enhancements to the `mcp-coaia-sequential-thinking` toolset. Analysis of four test scenarios and a review of the related `IAIP/rispecs` specifications have revealed critical architectural weaknesses, primarily the lack of state persistence between tool calls. This leads to significant failures in multi-step reasoning processes.

The proposed solution is to implement a **Stateful Inquiry Engine** that encapsulates the logic for sequential thinking and consensus decisions, using the existing `data_persistence.py` module to ensure continuity. This refactoring will fix the identified bugs and align the MCP more closely with the robust, relational principles outlined in the IAIP architecture.

## 2. Problem Analysis & Supporting Evidence

Execution of Scenarios 1 through 4 has highlighted a recurring structural flaw:

*   **State Management Failure:** Scenario 3 demonstrated a critical failure where `get_constitutional_audit_trail` could not find a `decision_id` created in a previous step. This is because the `ConsensusDecisionEngine` is stateless, using an in-memory dictionary that is lost between tool calls.
*   **Environment Brittleness:** Scenario 1 showed that the system is highly sensitive to environment configuration, with no mechanism to self-diagnose or report inconsistencies between the running code and the local source.
*   **Incomplete Implementations:** Scenario 4 revealed that key tools like `generate_active_pause_drafts` are functionally incomplete, returning placeholder data that undermines their utility.

These issues stem from a fundamentally stateless, tool-centric architecture. The `IAIP/rispecs` documents, in contrast, describe a stateful, **session-centric** or **inquiry-centric** architecture where a "Continuous Companion" maintains context over time.

## 3. Proposed Architectural Solution: The Stateful Inquiry Engine

I propose refactoring the core logic currently distributed across `server.py` into a new, stateful class: `StatefulInquiryEngine`. This engine will be instantiated **once** as a global singleton and will manage the lifecycle of all creative reasoning processes.

### 3.1. Core Responsibilities of the `StatefulInquiryEngine`

1.  **Inquiry Lifecycle Management:** It will manage a dictionary of active "inquiries" (currently analogous to `thinking_chains` or `emergence_sessions`). Each inquiry will have a unique ID and a defined state (e.g., `GATHERING_PERSPECTIVES`, `SYNTHESIZING`, `AWAITING_CONSENSUS`).
2.  **State Persistence:** All state changes (new perspectives, new decisions, synthesis results) will be immediately written to the SQLite database via `data_persistence.py`. When a tool is called with an `inquiry_id`, the engine will first load the inquiry's state from the database.
3.  **Encapsulation of Logic:** The business logic currently in the tool functions within `server.py` (e.g., the orchestration of advancing a thinking chain) will be moved into methods within the `StatefulInquiryEngine`.

### 3.2. Proposed Refactoring of Key Tools

The existing tools will be refactored to be thin wrappers around the new stateful engine.

**New Tool: `initiate_inquiry`**
*   **Replaces:** `initiate_creative_emergence` and `initiate_sequential_thinking`.
*   **Function:** Creates a new `Inquiry` object in the engine, persists it to the database, and returns an `inquiry_id`.
*   **Benefit:** Provides a single, unified entry point for any reasoning process, as inspired by `initiateHolisticInquiry` from the rispecs.

**Refactored Tool: `advance_inquiry`**
*   **Replaces:** `advance_thinking_chain`.
*   **Function:** Takes an `inquiry_id` and a new piece of information (e.g., a persona perspective). The engine loads the inquiry, applies the new information, saves the new state, and returns the result.
*   **Benefit:** Ensures that each step in the reasoning process builds upon a persisted state.

**Refactored Tool: `get_inquiry_status`**
*   **Replaces:** `get_thinking_chain_status`, `get_consensus_decision_status`, `get_task_status`.
*   **Function:** Takes an `inquiry_id` and returns a comprehensive status object including all associated perspectives, decisions, and tasks from the database.
*   **Benefit:** Provides a single, unified way to query the state of any reasoning process.

**Refactored `ConsensusDecisionEngine`**
*   The `ConsensusDecisionEngine` will be modified to use `data_persistence.py` for all `active_decisions` and `decision_history`. The `make_constitutional_decision` tool will now correctly persist the decision, and `get_constitutional_audit_trail` will be able to retrieve it.

### 3.3. Addressing Incomplete Tools

*   **`generate_active_pause_drafts`:** This tool's implementation in `constitutional_core.py` should be modified. Instead of returning placeholder strings, it should make a recursive call to the LLM, providing a meta-prompt for each risk profile (e.g., "Generate a conservative, low-risk response to the following context..."). This will produce substantive content for constitutional evaluation.

## 4. Implementation Steps

1.  **Create `mcp_coaia_sequential_thinking/inquiry_engine.py`:**
    *   Define the `StatefulInquiryEngine` class.
    *   Implement methods for `initiate_inquiry`, `advance_inquiry`, `get_inquiry_status`, etc.
    *   Integrate calls to `data_store` for all state modifications.

2.  **Refactor `mcp_coaia_sequential_thinking/server.py`:**
    *   Instantiate a single, global instance of the `StatefulInquiryEngine`.
    *   Rewrite the bodies of the existing tool functions to be simple calls to the engine's methods.
    *   Deprecate the separate `initiate_*` tools in favor of a single `initiate_inquiry`.

3.  **Modify `mcp_coaia_sequential_thinking/consensus_decision_engine.py`:**
    *   Remove the in-memory `self.active_decisions` dictionary.
    *   Modify all methods to read from and write to the `data_store` for consensus decisions.

4.  **Enhance `mcp_coaia_sequential_thinking/constitutional_core.py`:**
    *   Update `generate_active_pause_drafts` to call a generative model to create real content for the drafts.

## 5. Expected Impact

*   **Reliability:** The critical state-management failures will be eliminated, making multi-step reasoning processes reliable.
*   **Simplicity:** The tool surface will be simplified, with a more intuitive, inquiry-centric flow.
*   **Robustness:** The system will be more aligned with the proven architectural patterns of the IAIP project, making it more robust and scalable.
*   **Functionality:** Incomplete tools will become fully functional, unlocking their intended value.
