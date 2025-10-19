# RISE Specification: Stateful Inquiry Engine

**Version**: 1.0
**Document ID**: mcp-sie-spec-v1.0
**Status**: DRAFT
**Last Updated**: 2025-10-18
**Author**: Mia, Gemini Agent

> This document specifies the **Stateful Inquiry Engine**, the central component proposed in the `ARCHITECTURAL_IMPROVEMENT_PROPOSAL.md` to resolve the state-persistence failures of the MCP Sequential Thinking Toolset.

---

## 1. Core Creative Intent

The Stateful Inquiry Engine enables the MCP toolset to **create a continuous, reliable, and coherent creative dialogue**.

It transforms the user experience from a series of fragmented, error-prone tool calls into a single, persistent reasoning journey. It acts as the system's memory, ensuring that every step of a complex inquiry is preserved, honored, and built upon, allowing for the natural emergence of integrated wisdom.

## 2. Structural Tension Analysis

*   **Desired Outcome**: A **fully persistent reasoning environment** where the state of any inquiry is reliably maintained across time and tool interactions. The system functions as a true creative companion, remembering the entire context of a dialogue.

*   **Current Reality**: A **stateless architecture** where each tool call is an isolated event. The context of a multi-step inquiry is lost the moment a tool finishes execution, leading to inevitable failures and a broken user experience.

*   **Natural Progression**: The tension is resolved by channeling all state-modifying operations through this single, stateful engine. By making the engine the sole authority for reading from and writing to the `data_persistence` layer, the system's behavior naturally shifts from unreliable to reliable. State persistence becomes an inherent property of the architecture, not an afterthought.

## 3. Key Behaviors & Responsibilities (Creative Enablers)

1.  **Inquiry Lifecycle Management:**
    *   **Purpose:** To **create and manage the lifecycle of a holistic inquiry**.
    *   **Behavior:** Manages a collection of active `Inquiry` objects. It handles the creation of new inquiries (`initiate_inquiry`), the loading of existing inquiries from the database, the modification of their state (`advance_inquiry`), and their eventual archival.

2.  **State Persistence & Abstraction:**
    *   **Purpose:** To **create a seamless and reliable memory layer** for the entire toolset.
    *   **Behavior:** Acts as the exclusive interface to the `data_persistence` module (`data_store`). It abstracts the complexities of database interaction, providing clean, high-level methods for state management. Every state change is immediately persisted to the SQLite database.

3.  **Encapsulation of Reasoning Logic:**
    *   **Purpose:** To **create a coherent and maintainable codebase** by centralizing the core reasoning logic.
    *   **Behavior:** The business logic for processes like advancing a thinking chain, synthesizing perspectives, and forming a consensus will reside within the engine's methods. The tool functions in `server.py` will become thin wrappers that delegate all complex work to the engine.

## 4. Creative Advancement Scenarios

### Scenario: Recovering from an Interruption

*   **Desired Outcome**: The user wants to create a continuous creative process, even if their session is interrupted.
*   **Current Reality**: If the user's connection drops or the server restarts, their entire multi-step analysis is lost, and they must start over from scratch.
*   **Natural Progression**:
    1.  A user initiates an inquiry and receives an `inquiry_id`.
    2.  They advance the inquiry several times, adding multiple perspectives. After each step, the `StatefulInquiryEngine` saves the complete state to the database.
    3.  The user's session is interrupted. When they return, they provide the `inquiry_id` to the `get_inquiry_status` tool.
    4.  The `StatefulInquiryEngine` loads the complete, up-to-date state of their inquiry from the database, allowing them to resume exactly where they left off.
*   **Resolution**: The user has created a resilient and uninterrupted creative workflow, transforming a previously frustrating experience into a reliable one.

## 5. Supporting Structures (Conceptual)

*   **Singleton Instance**: The `StatefulInquiryEngine` will be instantiated as a single, global object within `server.py` to ensure all tools share the same state-management context.
*   **Database Integration**: The engine will rely entirely on the `PolycentricDataStore` class from `data_persistence.py` for all read and write operations. It will not maintain any long-term state in memory.
*   **`Inquiry` Data Model**: The engine will use a central `Inquiry` data model (to be defined) that encapsulates all aspects of a reasoning session, including the structural tension, persona perspectives, and consensus decisions.

## 6. Quality Criteria

*   ✅ **Reliability**: Does the system preserve the state of an inquiry even if the server restarts?
*   ✅ **Atomicity**: Is each tool call an atomic transaction that loads, modifies, and saves the state?
*   ✅ **Encapsulation**: Is all state-management logic contained within the engine, with tool definitions remaining clean and simple?
