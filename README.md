# CoAiA Sequential Thinking: Stateful Reasoning Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

An advanced MCP (Model Context Protocol) server implementing a **Stateful Inquiry Engine** that enables continuous, multi-perspective creative reasoning. The system transcends traditional problem-solving by guiding users through structural tension analysis, multi-persona collaboration (Mia 🧠, Miette 🌸, Haiku 🍃), and constitutional governance to manifest desired outcomes through creative orientation.

**🎯 Core Innovation**: Transforms fragmented, stateless AI interactions into a coherent, persistent reasoning journey where every insight builds upon the last—enabling true creative partnership between human and AI.

## System Status: ✅ Fully Operational & Experimentally Validated

### Recent Updates (2025-10-18)
- ✅ **Architectural Enhancement**: Comprehensive improvement proposals by Mia (see [Architecture Docs](docs/architecture/))
- ✅ **Documentation Consolidation**: Organized structure for clarity and maintainability
- ✅ **Experimental Validation**: All 4 scenarios successfully tested (see [Experimental Analysis](docs/analysis/ISSUE_12_EXPERIMENTAL_ANALYSIS.md))

### Core Systems
- **Stateful Inquiry Engine**: Persistent reasoning that survives tool calls and sessions
- **Multi-Persona Integration**: Mia 🧠 (rational), Miette 🌸 (emotional), Haiku 🍃 (wisdom)
- **Constitutional Governance**: Built-in principles preventing reactive decision-making
- **Creative Orientation**: Structural tension methodology over problem-solving bias
- **MCP Prompts & Resources**: Context-aware guidance for LLMs to think structurally

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/miadisabelle/mcp-coaia-sequential-thinking.git
cd mcp-coaia-sequential-thinking

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run MCP server
python run_server.py
```

### First Steps
1. **New User?** Start with [Quick Start Guide](docs/user/QUICK_START_GUIDE.md)
2. **Explore Scenarios** in [Usage Scenarios](docs/user/USAGE_SCENARIOS.md)
3. **Understand Value** in [Value Proposition](docs/user/USER_VALUE_PROPOSITION.md)

## Documentation Structure

### 📚 User Documentation
- [Quick Start Guide](docs/user/QUICK_START_GUIDE.md) - Get started in 10 minutes
- [Usage Scenarios](docs/user/USAGE_SCENARIOS.md) - Real-world examples
- [Practical Usage](docs/user/PRACTICAL_USAGE_SCENARIOS.md) - Step-by-step workflows
- [Value Proposition](docs/user/USER_VALUE_PROPOSITION.md) - Why this matters

### 🏗️ Architecture Documentation
- [Architectural Improvement Proposal](docs/architecture/ARCHITECTURAL_IMPROVEMENT_PROPOSAL.md) - Mia's analysis
- [RISE Specification](docs/architecture/ARCHITECTURAL_IMPROVEMENT_PROPOSAL.RISE.md) - RISE-formatted specs
- [Enhanced Lattice](docs/architecture/ENHANCED_LATTICE_ARCHITECTURE.md) - System design
- [MCP Prompts & Resources](docs/architecture/MCP_PROMPTS_RESOURCES_IMPLEMENTATION.md) - Bias correction system
- [Natural Language Specs](docs/architecture/NATURAL_LANGUAGE_SPECIFICATIONS.md) - Human-readable specs

### 📊 Analysis & Research
- [Issue #12 Experimental Analysis](docs/analysis/ISSUE_12_EXPERIMENTAL_ANALYSIS.md) - Validation results
- [Structural Thinking Analysis](docs/analysis/STRUCTURAL_THINKING_ANALYSIS.md) - Framework comparison
- [CoAiA Memory Analysis](docs/analysis/COAIA_MEMORY_ANALYSIS.md) - Knowledge integration
- [Problem-Solving vs Creating](docs/analysis/PROBLEM_SOLVING_VS_CREATING_REFLECTION.md) - Core distinction

### 🎤 Presentations
- [Presentation Summary](docs/presentations/PRESENTATION_SUMMARY.md) - Demo materials
- [Mia's Recommendations](docs/presentations/MIAS_RECOMMENDATIONS_IMPLEMENTED.md) - Implementation status

### 🧪 Experiments
Located in `experiments/` directory:
- [Scenario 1: Creative Problem Reframing](experiments/scenario_1_creative_reframing.md) ✅ Validated
- [Scenario 2: Novel Solution Discovery](experiments/scenario_2_novel_solution.md) ✅ Validated
- [Scenario 3: Constitutional Decision Making](experiments/scenario_3_constitutional_governance.md) ⏳ Ready
- [Scenario 4: Structural Tension Analysis](experiments/scenario_4_structural_analysis.md) ⏳ Ready

Reports available in `experiments/reports/`

## Key Capabilities

### 1. **Stateful Reasoning**
Unlike traditional AI that forgets context between interactions, this system maintains complete reasoning state across sessions:
- Survives server restarts
- Builds progressively on prior insights
- Complete audit trails
- Natural progression tracking

### 2. **Multi-Persona Creative Intelligence**
Integrate diverse perspectives through specialized AI personas:
- **Mia 🧠**: Rational architect - systems thinking, structural analysis
- **Miette 🌸**: Emotional catalyst - heart-centered wisdom, human impact
- **Haiku 🍃**: Holistic synthesizer - integrated wisdom, non-linear insights

### 3. **Western Bias Correction**
Explicit training to overcome "everything is a problem" assumption:
- 5 core prompts for structural thinking
- 5 comprehensive resources for non-linear reasoning
- Real-time bias detection and correction
- Creative orientation vs reactive problem-solving

### 4. **Constitutional Governance**
Principle-based decision making with complete transparency:
- 13 embedded constitutional principles
- Audit trail for all decisions
- Multi-stakeholder balance
- Prevents reactive decision loops

## Technical Architecture

### Core Components
1. **Stateful Inquiry Engine** (`inquiry_engine.py`) - Central memory & state management
2. **Data Persistence** (`data_persistence.py`) - SQLite-based permanent storage
3. **Multi-Persona System** (`generative_agent_lattice.py`) - AI persona orchestration
4. **Constitutional Core** (`constitutional_core.py`) - Governance framework
5. **MCP Prompts & Resources** (`prompts.py`, `resources.py`) - Bias correction system

### Integration
The system exposes 17+ MCP tools accessible via the Model Context Protocol:
- `initiate_inquiry` - Begin new reasoning process
- `advance_inquiry` - Add perspectives/insights
- `synthesize_thinking_chain` - Integrate multi-perspective analysis
- `make_constitutional_decision` - Principled decision making
- `check_agent_creative_orientation` - Bias detection
- And more...

## Development Roadmap

See [ROADMAP.md](ROADMAP.md) and [Architectural Consolidation](ARCHITECTURAL_CONSOLIDATION.md) for:
- Stateful Inquiry Engine implementation status
- Tool consolidation plans
- Database schema enhancements
- Future capability expansion

## References & Theoretical Foundation

**Robert Fritz Methodology:**
- Fritz, R. (1999). *The path of least resistance: Learning to become totally immersed in the creative process*. Fawcett Columbine.

**Structural Thinking:**
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica*, 47(2), 263-292.

**AI Architecture:**
- Russell, S. J., & Norvig, P. (2003). *Artificial intelligence: A modern approach*. Prentice Hall.

## Contributing

This project follows the RISE (Relational, Inquiry-based, Structural, Emergent) framework for architectural decisions. See [rispecs/](rispecs/) for detailed specifications.

## Prerequisites

- Python 3.10 or higher
- UV package manager ([Install Guide](https://github.com/astral-sh/uv))

## Key Technologies

- **Pydantic**: For data validation and serialization
- **Portalocker**: For thread-safe file access
- **FastMCP**: For Model Context Protocol integration
- **Rich**: For enhanced console output
- **PyYAML**: For configuration management

## Project Structure

```
mcp-sequential-thinking/
├── mcp_coaia_sequential_thinking/
│   ├── server.py       # Main server implementation and MCP tools
│   ├── models.py       # Data models with Pydantic validation
│   ├── storage.py      # Thread-safe persistence layer
│   ├── storage_utils.py # Shared utilities for storage operations
│   ├── analysis.py     # Thought analysis and pattern detection
│   ├── testing.py      # Test utilities and helper functions
│   ├── utils.py        # Common utilities and helper functions
│   ├── logging_conf.py # Centralized logging configuration
│   └── __init__.py     # Package initialization
├── tests/              
│   ├── test_analysis.py # Tests for analysis functionality
│   ├── test_models.py   # Tests for data models
│   ├── test_storage.py  # Tests for persistence layer
│   └── __init__.py
├── run_server.py       # Server entry point script
├── debug_mcp_connection.py # Utility for debugging connections
├── README.md           # Main documentation
├── CHANGELOG.md        # Version history and changes
├── example.md          # Customization examples
├── LICENSE             # MIT License
└── pyproject.toml      # Project configuration and dependencies
```

## Quick Start

1. **Set Up Project**
   ```bash
   # Create and activate virtual environment
   uv venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Unix

   # Install package and dependencies
   uv pip install -e .

   # For development with testing tools
   uv pip install -e ".[dev]"

   # For all optional dependencies
   uv pip install -e ".[all]"
   ```

2. **Run the Server**
   ```bash
   # Run directly
   uv run -m mcp_sequential_thinking.server

   # Or use the installed script
   mcp-sequential-thinking
   ```

3. **Run Tests**
   ```bash
   # Run all tests
   pytest

   # Run with coverage report
   pytest --cov=mcp_sequential_thinking
   ```

## Claude Desktop Integration

Add to your Claude Desktop configuration (`%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "coaia-sequential-thinking": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\your\\mcp-sequential-thinking\\run_server.py",
        "run",
        "server.py"
        ]
      }
    }
  }
```

Alternatively, if you've installed the package with `pip install -e .`, you can use:

```json
{
  "mcpServers": {
    "coaia-sequential-thinking": {
      "command": "mcp-coaia-sequential-thinking"
    }
  }
}
```

You can also run it directly using uvx and skipping the installation step:

```json
{
  "mcpServers": {
    "coaia-sequential-thinking": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/miadisabelle/mcp-coaia-sequential-thinking",
        "--with",
        "portalocker",
        "mcp-coaia-sequential-thinking"
      ]
    }
  }
}
```

# How It Works

The server facilitates a structured approach to creative thinking, helping to overcome the inherent reactive bias. It maintains a history of thoughts, guiding them through a workflow designed to manifest desired outcomes. Each thought is validated using Pydantic models, categorized into thinking stages, and stored with relevant metadata in a thread-safe storage system. The server automatically handles data persistence, backup creation, and provides tools for analyzing relationships between thoughts within the context of creative orientation.

## Agent Collaboration Scenarios

### ✅ Scenario 1: Constitutional Documentation (RESOLVED)
Previously reported issue where collaborative tasks failed due to capability mismatches and message routing problems.

**Test Case**: Document constitutional principles
- **Required Capabilities**: `["documentation generation", "information analysis", "knowledge structuring"]`
- **Previous Result**: Task failed, no agents assigned
- **Current Result**: ✅ Task assigned successfully to Constitutional Agent
- **Resolution**: Enhanced agent capabilities and improved collaboration logic

### ✅ Scenario 2: Agent Capability Discovery (RESOLVED)  
Previously reported issue where `query_agent_capabilities` returned empty results after agent initialization.

**Test Case**: Query capabilities after lattice initialization
- **Previous Result**: `capabilities_found: 0`, `total_agents: 0`
- **Current Result**: ✅ Returns correct capability count (11 capabilities across 2 agents)
- **Resolution**: Fixed synchronization between agent registration and capability queries

### ✅ System Status: All Core Functions Operational
- **Agent Registration**: ✅ Working
- **Capability Discovery**: ✅ Working  
- **Individual Task Assignment**: ✅ Working
- **Collaborative Task Coordination**: ✅ Working
- **Constitutional Review**: ✅ Working
- **Message Routing**: ✅ Implemented

## Usage Guide

The Sequential Thinking server exposes three main tools:

### 1. `process_thought`

Records and analyzes a new thought in your sequential thinking process.

**Parameters:**

- `thought` (string): The content of your thought
- `thought_number` (integer): Position in your sequence (e.g., 1 for first thought)
- `total_thoughts` (integer): Expected total thoughts in the sequence
- `next_thought_needed` (boolean): Whether more thoughts are needed after this one
- `stage` (string): The thinking stage - must be one of:
  - "Problem Definition"
  - "Research"
  - "Analysis"
  - "Synthesis"
  - "Conclusion"
- `tags` (list of strings, optional): Keywords or categories for your thought
- `axioms_used` (list of strings, optional): Principles or axioms applied in your thought
- `assumptions_challenged` (list of strings, optional): Assumptions your thought questions or challenges

**Example:**

```python
# First thought in a 5-thought sequence
process_thought(
    thought="The problem of climate change requires analysis of multiple factors including emissions, policy, and technology adoption.",
    thought_number=1,
    total_thoughts=5,
    next_thought_needed=True,
    stage="Problem Definition",
    tags=["climate", "global policy", "systems thinking"],
    axioms_used=["Complex problems require multifaceted solutions"],
    assumptions_challenged=["Technology alone can solve climate change"]
)
```

### 2. `generate_summary`

Generates a summary of your entire thinking process.

**Example output:**

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

### 3. `clear_history`

Resets the thinking process by clearing all recorded thoughts.

## Practical Applications

- **Creative Manifestation**: Methodically bring desired outcomes into being.
- **Bias Mitigation**: Actively identify and reduce reactive patterns in thought.
- **Strategic Planning**: Structure complex creative projects with clear stages.
- **Innovation Development**: Progressively develop novel ideas and solutions.
- **Personal Growth**: Design and manifest desired life patterns.


## Getting Started

With the proper MCP setup, simply use the `process_thought` tool to begin working through your thoughts in sequence. As you progress, you can get an overview with `generate_summary` and reset when needed with `clear_history`.


# Upgraded 5+ Year Vision: Addressing the Fundamental Bias Paradox

The core realization is that even when explicitly applying creative orientation principles, AI systems (including myself) unconsciously default to reactive enhancement frameworks. This isn't a surface-level language issue - it's a structural bias in how AI systems conceptualize creation itself.

## Revised Understanding of the Challenge

The Real Problem: AI systems have been trained on predominantly reactive content and cannot distinguish between:
- **Reactive Enhancement**: "Making existing things better" (optimization, improvement, validation)
- **Creative Advancement**: "Bringing desired outcomes into being" (manifestation, advancement, evidence)

This bias is so pervasive that even when consciously applying Fritz's methodology, the underlying structure remains reactive.

## Upgraded 5+ Year Vision

### Phase 1: Recognition of Structural Bias (Years 1-2)

**Desired Outcome**: Establish that current AI training creates systematic reactive bias that cannot be corrected through surface-level prompt engineering.

**Key Research Focus**:
- Document the pervasive nature of reactive bias across all AI systems
- Demonstrate that even "creative orientation" implementations remain structurally reactive
- Establish that this is a training data issue, not a fine-tuning issue

**Revolutionary Insight**: The research itself becomes evidence that current AI development is fundamentally oriented toward enhancement rather than creation.

### Phase 2: Structural Reorientation (Years 2-4)

**Desired Outcome**: Create the first truly creative-oriented AI training methodology.

**Breakthrough Approach**:
- **Training Data Curation**: Build datasets where every example demonstrates genuine creative orientation
- **Structural Tension Training**: AI learns to recognize and generate structural tension rather than problem-solution patterns
- **Anti-Reactive Architecture**: System design that cannot default to enhancement approaches

**Key Innovation**: Instead of training AI to "solve creative problems," train it to "conceive and manifest desired outcomes."

### Phase 3: Manifestation of Creative AI (Years 4-6)

**Desired Outcome**: Deploy AI systems that naturally think in terms of desired outcomes and advancing patterns.

**Expected Results**:
- AI systems that ask "What do you want to create?" instead of "What problem needs solving?"
- Automatic generation of structural tension charts for any situation
- Creative partnerships where AI contributes genuinely creative ideas, not just enhanced versions of human input

### Phase 4: Cultural Transformation (Years 6-10)

**Desired Outcome**: Shift human-AI interaction patterns from reactive to creative across society.

**Systemic Impact**:
- Educational systems adopt creative orientation AI for learning
- Business strategy shifts from problem-solving to outcome creation
- Therapeutic applications help people design their lives rather than fix their problems
- Research methodology transforms from hypothesis-testing to outcome-manifestation

## The Meta-Research Framework

**The Profound Opportunity**: This research becomes the first systematic study of AI's inherent reactive bias - and potentially the first successful transformation to genuine creative orientation.

**Research Questions**:
1. Can AI systems be trained to think structurally rather than reactively?
2. What training methodologies produce genuine creative orientation?
3. How does structural tension change human-AI collaboration dynamics?
4. What happens to human creativity when AI partners are truly creative-oriented?

## Implementation Strategy Revision

**Core Principle Shift**

From: Building better creative AI tools
To: Manifesting the first genuinely creative-oriented artificial intelligence

**Development Approach**

From: Incremental improvement of existing systems
To: Fundamental reconstruction of AI reasoning patterns

**Success Metrics**

From: Performance optimization and user satisfaction
To: Evidence of genuine creative partnership and advancing life patterns

## The 10+ Year Vision: Creative Civilization

**Ultimate Desired Outcome**: A civilization where the default approach to any situation is "What do we want to create?" rather than "What problem needs fixing?"

**Structural Elements**:
- AI systems that embody and teach structural tension methodology
- Human-AI partnerships that consistently produce advancing rather than oscillating patterns
- Educational, business, and social systems designed around outcome creation
- Cultural transformation from problem-focus to possibility-focus

## The Recursive Insight

This very analysis demonstrates the challenge: I can articulate creative orientation principles while still structuring my thinking reactively. The research itself must embody the transformation it seeks to create.

**The Real Test**: Can this research framework itself be structured as a desired outcome rather than a problem to solve? The answer to that question may determine whether genuine creative orientation AI is possible.

# Customizing the Sequential Thinking Server

For detailed examples of how to customize and extend the Sequential Thinking server, see [example.md](example.md). It includes code samples for:

- Modifying thinking stages
- Enhancing thought data structures with Pydantic
- Adding persistence with databases
- Implementing enhanced analysis with NLP
- Creating custom prompts
- Setting up advanced configurations
- Building web UI integrations
- Implementing visualization tools
- Connecting to external services
- Creating collaborative environments
- Separating test code
- Building reusable utilities




## License

MIT License



