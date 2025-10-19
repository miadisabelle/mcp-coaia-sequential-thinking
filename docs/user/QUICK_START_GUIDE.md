# Quick Start Guide
*Getting Started with the Polycentric Agentic Lattice*

## Overview
The Polycentric Agentic Lattice enables multi-persona sequential thinking for complex analysis and decision-making. This guide provides a simple tutorial for the core workflow.

## Prerequisites
```bash
# Install dependencies
pip install "mcp[cli]>=1.2.0" rich pyyaml portalocker numpy

# Verify installation
python test_all_tools.py
```

## Core Workflow: Sequential Thinking

### Step 1: Initiate Sequential Thinking
Start a multi-persona analysis chain:

```python
from mcp_coaia_sequential_thinking.server import initiate_sequential_thinking

# Initiate a new thinking chain
result = initiate_sequential_thinking(
    request="Analyze the effectiveness of remote work policies",
    primary_purpose="Develop comprehensive policy recommendations",
    cultural_context="Western corporate environment"
)

chain_id = result["chain_id"]  # e.g., "thinking_chain_20250915_123456"
print(f"Started thinking chain: {chain_id}")
```

### Step 2: Advance Through Persona Perspectives
Let each persona contribute their unique perspective:

```python
from mcp_coaia_sequential_thinking.server import advance_thinking_chain

# Mia's rational analysis
mia_result = advance_thinking_chain(
    chain_id=chain_id,
    persona_focus="rational_architect"
)
print(f"Mia's analysis confidence: {mia_result['confidence']}")

# Miette's empathetic insights  
miette_result = advance_thinking_chain(
    chain_id=chain_id,
    persona_focus="emotional_catalyst"
)
print(f"Miette's insights confidence: {miette_result['confidence']}")

# Haiku's wisdom synthesis
haiku_result = advance_thinking_chain(
    chain_id=chain_id,
    persona_focus="wisdom_synthesizer"
)
print(f"Haiku's synthesis confidence: {haiku_result['confidence']}")
```

### Step 3: Synthesize Final Wisdom
Integrate all perspectives into unified insights:

```python
from mcp_coaia_sequential_thinking.server import synthesize_thinking_chain

# Generate final synthesis
synthesis = synthesize_thinking_chain(
    chain_id=chain_id,
    synthesis_focus="integrated_wisdom"
)

print("Final Analysis:")
print(f"Confidence: {synthesis['confidence']}")
print(f"Synthesis: {synthesis['synthesis']}")
```

## Example: Complete Analysis

```python
# Complete workflow example
def analyze_topic(topic, purpose):
    # 1. Start the chain
    init_result = initiate_sequential_thinking(
        request=f"Analyze {topic}",
        primary_purpose=purpose
    )
    chain_id = init_result["chain_id"]
    
    # 2. Gather perspectives
    personas = ["rational_architect", "emotional_catalyst", "wisdom_synthesizer"]
    for persona in personas:
        advance_thinking_chain(chain_id=chain_id, persona_focus=persona)
    
    # 3. Synthesize insights
    synthesis = synthesize_thinking_chain(
        chain_id=chain_id,
        synthesis_focus="integrated_wisdom"
    )
    
    return synthesis

# Use it
result = analyze_topic(
    "team collaboration tools",
    "Select the best tool for our organization"
)
print(f"Recommendation: {result['synthesis']}")
```

## Advanced Features

### Consensus Decision Making
For decisions requiring multi-agent consensus:

```python
from mcp_coaia_sequential_thinking.server import (
    create_consensus_decision,
    get_consensus_decision_status
)

# Create a decision
decision_result = create_consensus_decision(
    decision_request="Implement new performance review process",
    primary_purpose="Improve employee development",
    decision_type="policy_change"
)

decision_id = decision_result["decision_id"]

# Check status
status = get_consensus_decision_status(decision_id)
print(f"Consensus status: {status['consensus_status']}")
```

### Human Consultation Loop
For complex decisions requiring human input:

```python
from mcp_coaia_sequential_thinking.server import (
    request_human_consultation,
    provide_human_response
)

# Request human consultation
consultation = request_human_consultation(
    decision_id=decision_id,
    consultation_request="Need stakeholder input on implementation timeline"
)

# Provide human response (simulated)
human_response = provide_human_response(
    decision_id=decision_id,
    human_input="Implement gradually over 6 months with pilot programs"
)
```

## High-Level Wrapper (Simplified Usage)

For quick analysis without managing chain IDs:

```python
from mcp_coaia_sequential_thinking.server import run_full_analysis_chain

# One-call complete analysis
result = run_full_analysis_chain(
    request="Evaluate cloud migration strategy",
    primary_purpose="Reduce infrastructure costs while maintaining performance"
)

print(f"Analysis: {result['final_synthesis']}")
print(f"Confidence: {result['confidence']}")
```

## Monitoring System Health

```python
from mcp_coaia_sequential_thinking.server import (
    get_lattice_status,
    get_active_thinking_chains,
    validate_constitutional_compliance
)

# Check system status
lattice_status = get_lattice_status()
print(f"Active agents: {lattice_status['active_agents']}")

# Check active chains
active_chains = get_active_thinking_chains()
print(f"Active thinking chains: {len(active_chains)}")

# Validate compliance
compliance = validate_constitutional_compliance()
print(f"Constitutional compliance: {compliance['compliance_score']}")
```

## Multi-Persona System Details

**🧠 Mia (Rational Architect)**
- Focus: Technical analysis, structured thinking, logical frameworks
- Best for: System design, process optimization, analytical tasks

**🌸 Miette (Emotional Catalyst)**  
- Focus: Human-centered design, empathy, user experience
- Best for: Team dynamics, user needs, emotional intelligence

**🌊 Haiku (Wisdom Synthesizer)**
- Focus: Pattern integration, essence distillation, holistic insights
- Best for: Strategic vision, connecting disparate ideas, synthesis

## Tips for Success

1. **Start Simple**: Begin with basic sequential thinking before adding consensus decisions
2. **Clear Purpose**: Always define a specific primary purpose for better results
3. **Iterative Refinement**: Use chain status checks to monitor progress
4. **Human Loop**: Use consultation for complex decisions requiring stakeholder input
5. **Monitor Health**: Regularly check lattice status for optimal performance

## Troubleshooting

**ImportError Issues:**
```bash
pip install "mcp[cli]>=1.2.0" rich pyyaml portalocker numpy
python test_all_tools.py  # Verify all tools work
```

**Low Confidence Scores:**
- Provide more specific requests
- Use appropriate cultural context
- Ensure primary purpose aligns with request

**System Performance:**
- Check lattice status regularly
- Monitor active thinking chains
- Validate constitutional compliance

---

**Next Steps:** Explore `ENHANCED_LATTICE_ARCHITECTURE.md` for advanced concepts and integration patterns.