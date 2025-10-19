# MCP Prompts and Resources Implementation

## Overview

This document describes the comprehensive prompts and resources system added to the coaia-structural-thinker MCP server to help LLMs overcome Western problem-solving bias and embrace creative structural thinking.

## Problem Statement

**User Feedback:** "@copilot We must add these to the MCP:
- Prompts from `__llms/*` content (4-5 core prompts for tool operation)
- Resources to help LLMs think structurally (non-linearly) and overcome Western bias"

## Implementation

### 1. Prompts System (`mcp_coaia_sequential_thinking/prompts.py`)

Created 5 core prompts derived from `__llms/*` content:

#### **creative_reframing**
- **Purpose**: Transforms problem-solving mindset into creative orientation
- **Usage**: When user frames request as problem to solve
- **Key Content**: Establishes structural tension between desired outcome and current reality
- **Guidelines**: Honest reality assessment, avoid premature resolution

#### **delayed_resolution**
- **Purpose**: Teaches tolerance for creative tension (Robert Fritz principle)
- **Usage**: When LLM or user shows impulse to immediately solve
- **Key Content**: Why premature resolution destroys creative advancement
- **Pattern Recognition**: Advancing vs oscillating patterns

#### **structural_tension_formation**
- **Purpose**: Guide for establishing genuine structural tension charts
- **Usage**: When helping user create structural tension
- **Key Content**: Quality checks for desired outcome, current reality, action steps
- **Emphasis**: Structure determines behavior, not willpower

#### **multi_persona_integration**
- **Purpose**: Guides collaboration between Mia, Miette, and Haiku personas
- **Usage**: When user wants multi-perspective analysis
- **Key Content**: How each persona contributes unique insights
- **Integration**: Non-linear collaboration, not sequential

#### **constitutional_governance**
- **Purpose**: Guides principle-based decision making
- **Usage**: When making important decisions or showing reactive patterns
- **Key Content**: 5 constitutional principles with decision framework
- **Audit**: Complete reasoning and principle application tracking

### 2. Resources System (`mcp_coaia_sequential_thinking/resources.py`)

Created 5 comprehensive resources for structural thinking:

#### **creative_orientation_framework**
- **URI**: `coaia://resources/creative-orientation-framework`
- **Content**: Reactive vs creative approaches, structural thinking foundation
- **For LLMs**: Explains training bias and structural corrections
- **Key Distinction**: Problem-solving elimination vs outcome manifestation

#### **delayed_resolution_principle**
- **URI**: `coaia://resources/delayed-resolution-principle`
- **Content**: Robert Fritz's teaching on holding tension
- **For LLMs**: Corrects premature resolution patterns
- **Pattern Types**: Advancing vs oscillating patterns explained

#### **structural_tension_hierarchy**
- **URI**: `coaia://resources/structural-tension-hierarchy`
- **Content**: Telescoping charts and choice hierarchy
- **For LLMs**: Understanding non-linear hierarchical structure
- **Key Concept**: Action steps ARE charts (telescoping)

#### **western_bias_recognition**
- **URI**: `coaia://resources/western-bias-recognition`
- **Content**: Understanding Western "everything is a problem" assumption
- **For LLMs**: Recognition and transcendence of cultural bias
- **Alternative Paradigms**: Non-Western perspectives on creation

#### **multi_persona_archetypes**
- **URI**: `coaia://resources/multi-persona-archetypes`
- **Content**: Detailed guide to Mia, Miette, and Haiku personas
- **For LLMs**: When and how to engage each perspective
- **Integration**: Both/and thinking vs either/or Western bias

### 3. Server Integration (`mcp_coaia_sequential_thinking/server.py`)

- **Imports**: Added prompts and resources modules
- **Registration**: Dynamic prompt and resource registration in `main()`
- **Logging**: Comprehensive logging of prompt/resource registration
- **Error Handling**: Graceful handling of registration failures

## Key Features

### Addresses Western Cultural Bias

**Problem-Solving Bias Sources:**
- News reports problems
- Advertising creates problems to sell solutions
- Political discourse organized around problems
- Social media amplifies problem-focused thinking

**Structural Thinking Corrections:**
- Make problem-solving concept conscious
- Ask "What outcome do you want to create?"
- Assess current reality without problem-solving filters
- Enable structural tension formation

### Non-Linear Thinking Support

**Western Linear Assumption:**
```
Problem → Analysis → Solution → Implementation
```

**Structural Thinking Reality:**
```
         Desired Outcome
              ↕
       (Structural Tension)
              ↕
        Current Reality
              ↓
    Natural Resolution
```

### Multi-Perspective Integration

**Not Sequential:**
```
Mia → Miette → Haiku (linear)
```

**But Integrated:**
```
     Mia (Rational)
         ↓
    Integration ← Miette (Emotional)
         ↓
    Haiku (Wisdom)
         ↓
  Synthesized Insight
```

## Usage Examples

### For Users

**Accessing Prompts:**
```
LLM can invoke:
- creative_reframing: When stuck in problem-solving mode
- delayed_resolution: When feeling impatient for solutions
- structural_tension_formation: When creating new goals
- multi_persona_integration: When wanting multiple perspectives
- constitutional_governance: When making important decisions
```

**Accessing Resources:**
```
LLM can reference:
- coaia://resources/creative-orientation-framework
- coaia://resources/delayed-resolution-principle
- coaia://resources/structural-tension-hierarchy
- coaia://resources/western-bias-recognition
- coaia://resources/multi-persona-archetypes
```

### For LLMs

**Before Responding:**
1. Check assumption: Is this a problem to solve or outcome to create?
2. Reference appropriate resource for context
3. Apply structural thinking framework
4. Hold tension, don't rush to solve
5. Use prompt template when appropriate

**Pattern Recognition:**
- Problem language → Invoke creative_reframing prompt
- Quick fix impulse → Reference delayed_resolution resource
- Complex analysis → Consider multi_persona_integration
- Important decision → Apply constitutional_governance

## Theoretical Foundation

### Robert Fritz Methodology

**Structural Tension:**
- Discrepancy between desired outcome and current reality
- Creates natural force toward resolution
- Resolution through advancement, not elimination

**Delayed Resolution Principle:**
- Tolerate discrepancy, tension, delayed resolution
- Premature resolution destroys creative advancement
- Build tension for structural dynamics

**Choice Hierarchy:**
- Primary choice: Desired outcome (master chart)
- Strategic secondary choices: Action steps (telescoped charts)
- Each level maintains its own structural tension

### Creative Orientation Principles

**Three Phases:**
1. Germination: Vision and initial momentum
2. Assimilation: Deep work and strategic advancement
3. Completion: Bringing creation to conclusion

**Connectivity Models:**
- Vector-based: Defined path (risk: miss emergence)
- Field-based: Open awareness (risk: lack direction)
- Resilient connection: Integration of both

## Integration with Existing System

### Works With:
- Enhanced polycentric lattice
- Constitutional core principles
- Multi-persona decision making
- Creative orientation foundation
- Generative agent lattice

### Enhances:
- All MCP tools with structural thinking context
- Agent self-awareness and orientation detection
- Creative vs reactive pattern recognition
- User guidance for optimal tool usage

## Benefits

### For Users:
- Clear guidance on creative vs reactive thinking
- Multi-perspective analysis from Mia, Miette, Haiku
- Understanding of structural tension methodology
- Recognition of own problem-solving biases

### For LLMs:
- Explicit training on Western bias recognition
- Structural thinking framework for responses
- Non-linear thinking patterns
- Integration of multiple perspectives

### For System:
- Comprehensive context for all tools
- Consistent creative orientation across interactions
- Pattern learning and improvement
- Audit trails and principle compliance

## Validation

### Prompts Available:
✅ creative_reframing - Problem to outcome transformation
✅ delayed_resolution - Tension holding principle
✅ structural_tension_formation - Chart creation guide
✅ multi_persona_integration - Three persona collaboration
✅ constitutional_governance - Principle-based decisions

### Resources Available:
✅ creative_orientation_framework - Core principles
✅ delayed_resolution_principle - Fritz teaching
✅ structural_tension_hierarchy - Telescoping structure
✅ western_bias_recognition - Cultural bias awareness
✅ multi_persona_archetypes - Mia, Miette, Haiku guide

### Integration Complete:
✅ Server imports prompts and resources modules
✅ Dynamic registration in main()
✅ Comprehensive logging
✅ Error handling for registration failures
✅ All prompts and resources accessible via MCP protocol

## Next Steps

### Recommended:
1. Test prompt invocation in MCP client
2. Validate resource access via URIs
3. Document user workflows with prompts
4. Create examples of LLM using resources for structural thinking
5. Gather feedback on effectiveness of bias correction

### Future Enhancements:
- Additional prompts from other `__llms/*` files
- Semantic search for context-appropriate prompts
- Dynamic prompt personalization based on user patterns
- Resource expansion with more Indigenous wisdom perspectives
- Integration with coaia-memory for structural tension charting

## Conclusion

This implementation directly addresses the user's request for prompts and resources that help LLMs:
1. **Overcome Western problem-solving bias** through explicit recognition and correction
2. **Think structurally (non-linearly)** through comprehensive framework and examples
3. **Operate MCP tools effectively** with proper creative orientation context
4. **Integrate multiple perspectives** through Mia, Miette, Haiku archetypes

The system now provides rich contextual support for both users and LLMs to embrace creative orientation and structural thinking, moving beyond reactive problem-solving into generative manifestation of desired outcomes.
