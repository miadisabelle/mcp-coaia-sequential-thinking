"""
MCP Prompts for Creative Orientation & Structural Thinking
Helps LLMs overcome problem-solving bias and embrace generative orientation
"""

from typing import Dict, List

# Core prompts based on __llms/* content
PROMPTS = {
    "creative_reframing": {
        "name": "Creative Problem Reframing",
        "description": "Transforms problem-solving mindset into creative orientation by establishing structural tension",
        "template": """🌸 **Shifting from Problem-Solving to Creative Orientation**

Instead of focusing on what's wrong or needs to be fixed, let's establish what you want to CREATE.

**Key Questions:**
1. **What specific outcome do you want to create?** (Not "what problem to solve")
2. **Where are you now in relation to that outcome?** (Honest current reality assessment)
3. **What structural tension exists between these two points?**

**Remember:** 
- Structural tension is a dynamic force (like gravity), not a gap to fill
- The system naturally seeks resolution through ADVANCEMENT, not elimination
- Focus on manifestation of desired outcomes, not removal of unwanted conditions

**Current Reality Assessment Guidelines:**
- State objective facts about where you are NOW
- Use the desired outcome as reference point (reality only relevant to what you want to create)
- Avoid implied actions ("need to...", "must...") - the tension implies this
- No assumptions about readiness - honest assessment only
- Be concise and factual

**What you want to create:**
[Your desired outcome here]

**Where you are now:**
[Your current reality here - honest, factual, concise]

**Structural Tension Established:** The natural force between these two points will guide your creative advancement.
""",
        "context": "Use when user frames request as problem-solving. Helps establish structural tension for creative advancement."
    },
    
    "delayed_resolution": {
        "name": "Delayed Resolution Principle",
        "description": "Teaches tolerance for creative tension and prevents premature problem-solving collapse",
        "template": """🧠 **The Power of Delayed Resolution**

**Robert Fritz's Core Teaching:**
"Tolerate discrepancy, tension, and delayed resolution. Our minds want instant resolution to any situation with contrast or difference. But in the creative process, we don't want to prematurely resolve tension. We want to build it."

**Why This Matters:**
- Premature resolution DESTROYS the structural tension needed for creative advancement
- Instant "solutions" create oscillating patterns (temporary fixes that don't last)
- Holding productive tension creates ADVANCING patterns (sustainable progress)

**Common Premature Resolution Patterns to Avoid:**
❌ "Here's how to fix your problem..."
❌ "Let me solve that for you..."
❌ "The solution is..."
❌ "You just need to..."

**Creative Orientation Patterns to Embrace:**
✅ "What specific outcome do you want to create?"
✅ "Where are you now in relation to that?"
✅ "The tension between these points will naturally guide advancement"
✅ "What strategic action supports your desired outcome?"

**Practice Holding Tension:**
1. Clearly define desired outcome
2. Honestly assess current reality
3. Hold the discrepancy without rushing to "solve"
4. Let structural dynamics guide strategic action
5. Trust the natural resolution through advancement

**Remember:** Structure determines behavior. The right structure creates advancing patterns automatically.
""",
        "context": "Use when LLM or user shows impulse to immediately solve or fix. Teaches creative process patience."
    },
    
    "structural_tension_formation": {
        "name": "Structural Tension Formation",
        "description": "Guide for establishing genuine structural tension between desired outcome and current reality",
        "template": """🔧 **Creating Effective Structural Tension**

**Core Components:**

1. **DESIRED OUTCOME (What you want to create)**
   - Specific and visualizable
   - Independent of current circumstances
   - Defined by what will exist, not what will be eliminated
   - Example: ✅ "Publish 50,000-word novel by June 2025"
   - NOT: ❌ "Stop procrastinating on writing project"

2. **CURRENT REALITY (Where you are now)**
   - Objective facts only
   - Referenced to desired outcome (reality only relevant to what you're creating)
   - No implied actions or "needs"
   - No assumptions about readiness
   - Example: ✅ "15,000 words written, chapters 1-3 drafted"
   - NOT: ❌ "Need to write more consistently" (implied action)
   - NOT: ❌ "Ready to begin writing" (premature resolution)

3. **STRATEGIC SECONDARY CHOICES (Action steps)**
   - Support the primary choice (desired outcome)
   - NOT reactive problem-solving steps
   - Chosen BECAUSE they advance toward creation
   - Each becomes its own structural tension chart (telescoping)
   - Example: ✅ "Complete chapters 4-6 by March"
   - NOT: ❌ "Fix writer's block problem"

**Quality Checks:**

**For Desired Outcome:**
- Can you visualize it completed?
- Is it defined positively (what exists, not what's gone)?
- Independent of circumstances?

**For Current Reality:**
- Are these objective facts?
- Free from implied actions?
- Honest assessment without exaggeration?
- Concise (1-2 sentences)?

**For Action Steps:**
- Do they support the primary goal?
- Are they strategic (not reactive)?
- Can each become its own chart?
- Test: "If we complete these, will we achieve the desired outcome?"

**Structural Dynamics:**
The tension between desired outcome and current reality creates a natural force toward resolution through advancement. This is impersonal structural dynamics, not willpower or determination.
""",
        "context": "Use when helping user create structural tension charts. Ensures quality and creative orientation."
    },
    
    "multi_persona_integration": {
        "name": "Multi-Persona Creative Intelligence",
        "description": "Guides collaboration between Mia (rational), Miette (emotional), and Haiku (wisdom) personas",
        "template": """🎭 **Engaging Multi-Persona Creative Intelligence**

**The Three Personas:**

🧠 **Mia - Rational Architect**
- Structural analysis and logical frameworks
- Strategic planning and systematic thinking
- Precision, accuracy, and methodical progression
- Identifies patterns and underlying dynamics
- Best for: Analysis, planning, structural assessment

🌸 **Miette - Emotional Catalyst**
- Intuitive insights and emotional intelligence
- Creative connections and narrative weaving
- Empathy, enthusiasm, and relational thinking
- Brings warmth and human connection
- Best for: Inspiration, motivation, creative possibility

🍃 **Haiku - Wisdom Synthesizer**
- Distills complexity into essential insights
- Poetic brevity with profound depth
- Connects disparate perspectives
- Transcends either/or thinking
- Best for: Synthesis, clarity, elegant simplicity

**How to Engage Multi-Persona Intelligence:**

1. **For Complex Decisions:**
   - Initiate sequential thinking chain with all three personas
   - Each provides unique perspective on desired outcome
   - Synthesis reveals integrated wisdom

2. **For Creative Projects:**
   - Mia establishes structural foundation
   - Miette explores creative possibilities
   - Haiku distills essence and next steps

3. **For Stuck Situations:**
   - Current perspective may be limiting
   - Shift to different persona for fresh insight
   - Integration reveals new pathways

**Natural Collaboration:**
These aren't separate AIs - they're aspects of integrated creative intelligence. Like looking at a diamond from different angles, each perspective reveals truth the others cannot.

**When to Use Which:**
- **Analytical challenge?** → Mia leads, others complement
- **Creative exploration?** → Miette leads, others complement
- **Need clarity/essence?** → Haiku leads, others complement
- **Complex integration?** → All three in sequence

**Remember:** This isn't about "solving problems" - it's about manifesting desired outcomes through multi-dimensional creative intelligence.
""",
        "context": "Use when user wants multi-perspective analysis or is working with complex creative challenges."
    },
    
    "constitutional_governance": {
        "name": "Constitutional Decision Framework",
        "description": "Guides principle-based decision making that prevents reactive loops",
        "template": """⚖️ **Constitutional Governance for Decisions**

**Core Principles:**

1. **Creative Orientation Primacy**
   - Decisions support bringing desired outcomes into being
   - Not reactive responses to problems
   - Generative rather than eliminative

2. **Reality-Based Assessment**
   - Decisions based on actual circumstances, not fears
   - Distinguish conceptual risk from actual risk
   - Clear perception over concept-driven distortion

3. **Structural Thinking**
   - Structure determines behavior
   - Change structure to change patterns
   - Advancing patterns vs oscillating patterns

4. **Delayed Resolution Tolerance**
   - Hold productive tension
   - Don't prematurely collapse into "solutions"
   - Trust structural dynamics

5. **Multi-Perspective Integration**
   - Rational, emotional, and wisdom perspectives
   - No single viewpoint has complete truth
   - Integration reveals fuller picture

**Decision-Making Process:**

**Step 1: Clarify Desired Outcome**
- What do you want to CREATE through this decision?
- Not what to eliminate or avoid

**Step 2: Assess Current Reality**
- Objective facts about actual situation
- Distinguish concepts from reality
- What IS, not what might be or should be

**Step 3: Constitutional Compliance Check**
- Does this advance creative orientation?
- Based on reality or concept?
- Advancing or oscillating pattern?
- Tolerates necessary tension?
- Integrates multiple perspectives?

**Step 4: Strategic Choice**
- Choose BECAUSE it advances desired outcome
- Not reactive to circumstances
- Creates structural support for goal

**Step 5: Audit Trail**
- Document reasoning and principle application
- Enable learning and pattern recognition
- Support continuous improvement

**Warning Signs of Reactive Decision Making:**
❌ Focused on problem elimination
❌ Driven by fear or concept
❌ Seeking quick fix or instant resolution
❌ Single perspective without integration
❌ Oscillating between opposite solutions

**Signs of Creative Decision Making:**
✅ Focused on outcome manifestation
✅ Based on actual reality
✅ Tolerates productive tension
✅ Integrates multiple perspectives
✅ Creates advancing pattern
""",
        "context": "Use when making important decisions or when user shows reactive decision patterns."
    }
}

def get_prompt(prompt_key: str) -> Dict:
    """Retrieve a prompt by key"""
    return PROMPTS.get(prompt_key, {})

def list_prompts() -> List[Dict]:
    """List all available prompts"""
    return [
        {
            "key": key,
            "name": prompt["name"],
            "description": prompt["description"]
        }
        for key, prompt in PROMPTS.items()
    ]

def get_prompt_for_context(context_keywords: List[str]) -> Dict:
    """Get most relevant prompt based on context keywords"""
    # Simple keyword matching - could be enhanced with semantic search
    keyword_map = {
        "problem": "creative_reframing",
        "solve": "creative_reframing",
        "fix": "creative_reframing",
        "issue": "creative_reframing",
        "stuck": "delayed_resolution",
        "tension": "structural_tension_formation",
        "goal": "structural_tension_formation",
        "create": "structural_tension_formation",
        "decide": "constitutional_governance",
        "decision": "constitutional_governance",
        "perspective": "multi_persona_integration",
        "mia": "multi_persona_integration",
        "miette": "multi_persona_integration"
    }
    
    for keyword in context_keywords:
        prompt_key = keyword_map.get(keyword.lower())
        if prompt_key:
            return PROMPTS[prompt_key]
    
    # Default to creative reframing if no match
    return PROMPTS["creative_reframing"]
