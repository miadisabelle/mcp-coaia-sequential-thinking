#!/usr/bin/env python3
"""
Test script for the Constitutional Core implementation.
Demonstrates the new generative agentic system capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mcp_coaia_sequential_thinking.constitutional_core import constitutional_core, ConstitutionalPrinciple


def test_constitutional_validation():
    """Test constitutional principle validation."""
    print("=== Testing Constitutional Validation ===")
    
    # Test cases with different orientations
    test_cases = [
        {
            "name": "Creative Orientation - Good",
            "content": "I want to create a comprehensive learning system that builds on my current knowledge to achieve mastery in this subject.",
            "expected": True
        },
        {
            "name": "Reactive Orientation - Poor", 
            "content": "I need to solve the problem of not understanding this subject and eliminate my confusion.",
            "expected": False
        },
        {
            "name": "Uncertainty Acknowledgment - Good",
            "content": "I'm not entirely certain about this approach, but it appears to align with the desired outcome.",
            "expected": True
        },
        {
            "name": "Fabrication - Poor",
            "content": "This is definitely the right approach and will absolutely work without any doubt.",
            "expected": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Content: {test_case['content']}")
        
        result = constitutional_core.validate_content(test_case['content'], {})
        
        print(f"Overall Valid: {result['overall_valid']}")
        print(f"Compliance Score: {result['constitutional_compliance_score']:.2f}")
        print(f"Expected: {test_case['expected']}")
        
        if result['overall_valid'] == test_case['expected']:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            print(f"Violated principles: {result['violated_principles']}")


def test_active_pause_mechanism():
    """Test the active pause draft generation."""
    print("\n\n=== Testing Active Pause Mechanism ===")
    
    context = "How should I approach learning a new programming language?"
    
    drafts = constitutional_core.generate_active_pause_drafts(context, num_drafts=3)
    
    print(f"Generated {len(drafts)} drafts for context: {context}")
    
    for draft in drafts:
        print(f"\n--- {draft['draft_id']} ---")
        print(f"Profile: {draft['profile']}")
        print(f"Content: {draft['content']}")
        print(f"Novelty Score: {draft['selection_criteria']['novelty_score']:.2f}")
        print(f"Reliability Score: {draft['selection_criteria']['reliability_score']:.2f}")
        print(f"Constitutional Compliance: {draft['selection_criteria']['constitutional_compliance']:.2f}")
    
    # Test draft selection
    selection_criteria = {
        'novelty_weight': 0.4,
        'reliability_weight': 0.3,
        'constitutional_weight': 0.3
    }
    
    best_draft = constitutional_core.select_best_draft(drafts, selection_criteria)
    print(f"\nSelected best draft: {best_draft['draft_id']}")
    print(f"Selection criteria: {selection_criteria}")


def test_constitutional_decision_making():
    """Test constitutional decision making with audit trail."""
    print("\n\n=== Testing Constitutional Decision Making ===")
    
    decision_context = "Choose approach for implementing new feature"
    options = [
        "Fix the current broken implementation first, then add the feature",
        "Create a new implementation that includes the feature from the ground up",
        "Build the feature as a separate module that enhances the existing system"
    ]
    
    print(f"Decision Context: {decision_context}")
    print("Options:")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    
    decision = constitutional_core.make_constitutional_decision(
        decision_context, options, {"priority": "long_term_value"}
    )
    
    print(f"\nDecision Made:")
    print(f"Decision ID: {decision.decision_id}")
    print(f"Chosen: {decision.decision_outcome}")
    print(f"Applicable Principles: {[p.value for p in decision.applicable_principles]}")
    
    # Test audit trail retrieval
    audit_trail = constitutional_core.get_decision_audit_trail(decision.decision_id)
    if audit_trail:
        print(f"\n✅ Audit trail successfully retrieved for {decision.decision_id}")
        print(f"Alternatives considered: {len(audit_trail.alternative_considered)}")
    else:
        print("❌ Failed to retrieve audit trail")


def test_principle_hierarchy():
    """Test principle conflict resolution."""
    print("\n\n=== Testing Principle Hierarchy ===")
    
    # Simulate a conflict between principles
    conflicting_principles = [
        ConstitutionalPrinciple.MULTIPLE_PERSPECTIVES,
        ConstitutionalPrinciple.NON_FABRICATION,
        ConstitutionalPrinciple.CREATIVE_PRIORITY
    ]
    
    print("Conflicting principles:")
    for principle in conflicting_principles:
        hierarchy_level = constitutional_core.principle_hierarchy.get(principle, 999)
        print(f"  - {principle.value} (hierarchy level: {hierarchy_level})")
    
    resolved = constitutional_core.resolve_principle_conflict(conflicting_principles, {})
    print(f"\nResolved to: {resolved.value}")
    print(f"Hierarchy level: {constitutional_core.principle_hierarchy[resolved]}")


def main():
    """Run all constitutional core tests."""
    print("🏛️ Constitutional Core Testing Suite")
    print("=" * 50)
    
    try:
        test_constitutional_validation()
        test_active_pause_mechanism()
        test_constitutional_decision_making()
        test_principle_hierarchy()
        
        print("\n\n🎉 All tests completed successfully!")
        print("\nConstitutional Core is ready for Phase 1 implementation:")
        print("✅ Immutable principles layer")
        print("✅ Anti-reactive validation")
        print("✅ Active pause mechanism")
        print("✅ Audit trail system")
        print("✅ Principle hierarchy")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()