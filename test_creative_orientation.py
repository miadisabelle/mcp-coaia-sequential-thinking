#!/usr/bin/env python3
"""
Test the new creative orientation tools
"""

import json
import subprocess
import sys

def test_creative_emergence():
    """Test the initiate_creative_emergence tool"""
    
    print("🧠 Mia: Testing creative orientation foundation...")
    print("🌸 Miette: Let's see our beautiful new system in action! ✨")
    
    # Test case 1: Good creative orientation
    test_cases = [
        {
            "name": "Good Creative Orientation",
            "request": "Currently have basic MCP server functionality",
            "desired_outcome": "Create a seamless multi-persona creative collaboration experience",
            "primary_purpose": "manifest_collaborative_intelligence"
        },
        {
            "name": "Problem-Solving Pattern (should be reframed)",
            "request": "The system has performance issues",
            "desired_outcome": "Fix the slow response times",
            "primary_purpose": "solve_performance"
        },
        {
            "name": "Archetype Focus Test", 
            "request": "Have theoretical framework from agents",
            "desired_outcome": "Manifest integrated wisdom architecture",
            "primary_purpose": "wisdom_synthesis",
            "archetype_focus": "mia"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test_case['name']}")
        print(f"{'='*60}")
        
        # Build tool call
        args = {
            "request": test_case["request"],
            "desired_outcome": test_case["desired_outcome"], 
            "primary_purpose": test_case["primary_purpose"]
        }
        
        if "archetype_focus" in test_case:
            args["archetype_focus"] = test_case["archetype_focus"]
        
        # Call the tool via the MCP server
        try:
            cmd = [
                "python", "-c", f"""
import sys
sys.path.append('.')
from mcp_coaia_sequential_thinking.server import mcp
from mcp_coaia_sequential_thinking.creative_orientation_foundation import creative_orientation_foundation
from mcp_coaia_sequential_thinking.generative_agent_lattice import generative_lattice

# Test the foundation directly
validation = creative_orientation_foundation.validate_creative_orientation(
    request="{test_case['request']}", 
    desired_outcome="{test_case['desired_outcome']}"
)

print("Constitutional Validation:")
print(f"  Is Generative: {{validation.is_generative}}")
print(f"  Orientation: {{validation.orientation_type.value}}")
print(f"  Tension Strength: {{validation.tension_strength.value}}")
print(f"  Creative Alignment: {{validation.creative_alignment_score:.2f}}")

if validation.guidance_for_advancement:
    print("\\nGuidance:")
    for guidance in validation.guidance_for_advancement:
        print(f"  - {{guidance}}")
"""
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"Error: {result.stderr}")
                
        except Exception as e:
            print(f"Test failed: {e}")

if __name__ == "__main__":
    test_creative_emergence()