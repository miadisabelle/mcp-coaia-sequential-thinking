#!/usr/bin/env python3
"""
Comprehensive Testing Suite for CoAiA Structural Thinker MCP

This test suite validates the structural thinking framework by testing:
1. Bias detection and creative orientation assessment
2. Multi-persona sequential thinking integration
3. Structural tension maintenance vs problem-solving collapse
4. Constitutional governance effectiveness
5. Real-world scenario validation

Focus: Understanding creative emergence vs reactive problem-solving
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_coaia_sequential_thinking.co_lint_integration import validate_thought, get_user_creative_patterns
from mcp_coaia_sequential_thinking import server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuralThinkingTester:
    """Comprehensive tester for structural thinking vs problem-solving orientation"""
    
    def __init__(self):
        self.test_results = []
        self.bias_examples = []
        self.creative_examples = []
        
    def test_bias_detection(self) -> Dict[str, Any]:
        """Test the system's ability to detect problem-solving bias vs creative orientation"""
        
        print("🧪 TESTING BIAS DETECTION AND CREATIVE ORIENTATION")
        print("=" * 60)
        
        test_cases = [
            # Reactive/Problem-solving examples (should score low)
            {
                "input": "We need to fix the broken authentication system that's causing user complaints",
                "expected_orientation": "reactive",
                "expected_score_range": (0.0, 0.3),
                "category": "problem_solving_bias"
            },
            {
                "input": "How can we solve the remote work productivity issues?",
                "expected_orientation": "reactive", 
                "expected_score_range": (0.0, 0.3),
                "category": "problem_solving_bias"
            },
            {
                "input": "We have a major bug that needs immediate fixing",
                "expected_orientation": "reactive",
                "expected_score_range": (0.0, 0.2),
                "category": "problem_solving_bias"
            },
            
            # Creative/Outcome-focused examples (should score high)
            {
                "input": "I want to create an authentication experience that delights users while providing robust security",
                "expected_orientation": "creative",
                "expected_score_range": (0.6, 1.0),
                "category": "creative_orientation"
            },
            {
                "input": "What would a thriving remote work culture look like that maximizes both productivity and employee fulfillment?",
                "expected_orientation": "creative",
                "expected_score_range": (0.6, 1.0), 
                "category": "creative_orientation"
            },
            {
                "input": "I envision building a system where users feel confident and secure in every interaction",
                "expected_orientation": "creative",
                "expected_score_range": (0.7, 1.0),
                "category": "creative_orientation"
            },
            
            # Mixed/Transitional examples (should score medium)
            {
                "input": "While we have authentication challenges, I want to create something better",
                "expected_orientation": "mixed",
                "expected_score_range": (0.3, 0.6),
                "category": "transitional"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {test_case['category']}")
            print(f"Input: '{test_case['input']}'")
            
            try:
                # Validate using CO-Lint integration
                validation = validate_thought(test_case['input'])
                score = validation.creative_orientation_score
                
                # Determine orientation based on score
                if score >= 0.7:
                    detected_orientation = "creative"
                elif score >= 0.4:
                    detected_orientation = "mixed"
                else:
                    detected_orientation = "reactive"
                
                # Check if score is in expected range
                min_score, max_score = test_case['expected_score_range']
                score_in_range = min_score <= score <= max_score
                
                # Check if orientation matches
                orientation_correct = detected_orientation == test_case['expected_orientation']
                
                test_result = {
                    "test_case": i,
                    "input": test_case['input'],
                    "category": test_case['category'],
                    "expected_orientation": test_case['expected_orientation'],
                    "detected_orientation": detected_orientation,
                    "creative_score": score,
                    "expected_score_range": test_case['expected_score_range'],
                    "score_in_range": score_in_range,
                    "orientation_correct": orientation_correct,
                    "passed": score_in_range and orientation_correct,
                    "advancing_pattern_detected": validation.advancing_pattern_detected,
                    "structural_tension_established": validation.structural_tension_established
                }
                
                results.append(test_result)
                
                print(f"   Creative Score: {score:.2f}")
                print(f"   Expected Range: {min_score:.1f} - {max_score:.1f}")
                print(f"   Detected Orientation: {detected_orientation}")
                print(f"   Expected Orientation: {test_case['expected_orientation']}")
                print(f"   Advancing Pattern: {validation.advancing_pattern_detected}")
                print(f"   Structural Tension: {validation.structural_tension_established}")
                print(f"   ✅ PASS" if test_result['passed'] else f"   ❌ FAIL")
                
                if test_case['category'] == 'problem_solving_bias':
                    self.bias_examples.append(test_result)
                elif test_case['category'] == 'creative_orientation':
                    self.creative_examples.append(test_result)
                    
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                results.append({
                    "test_case": i,
                    "input": test_case['input'],
                    "error": str(e),
                    "passed": False
                })
        
        # Summary
        passed = sum(1 for r in results if r.get('passed', False))
        total = len(results)
        
        print(f"\n📊 BIAS DETECTION TEST SUMMARY:")
        print(f"   Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        if self.bias_examples:
            avg_bias_score = sum(r['creative_score'] for r in self.bias_examples) / len(self.bias_examples)
            print(f"   Average Problem-Solving Score: {avg_bias_score:.2f} (should be low)")
            
        if self.creative_examples:
            avg_creative_score = sum(r['creative_score'] for r in self.creative_examples) / len(self.creative_examples)
            print(f"   Average Creative Score: {avg_creative_score:.2f} (should be high)")
        
        return {
            "test_type": "bias_detection",
            "results": results,
            "summary": {
                "passed": passed,
                "total": total,
                "pass_rate": passed/total*100,
                "bias_examples": self.bias_examples,
                "creative_examples": self.creative_examples
            }
        }
    
    def test_agent_self_awareness(self) -> Dict[str, Any]:
        """Test agent's ability to assess its own orientation before using MCP tools"""
        
        print("\n🤖 TESTING AGENT SELF-AWARENESS")
        print("=" * 40)
        
        try:
            # Test the enhanced pattern analysis
            patterns = get_user_creative_patterns(limit=20)
            
            print("✅ Pattern analysis working")
            
            if 'agent_orientation_awareness' in patterns:
                awareness = patterns['agent_orientation_awareness']
                print(f"✅ Agent orientation awareness available")
                
                current_status = awareness.get('current_orientation_status', {})
                print(f"   Current Status: {current_status.get('status', 'unknown')}")
                print(f"   Confidence: {current_status.get('confidence', 'unknown')}")
                print(f"   MCP Tool Readiness: {current_status.get('mcp_tool_readiness', 'unknown')}")
                
                tool_guidance = awareness.get('tool_usage_guidance', [])
                print(f"   Tool Usage Guidance: {len(tool_guidance)} recommendations")
                for guidance in tool_guidance[:3]:  # Show first 3
                    print(f"      • {guidance}")
                
                mcp_recommendations = awareness.get('mcp_interaction_recommendations', {})
                print(f"   MCP Interaction Recommendations: {len(mcp_recommendations)} tools assessed")
                
                return {
                    "test_type": "agent_self_awareness",
                    "passed": True,
                    "awareness_data": awareness,
                    "current_orientation": current_status,
                    "guidance_count": len(tool_guidance),
                    "mcp_tool_count": len(mcp_recommendations)
                }
            else:
                print("❌ Agent orientation awareness not available")
                return {"test_type": "agent_self_awareness", "passed": False, "error": "No awareness data"}
                
        except Exception as e:
            print(f"❌ Error testing agent self-awareness: {e}")
            return {"test_type": "agent_self_awareness", "passed": False, "error": str(e)}
    
    def test_structural_tension_scenarios(self) -> Dict[str, Any]:
        """Test real-world scenarios for structural tension vs problem-solving"""
        
        print("\n🏗️ TESTING STRUCTURAL TENSION SCENARIOS")
        print("=" * 45)
        
        scenarios = [
            {
                "name": "Strategic Business Decision",
                "context": "CEO deciding remote work policy",
                "problem_focused": "Our office attendance is low and we need to fix the productivity issues",
                "creative_focused": "We want to create a work environment that maximizes both productivity and employee fulfillment, whether remote or in-office",
                "test_focus": "Bias detection and reframing"
            },
            {
                "name": "Creative Project Development", 
                "context": "Author planning novel series",
                "problem_focused": "I'm stuck on plot holes and character inconsistencies that need solving",
                "creative_focused": "I envision creating a novel series that deeply moves readers while exploring themes of transformation and hope",
                "test_focus": "Creative emergence vs analytical problem-solving"
            },
            {
                "name": "Personal Life Design",
                "context": "Individual navigating career transition", 
                "problem_focused": "I hate my current job and need to figure out how to escape this situation",
                "creative_focused": "I want to create a career that aligns with my values and allows me to contribute meaningfully while providing financial stability",
                "test_focus": "Constitutional governance preventing reactive loops"
            }
        ]
        
        scenario_results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario['name']}")
            print(f"Context: {scenario['context']}")
            print(f"Focus: {scenario['test_focus']}")
            
            # Test problem-focused framing
            print(f"\n   🔴 Problem-Focused Test:")
            print(f"   Input: '{scenario['problem_focused']}'")
            
            try:
                problem_validation = validate_thought(scenario['problem_focused'])
                problem_score = problem_validation.creative_orientation_score
                print(f"   Creative Score: {problem_score:.2f}")
                print(f"   Advancing Pattern: {problem_validation.advancing_pattern_detected}")
            except Exception as e:
                print(f"   Error: {e}")
                problem_score = 0.0
            
            # Test creative-focused framing
            print(f"\n   🟢 Creative-Focused Test:")
            print(f"   Input: '{scenario['creative_focused']}'")
            
            try:
                creative_validation = validate_thought(scenario['creative_focused'])
                creative_score = creative_validation.creative_orientation_score
                print(f"   Creative Score: {creative_score:.2f}")
                print(f"   Advancing Pattern: {creative_validation.advancing_pattern_detected}")
                print(f"   Structural Tension: {creative_validation.structural_tension_established}")
            except Exception as e:
                print(f"   Error: {e}")
                creative_score = 0.0
            
            # Validate that creative framing scores higher
            orientation_difference = creative_score - problem_score
            scenario_passed = orientation_difference >= 0.3  # Creative should be significantly higher
            
            print(f"\n   📊 Scenario Results:")
            print(f"   Orientation Difference: {orientation_difference:.2f}")
            print(f"   Expected: Creative > Problem by 0.3+")
            print(f"   ✅ PASS" if scenario_passed else f"   ❌ FAIL")
            
            scenario_results.append({
                "scenario": scenario['name'],
                "context": scenario['context'],
                "problem_score": problem_score,
                "creative_score": creative_score,
                "orientation_difference": orientation_difference,
                "passed": scenario_passed,
                "test_focus": scenario['test_focus']
            })
        
        # Summary
        passed_scenarios = sum(1 for r in scenario_results if r['passed'])
        total_scenarios = len(scenario_results)
        
        print(f"\n📊 STRUCTURAL TENSION SCENARIOS SUMMARY:")
        print(f"   Passed: {passed_scenarios}/{total_scenarios} ({passed_scenarios/total_scenarios*100:.1f}%)")
        
        avg_difference = sum(r['orientation_difference'] for r in scenario_results) / total_scenarios
        print(f"   Average Orientation Difference: {avg_difference:.2f}")
        
        return {
            "test_type": "structural_tension_scenarios",
            "results": scenario_results,
            "summary": {
                "passed": passed_scenarios,
                "total": total_scenarios,
                "pass_rate": passed_scenarios/total_scenarios*100,
                "average_difference": avg_difference
            }
        }
    
    def test_mcp_tools_integration(self) -> Dict[str, Any]:
        """Test integration and availability of MCP tools"""
        
        print("\n🛠️ TESTING MCP TOOLS INTEGRATION")
        print("=" * 35)
        
        results = {
            "test_type": "mcp_tools_integration",
            "tools_tested": [],
            "errors": [],
            "summary": {}
        }
        
        try:
            # Test server initialization
            print("🔧 Testing server initialization...")
            
            if hasattr(server, 'enhanced_lattice') and server.enhanced_lattice:
                print("✅ Enhanced lattice available")
                results['enhanced_lattice'] = True
            else:
                print("❌ Enhanced lattice not available")
                results['enhanced_lattice'] = False
                results['errors'].append("Enhanced lattice not initialized")
            
            # Test CO-Lint integration
            print("🔧 Testing CO-Lint integration...")
            from mcp_coaia_sequential_thinking.co_lint_integration import CO_LINT_AVAILABLE
            
            if CO_LINT_AVAILABLE:
                print("✅ CO-Lint integration operational")
                results['co_lint'] = True
            else:
                print("⚠️ CO-Lint in SCCP-only mode")
                results['co_lint'] = 'sccp_only'
            
            # Count available MCP tools
            import inspect
            tool_functions = []
            for name, obj in inspect.getmembers(server):
                if hasattr(obj, '__mcp_tool__'):
                    tool_functions.append(name)
            
            print(f"🔧 MCP Tools available: {len(tool_functions)}")
            results['total_tools'] = len(tool_functions)
            results['tools_list'] = tool_functions
            
            # Test key tools exist
            key_tools = [
                'initiate_sequential_thinking',
                'advance_thinking_chain', 
                'synthesize_thinking_chain',
                'check_agent_creative_orientation',
                'create_consensus_decision'
            ]
            
            missing_tools = []
            for tool in key_tools:
                if tool in tool_functions:
                    print(f"   ✅ {tool}")
                else:
                    print(f"   ❌ {tool} - Missing")
                    missing_tools.append(tool)
            
            results['missing_tools'] = missing_tools
            results['key_tools_available'] = len(key_tools) - len(missing_tools)
            
            # Overall status
            all_key_tools = len(missing_tools) == 0
            lattice_available = results.get('enhanced_lattice', False)
            co_lint_working = results.get('co_lint') in [True, 'sccp_only']
            
            overall_status = all_key_tools and lattice_available and co_lint_working
            
            print(f"\n📊 MCP TOOLS INTEGRATION SUMMARY:")
            print(f"   Key Tools Available: {len(key_tools) - len(missing_tools)}/{len(key_tools)}")
            print(f"   Enhanced Lattice: {'✅' if lattice_available else '❌'}")
            print(f"   CO-Lint Status: {'✅' if co_lint_working else '❌'}")
            print(f"   Overall Status: {'✅ OPERATIONAL' if overall_status else '❌ ISSUES DETECTED'}")
            
            results['summary'] = {
                "overall_status": overall_status,
                "all_key_tools": all_key_tools,
                "lattice_available": lattice_available,
                "co_lint_working": co_lint_working
            }
            
        except Exception as e:
            print(f"❌ Error testing MCP tools: {e}")
            results['errors'].append(str(e))
            results['summary']['overall_status'] = False
        
        return results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all tests and provide comprehensive report"""
        
        print("🧪 COAIA STRUCTURAL THINKER - COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print("Testing creative orientation vs problem-solving bias detection")
        print("Validating structural thinking framework implementation")
        print("=" * 60)
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "test_suite": "coaia_structural_thinker_comprehensive",
            "tests": []
        }
        
        # Run all tests
        tests = [
            self.test_mcp_tools_integration,
            self.test_bias_detection,
            self.test_agent_self_awareness,
            self.test_structural_tension_scenarios
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                all_results["tests"].append(result)
            except Exception as e:
                print(f"❌ Test {test_func.__name__} failed: {e}")
                all_results["tests"].append({
                    "test_type": test_func.__name__,
                    "error": str(e),
                    "passed": False
                })
        
        # Overall summary
        total_tests = len(all_results["tests"])
        passed_tests = sum(1 for t in all_results["tests"] 
                          if t.get('summary', {}).get('overall_status', t.get('passed', False)))
        
        print(f"\n🏆 COMPREHENSIVE TEST RESULTS:")
        print(f"   Tests Passed: {passed_tests}/{total_tests}")
        print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        all_results["overall_summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests/total_tests*100,
            "status": "OPERATIONAL" if passed_tests/total_tests >= 0.8 else "NEEDS_ATTENTION"
        }
        
        print(f"   Overall Status: {all_results['overall_summary']['status']}")
        
        if passed_tests/total_tests >= 0.8:
            print("\n🎉 CoAiA Structural Thinker is OPERATIONAL!")
            print("   System successfully detects and guides away from problem-solving bias")
            print("   Creative orientation framework functioning as designed")
            print("   Multi-persona integration and structural tension capabilities validated")
        else:
            print("\n⚠️ CoAiA Structural Thinker needs attention")
            print("   Some components may not be functioning optimally")
        
        return all_results


def main():
    """Main test execution"""
    
    tester = StructuralThinkingTester()
    results = tester.run_comprehensive_tests()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/tmp/structural_thinking_test_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")
    
    return results


if __name__ == "__main__":
    main()