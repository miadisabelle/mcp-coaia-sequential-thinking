#!/bin/bash

# This script launches a Gemini instance to execute Scenario 4: Structural Tension Analysis & Pattern Recognition.

gemini -i "You are tasked with executing 'Scenario 4: Structural Tension Analysis & Pattern Recognition' as documented in 'experiments/scenario_4_structural_analysis.md'. Your goal is to validate the system's ability to detect creative vs reactive patterns, maintain structural tension, and provide agent self-awareness guidance for overcoming problem-solving biases.

Follow the steps outlined in the document. For each step, use the specified MCP tool with the provided arguments. Report the output of each tool call. After completing all steps, provide a final summary of the bias detection results, structural tension maintenance, and agent development guidance effectiveness. Pay special attention to how the system transforms problem-solving requests into creative orientation. Ensure all interactions with the MCP are clearly logged. Do not ask for confirmation for each tool call, assume auto-approval for this experiment." --approval-mode yolo