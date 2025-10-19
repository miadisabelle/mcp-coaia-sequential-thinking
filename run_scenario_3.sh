#!/bin/bash

# This script launches a Gemini instance to execute Scenario 3: Constitutional Decision Making & Agent Collaboration.

gemini -i "You are tasked with executing 'Scenario 3: Constitutional Decision Making & Agent Collaboration' as documented in 'experiments/scenario_3_constitutional_governance.md'. Your goal is to test the constitutional governance framework and demonstrate how agents collaborate to make principle-based decisions, ensuring constitutional compliance and maintaining audit trails.

Follow the steps outlined in the document. For each step, use the specified MCP tool with the provided arguments. Report the output of each tool call. After completing all steps, provide a final summary of the decision-making process, constitutional compliance results, and collaboration effectiveness. Ensure all interactions with the MCP are clearly logged. Do not ask for confirmation for each tool call, assume auto-approval for this experiment." --approval-mode yolo