#!/bin/bash

# This script launches a Gemini instance to execute Scenario 2: Novel Solution Discovery & Goal Integration.

gemini -i "You are tasked with executing 'Scenario 2: Novel Solution Discovery & Goal Integration' as documented in 'experiments/scenario_2_novel_solution.md'. Your goal is to use the novelty search capabilities to discover innovative solutions and then integrate them into existing system goals, maintaining a resilient connection between exploration and exploitation.

Follow the steps outlined in the document. For each step, use the specified MCP tool with the provided arguments. Report the output of each tool call. After completing all steps, provide a final summary of the discovered solutions, their integration, and the impact on the system goal. Ensure all interactions with the MCP are clearly logged. Do not ask for confirmation for each tool call, assume auto-approval for this experiment." --approval-mode yolo
