#!/bin/bash

# This script launches a Gemini instance to execute Scenario 1: Creative Problem Reframing & Multi-Persona Analysis.

gemini -i "You are tasked with executing 'Scenario 1: Creative Problem Reframing & Multi-Persona Analysis' as documented in 'experiments/scenario_1_creative_reframing.md'. Your goal is to reframe a traditional problem into a desired outcome and then use the sequential thinking chain with Mia, Miette, and Haiku personas to analyze it from multiple angles, synthesizing an integrated perspective.

Follow the steps outlined in the document. For each step, use the specified MCP tool with the provided arguments. Report the output of each tool call. After completing all steps, provide a final summary of the insights gained and the overall outcome of the scenario. Ensure all interactions with the MCP are clearly logged. Do not ask for confirmation for each tool call, assume auto-approval for this experiment." --approval-mode yolo
