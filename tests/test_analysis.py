import unittest
from mcp_coaia_sequential_thinking.models import ThoughtStage, ThoughtData
from mcp_coaia_sequential_thinking.analysis import ThoughtAnalyzer


class TestThoughtAnalyzer(unittest.TestCase):
    """Test cases for the ThoughtAnalyzer class."""

    def setUp(self):
        """Set up test data."""
        self.thought1 = ThoughtData(
            thought="Define the desired outcome for the project",
            thought_number=1,
            total_thoughts=5,
            next_thought_needed=True,
            stage=ThoughtStage.DESIRED_OUTCOME,
            tags=["project", "outcome"]
        )

        self.thought2 = ThoughtData(
            thought="Assess the current state of the project",
            thought_number=2,
            total_thoughts=5,
            next_thought_needed=True,
            stage=ThoughtStage.CURRENT_REALITY,
            tags=["project", "reality"]
        )

        self.thought3 = ThoughtData(
            thought="Identify the gap between outcome and reality",
            thought_number=3,
            total_thoughts=5,
            next_thought_needed=True,
            stage=ThoughtStage.STRUCTURAL_TENSION,
            tags=["gap", "tension"]
        )

        self.thought4 = ThoughtData(
            thought="Create an action plan to bridge the gap",
            thought_number=4,
            total_thoughts=5,
            next_thought_needed=True,
            stage=ThoughtStage.ACTION_STEP,
            tags=["action", "plan"]
        )
        
        self.thought5 = ThoughtData(
            thought="Complete the first action step",
            thought_number=5,
            total_thoughts=5,
            next_thought_needed=False,
            stage=ThoughtStage.ACTION_STEP,
            tags=["action", "completed"]
        )

        self.all_thoughts = [self.thought1, self.thought2, self.thought3, self.thought4, self.thought5]

    def test_generate_summary_empty(self):
        """Test generating summary with no thoughts."""
        summary = ThoughtAnalyzer.generate_summary([])
        self.assertEqual(summary, {"creativeProcessSummary": "No thoughts recorded yet"})

    def test_generate_summary(self):
        """Test generating summary with thoughts."""
        summary = ThoughtAnalyzer.generate_summary(self.all_thoughts)["creativeProcessSummary"]

        self.assertEqual(summary["desiredOutcome"], "Define the desired outcome for the project")
        self.assertEqual(summary["currentRealitySnapshot"], "Assess the current state of the project")
        self.assertEqual(summary["structuralTensionStatus"]["identifiedTensions"], 1)
        self.assertEqual(summary["structuralTensionStatus"]["activeTensions"], 1)
        self.assertEqual(summary["structuralTensionStatus"]["tensionResolutionProgress"], "50% resolved")
        self.assertEqual(summary["actionSteps"]["total"], 2)
        self.assertEqual(summary["actionSteps"]["completed"], 1)
        self.assertEqual(summary["actionSteps"]["nextSteps"], ["Create an action plan to bridge the gap"])
        self.assertEqual(summary["creativeElementsBreakdown"]["Desired Outcome"], 1)
        self.assertEqual(summary["creativeElementsBreakdown"]["Current Reality"], 1)
        self.assertEqual(summary["creativeElementsBreakdown"]["Structural Tension"], 1)
        self.assertEqual(summary["creativeElementsBreakdown"]["Action Step"], 2)


if __name__ == "__main__":
    unittest.main()
