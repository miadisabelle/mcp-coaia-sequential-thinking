import pathlib
from typing import List

from . import rules

def lint_text(text_content: str) -> List[dict]:
    """Lints a string of text against all CO-Lint rules."""
    findings = []
    # The check functions expect a list of bytes, so we encode the string
    # and split it into lines.
    content_lines = [line + b'\n' for line in text_content.encode('utf-8').splitlines()]
    
    # A dummy path is needed for the function signature, but it won't be used
    # when content is passed directly.
    dummy_path = pathlib.Path("realtime_buffer.md")

    # Call all check functions with the content
    for rule_id, check_function in rules.ALL_RULES.items():
        # Note: This simplified call doesn't pass specific configs like max_lines,
        # but could be extended to do so.
        findings.extend(check_function(dummy_path, content=content_lines, max_lines=120))

    return findings

if __name__ == '__main__':
    print("--- Testing Real-Time Filter ---")

    # Test Case 1: Failing Text
    failing_text = (
        "# My Document\n"
        "This is a test to solve a problem.\n"
        "## Observations\n"
        "We need to fix this."
    )
    print("\n--- Linting Failing Text ---")
    failing_findings = lint_text(failing_text)
    if failing_findings:
        print(f"Found {len(failing_findings)} issues:")
        for finding in failing_findings:
            print(f"  - [L{finding['line']} {finding['rule']}] {finding['message']}")
    else:
        print("No issues found.")

    # Test Case 2: Passing Text
    passing_text = (
        "- Desired Outcome: A passing test.\n"
        "- Current Reality: The test is being written.\n"
        "- Natural Progression: The test will be executed.\n\n"
        "## Observations\n"
        "The content is valid markdown.\n"
        "## Structural Assessment\n"
        "This is an advancing pattern."
    )
    print("\n--- Linting Passing Text ---")
    passing_findings = lint_text(passing_text)
    if passing_findings:
        print(f"Found {len(passing_findings)} issues:")
        for finding in passing_findings:
            print(f"  - [L{finding['line']} {finding['rule']}] {finding['message']}")
    else:
        print("No issues found.")
