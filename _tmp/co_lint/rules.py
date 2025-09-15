import pathlib
from typing import List, Optional

def _get_lines(file_path: pathlib.Path, content: Optional[List[bytes]]) -> List[bytes]:
    if content is not None:
        return content
    with open(file_path, 'rb') as f:
        return f.readlines()

def check_col001(file_path: pathlib.Path, max_lines: int, content: Optional[List[bytes]] = None):
    """Checks if the file contains the required Structural Tension block."""
    findings = []
    required_labels = [
        b"- Desired Outcome:",
        b"- Current Reality:",
        b"- Natural Progression:"
    ]
    
    try:
        lines = _get_lines(file_path, content)[:max_lines]
        found_labels = set()
        for label in required_labels:
            for line in lines:
                if line.strip().startswith(label):
                    found_labels.add(label)
                    break
        
        if len(found_labels) != len(required_labels):
            missing = [l.decode() for l in required_labels if l not in found_labels]
            findings.append({
                "path": str(file_path),
                "line": 1,
                "severity": "error",
                "rule": "COL001",
                "message": f"Required Structural Tension block is missing or incomplete. Missing: {', '.join(missing)}"
            })
    except Exception as e:
        print(f"Error in check_col001: {e}")

    return findings

def check_col002(file_path: pathlib.Path, content: Optional[List[bytes]] = None, **kwargs):
    """Checks for a neutral '## Observations' section."""
    findings = []
    in_observations = False
    observations_found = False
    forbidden_words = [b'fix', b'mitigate', b'eliminate', b'solve', b'address', b'remediate']

    try:
        lines = _get_lines(file_path, content)
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.lower().startswith(b'## observations'):
                observations_found = True
                in_observations = True
                continue

            if in_observations:
                if stripped_line.startswith(b'## '):
                    in_observations = False
                    continue
                
                for word in forbidden_words:
                    if word in line.lower():
                        findings.append({
                            "path": str(file_path),
                            "line": i + 1,
                            "severity": "warn",
                            "rule": "COL002",
                            "message": f"Observations section may contain non-neutral language. Found: '{word.decode()}'"
                        })
                        break 
        
        if not observations_found:
            findings.append({
                "path": str(file_path),
                "line": 1,
                "severity": "error",
                "rule": "COL002",
                "message": "Required '## Observations' section is missing."
            })

    except Exception as e:
        print(f"Error in check_col002: {e}")

    return findings

def check_col003(file_path: pathlib.Path, content: Optional[List[bytes]] = None, **kwargs):
    """Checks for a '## Structural Assessment' section with correct terminology."""
    findings = []
    in_assessment = False
    assessment_found = False
    required_words_found = False
    required_words = [b'advance', b'advancing', b'oscillate', b'oscillating']
    forbidden_word = b'problem'

    try:
        lines = _get_lines(file_path, content)
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.lower().startswith(b'## structural assessment'):
                assessment_found = True
                in_assessment = True
                continue

            if in_assessment:
                if stripped_line.startswith(b'## '):
                    in_assessment = False
                    continue
                
                for word in required_words:
                    if word in line.lower():
                        required_words_found = True
                        break
                
                if forbidden_word in line.lower():
                    findings.append({
                        "path": str(file_path),
                        "line": i + 1,
                        "severity": "error",
                        "rule": "COL003",
                        "message": f"Structural Assessment section should not contain problem-framing language like '{forbidden_word.decode()}'."
                    })

        if not assessment_found:
            findings.append({
                "path": str(file_path),
                "line": 1,
                "severity": "error",
                "rule": "COL003",
                "message": "Required '## Structural Assessment' section is missing."
            })
        elif not required_words_found:
            findings.append({
                "path": str(file_path),
                "line": 1,
                "severity": "error",
                "rule": "COL003",
                "message": "Structural Assessment section must contain advancing/oscillating terminology."
            })

    except Exception as e:
        print(f"Error in check_col003: {e}")

    return findings

def check_col004(file_path: pathlib.Path, content: Optional[List[bytes]] = None, **kwargs):
    """Checks for create-language in the optional '## Advancing Moves' section."""
    findings = []
    in_section = False
    forbidden_words = [b'fix', b'mitigate', b'eliminate', b'solve', b'address', b'remediate']

    try:
        lines = _get_lines(file_path, content)
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.lower().startswith(b'## advancing moves'):
                in_section = True
                continue

            if in_section:
                if stripped_line.startswith(b'## '):
                    in_section = False
                    continue
                
                for word in forbidden_words:
                    if word in line.lower():
                        findings.append({
                            "path": str(file_path),
                            "line": i + 1,
                            "severity": "warn",
                            "rule": "COL004",
                            "message": f"'Advancing Moves' section prefers create-language. Found elimination-focused term: '{word.decode()}'"
                        })
                        break

    except Exception as e:
        print(f"Error in check_col004: {e}")

    return findings

def check_col005(file_path: pathlib.Path, content: Optional[List[bytes]] = None, **kwargs):
    """Checks for forbidden language outside of technical contexts."""
    findings = []
    in_code_fence = False
    in_technical_section = False
    forbidden_words = [b'fix', b'mitigate', b'eliminate', b'solve', b'address', b'remediate']
    technical_section_keywords = [b'troubleshooting', b'error', b'command', b'cli', b'api', b'performance', b'benchmark', b'test', b'installation', b'uninstall']

    try:
        lines = _get_lines(file_path, content)
        for i, line in enumerate(lines):
            stripped_line = line.strip()

            if stripped_line.startswith(b'```'):
                in_code_fence = not in_code_fence
                continue

            if stripped_line.startswith(b'##'):
                in_technical_section = any(keyword in stripped_line.lower() for keyword in technical_section_keywords)

            if in_code_fence or in_technical_section:
                continue

            for word in forbidden_words:
                if word in line.lower():
                    findings.append({
                        "path": str(file_path),
                        "line": i + 1,
                        "severity": "warn",
                        "rule": "COL005",
                        "message": f"Consider conscious language choice. Found elimination-focused term: '{word.decode()}'. If this section is problem-focused, consider a dedicated 'Problem Definition' or 'Reactive Measures' heading."
                    })
                    break

    except Exception as e:
        print(f"Error in check_col005: {e}")

    return findings

ALL_RULES = {
    "COL001": check_col001,
    "COL002": check_col002,
    "COL003": check_col003,
    "COL004": check_col004,
    "COL005": check_col005,
}