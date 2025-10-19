import json
import pathlib
import sys

import click

from . import rules

@click.group()
def cli():
    """A linter to enforce Creative Orientation governance in repository documentation."""
    pass

@cli.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True, path_type=pathlib.Path))
@click.option("--config", "config_path", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=pathlib.Path), default=".co-lint.json", help="Path to the configuration file.")
@click.option("--format", "output_format", type=click.Choice(["text", "json", "github"], case_sensitive=False), default="text", help="The output format.")
@click.option("--rules", "rules_cli", help="Comma-separated list of rules to run (e.g., COL001,COL003).")
@click.option("--severity-threshold", type=click.Choice(["warn", "error"], case_sensitive=False), default="error", help="Set the severity threshold for reporting.")
@click.option("--fail-on", type=click.Choice(["warn", "error"], case_sensitive=False), default="error", help="Set the severity level that will trigger a non-zero exit code.")
@click.option("--no-color", is_flag=True, help="Disable colorized output.")
def lint(paths, config_path, output_format, rules_cli, severity_threshold, fail_on, no_color):
    """Lint the specified files or directories."""
    click.echo("Starting CO-Lint...")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        click.echo(f"Loaded configuration from: {config_path}")
    except FileNotFoundError:
        click.secho(f"Error: Configuration file not found at {config_path}", fg="red")
        sys.exit(1)
    except json.JSONDecodeError:
        click.secho(f"Error: Could not parse configuration file at {config_path}", fg="red")
        sys.exit(1)

    if rules_cli:
        active_rules = {r.strip().upper() for r in rules_cli.split(',')}
    else:
        active_rules = {r.strip().upper() for r in config.get("rule_filters", list(rules.ALL_RULES.keys()))}
    click.echo(f"Active rules: {sorted(list(active_rules))}")

    findings = []
    files_to_lint = find_files(paths, config.get("ignore", []))
    
    for file_path in files_to_lint:
        click.echo(f"Checking: {file_path}")
        for rule_id, check_function in rules.ALL_RULES.items():
            if rule_id in active_rules:
                try:
                    # Pass rule-specific config if needed in the future
                    rule_config = {"max_lines": config.get("max_structural_tension_search_lines", 120)}
                    findings.extend(check_function(file_path, **rule_config))
                except Exception as e:
                    click.secho(f"Error running rule {rule_id} on file {file_path}: {e}", fg="red")

    overrides = config.get("severity_overrides", {})
    for finding in findings:
        if finding["rule"] in overrides:
            new_severity = overrides[finding["rule"]]
            if new_severity in ["error", "warn", "off"]:
                finding["severity"] = new_severity
    
    findings = [f for f in findings if f["severity"] != "off"]

    report_findings(findings, output_format, fail_on)

def report_findings(findings, output_format, fail_on):
    has_error = any(f["severity"] == "error" for f in findings)
    has_warn = any(f["severity"] == "warn" for f in findings)

    if not findings:
        click.secho("Linting complete. No issues found.", fg="green")
    else:
        click.secho(f"\nFound {len(findings)} issue(s):", fg="yellow")
        for finding in sorted(findings, key=lambda x: (x['path'], x['line'])):
            if finding["severity"] == "error":
                click.secho(format_finding(finding, output_format), fg="red")
            else:
                click.secho(format_finding(finding, output_format), fg="yellow")

    if (fail_on == "error" and has_error) or (fail_on == "warn" and (has_error or has_warn)):
        sys.exit(1)

def find_files(paths, ignore_patterns):
    """Yields all markdown files to be linted, respecting ignore patterns."""
    for path in paths:
        if path.is_dir():
            for sub_path in path.rglob("*.md"):
                yield sub_path
        elif path.is_file() and path.suffix == ".md":
            yield path

def format_finding(finding, format):
    """Formats a finding for output."""
    return f"{finding['path']}:{finding['line']}  [{finding['severity'].upper()} {finding['rule']}] {finding['message']}"

if __name__ == "__main__":
    cli()