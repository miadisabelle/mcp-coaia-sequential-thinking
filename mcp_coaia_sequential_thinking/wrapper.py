import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="CoAiA Sequential Thinking MCP Server. This command starts the MCP server. "
                    "It offers a suite of tools for sequential thought processing and analysis."
    )
    args, unknown = parser.parse_known_args()

    if args.help:
        print(parser.description)
        print("\nAvailable Tools (accessed via MCP client after server starts):")
        print("  - process_thought: Add a sequential thought with its metadata.")
        print("  - generate_summary: Generate a summary of the entire thinking process.")
        print("  - clear_history: Clear the thought history.")
        print("  - export_session: Export the current thinking session to a file.")
        print("  - import_session: Import a thinking session from a file.")
        print("  - check_integration_status: Check the integration status with COAIA Memory system.")
        print("  - validate_thought_content: Validate thought content using SCCP-enhanced CO-Lint filtering.")
        print("\nTo interact with these tools, connect to the running server via an MCP client (e.g., `mcp call <tool_name>`).")
        sys.exit(0)

    # If --help is not requested, run the original server
    # We need to ensure the correct module is in sys.path for the server to import its dependencies
    # This assumes the wrapper is in the same directory as server.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Execute the original server.py's main function
    # This is a bit tricky as we want to run it as if it were the main entry point
    # We can use subprocess to run it as a separate process
    # Or, more cleanly, import and call its main function if it's designed for that
    # Given the FastMCP setup, running it as a subprocess is safer to avoid conflicts
    # with FastMCP's own argument parsing and server startup logic.
    
    # Construct the command to run the original server.py
    server_path = os.path.join(script_dir, 'server.py')
    command = [sys.executable, server_path] + sys.argv[1:] # Pass along other arguments

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
