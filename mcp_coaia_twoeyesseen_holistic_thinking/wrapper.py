import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="CoAiA Sequential Thinking MCP Server. This command starts the MCP server. "
                    "It offers a suite of tools for sequential thought processing and analysis."
    )
    # No explicit add_argument for --help here, argparse handles it automatically

    # Parse known arguments to allow FastMCP to handle its own args later
    args, unknown_args = parser.parse_known_args()

    # If --help was requested, argparse will handle printing the help message and exiting.
    # So, if we reach this point, it means --help was NOT requested, or it was handled.
    # We can then proceed to run the server.

    # Construct the command to run the original server.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(script_dir, 'server.py')
    
    # Pass along all arguments that were not parsed by our wrapper's argparse
    command = [sys.executable, server_path] + unknown_args 

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting MCP server: {e}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
