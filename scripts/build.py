#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import os


def run_command(
    command: list[str], description: str, exit_on_error: bool = False
) -> None:
    """Run a command and optionally exit if it fails."""
    print(f"\n=== Running {description} ===")
    result = subprocess.run(command)
    if result.returncode != 0:
        if exit_on_error:
            print(f"Error: {description} failed!")
            sys.exit(1)
        else:
            print(f"Warning: {description} encountered issues.")
    else:
        print(f"✓ {description} passed")


def main() -> None:
    # Ensure we're operating on the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Clean up any previous builds
    run_command(["rm", "-rf", "dist", "build", "*.egg-info"], "cleanup")

    # Rest of commands...
    run_command(["ruff", "check", "--fix", "--exit-zero", "."], "ruff fix")
    run_command(["ruff", "check", "--exit-zero", "."], "ruff linting")
    run_command(["ruff", "format", "."], "ruff format fix")
    run_command(["mypy", "."], "mypy type checking", exit_on_error=True)
    run_command(["pytest", "-v"], "pytest", exit_on_error=True)
    run_command(["python", "-m", "build"], "package build", exit_on_error=True)

    print("\n��� Build completed! ✨")


if __name__ == "__main__":
    main()
