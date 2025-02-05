#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os

def bump_version(version_type):
    """Bump the version."""
    subprocess.run(["bumpversion", version_type], check=True)

def run_tests():
    """Run tests using pytest."""
    result = subprocess.run(["pytest", "--cov=pytorch_tabnet"], check=False)
    if result.returncode != 0:
        print("Tests failed. Aborting build.")
        sys.exit(result.returncode)

def build_package():
    """Build the package."""
    subprocess.run(["python", "setup.py", "sdist", "bdist_wheel"], check=True)

def release():
    """Release the package."""
    run_tests()
    bump_version("patch")
    build_package()
    subprocess.run(["git", "push", "--tags"], check=True)

def main():
    parser = argparse.ArgumentParser(description="Build and release script.")
    parser.add_argument("command", choices=["bump", "build", "release"], help="Command to execute")
    parser.add_argument("--type", choices=["major", "minor", "patch"], help="Version type to bump")

    args = parser.parse_args()

    if args.command == "bump":
        if not args.type:
            print("Please specify a version type to bump: major, minor, or patch.")
            sys.exit(1)
        bump_version(args.type)
    elif args.command == "build":
        run_tests()
        build_package()
    elif args.command == "release":
        release()

if __name__ == "__main__":
    main()
