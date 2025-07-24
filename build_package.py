#!/usr/bin/env python3
"""
Package building and distribution script for Knowledge QA System.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Command: {cmd}")
        print(f"Error: {e.stderr}")
        return False


def clean_build_artifacts():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    artifacts = [
        "build/",
        "dist/",
        "*.egg-info/",
        "__pycache__/",
        ".pytest_cache/",
        "htmlcov/",
        ".coverage",
        "coverage.xml"
    ]
    
    for pattern in artifacts:
        if pattern.endswith("/"):
            # Directory
            for path in Path(".").glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"  Removed directory: {path}")
        else:
            # File or glob pattern
            for path in Path(".").glob(pattern):
                if path.is_file():
                    path.unlink()
                    print(f"  Removed file: {path}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    print(f"  Removed directory: {path}")


def run_tests():
    """Run the test suite."""
    return run_command("python -m pytest tests/ -v", "Running tests")


def run_linting():
    """Run code linting and formatting checks."""
    commands = [
        ("python -m black --check src/ tests/", "Black formatting check"),
        ("python -m isort --check-only src/ tests/", "Import sorting check"),
        ("python -m flake8 src/ tests/", "Flake8 linting"),
        ("python -m mypy src/", "Type checking"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    return True


def build_package():
    """Build the package."""
    return run_command("python -m build", "Building package")


def upload_to_pypi(test=True):
    """Upload package to PyPI (or TestPyPI)."""
    if test:
        cmd = "python -m twine upload --repository testpypi dist/*"
        desc = "Uploading to TestPyPI"
    else:
        cmd = "python -m twine upload dist/*"
        desc = "Uploading to PyPI"
    
    return run_command(cmd, desc)


def main():
    """Main build script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build and distribute Knowledge QA System")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-lint", action="store_true", help="Skip linting checks")
    parser.add_argument("--upload", choices=["test", "prod"], help="Upload to PyPI")
    parser.add_argument("--clean-only", action="store_true", help="Only clean artifacts")
    
    args = parser.parse_args()
    
    # Always clean first
    clean_build_artifacts()
    
    if args.clean_only:
        print("✅ Cleanup completed")
        return
    
    # Check if required tools are installed
    required_tools = ["build", "twine"]
    for tool in required_tools:
        try:
            subprocess.run([sys.executable, "-m", tool, "--help"], 
                         capture_output=True, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Required tool '{tool}' not found. Install with: pip install {tool}")
            sys.exit(1)
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            print("❌ Tests failed. Aborting build.")
            sys.exit(1)
    
    # Run linting
    if not args.skip_lint:
        if not run_linting():
            print("❌ Linting failed. Aborting build.")
            sys.exit(1)
    
    # Build package
    if not build_package():
        print("❌ Package build failed.")
        sys.exit(1)
    
    # Upload if requested
    if args.upload:
        test_upload = args.upload == "test"
        if not upload_to_pypi(test=test_upload):
            print("❌ Upload failed.")
            sys.exit(1)
    
    print("🎉 Build process completed successfully!")
    print("\nGenerated files:")
    dist_path = Path("dist")
    if dist_path.exists():
        for file in dist_path.iterdir():
            print(f"  📦 {file}")


if __name__ == "__main__":
    main()