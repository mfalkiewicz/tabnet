"""Script to update TabNet PySpark implementation."""

import os
import shutil
from pathlib import Path

def update_tabnet_pyspark():
    """Update TabNet PySpark implementation with new persistence logic."""
    project_root = Path(__file__).parent.parent
    old_file = project_root / "pytorch_tabnet/spark/tabnet_pyspark.py"
    new_file = project_root / "pytorch_tabnet/spark/tabnet_pyspark_new.py"
    backup_file = old_file.with_suffix('.py.bak')

    # Create backup of original file
    shutil.copy2(old_file, backup_file)
    print(f"Created backup at: {backup_file}")

    # Read the original file to get the TabNetParams and transform implementation
    with open(old_file, 'r') as f:
        old_content = f.read()

    # Read the new file
    with open(new_file, 'r') as f:
        new_content = f.read()

    # Extract TabNetParams implementation from old file
    params_start = old_content.find("class TabNetParams")
    params_end = old_content.find("class TabNetEstimator")
    params_impl = old_content[params_start:params_end].strip()

    # Extract transform implementation from old file
    transform_start = old_content.find("def _transform")
    transform_end = old_content.find("def write", transform_start)
    transform_impl = old_content[transform_start:transform_end].strip()

    # Replace placeholders in new content
    new_content = new_content.replace(
        "# ... [Keep all parameter definitions and methods unchanged] ...",
        params_impl
    )
    new_content = new_content.replace(
        "# ... [Keep transform implementation unchanged] ...",
        transform_impl
    )

    # Write updated content to original file
    with open(old_file, 'w') as f:
        f.write(new_content)
    print(f"Updated: {old_file}")

    # Clean up temporary file
    new_file.unlink()
    print(f"Removed temporary file: {new_file}")

    print("\nUpdate complete! The original file has been backed up and updated with the new implementation.")
    print("Please verify the changes and run tests to ensure everything works correctly.")

if __name__ == "__main__":
    update_tabnet_pyspark()