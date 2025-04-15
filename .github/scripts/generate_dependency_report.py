#!/usr/bin/env python3
"""
Convert SPDX JSON output from Trivy to Excel format with enhanced reporting
"""

import json
import pandas as pd
from pathlib import Path
import sys


def determine_origin(pkg_info: dict) -> str:
    """
    Enhanced origin detection with comprehensive pattern matching

    Args:
        pkg_info: Dictionary containing package info from SPDX
        Expected format:
        {
                "name": "PyYAML",
                "SPDXID": "SPDXRef-Package-2525510643a08a59",
                "versionInfo": "6.0.2",
                "supplier": "NOASSERTION",
                "downloadLocation": "NONE",
                "filesAnalyzed": true,
                "packageVerificationCode": {}
                "licenseConcluded": "MIT",
                "licenseDeclared": "MIT",
                "externalRefs": [],
                "primaryPackagePurpose": "LIBRARY",
                "annotations": []
        }
    Returns:
        str: Detected origin (e.g., 'PyPI', 'Ubuntu', 'DockerHub')
    """
    name = pkg_info.get("name", "").lower()
    download_loc = pkg_info.get("downloadLocation", "").lower()
    supplier = pkg_info.get("supplier", "").lower()

    # Check for Python packages
    if any(x in name for x in ['python', 'pip', 'pypi', 'wheel', 'setuptools']):
        return 'PyPI'

    # Check for system packages
    if any(x in name for x in ['ubuntu', 'debian', 'apt', 'dpkg', 'libc']):
        return 'Ubuntu'

    # Check download location patterns
    if 'github.com' in download_loc:
        return 'GitHub'
    if 'docker.io' in download_loc or 'docker.com' in download_loc:
        return 'DockerHub'
    if 'pypi.org' in download_loc or 'pypi.python.org' in download_loc:
        return 'PyPI'

    # Check supplier information
    if 'ubuntu' in supplier:
        return 'Ubuntu'
    if 'debian' in supplier:
        return 'Debian'

    # Fallback to PyPI as default
    return 'PyPI'

def convert_spdx_to_excel(spdx_path: Path, excel_path: Path) -> bool:
    """
    Convert SPDX JSON file to Excel format with enhanced reporting
    """
    try:
        if not spdx_path.exists():
            raise FileNotFoundError(f"SPDX JSON file not found at {spdx_path}")

        with open(spdx_path) as f:
            data = json.load(f)

        # Process packages, remove duplicates and filter out openfl:latest
        seen_components = set()
        packages = []

        # Add Ubuntu as the first component
        packages.append({
            "Dockerfile": "openfl-docker/Dockerfile.base",
            "Component": "Ubuntu",
            "Origin": "Ubuntu",
            "License": "Collection of licences",
            "Distributed by you?": "No",
            "Comments": "Base image"
        })

        for pkg in data.get("packages", []):
            component_name = pkg.get("name", "Unknown")
            if component_name.lower() not in ['openfl', 'openfl:latest'] and component_name not in seen_components:
                seen_components.add(component_name)
                packages.append({
                    "Dockerfile": "openfl-docker/Dockerfile.base",
                    "Component": component_name,
                    "Origin": determine_origin(pkg),
                    "License": pkg.get("licenseConcluded", "Unknown"),
                    "Distributed by you?": "Yes",
                    "Comments": ""
                })

        # Create Component List DataFrame and sort alphabetically (keeping Ubuntu first)
        component_df = pd.DataFrame(packages)
        if len(component_df) > 1:
            # Sort all rows except the first one (Ubuntu)
            sorted_df = component_df.iloc[1:].sort_values(by='Component', key=lambda x: x.str.lower())
            component_df = pd.concat([component_df.iloc[[0]], sorted_df])

        if component_df.empty:
            print("Warning: No package data found in SPDX report")
            component_df = pd.DataFrame([{"Status": "No package data found in scan"}])

        # Create Container List DataFrame with additional row
        container_data = [
            {
            "Container": "OpenFL Base Docker",
                "Dockerfile": "openfl-docker/Dockerfile.base",
            "Container Distribution": "Github Container Registry",
            "Dockerfile Distribution": "Distributed as a part of code",
                "Comments": ""
            },
            {
                "Container": "OpenFL Workspace Image",
                "Dockerfile": "openfl-docker/Dockerfile.workspace",
                "Container Distribution": "N/A",
                "Dockerfile Distribution": "Distributed as a part of code",
                "Comments": "Reference for users to create the workload."
            }
        ]

        container_df = pd.DataFrame(container_data)

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write Container List sheet
            container_df.to_excel(
                writer,
                index=False,
                sheet_name='Container List'
            )

            # Write Component List sheet
            component_df.to_excel(
                writer,
                index=False,
                sheet_name='Component List'
            )

            # Auto-adjust column widths for both sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = max(len(str(cell.value)) for cell in column)
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

        print(f"Successfully generated Excel report at {excel_path}")
        return True

    except Exception as e:
        print(f"Error processing SPDX data: {str(e)}", file=sys.stderr)

        error_df = pd.DataFrame([{
            "Error": str(e),
            "InputFile": str(spdx_path),
            "OutputFile": str(excel_path)
        }])
        error_df.to_excel(excel_path, index=False)
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert SPDX JSON to Excel format')
    parser.add_argument('input_json', help='Path to SPDX JSON file')
    parser.add_argument('output_excel', help='Path for output Excel file')

    args = parser.parse_args()

    success = convert_spdx_to_excel(Path(args.input_json), Path(args.output_excel))
    sys.exit(0 if success else 1)
