import re
import os
import sys
import json

def extract_emails(filepath):
    """
    Extract all unique email addresses from the given file.
    """
    email_pattern = r'[\w.+-]+@[\w-]+\.[\w.-]+'
    unique_emails = set()

    try:
        with open(filepath, 'r') as file:
            for line in file:
                # Skip comment lines that don't contain emails
                if line.strip().startswith('#') and '@' not in line:
                    continue

                # Find all email addresses in the line
                emails = re.findall(email_pattern, line)
                unique_emails.update(emails)
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}", file=sys.stderr)

    return sorted(unique_emails)

if __name__ == "__main__":
    # Check CODEOWNERS in standard locations
    codeowners_path = None
    for path in ['.github/CODEOWNERS', 'CODEOWNERS', 'docs/CODEOWNERS']:
        if os.path.exists(path):
            codeowners_path = path
            break

    result = {
        "emails": [],
        "codeowners_path": codeowners_path
    }

    if codeowners_path:
        emails = extract_emails(codeowners_path)
        result["emails"] = emails

    print(json.dumps(result))
