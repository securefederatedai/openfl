import json
import pandas as pd

# Path to the Bandit JSON report
json_report_path = 'bandit_results.json'
html_report_path = 'results.html'

def read_json_report(json_report_path):
    """
    Read the JSON report from the specified path.

    Args:
        json_report_path (str): Path to the JSON report file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(json_report_path, 'r') as file:
        data = json.load(file)
    return data

def filter_results(data, exclude_dirs):
    """
    Filter the results to exclude specified directories.

    Args:
        data (dict): Parsed JSON data.
        exclude_dirs (list): List of directories to exclude.

    Returns:
        list: Filtered results.
    """
    filtered_results = []
    for result in data['results']:
        file_path = result['filename']
        if not any(file_path.startswith(exclude_dir) for exclude_dir in exclude_dirs):
            filtered_results.append(result)
    return filtered_results

def generate_summary_and_details(filtered_results):
    """
    Generate summary and details from the filtered results.

    Args:
        filtered_results (list): Filtered results.

    Returns:
        tuple: Summary dictionary and details list.
    """
    summary = {
        'HIGH': {'file_count': 0, 'issue_count': 0},
        'MEDIUM': {'file_count': 0, 'issue_count': 0},
        'LOW': {'file_count': 0, 'issue_count': 0}
    }
    details = []

    for result in filtered_results:
        severity = result['issue_severity']
        confidence = result['issue_confidence']
        file_path = result['filename']
        line_number = result['line_number']
        test_id = result['test_id']
        issue_text = result['issue_text']
        
        summary[severity]['issue_count'] += 1
        if file_path not in [detail['file'] for detail in details]:
            summary[severity]['file_count'] += 1
        
        details.append({
            'file': file_path,
            'line_numbers': line_number,
            'test': test_id,
            'issue': issue_text,
            'severity': severity,
            'confidence': confidence
        })

    return summary, details

def create_html_report(summary, details_sorted, html_report_path):
    """
    Create an HTML report from the summary and details.

    Args:
        summary (dict): Summary dictionary.
        details_sorted (list): Sorted details list.
        html_report_path (str): Path to save the HTML report.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bandit Report Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f9f9f9; color: #333; }}
            h1 {{ text-align: center; padding: 20px; background-color: #4CAF50; color: white; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #f4f4f4; }}
            .severity-high {{ color: #D32F2F; font-weight: bold; }}
            .severity-medium {{ color: #FFA000; font-weight: bold; }}
            .severity-low {{ color: #388E3C; font-weight: bold; }}
            pre {{ white-space: pre-wrap; }} /* Preserve line breaks */
        </style>
    </head>
    <body>
        <h1>Bandit Report Summary</h1>

        <h2>Summary of Issues</h2>
        <table>
            <thead>
                <tr>
                    <th>Severity</th>
                    <th>File Count</th>
                    <th>Issue Count</th>
                </tr>
            </thead>
            <tbody>
                <tr class="severity-high">
                    <td>HIGH</td>
                    <td>{summary['HIGH']['file_count']}</td>
                    <td>{summary['HIGH']['issue_count']}</td>
                </tr>
                <tr class="severity-medium">
                    <td>MEDIUM</td>
                    <td>{summary['MEDIUM']['file_count']}</td>
                    <td>{summary['MEDIUM']['issue_count']}</td>
                </tr>
                <tr class="severity-low">
                    <td>LOW</td>
                    <td>{summary['LOW']['file_count']}</td>
                    <td>{summary['LOW']['issue_count']}</td>
                </tr>
            </tbody>
        </table>

        <h2>Details of Issues</h2>
        <table>
            <thead>
                <tr>
                    <th>File (Line Numbers)</th>
                    <th>Test</th>
                    <th>Issue</th>
                    <th>Severity</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
    """

    for detail in details_sorted:
        html_content += f"""
                <tr class="severity-{detail['severity'].lower()}">
                    <td><pre>{detail['file']}:{detail['line_numbers']}</pre></td>
                    <td>{detail['test']}</td>
                    <td>{detail['issue']}</td>
                    <td>{detail['severity']}</td>
                    <td>{detail['confidence']}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    with open(html_report_path, 'w') as file:
        file.write(html_content)

def main():
    """
    Main function to read the JSON report, filter results, generate summary and details,
    and create an HTML report.
    """
    exclude_dirs = ['./openfl-tutorials', './openfl-workspace', './tests']
    data = read_json_report(json_report_path)
    filtered_results = filter_results(data, exclude_dirs)
    summary, details = generate_summary_and_details(filtered_results)
    details_sorted = sorted(details, key=lambda x: ['HIGH', 'MEDIUM', 'LOW'].index(x['severity']))
    create_html_report(summary, details_sorted, html_report_path)

if __name__ == "__main__":
    main()