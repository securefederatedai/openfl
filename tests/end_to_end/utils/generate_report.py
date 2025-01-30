# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import linregress
from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)

    def chapter_title(self, title):
        self.add_page()
        self.set_font("Arial", "B", 14)  # Set font to bold for title
        self.cell(0, 10, title, 0, 1, "L")

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, body)


def generate_memory_report(memory_usage_dict, workspace_path):
    """
    Generates a memory usage report using input dictionary
    and saves it to a PDF file.
    Content of memory_usage_dict comes from reading the aggregator
    and collaborator memory usage json files inside respective logs folder.

    Parameters:
        memory_usage_dict (dict): A dictionary containing memory usage data.
        workspace_path (str): The path to the workspace where the report will be saved.

    Returns:
    None
    """
    # Load data
    data = pd.DataFrame(memory_usage_dict)

    # Plotting the chart
    plt.figure(figsize=(10, 5))
    plt.plot(data["round_number"], data["process_memory"], marker="o")
    plt.title("Memory Usage per Round")
    plt.xlabel("round_number")
    plt.ylabel("Process Memory Used (MB)")
    plt.grid(True)
    output_path = f"{workspace_path}/mem_usage_plot.png"
    plt.savefig(output_path)
    plt.close()

    # Calculate statistics
    min_mem = round(data["process_memory"].min(), 2)
    max_mem = round(data["process_memory"].max(), 2)
    mean_mem = round(data["process_memory"].mean(), 2)
    variance_mem = round(data["process_memory"].var(), 2)
    std_dev_mem = round(data["process_memory"].std(), 2)
    slope, _, _, _, _ = linregress(data.index, data["process_memory"])
    slope = round(slope, 2)
    stats_path = f"{workspace_path}/mem_stats.txt"
    with open(stats_path, "w") as file:
        file.write(f"Minimum Memory Used: {min_mem} MB\n")
        file.write(f"Maximum Memory Used: {max_mem} MB\n")
        file.write(f"Mean Memory Used: {mean_mem} MB\n")
        file.write(f"Variance: {variance_mem}\n")
        file.write(f"Standard Deviation: {std_dev_mem}\n")
        file.write(f"Slope: {slope}\n")

    # Generate PDF report
    pdf = PDF()
    add_introduction(pdf)
    add_chart_analysis(pdf, output_path, data)
    add_statistical_overview(pdf, stats_path)
    add_conclusion(pdf, slope)
    pdf_output_path = f"{workspace_path}/MemAnalysis.pdf"
    pdf.output(pdf_output_path)

    print("Memory report generation completed. Report saved to:", pdf_output_path)


def add_introduction(pdf):
    pdf.chapter_title("Introduction")
    intro_text = (
        "The purpose of this memory analysis is to identify memory usage trends and potential bottlenecks. "
        "This analysis focuses on the relationship between round information and memory usage."
    )
    pdf.chapter_body(intro_text)


def add_chart_analysis(pdf, output_path, data):
    pdf.chapter_title("Chart Analysis")
    pdf.image(output_path, w=180)
    diffs = data["process_memory"].diff().round(2)
    significant_changes = diffs[diffs.abs() > 500]
    for index, value in significant_changes.items():
        pdf.chapter_body(
            f"Significant memory change: {value} MB at Round {data['round_number'][index]}"
        )


def add_statistical_overview(pdf, stats_path):
    pdf.chapter_title("Statistical Overview")
    with open(stats_path, "r") as file:
        stats = file.read()
    pdf.chapter_body(stats)


def add_conclusion(pdf, slope):
    pdf.chapter_title("Conclusion")
    if slope > 0:
        conclusion_text = "The upward slope in the graph indicates a trend of increasing memory usage over rounds."
    else:
        conclusion_text = "There is no continuous memory growth."
    pdf.chapter_body(conclusion_text)


def convert_to_json(file):
    """
    Reads a file containing JSON objects, one per line, and converts them into a list of parsed JSON objects.
    Args:
        file (str): The path to the file containing JSON objects.
    Returns:
        list: A list of parsed JSON objects.
    """
    with open(file, "r") as infile:
        json_objects = infile.readlines()

    # Parse each JSON object
    parsed_json_objects = [json.loads(obj) for obj in json_objects]
    return parsed_json_objects
