# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as et
import json
import os
import math


def xml_to_json(input_base_folder, output_folder):

    if not os.path.exists(input_base_folder):
        raise SystemExit(f"The folder '{input_base_folder}' does not exist.")
    
    train_data = []
    test_data = []
    train_count, test_count = 0, 0

    subfolders = ["1_CancerGov_QA", "2_GARD_QA", "3_GHR_QA", "4_MPlus_Health_Topics_QA", 
                  "5_NIDDK_QA", "6_NINDS_QA", "7_SeniorHealth_QA", "8_NHLBI_QA_XML", "9_CDC_QA"]

    for subfolder in subfolders:
        folder_path = os.path.join(input_base_folder, subfolder)
        if os.path.isdir(folder_path):
            xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
            test_file_count = math.ceil(len(xml_files) * 0.01)

            # Process files for training data
            for xml_file in xml_files[:-test_file_count]:
                new_data, count = process_xml_file(folder_path, xml_file)
                train_data.extend(new_data)
                train_count += count

            # Process files for test data
            for xml_file in xml_files[-test_file_count:]:
                new_data, count = process_xml_file(folder_path, xml_file)
                test_data.extend(new_data)
                test_count += count

    # Save the data to JSON files
    save_json(train_data, os.path.join(output_folder, 'medquad_alpaca_train.json'))
    save_json(test_data, os.path.join(output_folder, 'medquad_alpaca_test.json'))

    # Write the counts to a text file
    with open(os.path.join(output_folder, 'data_counts.txt'), 'w') as f:
        f.write(f"Training data pairs: {train_count}\n")
        f.write(f"Test data pairs: {test_count}\n")


def process_xml_file(folder, xml_file):
    xml_path = os.path.join(folder, xml_file)
    tree = et.parse(xml_path)
    root = tree.getroot()

    data = []
    count = 0
    for qa_pair in root.findall(".//QAPair"):
        question = qa_pair.find('Question').text
        answer = qa_pair.find('Answer').text

        if not question or not answer:
            continue

        question = question.strip()
        answer = answer.strip().replace('\n', ' ').replace('  ', ' ')
        
        json_obj = {
            "instruction": question,
            "input": "",
            "output": answer
        }
        data.append(json_obj)
        count += 1

    return data, count


def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
