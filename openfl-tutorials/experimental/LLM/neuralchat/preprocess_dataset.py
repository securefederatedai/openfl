# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ETree
import json
import os
import math
import hashlib


def xml_to_json(input_base_folder, output_folder, verify_hash=1):

    if not os.path.exists(input_base_folder):
        raise SystemExit(f"The folder '{input_base_folder}' does not exist.")

    train_data = []
    test_data = []
    train_count, test_count = 0, 0

    subfolders = ["1_CancerGov_QA", "2_GARD_QA", "3_GHR_QA", "4_MPlus_Health_Topics_QA",
                  "5_NIDDK_QA", "6_NINDS_QA", "7_SeniorHealth_QA", "8_NHLBI_QA_XML", "9_CDC_QA"]

    if verify_hash == 1:
        verify_hashes(input_base_folder, subfolders)

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
        else:
            raise SystemError(f"{folder_path} does not exist")

    # Save the data to JSON files
    save_json(train_data, os.path.join(output_folder, 'medquad_alpaca_train.json'))
    save_json(test_data, os.path.join(output_folder, 'medquad_alpaca_test.json'))

    # Write the counts to a text file
    with open(os.path.join(output_folder, 'data_counts.txt'), 'w') as f:
        f.write(f"Training data pairs: {train_count}\n")
        f.write(f"Test data pairs: {test_count}\n")
        
    print("Preprocessing complete")


def process_xml_file(folder, xml_file):
    xml_path = os.path.join(folder, xml_file)
    tree = ETree.parse(xml_path)
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


def compute_hash(file_path, hash_name='sha256'):
    """Compute the hash of a single file."""
    hash_func = getattr(hashlib, hash_name)()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def verify_hashes(input_base_folder, dir_list, hash_file='./hashes.txt'):
    """Verify the hashes of files against the saved hash records."""
    # Load the saved hashes
    saved_hashes = {}
    with open(hash_file, 'r') as f:
        for line in f:
            if line.startswith('Directory:'):
                directory = line.strip().split(": ")[1]
            elif line.strip():
                file_path, file_hash = line.strip().split(': ')
                saved_hashes[file_path] = file_hash

    # Verify each file's hash
    for sub_directory in dir_list:
        directory = os.path.join(input_base_folder, sub_directory)
        if os.path.isdir(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    current_hash = compute_hash(file_path)
                    if file_path in saved_hashes and saved_hashes[file_path] != current_hash:
                        raise SystemError(f"Verification failed for {file_path}. The file may be corrupted or tampered with.")
        else:
            raise SystemError(f"{directory} does not exist")
    print("Verification passed")