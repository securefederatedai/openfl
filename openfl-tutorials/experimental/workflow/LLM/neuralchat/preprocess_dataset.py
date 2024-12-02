# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ETree
import json
import os
import math
import hashlib


def xml_to_json(input_base_folder, subfolders, output_folder, verify_hash=1):

    if not os.path.exists(input_base_folder):
        raise SystemExit(f"The folder '{input_base_folder}' does not exist.")

    train_data = []
    test_data = []
    train_count, test_count = 0, 0

    if verify_hash == 1:
        expected_hash = ('9d645c469ba37eb9ec2e121ae6ac90fbebccfb91f2aff7f'
                         'faabc0531f2ede54ab4c91bea775922e5910b276340c040e8')
        verify_aggregated_hashes(input_base_folder, subfolders,
                                 expected_hash=expected_hash)

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


def compute_hash(file_path, hash_name='sha384'):
    """Compute the hash of a single file using SHA-384."""
    hash_func = getattr(hashlib, hash_name)()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def verify_aggregated_hashes(input_base_folder, dir_list, expected_hash):
    """Verify the aggregated hash of all files against a single, hardcoded hash."""
    aggregated_hash_func = hashlib.sha384()

    for sub_directory in dir_list:
        directory = os.path.join(input_base_folder, sub_directory)
        if os.path.isdir(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_hash = compute_hash(file_path)
                    aggregated_hash_func.update(file_hash.encode('utf-8'))
        else:
            raise SystemError(f"{directory} does not exist")

    # Compute the aggregated hash
    aggregated_hash = aggregated_hash_func.hexdigest()

    # Compare the aggregated hash with the expected, hardcoded hash
    if aggregated_hash != expected_hash:
        raise SystemError(
            "Verification failed. Downloaded hash doesn\'t match expected hash.")
    else:
        print("Verification passed")
