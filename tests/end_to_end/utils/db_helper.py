# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sqlite3
import time

import tests.end_to_end.utils.exceptions as ex

# Database schema:
# Table: key_value_store
# Columns: key, value
# Table: next_round_tensors
# Columns: id, tensor_name, origin, round, report, tags, nparray
# Table: tensors
# Columns: id, tensor_name, origin, round, report, tags, nparray


class DBHelper:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None
        self.cursor = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

    def close(self):
        if self.conn:
            self.conn.close()

    def read_key_value_store(self):
        """
        This method connects to the database, executes a query to fetch all key-value pairs
        from the key_value_store table, and then closes the connection.
        Currently, it only holds best_score and round_number.

        Raises:
            ValueError: If either 'round_number' or 'best_score' keys are not found in the key_value_store.
        Returns:
            dict: A dictionary containing all key-value pairs from the key_value_store table.
        """
        self.connect()
        self.cursor.execute("SELECT key, value FROM key_value_store")
        rows = self.cursor.fetchall()
        self.close()

        key_value_dict = {row[0]: row[1] for row in rows}

        # DO NOT add any exception here as the calling functions have retries and will handle the exception.

        return key_value_dict


def get_key_value_from_db(key, database_file, max_retries=10, sleep_interval=5):
    """
    Get value by key from the database file
    Args:
        key (str): Key to search. For example - round_number, best_score.
        database_file (str): Database file
        max_retries (int): Maximum number of retries if the file does not exist
        sleep_interval (int): Time to wait between retries in seconds
    Returns:
        str: Value for the key
    """
    retries = 0
    # Observation - it always takes a few attempts in the beginning to get the values from the database.
    while retries < max_retries:
        if os.path.exists(database_file):
            db_obj = DBHelper(database_file)
            val = db_obj.read_key_value_store().get(key)
            if val:
                return val
            print(f"Value not found in the database. Retrying in {sleep_interval} seconds...")
        else:
            print(f"Database file not found. Retrying in {sleep_interval} seconds...")

        time.sleep(sleep_interval)
        retries += 1

    raise ex.TensorDBException(f"Failed to get value for key {key} from the database after {max_retries} retries.")
