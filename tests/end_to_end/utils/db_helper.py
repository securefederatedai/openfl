# Copyright 2020-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sqlite3

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
        key_value_store - This table holds key-value pairs. It only holds best_score and round_number.
                        This table holds which round had best score and what is best score until now in experiment.
        Reads key-value pairs from the key_value_store table in the database.
        This method connects to the database, executes a query to fetch all key-value pairs
        from the key_value_store table, and then closes the connection. It returns the values
        associated with the 'round_number' and 'best_score' keys.
        Raises:
            ValueError: If either 'round_number' or 'best_score' keys are not found in the key_value_store.
        Returns:
            tuple: A tuple containing the values of 'round_number' and 'best_score'.
        """
        self.connect()
        self.cursor.execute("SELECT key, value FROM key_value_store")
        rows = self.cursor.fetchall()
        self.close()

        key_value_dict = {row[0]: row[1] for row in rows}

        if 'round_number' not in key_value_dict or 'best_score' not in key_value_dict:
            raise ValueError("Required keys 'round_number' and 'best_score' not found in key_value_store")

        return key_value_dict['round_number'], key_value_dict['best_score']
