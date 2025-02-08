import json
import logging
import pickle
import sqlite3
from threading import Lock
from typing import Dict, Optional

import numpy as np

from openfl.utilities import TensorKey

logger = logging.getLogger(__name__)

__all__ = ["PersistentTensorDB"]


class PersistentTensorDB:
    """
    The PersistentTensorDB class implements a database
    for storing tensors and metadata using SQLite.

    Attributes:
        conn: The SQLite connection object.
        cursor: The SQLite cursor object.
        lock: A threading Lock object used to ensure thread-safe operations.
    """

    TENSORS_TABLE = "tensors"
    NEXT_ROUND_TENSORS_TABLE = "next_round_tensors"
    TASK_RESULT_TABLE = "task_results"
    KEY_VALUE_TABLE = "key_value_store"

    def __init__(self, db_path) -> None:
        """Initializes a new instance of the PersistentTensorDB class."""

        logger.info("Initializing persistent db at %s", db_path)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = Lock()

        cursor = self.conn.cursor()
        self._create_model_tensors_table(cursor, PersistentTensorDB.TENSORS_TABLE)
        self._create_model_tensors_table(cursor, PersistentTensorDB.NEXT_ROUND_TENSORS_TABLE)
        self._create_task_results_table(cursor)
        self._create_key_value_store(cursor)
        self.conn.commit()

    def _create_model_tensors_table(self, cursor, table_name) -> None:
        """Create the database table for storing tensors if it does not exist."""
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tensor_name TEXT NOT NULL,
                origin TEXT NOT NULL,
                round INTEGER NOT NULL,
                report INTEGER NOT NULL,
                tags TEXT,
                nparray BLOB NOT NULL
            )
        """
        cursor.execute(query)

    def _create_task_results_table(self, cursor) -> None:
        """Creates a table for storing task results."""
        query = f"""
            CREATE TABLE IF NOT EXISTS {PersistentTensorDB.TASK_RESULT_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collaborator_name TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                task_name TEXT NOT NULL,
                data_size INTEGER NOT NULL,
                named_tensors BLOB NOT NULL
            )
        """
        cursor.execute(query)

    def _create_key_value_store(self, cursor) -> None:
        """Create a key-value store table for storing additional metadata."""
        query = f"""
            CREATE TABLE IF NOT EXISTS {PersistentTensorDB.KEY_VALUE_TABLE} (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
        """
        cursor.execute(query)

    def save_task_results(
        self,
        collaborator_name: str,
        round_number: int,
        task_name: str,
        data_size: int,
        named_tensors,
    ) -> int:
        """
        Saves task results to the task_results table.

        Args:
            collaborator_name (str): Collaborator name.
            round_number (int): Round number.
            task_name (str): Task name.
            data_size (int): Data size.
            named_tensors(List): list of binary representation of tensors.
        """
        serialized_blob = pickle.dumps(named_tensors)

        # Insert into the database
        insert_query = f"""
        INSERT INTO {PersistentTensorDB.TASK_RESULT_TABLE}
        (collaborator_name, round_number, task_name, data_size, named_tensors)
        VALUES (?, ?, ?, ?, ?);
        """
        new_id = 0
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                insert_query,
                (collaborator_name, round_number, task_name, data_size, serialized_blob),
            )
            new_id = cursor.lastrowid
            self.conn.commit()
        return new_id

    def get_task_result_by_id(self, task_result_id: int):
        """
        Retrieve a task result by its ID.

        Args:
            task_result_id (int): The ID of the task result to retrieve.

        Returns:
            A dictionary containing the task result details, or None if not found.
        """
        with self.lock:
            cursor = self.conn.cursor()
            query = f"""
                SELECT collaborator_name, round_number, task_name, data_size, named_tensors
                FROM {PersistentTensorDB.TASK_RESULT_TABLE}
                WHERE id = ?
            """
            cursor.execute(query, (task_result_id,))
            result = cursor.fetchone()
            if result:
                collaborator_name, round_number, task_name, data_size, serialized_blob = result
                serialized_tensors = pickle.loads(serialized_blob)
                return {
                    "collaborator_name": collaborator_name,
                    "round_number": round_number,
                    "task_name": task_name,
                    "data_size": data_size,
                    "named_tensors": serialized_tensors,
                }
            return None

    def _serialize_array(self, array: np.ndarray) -> bytes:
        """Serialize a NumPy array into bytes for storing in SQLite.
        note: using pickle since in some cases the array is actually a scalar.
        """
        return pickle.dumps(array)

    def _deserialize_array(self, blob: bytes, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """Deserialize bytes from SQLite into a NumPy array."""
        try:
            return pickle.loads(blob)
        except Exception as e:
            raise ValueError(f"Failed to deserialize array: {e}")

    def __repr__(self) -> str:
        """Returns a string representation of the PersistentTensorDB."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT tensor_name, origin, round, report, tags FROM tensors")
            rows = cursor.fetchall()
            return f"PersistentTensorDB contents:\n{rows}"

    def finalize_round(
        self,
        tensor_key_dict: Dict[TensorKey, np.ndarray],
        next_round_tensor_key_dict: Dict[TensorKey, np.ndarray],
        round_number: int,
        best_score: float,
    ):
        """Finalize a training round by saving tensors, preparing for the next round,
        and updating metadata in the database.

        This function performs the following steps as a single transaction:
        1. Persist the tensors of the current round into the database.
        2. Persist the tensors for the next training round into the database.
        3. Reinitialize the task results table to prepare for new tasks.
        4. Update the round number and best score in the key-value store.

        If any step fails, the transaction is rolled back to ensure data integrity.

        Args:
            tensor_key_dict (Dict[TensorKey, np.ndarray]):
                A dictionary mapping tensor keys to their corresponding
                  NumPy arrays for the current round.
            next_round_tensor_key_dict (Dict[TensorKey, np.ndarray]):
                A dictionary mapping tensor keys to their corresponding
                 NumPy arrays for the next round.
            round_number (int):
                The current training round number.
            best_score (float):
                The best score achieved during the current round.

        Raises:
            RuntimeError: If an error occurs during the transaction, the transaction is rolled back,
                        and a RuntimeError is raised with the details of the failure.
        """
        with self.lock:
            try:
                # Begin transaction
                cursor = self.conn.cursor()
                cursor.execute("BEGIN TRANSACTION")
                self._persist_tensors(cursor, PersistentTensorDB.TENSORS_TABLE, tensor_key_dict)
                self._persist_next_round_tensors(cursor, next_round_tensor_key_dict)
                self._init_task_results_table(cursor)
                self._save_round_and_best_score(cursor, round_number, best_score)
                # Commit transaction
                self.conn.commit()
                logger.info(
                    f"Committed model for round {round_number}, saved {len(tensor_key_dict)}"
                    f" model tensors and {len(next_round_tensor_key_dict)}"
                    f" next round model tensors  with best_score {best_score}"
                )
            except Exception as e:
                # Rollback transaction in case of an error
                self.conn.rollback()
                raise RuntimeError(f"Failed to finalize round: {e}")

    def _persist_tensors(
        self, cursor, table_name, tensor_key_dict: Dict[TensorKey, np.ndarray]
    ) -> None:
        """Insert a dictionary of tensors into the SQLite as part of transaction"""
        for tensor_key, nparray in tensor_key_dict.items():
            tensor_name, origin, fl_round, report, tags = tensor_key
            serialized_array = self._serialize_array(nparray)
            serialized_tags = json.dumps(tags)
            query = f"""
                    INSERT INTO {table_name} (tensor_name, origin, round, report, tags, nparray)
                    VALUES (?, ?, ?, ?, ?, ?)
                """
            cursor.execute(
                query,
                (tensor_name, origin, fl_round, int(report), serialized_tags, serialized_array),
            )

    def _persist_next_round_tensors(
        self, cursor, tensor_key_dict: Dict[TensorKey, np.ndarray]
    ) -> None:
        """Persisting the last round next_round tensors."""
        drop_table_query = f"DROP TABLE IF EXISTS {PersistentTensorDB.NEXT_ROUND_TENSORS_TABLE}"
        cursor.execute(drop_table_query)
        self._create_model_tensors_table(cursor, PersistentTensorDB.NEXT_ROUND_TENSORS_TABLE)
        self._persist_tensors(cursor, PersistentTensorDB.NEXT_ROUND_TENSORS_TABLE, tensor_key_dict)

    def _init_task_results_table(self, cursor):
        """
        Creates a table for storing task results. Drops the table first if it already exists.
        """
        drop_table_query = "DROP TABLE IF EXISTS task_results"
        cursor.execute(drop_table_query)
        self._create_task_results_table(cursor)

    def _save_round_and_best_score(self, cursor, round_number: int, best_score: float) -> None:
        """Save the round number and best score as key-value pairs in the database."""
        # Create a table with key-value structure where values can be integer or float
        # Insert or update the round_number
        cursor.execute(
            """
            INSERT OR REPLACE INTO key_value_store (key, value)
            VALUES (?, ?)
        """,
            ("round_number", float(round_number)),
        )

        # Insert or update the best_score
        cursor.execute(
            """
            INSERT OR REPLACE INTO key_value_store (key, value)
            VALUES (?, ?)
        """,
            ("best_score", float(best_score)),
        )

    def get_tensors_table_name(self) -> str:
        return PersistentTensorDB.TENSORS_TABLE

    def get_next_round_tensors_table_name(self) -> str:
        return PersistentTensorDB.NEXT_ROUND_TENSORS_TABLE

    def load_tensors(self, tensor_table) -> Dict[TensorKey, np.ndarray]:
        """Load all tensors from the SQLite database and return them as a dictionary."""
        tensor_dict = {}
        with self.lock:
            cursor = self.conn.cursor()
            query = f"SELECT tensor_name, origin, round, report, tags, nparray FROM {tensor_table}"
            cursor.execute(query)
            rows = cursor.fetchall()
            for row in rows:
                tensor_name, origin, fl_round, report, tags, nparray = row
                # Deserialize the JSON string back to a Python list
                deserialized_tags = tuple(json.loads(tags))
                tensor_key = TensorKey(tensor_name, origin, fl_round, report, deserialized_tags)
                tensor_dict[tensor_key] = self._deserialize_array(nparray)
        return tensor_dict

    def get_round_and_best_score(self) -> tuple[int, float]:
        """Retrieve the round number and best score from the database."""
        with self.lock:
            cursor = self.conn.cursor()
            # Fetch the round_number
            cursor.execute(
                """
                SELECT value FROM key_value_store WHERE key = ?
            """,
                ("round_number",),
            )
            round_number = cursor.fetchone()
            if round_number is None:
                round_number = -1
            else:
                round_number = int(round_number[0])  # Cast to int

            # Fetch the best_score
            cursor.execute(
                """
                SELECT value FROM key_value_store WHERE key = ?
            """,
                ("best_score",),
            )
            best_score = cursor.fetchone()
            if best_score is None:
                best_score = 0
            else:
                best_score = float(best_score[0])  # Cast to float
        return round_number, best_score

    def clean_up(self, remove_older_than: int = 1) -> None:
        """Remove old entries from the database."""
        if remove_older_than < 0:
            return
        with self.lock:
            cursor = self.conn.cursor()
            query = f"SELECT MAX(round) FROM {PersistentTensorDB.TENSORS_TABLE}"
            cursor.execute(query)
            current_round = cursor.fetchone()[0]
            if current_round is None:
                return
            cursor.execute(
                """
                DELETE FROM tensors
                WHERE round <= ? AND report = 0
            """,
                (current_round - remove_older_than,),
            )
            self.conn.commit()

    def close(self) -> None:
        """Close the SQLite database connection."""
        self.conn.close()

    def is_task_table_empty(self) -> bool:
        """Check if the task table is empty."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM task_results")
            count = cursor.fetchone()[0]
            return count == 0
