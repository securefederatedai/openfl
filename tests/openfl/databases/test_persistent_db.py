import numpy as np
import pytest

from openfl.databases.persistent_db import PersistentTensorDB
from openfl.utilities.types import TensorKey


#############################################
#           Pytest Fixtures                 #
#############################################

@pytest.fixture
def persistent_db():
    """Create a PersistentTensorDB using an in-memory SQLite database."""
    db = PersistentTensorDB(":memory:")
    yield db
    db.close()


@pytest.fixture
def tensor_key():
    """Return a dummy TensorKey for testing."""
    # Note: The PersistentTensorDB code expects a TensorKey that can be unpacked
    # into (tensor_name, origin, fl_round, report, tags).
    return TensorKey("tensor1", "origin1", 1, False, ("tag1",))

@pytest.fixture
def random_array_factory():
    """Return a function that generates a random NumPy array of shape (3,) and dtype np.float32."""
    def _generate():
        return np.random.rand(3).astype(np.float32)
    return _generate


#############################################
#          Unit Tests for PersistentTensorDB #
#############################################

def test_save_and_get_task_results(persistent_db, random_array_factory):
    """
    Test that saving task results and then retrieving them by ID works correctly.
    """
    assert persistent_db.is_task_table_empty()
    for idx in range(0,3):
        collaborator_name = "collab1"
        round_number = 1
        task_name = "taskA"
        data_size = 100
        # Create a list of named tensors.
        named_tensors = [random_array_factory() for _ in range(5)]

        # Save the task result.
        id = persistent_db.save_task_results(collaborator_name, round_number, task_name, data_size, named_tensors)

        # Retrieve the task result (assuming the first inserted row gets id 1).
        result = persistent_db.get_task_result_by_id(id)
        assert result is not None
        assert result["collaborator_name"] == collaborator_name
        assert result["round_number"] == round_number
        assert result["task_name"] == task_name
        assert result["data_size"] == data_size
        for i in range(0,len(named_tensors)):
            np.testing.assert_array_equal(result["named_tensors"][i], named_tensors[i])




def test_finalize_round_and_load_tensors(persistent_db, tensor_key, random_array_factory):
    """
    Test that finalize_round correctly stores current round tensors, next round tensors,
    and updates the round and best score in the key-value store.
    Then verify that load_tensors returns the stored tensors.
    """
    for i in range(0,3):
        # populate a task
        collaborator_name = "collab1"
        round_number = i + 1
        task_name = "taskA"
        data_size = 100
        # Create a list of named tensors.
        named_tensors = [random_array_factory() for _ in range(5)]
        # Save the task result.
        assert persistent_db.is_task_table_empty()
        persistent_db.save_task_results(collaborator_name, round_number, task_name, data_size, named_tensors)

        assert not persistent_db.is_task_table_empty()

        current_tensors = {tensor_key: random_array_factory()}
        # Create a different tensor key for next round.
        next_tensor_key = TensorKey("tensor2", "origin2", 1, False, ("tag2",))
        next_round_tensors = {next_tensor_key: random_array_factory()}
        round_number = i + 1
        best_score = 0.95 + i

        persistent_db.finalize_round(current_tensors, next_round_tensors, round_number, best_score)

        assert persistent_db.is_task_table_empty()

        # Check that the key-value store has been updated.
        rn, bs = persistent_db.get_round_and_best_score()
        assert rn == round_number
        assert bs == best_score

        # Verify that the current round tensor is in the main tensors table.
        main_table = persistent_db.get_tensors_table_name()
        tensors = persistent_db.load_tensors(main_table)
        found = False
        for key, arr in tensors.items():
            if key.tensor_name == "tensor1":
                np.testing.assert_array_equal(arr, current_tensors[key])
                found = True
        assert found, "Expected tensor1 not found in the main tensors table."

        # Verify that the next round tensor is in the next round tensors table.
        next_table = persistent_db.get_next_round_tensors_table_name()
        next_tensors = persistent_db.load_tensors(next_table)
        found = False
        for key, arr in next_tensors.items():
            if key.tensor_name == "tensor2":
                np.testing.assert_array_equal(arr, next_round_tensors[key])
                found = True
        assert found, "Expected tensor2 not found in the next round tensors table."
