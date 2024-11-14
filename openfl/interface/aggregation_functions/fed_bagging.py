# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Federated Boostrap Aggregation for XGBoost module."""

import json
from openfl.interface.aggregation_functions.core import AggregationFunction
import numpy as np

def get_global_model(iterator, target_round):
    for item in iterator:
        # Items tagged with ('model',) are the global model of that round
        if 'tags' in item and item['tags'] == ('model',) and item['round'] == target_round:
            return item['nparray']
    raise ValueError(f"No item found with tag 'model' and round {target_round}")

        
def append_trees(global_model, local_trees):

    num_global_trees = int(global_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
    num_local_trees = len(local_trees)

    global_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(
        num_global_trees + num_local_trees
    )
    global_model["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        num_global_trees + num_local_trees
    )
    for new_tree in range(num_local_trees):
            local_trees[new_tree]["id"] = num_global_trees + new_tree
            global_model["learner"]["gradient_booster"]["model"]["trees"].append(local_trees[new_tree])
            global_model["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    return global_model


class FedBaggingXGBoost(AggregationFunction):
    """Federated Boostrap Aggregation for XGBoost."""

    def call(self, local_tensors, db_iterator, tensor_name, fl_round, *_):
        """Aggregate tensors.

        Args:
            local_tensors (list[openfl.utilities.LocalTensor]): List of local
                tensors to aggregate.
            db_iterator: iterator over history of all tensors. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight','fc2.bias'.
                - 'round': 0-based number of round corresponding to this
                    tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'model' indicates that the tensor is a model parameter.
                    - 'trained' indicates that tensor is a part of a training
                        result.
                        These tensors are passed to the aggregator node after
                        local learning.
                    - 'aggregated' indicates that tensor is a result of
                        aggregation.
                        These tensors are sent to collaborators for the next
                        round.
                    - 'delta' indicates that value is a difference between
                        rounds for a specific tensor.
                    also one of the tags is a collaborator name
                    if it corresponds to a result of a local task.

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
            tags: tuple of tags for this tensor
        Returns:
            bytearray: aggregated tensor
        """
        # global_model = None
        global_model = get_global_model(db_iterator, fl_round)
        
        if (isinstance(global_model, np.ndarray) and global_model.size == 0) or global_model is None:
            for local_tensor in local_tensors:
                local_tree_bytearray = bytearray(local_tensor.tensor.astype(np.uint8).tobytes())
                local_tree_json = json.loads(local_tree_bytearray)
                
                if (isinstance(global_model, np.ndarray) and global_model.size == 0) or global_model is None:
                    # the first tree becomes the global model to append to
                    global_model = local_tree_json
                else:
                    # append subsequent trees
                    local_model = local_tree_json
                    local_trees = local_model['learner']['gradient_booster']['model']['trees']
                    global_model = append_trees(global_model, local_trees)
        else:
            global_model_bytearray = bytearray(global_model.astype(np.uint8).tobytes())
            global_model = json.loads(global_model_bytearray)

            for local_tensor in local_tensors:
                local_tree_bytearray = bytearray(local_tensor.tensor.astype(np.uint8).tobytes())
                local_trees = json.loads(local_tree_bytearray)
                global_model = append_trees(global_model, local_trees)

        ## Ensures that model is recoverable. TODO put in function
        # Convert latest_trees to a JSON string
        global_model_json = json.dumps(global_model)
        global_model_bytes = global_model_json.encode('utf-8')
        
        return np.frombuffer(global_model_bytes, dtype=np.uint8).astype(np.float32)