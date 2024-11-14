# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Federated Boostrap Aggregation for XGBoost module."""

import json
from openfl.interface.aggregation_functions.core import AggregationFunction
from openfl.federated.task.runner_xgb import convert_back_to_json
import numpy as np
import base64

def get_global_model(iterator, target_round):
    for item in iterator:
        # Items tagged with ('model',) are the global model of that round
        if 'tags' in item and item['tags'] == ('model',) and item['round'] == target_round:
            return item['nparray']
    raise ValueError(f"No item found with tag 'model' and round {target_round}")

# def convert_back_to_json(booster_float32_array):
#     # Convert np.float32 array back to base64 string
#     booster_uint8_array = booster_float32_array.view(np.uint8)
#     booster_base64 = booster_uint8_array.tobytes().decode('utf-8')

#     # Decode base64 string back to original JSON string
#     booster_bytes = base64.b64decode(booster_base64)
#     booster_array = booster_bytes.decode('utf-8')
#     return booster_array
        
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
                local_tree_json = json.loads(convert_back_to_json(local_tensor.tensor))
                
                if (isinstance(global_model, np.ndarray) and global_model.size == 0) or global_model is None:
                    # the first tree becomes the global model to append to
                    global_model = local_tree_json
                else:
                    # append subsequent trees
                    local_model = local_tree_json
                    local_trees = local_model['learner']['gradient_booster']['model']['trees']
                    global_model = append_trees(global_model, local_trees)
        else:
            global_model = json.loads(convert_back_to_json(global_model))

            for local_tensor in local_tensors:
                local_trees = json.loads(convert_back_to_json(local_tensor.tensor))
                global_model = append_trees(global_model, local_trees)

        ## Ensures that model is recoverable. TODO put in function
        # Convert latest_trees to a JSON string
        global_model_json = json.dumps(global_model)

        # Convert JSON string to np.float32 array
        global_model_bytes = global_model_json.encode('utf-8')
        global_model_base64 = base64.b64encode(global_model_bytes).decode('utf-8')
        global_model_float32_array = np.frombuffer(global_model_base64.encode('utf-8'), dtype=np.uint8).view(np.float32)

        return global_model_float32_array

        # # global_model = None
        # import pdb; pdb.set_trace()
        # global_model = get_global_model(db_iterator, fl_round)
        
        # for local_tensor in local_tensors:
        #     local_tree_np_array = local_tensor.tensor[:-2]
        #     # local_tree_np_array = local_tensor.tensor['local_tree']
        #     local_tree_json = convert_back_to_json(local_tree_np_array)
            
        #     if global_model.size == 0:
        #         # the first tree becomes the global model to append to
        #         global_model = local_tree_json
        #     else:
        #         # append subsequent trees
        #         local_model = local_tree_json
        #         # Assertion to check if the original trees in the local model match the global model trees
        #         num_global_trees = int(local_tensor.tensor[-2])
        #         # num_global_trees = local_tensor.tensor['num_global_trees']
        #         verify_global_model(global_model, local_model, num_global_trees)
                
        #         num_global_trees = int(global_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
        #         num_latest_trees = int(local_tensor.tensor[-1])
        #         # num_latest_trees = local_tensor.tensor['num_latest_trees']
        #         local_trees = local_model['learner']['gradient_booster']['model']['trees'][-num_latest_trees:]

        #         global_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(
        #             num_global_trees + num_latest_trees
        #         )
        #         global_model["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        #             num_global_trees + num_latest_trees
        #         )

        #         for new_tree in range(num_latest_trees):
        #             local_trees[new_tree]["id"] = num_global_trees + new_tree
        #             global_model["learner"]["gradient_booster"]["model"]["trees"].append(local_trees[new_tree])
        #             global_model["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

        # # TODO: this will probably be problematic, make sure that the conversion is working
        # return bytearray(json.dumps(global_model, default=int), "utf-8")
