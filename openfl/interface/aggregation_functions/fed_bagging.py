# Copyright 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Federated Boostrap Aggregation for XGBoost module."""

import json
from openfl.interface.aggregation_functions.core import AggregationFunction


def verify_global_model(global_model, local_model, num_global_trees):
    for i in range(num_global_trees):
        global_tree = global_model['learner']['gradient_booster']['model']['trees'][i]
        global_tree_local = local_model['learner']['gradient_booster']['model']['trees'][i]
        
        assert global_tree == global_tree_local, \
            "Mismatch found in trees. Models are not from the same global model."


class FedBaggingXGBoost(AggregationFunction):
    """Federated Boostrap Aggregation for XGBoost."""

    def call(self, local_tensors, *_):
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
        global_model = None
        
        for local_tensor in local_tensors:
            if global_model is None:
                global_model = json.loads(local_tensor.tensor['local_tree'])
            else:
                local_model = json.loads(local_tensor.tensor['local_tree'])
                
                # Assertion to check if the original trees in the local model match the global model trees
                num_global_trees = local_tensor.tensor['num_global_trees']
                verify_global_model(global_model, local_model, num_global_trees)
                
                num_global_trees = int(global_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"])
                num_latest_trees = local_tensor.tensor['num_latest_trees']
                local_trees = local_model['learner']['gradient_booster']['model']['trees'][-num_latest_trees:]

                global_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"]["num_trees"] = str(
                    num_global_trees + num_latest_trees
                )
                global_model["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
                    num_global_trees + num_latest_trees
                )

                for new_tree in range(num_latest_trees):
                    local_trees[new_tree]["id"] = num_global_trees + new_tree
                    global_model["learner"]["gradient_booster"]["model"]["trees"].append(local_trees[new_tree])
                    global_model["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

        return bytearray(json.dumps(global_model), "utf-8")
