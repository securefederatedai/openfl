# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TensorDB Module."""

import pandas as pd
import numpy as np

from threading import Lock

from openfl.utilities import TensorKey


class TensorDB:
    """
    The TensorDB stores a tensor key and the data that it corresponds to.

    It is built on top of a pandas dataframe
    for it's easy insertion, retreival and aggregation capabilities. Each
    collaborator and aggregator has its own TensorDB.
    """

    def __init__(self):
        """Initialize."""
        self.tensor_db = pd.DataFrame([], columns=[
            'tensor_name', 'origin', 'round', 'report', 'tags', 'nparray'
        ])
        self.mutex = Lock()

    def __repr__(self):
        """Representation of the object."""
        with pd.option_context('display.max_rows', None):
            return 'TensorDB contents:\n{}'.format(
                self.tensor_db[
                    ['tensor_name', 'origin', 'round', 'report', 'tags']
                ])

    def __str__(self):
        """Printable string representation."""
        return self.__repr__()

    def clean_up(self, remove_older_than=1):
        """Remove old entries from database preventing the db from becoming too large and slow."""
        if remove_older_than < 0:
            # Getting a negative argument calls off cleaning
            return
        current_round = int(self.tensor_db['round'].max())
        self.tensor_db = self.tensor_db[
            self.tensor_db['round'] > current_round - remove_older_than
        ].reset_index(drop=True)

    def cache_tensor(self, tensor_key_dict):
        """Insert tensor into TensorDB (dataframe).

        Args:
            tensor_key_dict: The Tensor Key

        Returns:
            None
        """
        entries_to_add = []
        with self.mutex:
            for tensor_key, nparray in tensor_key_dict.items():
                tensor_name, origin, fl_round, report, tags = tensor_key
                entries_to_add.append(
                    pd.DataFrame([
                        [tensor_name, origin, fl_round, report, tags, nparray]
                    ],
                        columns=[
                            'tensor_name',
                            'origin',
                            'round',
                            'report',
                            'tags',
                            'nparray']
                    )
                )

            self.tensor_db = pd.concat(
                [self.tensor_db, *entries_to_add], ignore_index=True
            )

    def get_tensor_from_cache(self, tensor_key):
        """
        Perform a lookup of the tensor_key in the TensorDB.

        Returns the nparray if it is available
        Otherwise, it returns 'None'
        """
        tensor_name, origin, fl_round, report, tags = tensor_key

        # TODO come up with easy way to ignore compression
        df = self.tensor_db[(self.tensor_db['tensor_name'] == tensor_name)
                            & (self.tensor_db['origin'] == origin)
                            & (self.tensor_db['round'] == fl_round)
                            & (self.tensor_db['report'] == report)
                            & (self.tensor_db['tags'] == tags)]

        if len(df) == 0:
            return None
        return np.array(df['nparray'].iloc[0])

    def get_aggregated_tensor(self, tensor_key, collaborator_weight_dict,
                              aggregation_functions):
        """
        Determine whether all of the collaborator tensors are present for a given tensor key.

        Returns their weighted average.

        Args:
            tensor_key: The tensor key to be resolved. If origin 'agg_uuid' is
                        present, can be returned directly. Otherwise must
                        compute weighted average of all collaborators
            collaborator_weight_dict: List of collaborator names in federation
                                      and their respective weights
            aggregation_functions: Call the underlying numpy aggregation
                                   function. Default is just the weighted
                                   average. ONLY THE FIRST FUNCTION will get
                                   recorded as the aggregated tensor

        Returns:
            weighted_nparray if all collaborator values are present
            None if not all values are present

        """
        if len(collaborator_weight_dict) != 0:
            assert (np.abs(
                1.0 - sum(collaborator_weight_dict.values())
            ) < 0.01), \
                'Collaborator weights do not sum to 1.0:' \
                ' {}'.format(collaborator_weight_dict)

        if aggregation_functions is None:
            aggregation_functions = ['weighted_average']

        collaborator_names = collaborator_weight_dict.keys()
        agg_tensor_dict = {}

        # Check if the aggregated tensor is already present in TensorDB
        tensor_name, origin, fl_round, report, tags = tensor_key

        raw_df = self.tensor_db[(self.tensor_db['tensor_name'] == tensor_name)
                                & (self.tensor_db['origin'] == origin)
                                & (self.tensor_db['round'] == fl_round)
                                & (self.tensor_db['report'] == report)
                                & (self.tensor_db['tags'] == tags)]['nparray']
        if len(raw_df) > 0:
            return np.array(raw_df.iloc[0]), {}

        for col in collaborator_names:
            if type(tags) == str:
                new_tags = tuple([tags] + [col])
            else:
                new_tags = tuple(list(tags) + [col])
            raw_df = self.tensor_db[
                (self.tensor_db['tensor_name'] == tensor_name)
                & (self.tensor_db['origin'] == origin)
                & (self.tensor_db['round'] == fl_round)
                & (self.tensor_db['report'] == report)
                & (self.tensor_db['tags'] == new_tags)]['nparray']
            # print(raw_df)
            if len(raw_df) == 0:
                print('No results for collaborator {}, TensorKey={}'.format(
                    col, TensorKey(
                        tensor_name, origin, report, fl_round, new_tags
                    )))
                return None, {}
            else:
                agg_tensor_dict[col] = raw_df.iloc[0]
            # agg_tensor_dict[col] = agg_tensor_dict[col]
            # * collaborator_weight_dict[col]

        concat_nparray = np.array(list(agg_tensor_dict.values()))

        aggregated_tensorkey_is_set = False
        agg_metadata_dict = {}

        for aggregation_function in aggregation_functions:
            if aggregation_function == 'weighted_average':
                agg_nparray = np.average(
                    concat_nparray,
                    weights=np.array(list(collaborator_weight_dict.values())),
                    axis=0
                )
            else:
                agg_func = getattr(np, aggregation_function, None)
                if callable(agg_func):
                    agg_nparray = agg_func(concat_nparray, axis=0)
                else:
                    raise KeyError(
                        '{} is not a valid numpy function'.format(
                            aggregation_function))

            # TODO This enforces that only the first aggregation function
            #  will be associated with
            if not aggregated_tensorkey_is_set:
                # Cache aggregated tensor in TensorDB
                self.cache_tensor({tensor_key: agg_nparray})
                aggregated_tensorkey_is_set = True
                primary_agg = agg_nparray
            else:
                agg_metadata_dict[aggregation_function] = np.array(agg_nparray)

        return np.array(primary_agg), agg_metadata_dict
