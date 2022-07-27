# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TensorDB Module."""

from threading import Lock
from typing import Dict
from typing import Iterator
from typing import Optional
from types import MethodType

import numpy as np
import pandas as pd

from openfl.interface.aggregation_functions import AggregationFunction
from openfl.utilities import change_tags
from openfl.utilities import LocalTensor
from openfl.utilities import TensorKey
from openfl.databases.utilities import _search,_store,_retrieve, ROUND_PLACEHOLDER


class TensorDB:
    """
    The TensorDB stores a tensor key and the data that it corresponds to.

    It is built on top of a pandas dataframe
    for it's easy insertion, retreival and aggregation capabilities. Each
    collaborator and aggregator has its own TensorDB.
    """

    def __init__(self) -> None:
        """Initialize."""
        self.tensor_db = pd.DataFrame([], columns=[
            'tensor_name', 'origin', 'round', 'report', 'tags', 'nparray'
        ])
        self._bind_convenience_methods()
        
        self.mutex = Lock()

    def _bind_convenience_methods(self):
        # Bind convenience methods for TensorDB dataframe to make storage, retrieval, and search easier
        if not hasattr(self.tensor_db, 'store'):
            self.tensor_db.store = MethodType(_store, self.tensor_db)
        if not hasattr(self.tensor_db, 'retrieve'):
            self.tensor_db.retrieve = MethodType(_retrieve, self.tensor_db)
        if not hasattr(self.tensor_db, 'search'):
            self.tensor_db.search = MethodType(_search, self.tensor_db)

    def __repr__(self) -> str:
        """Representation of the object."""
        with pd.option_context('display.max_rows', None):
            content = self.tensor_db[['tensor_name', 'origin', 'round', 'report', 'tags']]
            return f'TensorDB contents:\n{content}'

    def __str__(self) -> str:
        """Printable string representation."""
        return self.__repr__()
        

    def clean_up(self, remove_older_than: int = 1) -> None:
        """Remove old entries from database preventing the db from becoming too large and slow."""
        if remove_older_than < 0:
            # Getting a negative argument calls off cleaning
            return
        current_round = self.tensor_db['round'].astype(int).max()
        if current_round == ROUND_PLACEHOLDER:
            current_round = np.sort(self.tensor_db['round'].astype(int).unique())[-2]
        self.tensor_db = self.tensor_db[
            (self.tensor_db['round'].astype(int) > current_round - remove_older_than) |
            (self.tensor_db['report'] == True)
        ].reset_index(drop=True)

    def cache_tensor(self, tensor_key_dict: Dict[TensorKey, np.ndarray]) -> None:
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

    def get_tensor_from_cache(self, tensor_key: TensorKey) -> Optional[np.ndarray]:
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

    def get_aggregated_tensor(self, tensor_key: TensorKey, collaborator_weight_dict: dict,
                              aggregation_function: AggregationFunction
                              ) -> Optional[np.ndarray]:
        """
        Determine whether all of the collaborator tensors are present for a given tensor key.

        Returns their weighted average.

        Args:
            tensor_key: The tensor key to be resolved. If origin 'agg_uuid' is
                        present, can be returned directly. Otherwise must
                        compute weighted average of all collaborators
            collaborator_weight_dict: List of collaborator names in federation
                                      and their respective weights
            aggregation_function: Call the underlying numpy aggregation
                                   function. Default is just the weighted
                                   average.
        Returns:
            weighted_nparray if all collaborator values are present
            None if not all values are present

        """
        if len(collaborator_weight_dict) != 0:
            assert np.abs(1.0 - sum(collaborator_weight_dict.values())) < 0.01, (
                f'Collaborator weights do not sum to 1.0: {collaborator_weight_dict}'
            )

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
            new_tags = change_tags(tags, add_field=col)
            raw_df = self.tensor_db[
                (self.tensor_db['tensor_name'] == tensor_name)
                & (self.tensor_db['origin'] == origin)
                & (self.tensor_db['round'] == fl_round)
                & (self.tensor_db['report'] == report)
                & (self.tensor_db['tags'] == new_tags)]['nparray']
            if len(raw_df) == 0:
                tk = TensorKey(tensor_name, origin, report, fl_round, new_tags)
                print(f'No results for collaborator {col}, TensorKey={tk}')
                return None
            else:
                agg_tensor_dict[col] = raw_df.iloc[0]

        local_tensors = [LocalTensor(col_name=col_name,
                                     tensor=agg_tensor_dict[col_name],
                                     weight=collaborator_weight_dict[col_name])
                         for col_name in collaborator_names]

        if hasattr(aggregation_function, '_privileged'):
            if(aggregation_function._privileged):
                with self.mutex:
                    #self.tensor_db.store = MethodType(_store, self.tensor_db)
                    #self.tensor_db.retrieve = MethodType(_retrieve, self.tensor_db)
                    #self.tensor_db.search = MethodType(_search, self.tensor_db)
                    self._bind_convenience_methods()
                    agg_nparray = aggregation_function(local_tensors,
                                                       self.tensor_db,
                                                       tensor_name,
                                                       fl_round,
                                                       tags)
                self.cache_tensor({tensor_key: agg_nparray})

                return np.array(agg_nparray)

        db_iterator = self._iterate()
        agg_nparray = aggregation_function(local_tensors,
                                           db_iterator,
                                           tensor_name,
                                           fl_round,
                                           tags)
        self.cache_tensor({tensor_key: agg_nparray})

        return np.array(agg_nparray)

    def _iterate(self, order_by: str = 'round', ascending: bool = False) -> Iterator[pd.Series]:
        columns = ['round', 'nparray', 'tensor_name', 'tags']
        rows = self.tensor_db[columns].sort_values(by=order_by, ascending=ascending).iterrows()
        for _, row in rows:
            yield row
