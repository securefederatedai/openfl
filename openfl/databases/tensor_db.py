# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""TensorDB Module."""

from threading import Lock
from types import MethodType
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd

from openfl.databases.utilities import ROUND_PLACEHOLDER, _retrieve, _search, _store
from openfl.interface.aggregation_functions import AggregationFunction
from openfl.utilities import LocalTensor, TensorKey, change_tags


class TensorDB:
    """The TensorDB stores a tensor key and the data that it corresponds to.

    It is built on top of a pandas dataframe for it's easy insertion, retreival
    and aggregation capabilities. Each collaborator and aggregator has its own
    TensorDB.

    Attributes:
        tensor_db: A pandas DataFrame that stores the tensor key and the data
            that it corresponds to.
        mutex: A threading Lock object used to ensure thread-safe operations
            on the tensor_db Dataframe.
    """

    def __init__(self) -> None:
        """Initializes a new instance of the TensorDB class."""
        types_dict = {
            "tensor_name": "string",
            "origin": "string",
            "round": "int32",
            "report": "bool",
            "tags": "object",
            "nparray": "object",
        }
        self.tensor_db = pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in types_dict.items()}
        )
        self._bind_convenience_methods()

        self.mutex = Lock()

    def _bind_convenience_methods(self):
        """Bind convenience methods for the TensorDB dataframe to make storage,
        retrieval, and search easier."""
        if not hasattr(self.tensor_db, "store"):
            self.tensor_db.store = MethodType(_store, self.tensor_db)
        if not hasattr(self.tensor_db, "retrieve"):
            self.tensor_db.retrieve = MethodType(_retrieve, self.tensor_db)
        if not hasattr(self.tensor_db, "search"):
            self.tensor_db.search = MethodType(_search, self.tensor_db)

    def __repr__(self) -> str:
        """Returns the string representation of the TensorDB object.

        Returns:
            content (str): The string representation of the TensorDB object.
        """
        with pd.option_context("display.max_rows", None):
            content = self.tensor_db[["tensor_name", "origin", "round", "report", "tags"]]
            return f"TensorDB contents:\n{content}"

    def __str__(self) -> str:
        """Returns the string representation of the TensorDB object.

        Returns:
        __repr__ (str): The string representation of the TensorDB object.
        """
        return self.__repr__()

    def clean_up(self, remove_older_than: int = 1) -> None:
        """Removes old entries from the database to prevent it from becoming
        too large and slow.

        Args:
            remove_older_than (int, optional): Entries older than this number
                of rounds are removed. Defaults to 1.
        """
        if remove_older_than < 0:
            # Getting a negative argument calls off cleaning
            return
        current_round = self.tensor_db["round"].astype(int).max()
        if current_round == ROUND_PLACEHOLDER:
            current_round = np.sort(self.tensor_db["round"].astype(int).unique())[-2]
        self.tensor_db = self.tensor_db[
            (self.tensor_db["round"].astype(int) > current_round - remove_older_than)
            | self.tensor_db["report"]
        ].reset_index(drop=True)

    def cache_tensor(self, tensor_key_dict: Dict[TensorKey, np.ndarray]) -> None:
        """Insert a tensor into TensorDB (dataframe).

        Args:
            tensor_key_dict (Dict[TensorKey, np.ndarray]): A dictionary where
                the key is a TensorKey and the value is a numpy array.

        Returns:
            None
        """
        entries_to_add = []
        with self.mutex:
            for tensor_key, nparray in tensor_key_dict.items():
                tensor_name, origin, fl_round, report, tags = tensor_key
                entries_to_add.append(
                    pd.DataFrame(
                        [
                            [
                                tensor_name,
                                origin,
                                fl_round,
                                report,
                                tags,
                                nparray,
                            ]
                        ],
                        columns=list(self.tensor_db.columns),
                    )
                )

            self.tensor_db = pd.concat([self.tensor_db, *entries_to_add], ignore_index=True)

    def get_tensor_from_cache(self, tensor_key: TensorKey) -> Optional[np.ndarray]:
        """Perform a lookup of the tensor_key in the TensorDB.

        Args:
            tensor_key (TensorKey): The key of the tensor to look up.

        Returns:
            Optional[np.ndarray]: The numpy array if it is available.
                Otherwise, returns None.
        """
        tensor_name, origin, fl_round, report, tags = tensor_key

        # TODO come up with easy way to ignore compression
        df = self.tensor_db[
            (self.tensor_db["tensor_name"] == tensor_name)
            & (self.tensor_db["origin"] == origin)
            & (self.tensor_db["round"] == fl_round)
            & (self.tensor_db["report"] == report)
            & (self.tensor_db["tags"] == tags)
        ]

        if len(df) == 0:
            return None
        return np.array(df["nparray"].iloc[0])

    def get_aggregated_tensor(
        self,
        tensor_key: TensorKey,
        collaborator_weight_dict: dict,
        aggregation_function: AggregationFunction,
    ) -> Optional[np.ndarray]:
        """
        Determine whether all of the collaborator tensors are present for a
        given tensor key

        Returns their weighted average.

        Args:
            tensor_key (TensorKey): The tensor key to be resolved. If origin
                'agg_uuid' is present, can be returned directly. Otherwise
                must compute weighted average of all collaborators.
            collaborator_weight_dict (dict): A dictionary where the keys are
                collaborator names and the values are their respective weights.
            aggregation_function (AggregationFunction): Call the underlying
                numpy aggregation function to use to compute the weighted
                average. Default is just the weighted average.

        Returns:
            agg_nparray Optional[np.ndarray]: weighted_nparray The weighted
                average if all collaborator values are present. Otherwise,
                returns None.
            None: if not all values are present.
        """
        if len(collaborator_weight_dict) != 0:
            assert (
                np.abs(1.0 - sum(collaborator_weight_dict.values())) < 0.01
            ), f"Collaborator weights do not sum to 1.0: {collaborator_weight_dict}"

        collaborator_names = collaborator_weight_dict.keys()
        agg_tensor_dict = {}

        # Check if the aggregated tensor is already present in TensorDB
        tensor_name, origin, fl_round, report, tags = tensor_key

        raw_df = self.tensor_db[
            (self.tensor_db["tensor_name"] == tensor_name)
            & (self.tensor_db["origin"] == origin)
            & (self.tensor_db["round"] == fl_round)
            & (self.tensor_db["report"] == report)
            & (self.tensor_db["tags"] == tags)
        ]["nparray"]
        if len(raw_df) > 0:
            return np.array(raw_df.iloc[0]), {}

        for col in collaborator_names:
            new_tags = change_tags(tags, add_field=col)
            raw_df = self.tensor_db[
                (self.tensor_db["tensor_name"] == tensor_name)
                & (self.tensor_db["origin"] == origin)
                & (self.tensor_db["round"] == fl_round)
                & (self.tensor_db["report"] == report)
                & (self.tensor_db["tags"] == new_tags)
            ]["nparray"]
            if len(raw_df) == 0:
                tk = TensorKey(tensor_name, origin, report, fl_round, new_tags)
                print(f"No results for collaborator {col}, TensorKey={tk}")
                return None
            else:
                agg_tensor_dict[col] = raw_df.iloc[0]

        local_tensors = [
            LocalTensor(
                col_name=col_name,
                tensor=agg_tensor_dict[col_name],
                weight=collaborator_weight_dict[col_name],
            )
            for col_name in collaborator_names
        ]

        if hasattr(aggregation_function, "_privileged"):
            if aggregation_function._privileged:
                with self.mutex:
                    self._bind_convenience_methods()
                    agg_nparray = aggregation_function(
                        local_tensors,
                        self.tensor_db,
                        tensor_name,
                        fl_round,
                        tags,
                    )
                self.cache_tensor({tensor_key: agg_nparray})

                return np.array(agg_nparray)

        db_iterator = self._iterate()
        agg_nparray = aggregation_function(local_tensors, db_iterator, tensor_name, fl_round, tags)
        self.cache_tensor({tensor_key: agg_nparray})

        return np.array(agg_nparray)

    def _iterate(self, order_by: str = "round", ascending: bool = False) -> Iterator[pd.Series]:
        """Returns an iterator over the rows of the TensorDB, sorted by a
        specified column.

        Args:
            order_by (str, optional): The column to sort by. Defaults to
                'round'.
            ascending (bool, optional): Whether to sort in ascending order.
                Defaults to False.

        Returns:
            Iterator[pd.Series]: An iterator over the rows of the TensorDB.
        """
        columns = ["round", "nparray", "tensor_name", "tags"]
        rows = self.tensor_db[columns].sort_values(by=order_by, ascending=ascending).iterrows()
        for _, row in rows:
            yield row
