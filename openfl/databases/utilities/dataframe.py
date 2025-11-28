# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Convenience Utilities for DataFrame."""

from typing import Optional

import numpy as np
import pandas as pd

ROUND_PLACEHOLDER = 1000000


def _search(
    self,
    tensor_name: str = None,
    origin: str = None,
    fl_round: int = None,
    metric: bool = None,
    tags: tuple = None,
) -> pd.DataFrame:
    """
    Search the tensor_db dataframe based on:
        - tensor_name
        - origin
        - fl_round
        - metric
        -tags

    Returns a new dataframe that matched the query

    Args:
        tensor_name: The name of the tensor (or metric) to be searched
        origin:      Origin of the tensor
        fl_round:    Round the tensor is associated with
        metric:      Is the tensor a metric?
        tags:        Tuple of unstructured tags associated with the tensor

    Returns:
        pd.DataFrame : New dataframe that matches the search query from
                       the tensor_db dataframe
    """
    df = pd.DataFrame()
    query_string = []
    if tensor_name is not None:
        query_string.append(f"(tensor_name == '{tensor_name}')")
    if origin is not None:
        query_string.append(f"(origin == '{origin}')")
    if fl_round is not None:
        query_string.append(f"(round == {fl_round})")
    if metric is not None:
        query_string.append(f"(report == {metric})")

    if len(query_string) > 0:
        query_string = (" and ").join(query_string)
        df = self.query(query_string)
    if tags is not None:
        if not df.empty:
            df = df[df["tags"] == tags]
        else:
            df = self[self["tags"] == tags]

    if not df.empty:
        return df
    else:
        return self


def _store(
    self,
    tensor_name: str = "_",
    origin: str = "_",
    fl_round: int = ROUND_PLACEHOLDER,
    metric: bool = False,
    tags: tuple = ("_",),
    nparray: np.array = None,
    overwrite: bool = True,
) -> None:
    """
    Convenience method to store a new tensor in the dataframe.

    Args:
        tensor_name [ optional ] : The name of the tensor (or metric) to be saved
        origin      [ optional ] : Origin of the tensor
        fl_round    [ optional ] : Round the tensor is associated with
        metric      [ optional ] : Is the tensor a metric?
        tags        [ optional ] : Tuple of unstructured tags associated with the tensor
        nparray     [ required ] : Value to store associated with the other
                                   included information (i.e. TensorKey info)
        overwrite   [ optional ] : If the tensor is already present in the dataframe
                                   should it be overwritten?

    Returns:
        None
    """

    if nparray is None:
        print("nparray not provided. Nothing to store.")
        return
    idx = self[
        (self["tensor_name"] == tensor_name)
        & (self["origin"] == origin)
        & (self["round"] == fl_round)
        & (self["tags"] == tags)
    ].index
    if len(idx) > 0:
        if not overwrite:
            return
        idx = idx[0]
    else:
        idx = self.shape[0]
    self.loc[idx] = np.array([tensor_name, origin, fl_round, metric, tags, nparray], dtype=object)


def _retrieve(
    self,
    tensor_name: str = "_",
    origin: str = "_",
    fl_round: int = ROUND_PLACEHOLDER,
    metric: bool = False,
    tags: tuple = ("_",),
) -> Optional[np.array]:
    """
    Convenience method to retrieve tensor from the dataframe.

    Args:
        tensor_name [ optional ] : The name of the tensor (or metric) to retrieve
        origin      [ optional ] : Origin of the tensor
        fl_round    [ optional ] : Round the tensor is associated with
        metric:     [ optional ] : Is the tensor a metric?
        tags:       [ optional ] : Tuple of unstructured tags associated with the tensor
                                   should it be overwritten?

    Returns:
        Optional[ np.array ]     : If there is a match, return the first row
    """

    df = self[
        (self["tensor_name"] == tensor_name)
        & (self["origin"] == origin)
        & (self["round"] == fl_round)
        & (self["report"] == metric)
        & (self["tags"] == tags)
    ]["nparray"]

    if len(df) > 0:
        return df.iloc[0]
    else:
        return None
