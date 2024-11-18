# Copyright (C) 2024 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between
# Intel Corporation and you.

"""You may copy this file as the starting point of your own model."""
import numpy as np
import xgboost as xgb

from openfl.federated import XGBoostTaskRunner
from openfl.utilities import Metric
from sklearn.metrics import accuracy_score


class XGBoostRunner(XGBoostTaskRunner):
    """
    A class to run XGBoost training and validation tasks.

    This class inherits from XGBoostTaskRunner and provides methods to train and validate
    an XGBoost model using federated learning.

    Attributes:
        bst (xgb.Booster): The XGBoost model.
        params (dict): Parameters for the XGBoost model.
        num_rounds (int): Number of boosting rounds.
    """
    def __init__(self, params=None, num_rounds=1, **kwargs):
        """
        Initialize the XGBoostRunner.

        Args:
            params (dict, optional): Parameters for the XGBoost model. Defaults to None.
            num_rounds (int, optional): Number of boosting rounds. Defaults to 1.
            **kwargs: Additional arguments to pass to the function.
        """
        super().__init__(**kwargs)

        self.bst = None
        self.params = params
        self.num_rounds = num_rounds

    def train_(self, data) -> Metric:
        """
        Train the XGBoost model.

        Args:
            train_dataloader (dict): A dictionary containing the training data with keys 'dmatrix'.

        Returns:
            Metric: A Metric object containing the training loss.
        """
        dtrain = data['dmatrix']
        evals = [(dtrain, 'train')]
        evals_result = {}

        self.bst = xgb.train(self.params, dtrain, self.num_rounds, xgb_model=self.bst,
                             evals=evals, evals_result=evals_result, verbose_eval=False)

        loss = evals_result['train']['logloss'][-1]
        return Metric(name=self.params['eval_metric'], value=np.array(loss))

    def validate_(self, data) -> Metric:
        """
        Validate the XGBoost model.

        Args:
            validation_dataloader (dict): A dictionary containing the validation data with keys 'dmatrix' and 'labels'.

        Returns:
            Metric: A Metric object containing the validation accuracy.
        """
        dtest = data['dmatrix']
        y_test = data['labels']
        preds = self.bst.predict(dtest)
        y_pred_binary = np.where(preds > 0.5, 1, 0)
        acc = accuracy_score(y_test, y_pred_binary)

        return Metric(name="accuracy", value=np.array(acc))
