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
    Simple CNN for classification.

    PyTorchTaskRunner inherits from nn.module, so you can define your model
    in the same way that you would for PyTorch
    """

    def __init__(self, params=None, num_rounds=1, **kwargs):
        """Initialize.

        Args:
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(**kwargs)

        self.bst = None
        self.params = params
        self.num_rounds = num_rounds

    def train_(self, train_dataloader) -> Metric:
        """Train model."""
        dtrain = train_dataloader['dmatrix']
        evals = [(dtrain, 'train')]
        evals_result = {}

        self.bst = xgb.train(self.params, dtrain, self.num_rounds, xgb_model=self.bst,
                             evals=evals, evals_result=evals_result, verbose_eval=False)

        loss = evals_result['train']['logloss'][-1]
        return Metric(name=self.params['eval_metric'], value=np.array(loss))

    def validate_(self, validation_dataloader) -> Metric:
        """Validate model."""

        dtest = validation_dataloader['dmatrix']
        y_test = validation_dataloader['labels']
        preds = self.bst.predict(dtest)
        y_pred_binary = np.where(preds > 0.5, 1, 0)
        acc = accuracy_score(y_test, y_pred_binary)

        return Metric(name="accuracy", value=np.array(acc))
