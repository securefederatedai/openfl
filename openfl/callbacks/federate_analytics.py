import json
import logging

import numpy as np

from openfl.callbacks.callback import Callback

logger = logging.getLogger(__name__)


class FederateAnalyticsCallback(Callback):
    """
    FederateAnalyticsCallback is a custom callback class for federated analytics.
    This callback is triggered at the end of federated analytics to perform
    analytics-related operations, such as retrieving tensors tagged with "analytics"
    from a tensor database and saving them to a JSON file.

    Methods:
        __init__(self):Initializes the callback with an optional log directory.
        end_of_round(round_number, metrics, context): process and save analytics data
            after round completion.
    """

    def __init__(self):
        super().__init__()

    def on_round_end(self, round_num: int, logs=None):
        """
        Callback triggered at the end of each federated analytics round to save
        analytical result in JSON file.

        Args:
            round_number (int): The current round number.
            metrics (dict): A dictionary containing metrics for the round.
            context (dict): Additional context or state information.
        """
        analytics_result = self.tensor_db.get_tensors_by_round_and_tags(round_num, ("analytics",))
        if len(analytics_result) > 0 and self.params.get("last_state_path"):
            with open(self.params.get("last_state_path"), "w") as jsonfile:
                analytics_result_json = {}
                for tensorkey, values in analytics_result.items():
                    if isinstance(values, np.ndarray):
                        values = values.tolist()
                    analytics_result_json[tensorkey.tensor_name] = values
                json.dump(analytics_result_json, jsonfile, indent=4)
            logger.info(f"Analytics result: {analytics_result_json}")
