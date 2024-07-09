# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from openfl.interface.aggregation_functions.adagrad_adaptive_aggregation import (
    AdagradAdaptiveAggregation,
)
from openfl.interface.aggregation_functions.adam_adaptive_aggregation import (
    AdamAdaptiveAggregation,
)
from openfl.interface.aggregation_functions.core import AggregationFunction
from openfl.interface.aggregation_functions.fedcurv_weighted_average import (
    FedCurvWeightedAverage,
)
from openfl.interface.aggregation_functions.geometric_median import (
    GeometricMedian,
)
from openfl.interface.aggregation_functions.median import Median
from openfl.interface.aggregation_functions.weighted_average import (
    WeightedAverage,
)
from openfl.interface.aggregation_functions.yogi_adaptive_aggregation import (
    YogiAdaptiveAggregation,
)
