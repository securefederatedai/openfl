# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Update plan test module."""
import pytest
from pathlib import Path

from openfl.federated import Plan
from openfl.native import update_plan


@pytest.mark.parametrize(
    'override_config,expected_result', [
        ({},
         {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': 'Portland'}}}},
                     'Mars': ['Water', 'Ice'],
                     'Pluto': []}}),
        ({'Planet.Earth.Continent.Australia': 'Sydney'},
         {'Planet': {'Earth': {'Continent': {'Australia': 'Sydney',
                                             'North-America': {'USA': {'Oregon': 'Portland'}}}},
                     'Mars': ['Water', 'Ice'],
                     'Pluto': []}})
    ])
def test_update_plan_new_key_value_addition(override_config, expected_result):
    """Test update_plan for adding a new key value pair."""
    plan = Plan()
    plan.config = Plan.load(Path('./tests/openfl/native/base_example.yaml'))
    result = update_plan(override_config, plan=plan, resolve=False)
    assert result.config == expected_result


@pytest.mark.parametrize(
    'override_config,expected_result', [
        ({'Planet.Jupiter': ['Sun', 'Rings']},
         {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': 'Portland'}}}},
                     'Mars': ['Water', 'Ice'],
                     'Pluto': [],
                     'Jupiter': ['Sun', 'Rings']}}),
        ({'Planet.Earth.Continent.Australia': ['Sydney', 'Melbourne']},
         {'Planet': {'Earth': {'Continent': {'Australia': ['Sydney', 'Melbourne'],
                                             'North-America': {'USA': {'Oregon': 'Portland'}}}},
                     'Mars': ['Water', 'Ice'],
                     'Pluto': []}})
    ])
def test_update_plan_new_key_list_value_addition(override_config, expected_result):
    """Test update_plan or adding a new key with value as a list."""
    plan = Plan()
    plan.config = Plan.load(Path('./tests/openfl/native/base_example.yaml'))
    result = update_plan(override_config, plan=plan, resolve=False)
    assert result.config == expected_result


@pytest.mark.parametrize(
    'override_config,expected_result', [
        ({'Planet.Earth.Continent.North-America.USA.Oregon': 'Salem'},
         {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': 'Salem'}}}},
                     'Mars': ['Water', 'Ice'],
                     'Pluto': []}}),
        ({'Planet.Mars': 'Moon'},
         {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': 'Portland'}}}},
                     'Mars': 'Moon',
                     'Pluto': []}}),
        ({'Planet.Pluto': 'Tiny'},
         {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': 'Portland'}}}},
                     'Mars': ['Water', 'Ice'],
                     'Pluto': 'Tiny'}})
    ])
def test_update_plan_existing_key_value_updation(override_config, expected_result):
    """Test update_plan for adding a new key value pair."""
    plan = Plan()
    plan.config = Plan.load(Path('./tests/openfl/native/base_example.yaml'))
    result = update_plan(override_config, plan=plan, resolve=False)
    assert result.config == expected_result


@pytest.mark.parametrize(
    'override_config,expected_result', [
        ({'Planet.Mars': ['Water', 'Moon', 'Ice']},
         {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': 'Portland'}}}},
                     'Mars': ['Water', 'Moon', 'Ice'],
                     'Pluto': []}}),
        ({'Planet.Mars': ['Water']},
         {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': 'Portland'}}}},
                     'Mars': ['Water'],
                     'Pluto': []}}),
        ({'Planet.Earth.Continent.North-America.USA.Oregon': ['Portland', 'Salem']},
         {'Planet': {'Earth': {'Continent': {'North-America':
                                             {'USA': {'Oregon': ['Portland', 'Salem']}}}},
                     'Mars': ['Water', 'Ice'],
                     'Pluto': []}}),
        ({'Planet.Earth.Continent.North-America.USA.Oregon': ['Salem']},
         {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': ['Salem']}}}},
                     'Mars': ['Water', 'Ice'],
                     'Pluto': []}}),
        ({'Planet.Pluto': ['Tiny', 'Far']},
         {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': 'Portland'}}}},
                     'Mars': ['Water', 'Ice'],
                     'Pluto': ['Tiny', 'Far']}})
    ])
def test_update_plan_existing_key_list_value_updation(override_config, expected_result):
    """Test update_plan or adding a new key with value as a list."""
    plan = Plan()
    plan.config = Plan.load(Path('./tests/openfl/native/base_example.yaml'))
    result = update_plan(override_config, plan=plan, resolve=False)
    assert result.config == expected_result
