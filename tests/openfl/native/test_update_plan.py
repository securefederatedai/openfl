# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Update plan test module."""
import pytest
from pathlib import Path

from openfl.federated import Plan
from openfl.native import update_plan


@pytest.mark.parametrize(
    'override_config,expected_result', [
        ({},
        {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': 'Portland'}}}}}}),
        ({'Planet.Earth.Continent.Australia': 'Sydney'},
        {'Planet': {'Earth': {'Continent': {'Australia': 'Sydney',
                                            'North-America': {'USA': {'Oregon': 'Portland'}}}}}})
    ])
def test_update_plan_new_key_value_addition(override_config,expected_result):
    """Test update_plan for adding a new key value pair."""
    plan = Plan()
    plan.config = Plan.load(Path('./base_example.yaml'))
    result = update_plan(override_config, plan=plan, resolve=False)
    assert result.config == expected_result


@pytest.mark.parametrize(
    'override_config,expected_result', [
        ({'Planet.Mars.0': 'Water', 'Planet.Mars.1': 'Ice'},
        {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': 'Portland'}}}},
                    'Mars': ['Water', 'Ice']}}),
        ({'Planet.Earth.Continent.Australia.0': 'Sydney', 'Planet.Earth.Continent.Australia.1': 'Melbourne'},
        {'Planet': {'Earth': {'Continent': {'Australia': ['Sydney', 'Melbourne'],
                                            'North-America': {'USA': {'Oregon': 'Portland'}}}}}})
    ])
def test_update_plan_new_key_list_value_addition(override_config,expected_result):
    """Test update_plan or adding a new key with value as a list."""
    plan = Plan()
    plan.config = Plan.load(Path('./base_example.yaml'))
    result = update_plan(override_config, plan=plan, resolve=False)
    assert result.config == expected_result


@pytest.mark.parametrize(
    'override_config,expected_result', [
        ({'Planet.Earth.Continent.North-America.USA.Oregon': 'Salem'},
        {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': 'Salem'}}}}}})
    ])
def test_update_plan_existing_key_value_updation(override_config,expected_result):
    """Test update_plan for adding a new key value pair."""
    plan = Plan()
    plan.config = Plan.load(Path('./base_example.yaml'))
    result = update_plan(override_config, plan=plan, resolve=False)
    assert result.config == expected_result


@pytest.mark.parametrize(
    'override_config,expected_result', [
        ({'Planet.Earth.Continent.North-America.USA.Oregon.0': 'Portland', 'Planet.Earth.Continent.North-America.USA.Oregon.1': 'Salem'},
        {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': ['Portland', 'Salem']}}}}}}),
        ({'Planet.Earth.Continent.North-America.USA.Oregon.0': 'Portland'},
        {'Planet': {'Earth': {'Continent': {'North-America': {'USA': {'Oregon': ['Portland']}}}}}})
    ])
def test_update_plan_existing_key_list_value_updation(override_config,expected_result):
    """Test update_plan or adding a new key with value as a list."""
    plan = Plan()
    plan.config = Plan.load(Path('./base_example.yaml'))
    result = update_plan(override_config, plan=plan, resolve=False)
    assert result.config == expected_result