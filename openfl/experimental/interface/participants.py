# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.interface.participants module."""


class Participant:
    def __init__(self, name=""):
        self.private_attributes = {}
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def private_attributes(self, attrs):
        self.private_attributes = attrs


class Collaborator(Participant):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Aggregator(Participant):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
