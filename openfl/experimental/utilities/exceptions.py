# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class SerializationError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        pass


class GPUResourcesNotAvailableError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        pass
