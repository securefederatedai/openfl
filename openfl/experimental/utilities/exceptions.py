# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class SerializationException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        pass


class GPUResourcesNotAvailable(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        pass
