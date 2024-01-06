# Copyright (C) 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class SerializationError(Exception):
    """Raised when there is an error in serialization process."""

    def __init__(self, *args: object) -> None:
        """
        Initializes the SerializationError with the provided arguments.

        Args:
            *args (object): Variable length argument list.
        """
        super().__init__(*args)
        pass


class GPUResourcesNotAvailableError(Exception):
    """Raised when the required GPU resources are not available."""

    def __init__(self, *args: object) -> None:
        """Initializes the GPUResourcesNotAvailableError with the provided arguments.

        Args:
            *args (object): Variable length argument list.
        """
        super().__init__(*args)
        pass
