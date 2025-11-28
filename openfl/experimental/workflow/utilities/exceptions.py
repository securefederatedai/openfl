# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class SerializationError(Exception):
    """Raised when there is an error in serialization process."""

    def __init__(self, *args: object) -> None:
        """Initializes the SerializationError with the provided arguments.

        Args:
            *args (object): Variable length argument list.
        """
        super().__init__(*args)
        pass


class ResourcesNotAvailableError(Exception):
    """Exception raised when the required resources are not available."""

    def __init__(self, *args: object) -> None:
        """Initializes the ResourcesNotAvailableError with the provided
        arguments.

        Args:
            *args (object): Variable length argument list.
        """
        super().__init__(*args)
        pass


class ResourcesAllocationError(Exception):
    """Exception raised when there is an error in the resources allocation
    process."""

    def __init__(self, *args: object) -> None:
        """Initializes the ResourcesAllocationError with the provided
        arguments.

        Args:
            *args (object): Variable length argument list.
        """
        super().__init__(*args)
        pass
