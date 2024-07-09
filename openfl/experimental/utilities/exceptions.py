

class SerializationError(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        pass


class ResourcesNotAvailableError(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        pass


class ResourcesAllocationError(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        pass
