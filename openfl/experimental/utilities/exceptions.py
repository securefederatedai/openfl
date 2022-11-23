class SerializationException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        pass


class GPUResourcesNotAvailable(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        pass
