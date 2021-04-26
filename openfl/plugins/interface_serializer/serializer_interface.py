class Serializer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def serialize(object, filename):
        raise NotImplementedError

    @staticmethod
    def restore_object(filename):
        raise NotImplementedError