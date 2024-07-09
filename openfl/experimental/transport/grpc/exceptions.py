"""Exceptions that occur during service interaction."""


class ShardNotFoundError(Exception):
    """Indicates that director has no information about that shard."""
