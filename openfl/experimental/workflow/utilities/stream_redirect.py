# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.workflow.utilities.stream_redirect module."""

import io
import sys
from copy import deepcopy


class RedirectStdStreamBuffer:
    """Buffer object used to store stdout and stderr.

    Attributes:
        _stdoutbuff (io.StringIO): Buffer for stdout.
        _stderrbuff (io.StringIO): Buffer for stderr.
    """

    def __init__(self):
        """Initializes the RedirectStdStreamBuffer with empty stdout and stderr
        buffers."""
        self._stdoutbuff = io.StringIO()
        self._stderrbuff = io.StringIO()

    def get_stdstream(self):
        """Returns the contents of stdout and stderr buffers.

        Returns:
            tuple: A tuple containing the contents of stdout and stderr
                buffers.
        """
        self._stdoutbuff.seek(0)
        self._stderrbuff.seek(0)

        step_stdout = deepcopy(self._stdoutbuff)
        step_stderr = deepcopy(self._stderrbuff)

        self._stdoutbuff.truncate(0)
        self._stderrbuff.truncate(0)

        return step_stdout, step_stderr


class RedirectStdStream:
    """
    Class used to intercept stdout and stderr, so that stdout and stderr is
    written to buffer as well as terminal.

    Attributes:
        __stdDestination (io.TextIOWrapper): Destination for standard outputs.
        __stdBuffer (RedirectStdStreamBuffer): Buffer for standard outputs.
    """

    def __init__(self, buffer, destination):
        """Initializes the RedirectStdStream with a buffer and a destination.

        Args:
            buffer (RedirectStdStreamBuffer): Buffer for standard outputs.
            destination (io.TextIOWrapper): Destination for standard outputs.
        """
        self.__stdDestination = destination
        self.__stdBuffer = buffer

    def write(self, message):
        """Writes the message to the standard destination and buffer.

        Args:
            message (str): The message to write.
        """
        message = f"\33[94m{message}\33[0m"
        self.__stdDestination.write(message)
        self.__stdBuffer.write(message)

    def flush(self):
        pass


class RedirectStdStreamContext:
    """Context Manager that enables redirection of stdout and stderr.

    Attributes:
        stdstreambuffer (RedirectStdStreamBuffer): Buffer for standard outputs.
    """

    def __init__(self):
        """Initializes the RedirectStdStreamContext with a
        RedirectStdStreamBuffer."""
        self.stdstreambuffer = RedirectStdStreamBuffer()

    def __enter__(self):
        """Creates a context to redirect stdout and stderr.

        Returns:
            RedirectStdStreamBuffer: The buffer for standard outputs.
        """
        self.__old_stdout = sys.stdout
        self.__old_stderr = sys.stderr
        sys.stdout = RedirectStdStream(self.stdstreambuffer._stdoutbuff, sys.stdout)
        sys.stderr = RedirectStdStream(self.stdstreambuffer._stderrbuff, sys.stderr)

        return self.stdstreambuffer

    def __exit__(self, et, ev, tb):
        """Exits the context and restores the stdout and stderr.

        Args:
            et (type): The type of exception.
            ev (BaseException): The instance of exception.
            tb (traceback): A traceback object encapsulating the call stack.
        """
        sys.stdout = self.__old_stdout
        sys.stderr = self.__old_stderr
