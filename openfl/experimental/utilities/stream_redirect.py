# Copyright 2020-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""openfl.experimental.utilities.stream_redirect module."""

import io
import sys
from copy import deepcopy


class RedirectStdStreamBuffer:
    """
    Buffer object used to store stdout & stderr
    """

    def __init__(self):
        self._stdoutbuff = io.StringIO()
        self._stderrbuff = io.StringIO()

    def get_stdstream(self):
        """
        Return the contents of stdout and stderr buffers
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
    This class used to intercept stdout and stderr, so that
    stdout and stderr is written to buffer as well as terminal
    """

    def __init__(self, buffer, destination):
        self.__stdDestination = destination
        self.__stdBuffer = buffer

    def write(self, message):
        message = f"\33[94m{message}\33[0m"
        self.__stdDestination.write(message)
        self.__stdBuffer.write(message)

    def flush(self):
        pass


class RedirectStdStreamContext:
    """
    Context Manager that enables redirection of stdout & stderr
    """

    def __init__(self):
        self.stdstreambuffer = RedirectStdStreamBuffer()

    def __enter__(self):
        """
        Create context to redirect stdout & stderr
        """
        self.__old_stdout = sys.stdout
        self.__old_stderr = sys.stderr
        sys.stdout = RedirectStdStream(self.stdstreambuffer._stdoutbuff, sys.stdout)
        sys.stderr = RedirectStdStream(self.stdstreambuffer._stderrbuff, sys.stderr)

        return self.stdstreambuffer

    def __exit__(self, et, ev, tb):
        """
        Exit the context and restore the stdout & stderr
        """
        sys.stdout = self.__old_stdout
        sys.stderr = self.__old_stderr
