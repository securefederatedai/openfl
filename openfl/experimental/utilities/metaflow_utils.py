# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""openfl.experimental.utilities.metaflow_utils module."""

from __future__ import annotations

import ast
import fcntl
import hashlib
from datetime import datetime
from pathlib import Path

# getsource only used to determine structure of FlowGraph
from typing import TYPE_CHECKING

import cloudpickle as pickle
import ray
from dill.source import getsource  # nosec
from metaflow.datastore import DATASTORES, FlowDataStore
from metaflow.datastore.exceptions import DataException, UnpicklableArtifactException
from metaflow.datastore.task_datastore import TaskDataStore, only_if_not_done, require_mode
from metaflow.graph import DAGNode, FlowGraph, StepVisitor, deindent_docstring
from metaflow.metaflow_environment import MetaflowEnvironment
from metaflow.mflog import RUNTIME_LOG_SOURCE
from metaflow.plugins import LocalMetadataProvider
from metaflow.runtime import MAX_LOG_SIZE, TruncatedBuffer, mflog_msg
from metaflow.task import MetaDatum

if TYPE_CHECKING:
    from openfl.experimental.interface import FLSpec

import base64
import json
import uuid
from io import StringIO
from typing import Any, Generator, Type

from metaflow import __version__ as mf_version
from metaflow.plugins.cards.card_modules.basic import (
    CSS_PATH,
    JS_PATH,
    RENDER_TEMPLATE_PATH,
    DagComponent,
    DefaultCard,
    PageComponent,
    SectionComponent,
    TaskInfoComponent,
    read_file,
    transform_flow_graph,
)


class SystemMutex:
    """Provides a system-wide mutex that locks a file until the lock is
    released."""

    def __init__(self, name):
        """Initializes the SystemMutex with the provided name.

        Args:
            name (str): The name of the mutex.
        """
        self.name = name

    def __enter__(self):
        lock_id = hashlib.new(
            "md5", self.name.encode("utf8"), usedforsecurity=False
        ).hexdigest()  # nosec
        # MD5sum used for concurrency purposes, not security
        self.fp = open(f"/tmp/.lock-{lock_id}.lck", "wb")
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__(self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()


class Flow:
    """A mock class representing a flow for Metaflow's internal use."""

    def __init__(self, name):
        """Mock flow for metaflow internals.

        Args:
            name (str): The name of the flow.
        """
        self.name = name


@ray.remote
class Counter:

    def __init__(self):
        """Initializes the Counter with value set to 0."""
        self.value = 0

    def increment(self):
        """Increments the counter by 1.

        Returns:
            int: The incremented value of the counter.
        """
        self.value += 1
        return self.value

    def get_counter(self):
        """Retrieves the current value of the counter.

        Returns:
            int: The current value of the counter.
        """
        return self.value


class DAGnode(DAGNode):
    """A custom DAGNode class for the Metaflow graph.

    Attributes:
        name (str): The name of the DAGNode.
        func_lineno (int): The line number of the function in the source code.
        decorators (list): The decorators applied to the function.
        doc (str): The docstring of the function.
        parallel_step (bool): A flag indicating if the step is parallelized.
    """

    def __init__(self, func_ast, decos, doc):
        """Initializes the DAGNode with the provided function AST, decorators,
        and docstring.

        Args:
            func_ast (ast.FunctionDef): The function's abstract syntax tree.
            decos (list): The decorators applied to the function.
            doc (str): The docstring of the function.
        """
        self.name = func_ast.name
        self.func_lineno = func_ast.lineno
        self.decorators = decos
        self.doc = deindent_docstring(doc)
        self.parallel_step = any(getattr(deco, "IS_PARALLEL", False) for deco in decos)

        # these attributes are populated by _parse
        self.tail_next_lineno = 0
        self.type = None
        self.out_funcs = []
        self.has_tail_next = False
        self.invalid_tail_next = False
        self.num_args = 0
        self.foreach_param = None
        self.num_parallel = 0
        self.parallel_foreach = False
        self._parse(func_ast)

        # these attributes are populated by _traverse_graph
        self.in_funcs = set()
        self.split_parents = []
        self.matching_join = None
        # these attributes are populated by _postprocess
        self.is_inside_foreach = False

    def _parse(self, func_ast):
        self.num_args = len(func_ast.args.args)
        tail = func_ast.body[-1]

        # end doesn't need a transition
        if self.name == "end":
            # TYPE is end
            self.type = "end"

        # ensure that the tail an expression
        if not isinstance(tail, ast.Expr):
            return

        # determine the type of self.next transition
        try:
            if not self._expr_str(tail.value.func) == "self.next":
                return

            self.has_tail_next = True
            self.invalid_tail_next = True
            self.tail_next_lineno = tail.lineno
            self.out_funcs = [e.attr for e in tail.value.args]

            keywords = {k.arg: getattr(k.value, "s", None) for k in tail.value.keywords}
            # Second condition in the folliwing line added,
            # To add the support for up to 2 keyword arguments in Flowgraph
            if len(keywords) == 1 or len(keywords) == 2:
                if "foreach" in keywords:
                    # TYPE is foreach
                    self.type = "foreach"
                    if len(self.out_funcs) == 1:
                        self.foreach_param = keywords["foreach"]
                        self.invalid_tail_next = False
                elif "num_parallel" in keywords:
                    self.type = "foreach"
                    self.parallel_foreach = True
                    if len(self.out_funcs) == 1:
                        self.num_parallel = keywords["num_parallel"]
                        self.invalid_tail_next = False
                # Following else parse is also added,
                # If there are no keyword arguments it is a linear flow
                else:
                    self.type = "linear"
            elif len(keywords) == 0:
                if len(self.out_funcs) > 1:
                    # TYPE is split
                    self.type = "split"
                    self.invalid_tail_next = False
                elif len(self.out_funcs) == 1:
                    # TYPE is linear
                    if self.name == "start":
                        self.type = "start"
                    elif self.num_args > 1:
                        self.type = "join"
                    else:
                        self.type = "linear"
                    self.invalid_tail_next = False
        except AttributeError:
            return


class StepVisitor(StepVisitor):
    """A custom StepVisitor class for visiting the steps in a Metaflow
    graph."""

    def __init__(self, nodes, flow):
        """Initializes the StepVisitor with the provided nodes and flow.

        Args:
            nodes (dict): The nodes in the graph.
            flow (Flow): The flow object.
        """
        super().__init__(nodes, flow)

    def visit_FunctionDef(self, node):  # NOQA: N802
        """Visits a FunctionDef node in the flow and adds it to the nodes
        dictionary if it's a step.

        Args:
            node (ast.FunctionDef): The function definition node to visit.
        """
        func = getattr(self.flow, node.name)
        if hasattr(func, "is_step"):
            self.nodes[node.name] = DAGnode(node, func.decorators, func.__doc__)


class FlowGraph(FlowGraph):
    """A custom FlowGraph class for representing a Metaflow graph."""

    def __init__(self, flow):
        """Initializes the FlowGraph with the provided flow.

        Args:
            flow (Flow): The flow object.
        """
        self.name = flow.__name__
        self.nodes = self._create_nodes(flow)
        self.doc = deindent_docstring(flow.__doc__)
        self._traverse_graph()
        self._postprocess()

    def _create_nodes(self, flow):
        """Creates nodes for the flow graph by parsing the source code of the
        flow's module.

        Args:
            flow (Flow): The flow object.

        Returns:
            nodes (dict): A dictionary of nodes in the graph.
        """
        module = __import__(flow.__module__)
        tree = ast.parse(getsource(module)).body
        root = [n for n in tree if isinstance(n, ast.ClassDef) and n.name == self.name][0]
        nodes = {}
        StepVisitor(nodes, flow).visit(root)
        return nodes


class TaskDataStore(TaskDataStore):
    """A custom TaskDataStore class for storing task data in Metaflow."""

    def __init__(
        self,
        flow_datastore,
        run_id,
        step_name,
        task_id,
        attempt=None,
        data_metadata=None,
        mode="r",
        allow_not_done=False,
    ):
        """Initializes the TaskDataStore with the provided parameters.

        Args:
            flow_datastore (FlowDataStore): The flow datastore.
            run_id (str): The run id.
            step_name (str): The step name.
            task_id (str): The task id.
            attempt (int, optional): The attempt number. Defaults to None.
            data_metadata (DataMetadata, optional): The data metadata.
                Defaults to None.
            mode (str, optional): The mode (read 'r' or write 'w'). Defaults
                to 'r'.
            allow_not_done (bool, optional): A flag indicating whether to
                allow tasks that are not done. Defaults to False.
        """
        super().__init__(
            flow_datastore,
            run_id,
            step_name,
            task_id,
            attempt,
            data_metadata,
            mode,
            allow_not_done,
        )

    @only_if_not_done
    @require_mode("w")
    def save_artifacts(self, artifacts_iter, force_v4=False, len_hint=0):
        """Saves Metaflow Artifacts (Python objects) to the datastore and
        stores any relevant metadata needed to retrieve them.

        Typically, objects are pickled but the datastore may perform any
        operation that it deems necessary. You should only access artifacts
        using load_artifacts

        This method requires mode 'w'.

        Args:
            artifacts_iter (Iterator[(string, object)]): Iterator over the
                human-readable name of the object to save and the object
                itself.
        force_v4 (Union[bool, Dict[string -> boolean]], optional): Indicates
            whether the artifact should be pickled using the v4 version of
            pickle. If a single boolean, applies to all artifacts. If a
            dictionary, applies to the object named only. Defaults to False if
            not present or not specified.
        len_hint (int, optional): Estimated number of items in artifacts_iter.
            Defaults to 0.
        """
        artifact_names = []

        def pickle_iter():
            for name, obj in artifacts_iter:
                do_v4 = (
                    force_v4 and force_v4
                    if isinstance(force_v4, bool)
                    else force_v4.get(name, False)
                )
                if do_v4:
                    encode_type = "gzip+pickle-v4"
                    if encode_type not in self._encodings:
                        raise DataException(
                            f"Artifact {name} requires a serialization encoding that "
                            + "requires Python 3.4 or newer."
                        )
                    try:
                        blob = pickle.dumps(obj, protocol=4)
                    except TypeError:
                        raise UnpicklableArtifactException(name)
                else:
                    try:
                        blob = pickle.dumps(obj, protocol=2)
                        encode_type = "gzip+pickle-v2"
                    except (SystemError, OverflowError):
                        encode_type = "gzip+pickle-v4"
                        if encode_type not in self._encodings:
                            raise DataException(
                                f"Artifact {name} is very large (over 2GB). "
                                + "You need to use Python 3.4 or newer if you want to "
                                + "serialize large objects."
                            )
                        try:
                            blob = pickle.dumps(obj, protocol=4)
                        except TypeError:
                            raise UnpicklableArtifactException(name)
                    except TypeError:
                        raise UnpicklableArtifactException(name)

                self._info[name] = {
                    "size": len(blob),
                    "type": str(type(obj)),
                    "encoding": encode_type,
                }
                artifact_names.append(name)
                yield blob

        # Use the content-addressed store to store all artifacts
        save_result = self._ca_store.save_blobs(pickle_iter(), len_hint=len_hint)
        for name, result in zip(artifact_names, save_result):
            self._objects[name] = result.key


class FlowDataStore(FlowDataStore):
    """A custom FlowDataStore class for storing flow data in Metaflow."""

    def __init__(
        self,
        flow_name,
        environment,
        metadata=None,
        event_logger=None,
        monitor=None,
        storage_impl=None,
        ds_root=None,
    ):
        """Initializes the FlowDataStore with the provided parameters.

        Args:
            flow_name (str): The name of the flow.
            environment (MetaflowEnvironment): The Metaflow environment.
            metadata (MetadataProvider, optional): The metadata provider.
                Defaults to None.
            event_logger (EventLogger, optional): The event logger. Defaults
                to None.
            monitor (Monitor, optional): The monitor. Defaults to None.
            storage_impl (DataStore, optional): The storage implementation.
                Defaults to None.
            ds_root (str, optional): The root of the datastore. Defaults to
                None.
        """

        super().__init__(
            flow_name,
            environment,
            metadata,
            event_logger,
            monitor,
            storage_impl,
            ds_root,
        )

    def get_task_datastore(
        self,
        run_id,
        step_name,
        task_id,
        attempt=None,
        data_metadata=None,
        mode="r",
        allow_not_done=False,
    ):
        """Returns a TaskDataStore for the specified task.

        Args:
            run_id (str): The run id.
            step_name (str): The step name.
            task_id (str): The task id.
            attempt (int, optional): The attempt number. Defaults to None.
            data_metadata (DataMetadata, optional): The data metadata.
                Defaults to None.
            mode (str, optional): The mode (read 'r' or write 'w'). Defaults
                to 'r'.
            allow_not_done (bool, optional): A flag indicating whether to
                allow tasks that are not done. Defaults to False.

        Returns:
            TaskDataStore: A TaskDataStore for the specified task.
        """
        return TaskDataStore(
            self,
            run_id,
            step_name,
            task_id,
            attempt=attempt,
            data_metadata=data_metadata,
            mode=mode,
            allow_not_done=allow_not_done,
        )


class MetaflowInterface:
    """A wrapper class for Metaflow's tooling, modified to work with the
    workflow interface."""

    def __init__(self, flow: Type[FLSpec], backend: str = "ray"):
        """Wrapper class for the metaflow tooling modified to work with the
        workflow interface. Keeps track of the current flow run, tasks, and
        data artifacts.

        Args:
            flow (Type[FLSpec]): The current flow that will be serialized /
                tracked using metaflow tooling.
            backend (str, optional): The backend selected by the runtime.
                Permitted selections are 'ray' and 'single_process'. Defaults
                to 'ray'.
        """
        self.backend = backend
        self.flow_name = flow.__name__
        if backend == "ray":
            self.counter = Counter.remote()
        else:
            self.counter = 0

    def create_run(self) -> int:
        """Creates a run for the current flow using metaflow internal
        functions.

        Args:
            None

        Returns:
            run_id [int]
        """
        flow = Flow(self.flow_name)
        env = MetaflowEnvironment(self.flow_name)
        env.get_environment_info()
        self.local_metadata = LocalMetadataProvider(env, flow, None, None)
        self.run_id = self.local_metadata.new_run_id()
        self.flow_datastore = FlowDataStore(
            self.flow_name,
            env,
            metadata=self.local_metadata,
            storage_impl=DATASTORES["local"],
            ds_root=f"{Path.home()}/.metaflow",
        )
        return self.run_id

    def create_task(self, task_name: str) -> int:
        """Creates a task for the current run. The generated task_id is unique
        for each task and can be recalled later with the metaflow client.

        Args:
            task_name (str): The name of the new task.

        Returns:
            task_id [int]
        """
        with SystemMutex("critical_section"):
            if self.backend == "ray":
                task_id = ray.get(self.counter.get_counter.remote())
                self.local_metadata._task_id_seq = task_id
                self.local_metadata.new_task_id(self.run_id, task_name)
                return ray.get(self.counter.increment.remote())
            else:
                # Keeping single_process in critical_section
                # because gRPC calls may cause problems.
                task_id = self.counter
                self.local_metadata._task_id_seq = task_id
                self.local_metadata.new_task_id(self.run_id, task_name)
                self.counter += 1
                return self.counter

    def save_artifacts(
        self,
        data_pairs: Generator[str, Any],
        task_name: str,
        task_id: int,
        buffer_out: Type[StringIO],
        buffer_err: Type[StringIO],
    ) -> None:
        """Use metaflow task datastore to save flow attributes, stdout, and
        stderr for a specific task (identified by the task_name + task_id).

        Args:
            data_pairs (Generator[str, Any]): Generator that returns the name
                of the attribute, and it's corresponding object.
            task_name (str): The name of the task for which an artifact is
                being saved.
            task_id (int): A unique id (within the flow) that will be used to
                recover these data artifacts by the metaflow client.
            buffer_out (StringIO): StringIO buffer containing stdout.
            buffer_err (StringIO): StringIO buffer containing stderr.
        """
        task_datastore = self.flow_datastore.get_task_datastore(
            self.run_id, task_name, str(task_id), attempt=0, mode="w"
        )
        task_datastore.init_task()
        task_datastore.save_artifacts(data_pairs)

        # Register metadata for task
        retry_count = 0
        metadata_tags = [f"attempt_id:{retry_count}"]
        self.local_metadata.register_metadata(
            self.run_id,
            task_name,
            str(task_id),
            [
                MetaDatum(
                    field="attempt",
                    value=str(retry_count),
                    type="attempt",
                    tags=metadata_tags,
                ),
                MetaDatum(
                    field="origin-run-id",
                    value=str(0),
                    type="origin-run-id",
                    tags=metadata_tags,
                ),
                MetaDatum(
                    field="ds-type",
                    value=self.flow_datastore.TYPE,
                    type="ds-type",
                    tags=metadata_tags,
                ),
                MetaDatum(
                    field="ds-root",
                    value=self.flow_datastore.datastore_root,
                    type="ds-root",
                    tags=metadata_tags,
                ),
            ],
        )

        self.emit_log(buffer_out, buffer_err, task_datastore)

        task_datastore.done()

    def load_artifacts(self, artifact_names, task_name, task_id):
        """Loads flow attributes from Metaflow's task datastore.

        Args:
            artifact_names (list): The names of the artifacts to load.
            task_name (str): The name of the task from which to load artifacts.
            task_id (int): The id of the task from which to load artifacts.

        Returns:
            dict: A dictionary of loaded artifacts.
        """
        task_datastore = self.flow_datastore.get_task_datastore(
            self.run_id, task_name, str(task_id), attempt=0, mode="r"
        )
        return task_datastore.load_artifacts(artifact_names)

    def emit_log(
        self,
        msgbuffer_out: Type[StringIO],
        msgbuffer_err: Type[StringIO],
        task_datastore: Type[TaskDataStore],
        system_msg: bool = False,
    ) -> None:
        """Writes stdout and stderr to Metaflow's TaskDatastore.

        Args:
            msgbuffer_out (StringIO): A StringIO buffer containing stdout.
            msgbuffer_err (StringIO): A StringIO buffer containing stderr.
            task_datastore (TaskDataStore): A Metaflow TaskDataStore instance.
            system_msg (bool, optional): A flag indicating whether the message
                is a system message. Defaults to False.
        """
        stdout_buffer = TruncatedBuffer("stdout", MAX_LOG_SIZE)
        stderr_buffer = TruncatedBuffer("stderr", MAX_LOG_SIZE)

        for std_output in msgbuffer_out.readlines():
            timestamp = datetime.utcnow()
            stdout_buffer.write(mflog_msg(std_output, now=timestamp), system_msg=system_msg)

        for std_error in msgbuffer_err.readlines():
            timestamp = datetime.utcnow()
            stderr_buffer.write(mflog_msg(std_error, now=timestamp), system_msg=system_msg)

        task_datastore.save_logs(
            RUNTIME_LOG_SOURCE,
            {
                "stdout": stdout_buffer.get_buffer(),
                "stderr": stderr_buffer.get_buffer(),
            },
        )


class DefaultCard(DefaultCard):
    """A custom DefaultCard class for Metaflow.

    Attributes:
        ALLOW_USER_COMPONENTS (bool): A flag indicating whether user
            components are allowed. Defaults to True.
        type (str): The type of the card. Defaults to "default".
    """

    ALLOW_USER_COMPONENTS = True

    type = "default"

    def __init__(self, options={"only_repr": True}, components=[], graph=None):
        """Initializes the DefaultCard with the provided options, components,
        and graph.

        Args:
            options (dict, optional): A dictionary of options. Defaults to
                {"only_repr": True}.
            components (list, optional): A list of components. Defaults to an
                empty list.
            graph (any, optional): The graph to use. Defaults to None.
        """
        self._only_repr = True
        self._graph = None if graph is None else transform_flow_graph(graph)
        if "only_repr" in options:
            self._only_repr = options["only_repr"]
        self._components = components

    # modified Defaultcard render function
    def render(self, task):
        """Renders the card with the provided task.

        Args:
            task (any): The task to render the card with.

        Returns:
            any: The rendered card.
        """
        # :param: task instead of metaflow.client.Task object task.pathspec
        # (string) is provided # NOQA
        RENDER_TEMPLATE = read_file(RENDER_TEMPLATE_PATH)  # NOQA: N806
        JS_DATA = read_file(JS_PATH)  # NOQA: N806
        CSS_DATA = read_file(CSS_PATH)  # NOQA: N806
        final_component_dict = dict(self._graph)
        final_component_dict = TaskInfoComponent(
            task,
            only_repr=self._only_repr,
            graph=self._graph,
            components=self._components,
        ).render()
        pt = self._get_mustache()
        data_dict = {
            "task_data": base64.b64encode(json.dumps(final_component_dict).encode("utf-8")).decode(
                "utf-8"
            ),
            "javascript": JS_DATA,
            "title": task,
            "css": CSS_DATA,
            "card_data_id": uuid.uuid4(),
        }
        return pt.render(RENDER_TEMPLATE, data_dict)


class TaskInfoComponent(TaskInfoComponent):
    """A custom TaskInfoComponent class for Metaflow.

    Properties:
        page_content (list): A list of MetaflowCardComponents going as task
            info.
        final_component (dict): The dictionary returned by the `render`
            function of this class.
    """

    def __init__(
        self,
        task,
        page_title="Task Info",
        only_repr=True,
        graph=None,
        components=[],
    ):
        """Initializes the TaskInfoComponent with the provided task, page
        title, representation flag, graph, and components.

        Args:
            task (any): The task to use.
            page_title (str, optional): The title of the page. Defaults to
                "Task Info".
            only_repr (bool, optional): A flag indicating whether to only use
                the representation. Defaults to True.
            graph (any, optional): The graph to use. Defaults to None.
            components (list, optional): A list of components. Defaults to an
                empty list.
        """
        self._task = task
        self._only_repr = only_repr
        self._graph = graph
        self._components = components
        self._page_title = page_title
        self.final_component = None
        self.page_component = None

    # modified TaskInfoComponent render function
    def render(self):
        """Renders the component and returns a dictionary of metadata and
        components.

        Returns:
            final_component_dict (dict): A dictionary of the form:
                dict(metadata={}, components=[]).
        """
        final_component_dict = {
            "metadata": {
                "metaflow_version": mf_version,
                "version": 1,
                "template": "defaultCardTemplate",
            },
            "components": [],
        }

        dag_component = SectionComponent(
            title="DAG", contents=[DagComponent(data=self._graph).render()]
        ).render()

        page_contents = []
        page_contents.append(dag_component)
        page_component = PageComponent(
            title=self._page_title,
            contents=page_contents,
        ).render()
        final_component_dict["components"].append(page_component)

        self.final_component = final_component_dict
        self.page_component = page_component

        return final_component_dict
