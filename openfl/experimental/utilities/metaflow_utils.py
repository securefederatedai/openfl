from metaflow.metaflow_environment import MetaflowEnvironment
from metaflow.plugins import LocalMetadataProvider
from metaflow.datastore import FlowDataStore, TaskDataStore, DATASTORES
from metaflow.datastore.task_datastore import only_if_not_done,require_mode
import multiprocessing
import cloudpickle
import ray

class Flow:
    def __init__(self,name):
        """Mock flow for metaflow internals"""
        self.name = name

@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value

class MetaflowInterface:

    def __init__(self,flow_name):
        self.flow_name = flow_name
        self.counter = Counter.remote()

    def create_run(self):
        flow = Flow(self.flow_name)
        env = MetaflowEnvironment(self.flow_name)
        env.get_environment_info()
        self.local_metadata = LocalMetadataProvider(env,flow,None,None)
        self.run_id = self.local_metadata.new_run_id()
        self.flow_datastore = ModifiedFlowDataStore(self.flow_name, env, metadata=self.local_metadata, storage_impl=DATASTORES['local'])
        return self.run_id

    def create_task(self, task_name):
        # May need a lock here
        task_id = ray.get(self.counter.get_counter.remote())
        self.local_metadata._task_id_seq = task_id
        self.local_metadata.new_task_id(self.run_id, task_name)
        return ray.get(self.counter.increment.remote())

    def save_artifacts(self,data_pairs, task_name, task_id):
        """Use metaflow task datastore to save federated flow attributes"""
        task_datastore = self.flow_datastore.get_task_datastore(self.run_id, task_name, str(task_id), attempt=0, mode='w')
        task_datastore.init_task()
        task_datastore.save_artifacts(data_pairs)
        task_datastore.done()
    
    def load_artifacts(self, artifact_names, task_name, task_id):
        """Use metaflow task datastore to load flow attributes"""
        task_datastore = self.flow_datastore.get_task_datastore(self.run_id, task_name, str(task_id), attempt=0, mode='r')
        return task_datastore.load_artifacts(artifact_names)

class ModifiedFlowDataStore(FlowDataStore):
    #def __init__(
    #    self,
    #    flow_name,
    #    environment,
    #    metadata=None,
    #    event_logger=None,
    #    monitor=None,
    #    storage_impl=None,
    #    ds_root=None,
    #):
    #    super().__init__(        
    #            flow_name,
    #            environment,
    #            metadata,
    #            event_logger,
    #            monitor,
    #            storage_impl,
    #            ds_root
    #            )



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

        return ModifiedTaskDataStore(
            self,
            run_id,
            step_name,
            task_id,
            attempt=attempt,
            data_metadata=data_metadata,
            mode=mode,
            allow_not_done=allow_not_done,
        )


class ModifiedTaskDataStore(TaskDataStore):

    #def __init__(
    #    self,
    #    flow_datastore,
    #    run_id,
    #    step_name,
    #    task_id,
    #    attempt=None,
    #    data_metadata=None,
    #    mode="r",
    #    allow_not_done=False,
    #):
    #    super().__init__(
    #            flow_datastore,
    #            run_id,
    #            step_name,
    #            task_id,
    #            attempt,
    #            data_metadata,
    #            mode,
    #            allow_not_done
    #            )


    #@only_if_not_done
    #@require_mode("w")
    def save_artifacts(self, artifacts_iter, force_v4=False, len_hint=0):
        """
        Saves Metaflow Artifacts (Python objects) to the datastore and stores
        any relevant metadata needed to retrieve them.

        Typically, objects are pickled but the datastore may perform any
        operation that it deems necessary. You should only access artifacts
        using load_artifacts

        This method requires mode 'w'.

        Parameters
        ----------
        artifacts : Iterator[(string, object)]
            Iterator over the human-readable name of the object to save
            and the object itself
        force_v4 : boolean or Dict[string -> boolean]
            Indicates whether the artifact should be pickled using the v4
            version of pickle. If a single boolean, applies to all artifacts.
            If a dictionary, applies to the object named only. Defaults to False
            if not present or not specified
        len_hint: integer
            Estimated number of items in artifacts_iter
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
                            "Artifact *%s* requires a serialization encoding that "
                            "requires Python 3.4 or newer." % name
                        )
                    try:
                        blob = cloudpickle.dumps(obj, protocol=4)
                    except TypeError as e:
                        raise UnpicklableArtifactException(name)
                else:
                    try:
                        blob = cloudpickle.dumps(obj, protocol=2)
                        #blob = pickle.dumps(obj, protocol=2)
                        encode_type = "gzip+pickle-v2"
                    except (SystemError, OverflowError):
                        encode_type = "gzip+pickle-v4"
                        if encode_type not in self._encodings:
                            raise DataException(
                                "Artifact *%s* is very large (over 2GB). "
                                "You need to use Python 3.4 or newer if you want to "
                                "serialize large objects." % name
                            )
                        try:
                            blob = pickle.dumps(obj, protocol=4)
                        except TypeError as e:
                            raise UnpicklableArtifactException(name)
                    except TypeError as e:
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

