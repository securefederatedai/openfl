.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _director_workflow:

************
Director-based workflow
************

Establishing a long-living Federation with Director
#######################################

#. Install OpenFL 
==================

Please, refer to :ref:`_install_software_root`

#. Implement Shard Descriptors
==================

OpenFL framework provides a ‘Shard descriptor’ interface that should be described on every collaborator node 
to provide a unified data interface for FL experiments. Abstract “Shard descriptor” should be subclassed and 
all its methods should be implemented to describe the way data samples and labels will be loaded from disk 
during training. Shard descriptor is a subscriptable object that implements `__getitem__()` and `len()` methods 
as well as several additional methods to access ‘sample shape’, ‘target shape’, and ‘shard description’ text 
that may be used to identify participants during experiment definition and execution.

Then the data owners need to implement `Shard Descriptors` Python classes. 

#. Start Director
==================

#. Start Envoys
==================

Describing an FL experimnet using Interactive Python API
#######################################

another story