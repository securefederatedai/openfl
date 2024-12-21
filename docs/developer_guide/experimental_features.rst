.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _experimental_features:

*********************
Experimental Features
*********************

This section includes a set of experimental features that our team wants feedback on before adding them into core OpenFL. 
Experimental features are *not* ready for production. These features are under active development and intended to make their way into core OpenFL, but there are several key considerations to make when building on top of these:

1. *Backward compatibility is not guaranteed* - Our goal is to maintain backward compatibility whenever possible, but user feedback (and our own internal research)
   may result in necessary changes to the APIs.

**Workflow Interface**

    Learn how to:
        - Chain a series of tasks that run on aggregator or collaborator.
        - Filter out information that should stay local
        - Use Metaflow tools to analyze and debug experiments  

    - :doc:`../about/features_index/workflowinterface`

.. toctree::
   :maxdepth: 4
   :hidden:

   ../about/features_index/workflowinterface
