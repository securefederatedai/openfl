.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _troubleshooting:

|productName| Troubleshooting
#############################

The following is a list of commonly reported issues. If you don't see your issue reported here, please submit a `Github issue
<https://github.com/intel/openfl/issues>`_ or contact us directly on `Slack <https://join.slack.com/t/openfl/shared_invite/zt-ovzbohvn-T5fApk05~YS_iZhjJ5yaTw>`_.

1. :code:`Cannot import name TensorFlowDataLoader from openfl.federated`

   |productName| currently uses conditional imports to attempt to be framework agnostic. If your task runner is derived from `KerasTaskRunner` or `TensorflowTaskRunner`, this error could come up if Tensorflow was not installed in your collaborator's virtual environment. If running on multi-node experiment, we recommend using the :code:`fx workspace export` and :code:`fx workspace import` commands, as this will ensure consistent modules between aggregator and collaborators.

2. **None of the collaborators can connect to my aggregator node**

   There are a few reasons that this can happen, but the most common is the aggregator node's FQDN (Fully qualified domain name) was incorrectly specified in the plan. By default, :code:`fx plan initialize` will attempt to resolve the FQDN for you (this should look something like :code:`hostname.domain.com`), but this can sometimes parse an incorrect domain name. If you face this issue, look at :code:`agg_addr` in :code:`plan/plan.yaml` and verify that you can access this address externally. If the address is externally accessible and you are running |productName| in an enterprise environment, verify that the aggregator's listening port is not blocked. In such cases, :code:`agg_port` should be manually specified in the plan and then redistributed to all participants. 

3. **After starting the collaborator, I see the error** :code:`Handshake failed with fatal error SSL_ERROR_SSL`

   This error likely results from a bad certificate presented by the collaborator. Steps for regenerating the collaborator certificate can be found :ref:`here <install_certs_colab>`.

4. **I am seeing some other error while running the experiment. Is there more verbose logging available so I can investigate this on my own?**

   Yes! You can turn on verbose logging with :code:`fx -l DEBUG collaborator start` or :code:`fx -l DEBUG aggregator start`. This will give verbose information related to gRPC, bidirectional tensor transfer, and compression related information.  
