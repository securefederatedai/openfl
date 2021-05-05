.. # Copyright (C) 2021 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _compression_settings:

***************
Compression Settings
***************

Federated Learning can enable tens to thousands of participants to work together on the same model, but with this scaling comes increased communication cost. Furthermore, large models exacerbate this problem. For this reason we make compression is a core capability of |productName|, and our framework supports several lossless and lossy compression pipelines out of the box. In general, the weights of a model are typically not robust to information loss, so no compression is applied by default to the model weights sent bidirectionally; however, the deltas between the model weights for each round are inherently more sparse and better suited for lossy compression. The following is the list of compression pipelines that |productName| currently supports:

* ``NoCompressionPipeline``: This is the default option applied to model weights
* ``RandomShiftPipeline``: A **lossless** pipeline that randomly shifts the weights during transport
* ``STCPipeline``: A **lossy** pipeline consisting of three transformations: *Sparsity Transform* (p_sparsity=0.1), which by default retains only the (p*100)% absolute values of greatest magnitude. *Ternary Transform*, which discretizes the sparse array into three buckets, and finally a *GZIP Transform*. 
* ``SKCPipeline``: A **lossy** pipeline consisting of three transformations: *Sparsity Transform* (p=0.1), which by default retains only the(p*100)% absolute values of greatest magnitude. *KMeans Transform* (k=6), which applies the KMeans algorithm to the sparse array with *k* centroids, and finally a *GZIP Transform*. 
* ``KCPipeline``: A **lossy** pipeline consisting of two transformations: *KMeans Transform* (k=6), which applies the KMeans algorithm to the original weight array with *k* centroids, and finally a *GZIP Transform*. 

We provide an example template, **keras_cnn_with_compression**, that utilizes the *KCPipeline* with 6 centroids for KMeans. To gain a better understanding of how experiments perform with greater or fewer centroids, you can modify the *n_clusters* parameter in the template's plan.yaml:

    .. code-block:: console
    
       compression_pipeline :
         defaults : plan/defaults/compression_pipeline.yaml
         template : openfl.pipelines.KCPipeline
         settings :
           n_clusters : 6

