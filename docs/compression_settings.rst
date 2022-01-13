.. # Copyright (C) 2020-2021 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

.. _compression_settings:

**************************
Apply Compression Settings
**************************

The Open Federated Learning (|productName|) framework supports lossless and lossy compression pipelines. Federated learning enables a large number of participants to work together on the same model. Without a compression pipeline, this scalability results in increased communication cost. Furthermore, large models exacerbate this problem.

.. note::
    In general, the weights of a model are typically not robust to information loss, so no compression is applied by default to the model weights sent bidirectionally; however, the deltas between the model weights for each round are inherently more sparse and better suited for lossy compression.

The following are the compression pipelines supported in |productName|:

``NoCompressionPipeline``
    The default option applied to model weights

``RandomShiftPipeline``
    A **lossless** pipeline that randomly shifts the weights during transport
    
``STCPipeline``
    A **lossy** pipeline consisting of three transformations: 
    
        - *Sparsity Transform* (p_sparsity=0.1), which by default retains only the (p*100)% absolute values of greatest magnitude. 
        - *Ternary Transform*, which discretizes the sparse array into three buckets
        - *GZIP Transform*

``SKCPipeline``
    A **lossy** pipeline consisting of three transformations:
    
        - *Sparsity Transform* (p=0.1), which by default retains only the(p*100)% absolute values of greatest magnitude. 
        - *KMeans Transform* (k=6), which applies the KMeans algorithm to the sparse array with *k* centroids
        - *GZIP Transform*
        
``KCPipeline``
    A **lossy** pipeline consisting of two transformations: 
    
        - *KMeans Transform* (k=6), which applies the KMeans algorithm to the original weight array with *k* centroids
        - *GZIP Transform* 


Demonstration of a Compression Pipeline
=======================================

The example template, **keras_cnn_with_compression**, uses the ``KCPipeline`` with six centroids for KMeans. To gain a better understanding of how experiments perform with greater or fewer centroids, you can modify the **n_clusters** parameter in the template **plan.yaml**:

    .. code-block:: console
    
       compression_pipeline :
         defaults : plan/defaults/compression_pipeline.yaml
         template : openfl.pipelines.KCPipeline
         settings :
           n_clusters : 6

