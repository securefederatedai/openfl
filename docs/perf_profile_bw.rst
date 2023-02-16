.. # Copyright (C) 2020-2023 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

**************************
Insights on |productName| Network bandwidth throttling performance
**************************


1: Overview
===================

|productName| Profiling to understand the bandwidth limitations & 
factors (network size, # of collaborators, memory utilizations, etc.) that affects the performance of the components in federation.


2: Experiment Setup
============================================

1. TC Command - Queuing Discipline
   
   .. code-block:: console
      
        # BW Values - 32, 64, 96, 128, 256, ...
        BW=$1
        
        # Delete the existing `dev` profile attached to `lo` adapter, if any.
        tc qdisc del dev lo root

        # interface throttle
        tc qdisc add dev lo root handle 1: htb default 12
        tc class add dev lo parent 1:1 classid 1:12 htb rate "$BW"kbps ceil "$BW"kbps

2. Experiment steps & dependencies

   In the config.yml, define the number of envoys and BW values to experiment.

   The `perf_profile.py` triggers the async backround thread from the `ThreadPool` to monitor and capture the CPU and memory utilization of all the director and envoy process.
   
   With the parameters defined in `config.yaml`, For a given Interactive API example, a director and a series of envoys are spawned with the throttled bandwidth values.
   
   The execution time and the detailed per second resource utilization of the `fx` process is captured in the file.

   .. code-block:: console

        $ cat config.yaml

        num_envoys: [2, 3, 4, 5, 6]
        bw_values: [32, 64, 96, 128, 256, 512, 1024]
        example_root_dir: 'repo/t_openfl/openfl-tutorials/interactive_api/Tensorflow_CIFAR_tfdata'
        profile_dir: 'repo/t_openfl/openfl-tutorials/interactive_api/profiling'


3: Usage
=========================================

Install dependencies required to run profiling script

   .. code-block:: console

      cd profiling
      pip install -r requirements.txt

Add `set_qdisc.sh` to the sudoers list with NOPASSWD Flag set for ALL or specific user (sudo required to throttle bandwidth of the given network interface at a kernel level).

   .. code-block:: console

      ALL ALL=(ALL) NOPASSWD: <ABSOLUTE_PATH>/set_qdisc.sh

Add below step in the `start_director.sh` and `start_envoy.sh` to log the process STDOUT and STDERR to `*.txt` file

   .. code-block:: console
      
      # edit director/start_director.sh and add below line

      # $1 -> Total Number of Envoys (Just a file naming convention)
      # $2 -> BW value
      exec > ../output/director_te_$1_qd_$2.txt 2>&1


      # edit envoy/start_envoy.sh and add below line

      # $1 -> Name of the Envoy (Just a file naming convention)
      exec > ../output/log_$1.txt 2>&1


Run `mem_profile.py` to initiate the experiment.

   .. code-block:: console

      cd profiling
      python mem_profile.py


4: Result
=========================================

Significant increase in the execution time as the BW value is throttled down to 32kbps.
Total number of spawned envoys linearly affects the execution time.
Minimum bandwidth of 512kbps is reasonable to run the federation beyond which the total execution time remains almost similar.
 

Number of federation round: 5

   .. image:: images/bw_experiment_values.png
      :alt: BW Throttled experiment values.

   .. image:: images/bw_vs_exec_time.png
      :alt: Execution time vs BW throttled values.
      

