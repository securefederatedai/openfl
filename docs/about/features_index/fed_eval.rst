.. # Copyright (C) 2020-2024 Intel Corporation
.. # SPDX-License-Identifier: Apache-2.0

Federated Evaluation
=======================================

Introduction to Federated Evaluation
-------------------------------------

Model evaluation is an essential part of the machine learning development cycle. In a traditional centralized learning system, all evaluation data is collected on a localized server. Because of this, centralized evaluation of machine learning models is a fairly straightforward task. However, in a federated learning system, data is distributed across multiple decentralized devices or nodes. In an effort to preserve the security and privacy of the distributed data, it is infeasible to simply aggregate all the data into a centralized system. Federated evaluation offers a solution by assessing the model at the client side and aggregating the accuracy without ever having to share the data. This is crucial for ensuring the model's effectiveness and reliability in diverse and real-world environments while respecting privacy and data locality

OpenFL's Support for Federated Evaluation
-------------------------------------------------

OpenFL, a flexible framework for Federated Learning, has the capability to perform federated evaluation by modifying the federation plan. In this document, we will show how OpenFL can facilitate this process through its task runner API (aggregator-based workflow), where the model evaluation is distributed across various collaborators before being sent to the aggregator. For the task runner API, this involves minor modifications to the ``plan.yaml`` file, which defines the workflow and tasks for the federation. In particular, the federation plan should be defined to run for one forward pass and perform only aggregated model validation

In general pipeline is as follows:

1. **Setup**: Initialize the federation with the modified ``plan.yaml`` set to run for one round and only perform aggregated model validation
2. **Execution**: Run the federation. The model is distributed across collaborators for evaluation.
3. **Evaluation**: Each collaborator evaluates the model on its local data.
4. **Aggregation**: The aggregator collects and aggregates these metrics to assess overall model performance.

Example Using the Task Runner API (Aggregator-based Workflow)
--------------------------------------------------------------

The following steps can be leveraged to achieve practical e2e usage of FedEval

*N.B*: We will be using torch_cnn_mnist plan itself for both training and with some minor changes for evaluation as well

*Prerequisites*: Please ensure that OpenFL version==1.7 is installed or you can also choose to install latest from source.

With OpenFL version==1.7 aggregator start command is enhanced to have an optional argument '--task_group' which, as the help suggest, will select the provided task_groups task to assigner for execution in the collaborator(s), since this defaults to 'learning'

.. code-block:: shell

    Usage: fx aggregator start [OPTIONS]

    Start the aggregator service.

    Args:     plan (str): Path to plan config file     authorized_cols (str): Path to authorized collaborators file
    task_group (str): Selected task-group for assignement - defaults to 'learning'

    Options:
    -p, --plan PATH             Federated learning plan [plan/plan.yaml]
    -c, --authorized_cols PATH  Authorized collaborator list [plan/cols.yaml]
    --task_group TEXT           Selected task-group for assignment - defaults to learning
    --help                      Show this message and exit.

1. **Setup**
We will use the `torch_cnn_mnist` workspace for training

Let's first configure a workspace with all necesary certificates

.. code-block:: shell

    fx workspace create --prefix ./cnn_train_eval --template torch_cnn_mnist
    cd cnn_train_eval
    fx workspace certify
    fx aggregator generate-cert-request
    fx aggregator certify --silent

Succesful run of this will show in console both the FL plan details and certificates generations

.. code-block:: shell

            INFO     Parsing Federated Learning Plan : SUCCESS :                                                    
        
                        settings:                                                                                    

                            best_state_path: save/best.pbuf                                                            

                            db_store_rounds: 2                                                                         

                            init_state_path: save/init.pbuf                                                            

                            last_state_path: save/last.pbuf                                                            

                            rounds_to_train: 2                                                                         

                            write_logs: false                                                                          

                        template: openfl.component.aggregator.Aggregator                                             

                        assigner:                                                                                      

                        settings:                                                                                    

                            task_groups:                                                                               

                            - name: learning                                                                           

                            percentage: 1.0                                                                          

                            tasks:                                                                                   

                            - aggregated_model_validation                                                            

                            - train                                                                                  

                            - locally_tuned_model_validation                                                         

                        template: openfl.component.RandomGroupedAssigner                                             

                        collaborator:                                                                                  

                        settings:                                                                                    
                                    
                            db_store_rounds: 1                                                                         

                            delta_updates: false                                                                       

                            opt_treatment: RESET                                                                       

                        template: openfl.component.collaborator.Collaborator                                         

                        compression_pipeline:                                                                          

                        settings: {}                                                                                 

                        template: openfl.pipelines.NoCompressionPipeline                                             

                        data_loader:                                                                                   

                        settings:                                                                                    

                            batch_size: 64                                                                             

                            collaborator_count: 2                                                                      

                        template: src.dataloader.PyTorchMNISTInMemory                                                

                        network:                                                                                       

                        settings:                                                                                    

                            agg_addr: devvm###.com                                                

                            agg_port: 55529                                                                            

                            cert_folder: cert                                                                          

                            client_reconnect_interval: 5                                                               

                            hash_salt: auto                                                                            

                            require_client_auth: true                                                                  

                            use_tls: true                                                                              

                        template: openfl.federation.Network                                                          

                        task_runner:                                                                                   

                        settings: {}                                                                                 

                        template: src.taskrunner.TemplateTaskRunner                                                  

                        tasks:                                                                                         

                        aggregated_model_validation:                                                                 

                            function: validate_task                                                                    

                            kwargs:                                                                                    

                            apply: global                                                                            

                            metrics:                                                                                 

                            - acc                                                                                    

                        locally_tuned_model_validation:                                                              

                            function: validate_task                                                                    

                            kwargs:                                                                                    

                            apply: local                                                                             

                            metrics:                                                                                 

                            - acc                                                                                    

                        settings: {}                                                                                 

                        train:                                                                                       

                            function: train_task                                                                       

                            kwargs:                                                                                    

                            epochs: 1                                                                                

                            metrics:                                                                                 

                            - loss                                                                                                                                                                                                 
    New workspace directory structure:
    cnn_train_eval
    ├── requirements.txt
    ├── .workspace
    ├── logs
    ├── data
    ├── cert
    ├── README.md
    ├── src
    │   ├── __init__.py
    │   ├── taskrunner.py
    │   ├── cnn_model.py
    │   └── dataloader.py
    ├── plan
    │   ├── cols.yaml
    │   ├── plan.yaml
    │   ├── data.yaml
    │   └── defaults
    └── save

    6 directories, 11 files

    ✔️ OK
    Setting Up Certificate Authority...

    Done.

    ✔️ OK
    Creating AGGREGATOR certificate key pair with following settings: CN=devvm###.com, SAN=DNS:devvm###.com
    
    ✔️ OK
    The CSR Hash for file server/agg_devvm###.com.csr = 3affa56ce391a084961c5f1ba634f223536173665daa6191e705e13557f36d58c844133758f804d1f85d93bfc113fd7b
    
    Signing AGGREGATOR certificate

    ✔️ OK

2. Initialize the plan

.. code-block:: shell

    cd ~/src/clean/openfl/cnn_train_eval
    fx plan initialize >~/plan.log 2>&1 &
    tail -f ~/plan.log

This should initialize the plan with random initial weights in ``init.pbuf``

.. code-block:: shell

            WARNING  Following parameters omitted from global initial model, local initialization will determine values: []                           plan.py:186
            INFO     Creating Initial Weights File    🠆 save/init.pbuf
                                    plan.py:196
    ✔️ OK

3. Next run the 'learning' federation with two collaborators

.. code-block:: shell

    ## Create two collaborators
    cd ~/src/clean/openfl/cnn_train_eval
    fx collaborator create -n collaborator1 -d 1
    fx collaborator generate-cert-request -n collaborator1
    fx collaborator certify -n collaborator1 --silent
    fx collaborator create -n collaborator2 -d 2
    fx collaborator generate-cert-request -n collaborator2
    fx collaborator certify -n collaborator2 --silent

    ## start the fedeval federation
    fx aggregator start > ~/fx_aggregator.log 2>&1 &
    fx collaborator start -n collaborator1 > ~/collab1.log 2>&1 &
    fx collaborator start -n collaborator2 > ~/collab2.log 2>&1 &
    cd ~
    tail -f plan.log fx_aggregator.log collab1.log collab2.log

This script will run two collaborator and start the aggregator with default `--task_group` 'learning'

The same is defined in the assigner section of the plan which comes from the defaults itself

.. code-block:: yaml

    assigner:                                                                                      

        settings:                                                                                    

            task_groups:                                                                               

            - name: learning                                                                           

            percentage: 1.0                                                                          

            tasks:                                                                                   

            - aggregated_model_validation                                                            

            - train                                                                                  

            - locally_tuned_model_validation 

This will run the 2 rounds of training across both the collaborators

.. code-block:: shell

    ==> fx_aggregator.log <==
            INFO     Sending tasks to collaborator collaborator2 for round 0                                        
                                aggregator.py:409

    ==> collab2.log <==
            INFO     Received Tasks: [name: "aggregated_model_validation"                                           
                            collaborator.py:184
                        , name: "train"                                                                                

                        , name: "locally_tuned_model_validation"                                                       

                        ]                                                                                              

Post the end of learning federation we can note what is the best model accuracy reported and save the ``best.pbuf`` file for next step - evaluation

.. code-block:: shell

        ==> fx_aggregator.log <==
    [06:09:27] INFO     Collaborator collaborator1 is sending task results for train, round 1                          
            
    [06:09:28] INFO     Collaborator collaborator1 is sending task results for locally_tuned_model_validation, round 1                             aggregator.py:629
            INFO     Round 1: Collaborators that have completed all tasks: ['collaborator2', 'collaborator1']                                  aggregator.py:1049
            INFO     Round 1: saved the best model with score 0.960096                                              
            
            INFO     Saving round 1 model...                                                                        

            INFO     Experiment Completed. Cleaning up...                                                           

In this case we can confirm that post the 2 rounds of training the model reported an accuracy of 0.960096

.. code-block:: shell

    Round 1: saved the best model with score 0.960096                                              
                                aggregator.py:955

Let's save this model (``best.pbuf``) for later usage

.. code-block:: shell

    cp cnn_train_eval/save/best.pbuf ~/trained_model.pbuf
    devuser@devvm:~/src/clean/openfl$ 

Now let's create another workspace using the same plan and steps as mentioned in learning Setup:

Post this we will do plan initialize and we shall replace the ``init.pbuf`` with the previously saved ``best.pbuf`` and then re-adjust the plan 
to use "evaluation" defaults.

Once all the pieces are in place we then run the aggregator in evaluation mode by supplying the `--task_group` as "evaluation" validating the 
accuracy of the previously trained model

The updated plan post initialization with edits to make it ready for evaluation will be as follows:

.. code-block:: yaml

    aggregator:
    settings:
        best_state_path: save/best.pbuf
        db_store_rounds: 2
        init_state_path: save/init.pbuf
        last_state_path: save/last.pbuf
        rounds_to_train: 1
        write_logs: false
    template: openfl.component.aggregator.Aggregator
    assigner:
    settings:
        task_groups:
        - name: evaluation
        percentage: 1.0
        tasks:
        - aggregated_model_validation
    template: openfl.component.RandomGroupedAssigner
    collaborator:
    settings:
        db_store_rounds: 1
        delta_updates: false
        opt_treatment: RESET
    template: openfl.component.collaborator.Collaborator
    compression_pipeline:
    settings: {}
    template: openfl.pipelines.NoCompressionPipeline
    data_loader:
    settings:
        batch_size: 64
        collaborator_count: 2
    template: src.dataloader.PyTorchMNISTInMemory
    network:
    settings:
        agg_addr: devvm###.com
        agg_port: 55529
        cert_folder: cert
        client_reconnect_interval: 5
        hash_salt: auto
        require_client_auth: true
        use_tls: true
    template: openfl.federation.Network
    task_runner:
    settings: {}
    template: src.taskrunner.TemplateTaskRunner
    tasks:
    aggregated_model_validation:
        function: validate_task
        kwargs:
        apply: global
        metrics:
        - acc
    locally_tuned_model_validation:
        function: validate_task
        kwargs:
        apply: local
        metrics:
        - acc
    settings: {}
    train:
        function: train_task
        kwargs:
        epochs: 1
        metrics:
        - loss

We have done following changes to the initialized torch_cnn_mnist plan in the new workspace:
 - Set the rounds_to_train to 1 as evaluation needs just one round of federation run across the collaborators
 - Removed all other training related tasks from assigner settings except "aggregated_model_validation"
Now let's replace the ``init.pbuf`` with the previously saved ``trained_model.pbuf``

.. code-block:: shell

    ll cnn_eval/save/init.pbuf 
    -rw------- 1 devuser devuser 1722958 Jan 14 09:44 cnn_eval/save/init.pbuf
    (venv) devuser@devvm:~/src/clean/openfl$ cp ~/trained_model.pbuf cnn_eval/save/init.pbuf 
    (venv) devuser@devvm:~/src/clean/openfl$ ll cnn_eval/save/init.pbuf
    -rw------- 1 devuser devuser 1722974 Jan 14 09:52 cnn_eval/save/init.pbuf
    (venv) devuser@devvm:~/src/clean/openfl$ 

Notice the size changes in the ``init.pbuf`` as its replaced by the trained model we saved from the training run of the federation

Now finally let's run the federation and this time we will launch the aggregator with overriding the default value of `--task_group` to "evaluation"

.. code-block:: shell

    ## Create two collaborators
    cd ~/src/clean/openfl/cnn_eval
    fx collaborator create -n collaborator1 -d 1
    fx collaborator generate-cert-request -n collaborator1
    fx collaborator certify -n collaborator1 --silent
    fx collaborator create -n collaborator2 -d 2
    fx collaborator generate-cert-request -n collaborator2
    fx collaborator certify -n collaborator2 --silent

    ## start the fedeval federation
    fx aggregator start --task_group evaluation > ~/fx_aggregator.log 2>&1 &
    fx collaborator start -n collaborator1 > ~/collab1.log 2>&1 &
    fx collaborator start -n collaborator2 > ~/collab2.log 2>&1 &
    cd ~
    tail -f plan.log fx_aggregator.log collab1.log collab2.log

Notice the only change in fedration run steps from previous training round is the additional argument `--task_group` to aggregator start

Now since the aggregators' task_group is set to "evaluation" it will skip the `round_number_check` and use the init model supplied just for evaluation

.. code-block:: shell

        INFO     Setting aggregator to assign: evaluation task_group                                            
                            aggregator.py:101
        INFO     🧿 Starting the Aggregator Service.                                                            
                            aggregator.py:103
       
        INFO     Skipping round_number check for evaluation task_group                                          
                            aggregator.py:215
        INFO     Starting Aggregator gRPC Server                                                                

In each collaborator logs we can see that the assigned task is only the evaluation task

.. code-block:: shell

    => collab1.log <==
            INFO     Waiting for tasks...                                                                           
                            collaborator.py:234
            INFO     Received Tasks: [name: "aggregated_model_validation"                                           
                            collaborator.py:184
                        ]                                     
    ==> collab2.log <==
            INFO     Waiting for tasks...                                                                           
                            collaborator.py:234
            INFO     Received Tasks: [name: "aggregated_model_validation"                                           
                            collaborator.py:184
                        ]

And post the federation run, since its only evaluation run, we get from the collaborator the accuracy of the init model which, as per successful 
evaluation, is same as previously trained best models' accuracy, in our case that was 0.960096

.. code-block:: shell

    ==> fx_aggregator.log <==
    [10:00:15] INFO     Collaborator collaborator2 is sending task results for aggregated_model_validation, round 0                                aggregator.py:629
            INFO     Round 0: Collaborators that have completed all tasks: ['collaborator2']                        
                            aggregator.py:1049
            INFO     Collaborator collaborator1 is sending task results for aggregated_model_validation, round 0                                aggregator.py:629
            INFO     Round 0: Collaborators that have completed all tasks: ['collaborator2', 'collaborator1']                                  aggregator.py:1049
            INFO     Round 0: saved the best model with score 0.960096                                              
                                aggregator.py:955
            INFO     Saving round 0 model...                                                                        
                                aggregator.py:994
            INFO     Experiment Completed. Cleaning up...                                                           
                            aggregator.py:1005
            INFO     Sending signal to collaborator collaborator1 to shutdown...                                    
                                aggregator.py:356

---

Congratulations, you have successfully performed federated evaluation across two decentralized collaborator nodes using the same plan with minor evaluation-related changes leveraging a previously trained OpenFL model protobuf as input.