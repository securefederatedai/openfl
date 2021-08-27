
workspace "OpenFL" "An open framework for Federated Learning." {
    model {
        group "Control" {
            user = person "Data scientist" "A person or group of people using OpenFL"
            shardOwner = person "Collaborator manager" "Data owner's representative controlling Envoy"
            centralManager = person "Director manager" 
            governor = softwareSystem "Governor" "CCF-based system for corporate clients"
        }
        openfl = softwareSystem "OpenFL" "An open framework for Federated Learning" {
            apiLayer = container "Python API component" "A set of tools to setup register FL Experiments" {
                federationInterface = component "Federaion Interface"
                experimentInterface = component "Experiment Interface"
                # TaskInterface = component ""
            }

            group "Central node" {
                director = container "Director" "A long-living entity that can spawn aggregators"
                aggregator = container "Aggregator" "Model server and collaborator orchestrator"{
                    assigner = component "Task Assigner" "Decides the policy for which collaborators should run FL tasks"
                    grpcServer = component "gRPC Server"
                }
            }
            group "Collaborator node" {
                envoy = container "Envoy" "A long-living entity that can adapt a local data set and spawn collaborators" {
                    shardDescriptor = component "Shard Descriptor" "Data manager's interface aimed to unify data access" {
                        tags "Interface"
                    }
                }
                collaborator = container "Collaborator" "Actor executing tasks on local data inside one experiment" {
                    pluginManager = component "Plugin Manager"
                    taskRunner = component "Task Runner"
                    tensorDB = component "Tensor Data Base"
                    tensorCodec = component "TensorCodec"
                    grpcClient = component "gRPC Client"
                    frameworkAdapter = component "Framework Adapter"
                }
            }
        }
        config = element "Config file"

        # relationships between people and software systems
        user -> openfl "Controls Fedarations. Provides FL plans, tasks, models, data"
        governor -> openfl "Controls Fedarations"

        # relationships to/from containers
        user -> apiLayer "Provides FL Plans, Tasks, Models, DataLoaders"
        shardOwner -> envoy "Launches. Provides local dataset ShardDescriptors"
        centralManager -> director "Launches. Sets up global Federation settings"
        apiLayer -> director "Registers FL experiments"
        director -> apiLayer "Sends information about the Federation. Returns training artifacts."
        director -> aggregator "Creates an instance to maintain an FL experiment"
        envoy -> collaborator "Creates an instance to maintain an FL experiment"
        envoy -> director "Communicates dataset info, Sends status updates"
        director -> envoy "Approves, Sends FL experiments"
        aggregator -> collaborator "Sends tasks and initial tensors"
        collaborator -> aggregator "Sends locally tuned tensors and training metrics"


        # relationships to/from components
        envoy -> taskRunner "Provides tasks' defenitions"
        grpcClient -> taskRunner "Invokes some tasks for the round"
        aggregator -> grpcClient "Communicates"
    }

    views
        theme default

        systemcontext openfl "SystemContext" {
            include *
            autoLayout
            
        }

        container openfl "Containers" {
            include *
            # include config
            # autoLayout
        }

        component collaborator "Collaborator" {
            include *
            autoLayout
        }

        component apiLayer "API" {
            include *
            autoLayout
        }

        component envoy "Envoy" {
            include *
            autoLayout
        }

}

