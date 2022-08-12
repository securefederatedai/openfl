def snykData = [
    'openfl-docker': 'openfl-docker/Dockerfile.base',
    'openfl': 'setup.py',
    'openfl-workspace_tf_2dunet': 'openfl-workspace/tf_2dunet/requirements.txt',
    'openfl-workspace_torch_cnn_mnist_straggler_check': 'openfl-workspace/torch_cnn_mnist_straggler_check/requirements.txt',
    // CN-14619 snyk test CLI does not support -f in requirements.txt file
    // 'openfl-workspace_torch_cnn_histology': 'openfl-workspace/torch_cnn_histology/requirements.txt',
    'openfl-workspace_torch_cnn_histology_src': 'openfl-workspace/torch_cnn_histology/src/requirements.txt',
    'openfl-workspace_keras_nlp': 'openfl-workspace/keras_nlp/requirements.txt',
    'openfl-workspace_fe_tf_adversarial_cifar': 'openfl-workspace/fe_tf_adversarial_cifar/requirements.txt',
    'openfl-workspace_torch_cnn_mnist': 'openfl-workspace/torch_cnn_mnist/requirements.txt',
    'openfl-workspace_keras_nlp_gramine_ready': 'openfl-workspace/keras_nlp_gramine_ready/requirements.txt',
    'openfl-workspace_torch_unet_kvasir': 'openfl-workspace/torch_unet_kvasir/requirements.txt',
    'openfl-workspace_fe_torch_adversarial_cifar': 'openfl-workspace/fe_torch_adversarial_cifar/requirements.txt',
    'openfl-workspace_tf_cnn_histology': 'openfl-workspace/tf_cnn_histology/requirements.txt',
    'openfl-workspace_torch_cnn_histology_gramine_ready': 'openfl-workspace/torch_cnn_histology_gramine_ready/requirements.txt',
    'openfl-workspace_tf_3dunet_brats': 'openfl-workspace/tf_3dunet_brats/requirements.txt',
    'openfl-workspace_keras_cnn_with_compression': 'openfl-workspace/keras_cnn_with_compression/requirements.txt',
    'openfl-workspace_keras_cnn_mnist': 'openfl-workspace/keras_cnn_mnist/requirements.txt',
    'openfl-workspace_torch_unet_kvasir_gramine_ready': 'openfl-workspace/torch_unet_kvasir_gramine_ready/requirements.txt',
    'openfl-tutorials_interactive_api_pytorch_medmnist_2d_envoy': 'openfl-tutorials/interactive_api/PyTorch_MedMNIST_2D/envoy/requirements.txt',
    'openfl-tutorials_interactive_api_pytorch_dogscats_vit_workspace': 'openfl-tutorials/interactive_api/PyTorch_DogsCats_ViT/workspace/requirements.txt',
    'openfl-tutorials_interactive_api_pytorch_histology_envoy': 'openfl-tutorials/interactive_api/PyTorch_Histology/envoy/requirements.txt',
    'openfl-tutorials_interactive_api_mxnet_landmarks_workspace': 'openfl-tutorials/interactive_api/MXNet_landmarks/workspace/requirements.txt',
    'openfl-tutorials_interactive_api_pytorch_histology_fedcurv_envoy': 'openfl-tutorials/interactive_api/PyTorch_Histology_FedCurv/envoy/requirements.txt',
    'openfl-tutorials_interactive_api_tensorflow_word_prediction_workspace': 'openfl-tutorials/interactive_api/Tensorflow_Word_Prediction/workspace/requirements.txt',
    'openfl-tutorials_interactive_api_jax_linear_regression_envoy': 'openfl-tutorials/interactive_api/jax_linear_regression/envoy/requirements.txt',
    'openfl-tutorials_interactive_api_pytorch_medmnist_3d_envoy': 'openfl-tutorials/interactive_api/PyTorch_MedMNIST_3D/envoy/requirements.txt',
    'openfl-tutorials_interactive_api_numpy_linear_regression_workspace': 'openfl-tutorials/interactive_api/numpy_linear_regression/workspace/requirements.txt',
    'openfl-tutorials_interactive_api_numpy_linear_regression_envoy': 'openfl-tutorials/interactive_api/numpy_linear_regression/envoy/requirements.txt',
    'openfl-tutorials_interactive_api_pytorch_market_re-id_workspace': 'openfl-tutorials/interactive_api/PyTorch_Market_Re-ID/workspace/requirements.txt',
    'openfl-tutorials_interactive_api_pytorch_market_re-id_envoy': 'openfl-tutorials/interactive_api/PyTorch_Market_Re-ID/envoy/requirements.txt',
    'openfl-tutorials_interactive_api_pytorch_tinyimagenet_workspace': 'openfl-tutorials/interactive_api/PyTorch_TinyImageNet/workspace/requirements.txt',
    'openfl-tutorials_interactive_api_pytorch_tinyimagenet_envoy': 'openfl-tutorials/interactive_api/PyTorch_TinyImageNet/envoy/requirements.txt'
]
pipeline {
    agent { 
        label 'rbhe'
    }
    stages {
        stage('Build') {
            environment {
                DOCKER_BUILD_ARGS = '--build-arg http_proxy --build-arg https_proxy --no-cache'
            }
            steps {
                sh '''
                    docker image build ${DOCKER_BUILD_ARGS} -t openfl-docker:latest . -f openfl-docker/Dockerfile.base
                    docker images | { grep openfl || true; }
                '''
            }
        }
        stage('Prep Code Scan') {
            steps {
                script {
                    // prep environment variables for Snyk Scan
                    env.SNYK_PROJECT_NAME = snykData.collect { project, manifest -> project }.join(',')
                    env.SNYK_MANIFEST_FILE = snykData.collect { project, manifest -> manifest }.join(',')
                    env.SNYK_DOCKER_IMAGE = snykData.collect { project, manifest -> manifest.contains('Dockerfile') ? project : '' }.join(',')
                    sh 'env | { grep SNYK || true; }'
                }
            }
        }
        stage('Code Scan') {
            environment {
                PROJECT_NAME = 'OpenFL-Main'
                SCANNERS = 'snyk,checkmarx,protex,bandit'

                SNYK_ALLOW_LONG_PROJECT_NAME = true
                SNYK_USE_MULTI_PROC = true
                SNYK_DEBUG = true
                SNYK_PYTHON_VERSION = '3.8'

                BANDIT_SOURCE_PATH = 'openfl/ openfl-workspace/ openfl-tutorials/'

                PUBLISH_TO_ARTIFACTORY = false
            }
            steps {
                rbheStaticCodeScan()
            }
        }
    }
    post {
        always {
            publishArtifacts()
            cleanWs()
        }
    }
}