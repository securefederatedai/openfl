def snykData = [
    'openfl-docker': 'openfl-docker/Dockerfile.base',
    'openfl': 'setup.py',
    'openfl-workspace_tf_2dunet': 'openfl-workspace/tf_2dunet/requirements.txt',
    'openfl-workspace_torch_cnn_mnist_straggler_check': 'openfl-workspace/torch_cnn_mnist_straggler_check/requirements.txt',
    // CN-14619 snyk test CLI does not support -f in requirements.txt file
    // 'openfl-workspace_torch_cnn_histology': 'openfl-workspace/torch_cnn_histology/requirements.txt',
    'openfl-workspace_torch_cnn_histology_src': 'openfl-workspace/torch_cnn_histology/src/requirements.txt',
    'openfl-workspace_keras_nlp': 'openfl-workspace/keras_nlp/requirements.txt',
    'openfl-workspace_torch_cnn_mnist': 'openfl-workspace/torch_cnn_mnist/requirements.txt',
    'openfl-workspace_torch_unet_kvasir': 'openfl-workspace/torch_unet_kvasir/requirements.txt',
    'openfl-workspace_tf_cnn_histology': 'openfl-workspace/tf_cnn_histology/requirements.txt',
    'openfl-workspace_tf_3dunet_brats': 'openfl-workspace/tf_3dunet_brats/requirements.txt',
    'openfl-workspace_keras_cnn_with_compression': 'openfl-workspace/keras_cnn_with_compression/requirements.txt',
    'openfl-workspace_keras_cnn_mnist': 'openfl-workspace/keras_cnn_mnist/requirements.txt',
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
    options {
        disableConcurrentBuilds()
    }
    triggers {
        cron('TZ=US/Arizona\nH 20 * * 0-4')
    }
    stages {
        stage('Code Scan Pipeline') {
            when {
                anyOf {
                    allOf {
                        expression { env.GIT_BRANCH == 'develop' }
                        triggeredBy 'TimerTrigger'
                    }
                    triggeredBy 'UserIdCause'
                    triggeredBy 'BranchIndexingCause'
                }
            }
            stages {
                stage('Build Docker Images') {
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
                        SCANNERS = 'snyk,checkmarx,protex,bandit,virus'

                        SNYK_ALLOW_LONG_PROJECT_NAME = true
                        SNYK_USE_MULTI_PROC = true
                        SNYK_DEBUG = true
                        SNYK_PYTHON_VERSION = '3.9'

                        BANDIT_SOURCE_PATH = 'openfl/ openfl-workspace/ openfl-tutorials/'
                        BANDIT_SEVERITY_LEVEL = 'high'

                        VIRUS_SCAN_DIR = '.'

                        PUBLISH_TO_ARTIFACTORY = false
                    }
                    steps {
                        rbheStaticCodeScan()
                    }
                }
            }
        }
        stage('Publish PyPi') {
            // only publish pypi package when these conditions are true:
            //   * commit is a release pypi publish commit
            //   * branch is a jenkins release branch
            // NOTE: ensure version in setup.py is updated accordingly
            when {
                allOf {
                    expression { env.GIT_BRANCH ==~ /(?i)(jenkins-v\d+.\d+)/ }
                    expression { common.isPyPiPublishCommit() }
                    not {
                        triggeredBy 'TimerTrigger'
                    }
                }
            }
            stages {
                stage('Build Package') {
                    agent {
                        docker {
                            image 'python:3.9'
                        }
                    }
                    steps {
                        sh 'scripts/build_wheel.sh'
                    }
                }
                stage('Publish Package') {
                    steps {
                        pypiPublish()
                    }
                }
            }
        }
        stage('Publish Docker Hub') {
            // only publish docker image when these conditions are true:
            //   * commit is a release docker publish commit
            //   * the docker image:tag is not already published
            //   * branch is a jenkins release branch
            // NOTE: ensure image tag is updated accordingly
            when {
                allOf {
                    expression { env.GIT_BRANCH ==~ /(?i)(jenkins-v\d+.\d+)/ }
                    expression { isDockerPublishCommit() }
                    expression { ! isDockerImagePublished('intel/openfl:1.5') }
                    not {
                        triggeredBy 'TimerTrigger'
                    }
                }
            }
            steps {
                rbheDocker(
                    image: 'intel/openfl',
                    dockerfile: 'openfl-docker/Dockerfile.base',
                    latest: true,
                    tags: ['1.5'],
                    pushOn: /^(?:jenkins-v\d+.\d+)$/,
                    scan: false,
                    registry: [[
                        url: 'docker.io',
                        credentialId: rbhe.credentials.intelDockerHub
                    ]]
                )
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

def isDockerPublishCommit() {
    def publish = common.isMatchingCommit(/(?s)^release\(.*docker.*\):.*publish image.*$/)
    echo "[isDockerPublishCommit]: publish Docker image: ${publish}"
    publish
}

def isDockerImagePublished(dockerImage) {
    // temporary method until this feature is made available within the rbheDocker function
    def status
    docker.withRegistry('https://docker.io', rbhe.credentials.intelDockerHub) {
        status = sh(script:"docker pull ${dockerImage}", returnStatus: true)
    }
    def published = status == 0
    echo "[isDockerImagePublished]: ${dockerImage} already published: ${published}"
    published
}
