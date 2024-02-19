
pip install -U pip --no-cache
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
export HOROVOD_WITH_PYTORCH=1 
export HOROVOD_WITHOUT_MPI=1
pip install -r openfl-workspace/torch_llm_horovod/requirements.txt --no-cache


