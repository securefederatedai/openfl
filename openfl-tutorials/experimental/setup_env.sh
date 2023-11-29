
pip install -U pip --no-cache
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install -r requirments_horovod.txt --no-cache
HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MPI=1 pip install horovod[pytorch] --no-cache


