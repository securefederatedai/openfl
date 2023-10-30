conda remove -n speed --all -y
conda create -n speed pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
source /home/oamontoy/miniconda3/bin/activate speed 
conda install -c nvidia cuda-compiler
pip install -U pip
pip install -r reqs.txt
pip install ../../.
