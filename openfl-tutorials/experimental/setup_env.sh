conda remove -n speed --all -y
conda create -n speed python=3.8 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
source /home/oamontoy/miniconda3/bin/activate speed 
conda install -c nvidia cuda-compiler -y
pip install -U pip setuptools
pip install -r reqs.txt
pip install ../../.
echo conda activate speed
