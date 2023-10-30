conda remove -n speed --all -y
conda create -n speed pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c nvidia cuda-compiler
source /home/oamontoy/miniconda3/bin/activate speed 
pip install -U pip
#pip install -r reqs.txt
#cd ../..
#pip install .
