vname=meta-env
rm -rf ~/workspace/$vname
python3.8 -m venv ~/workspace/$vname

source ~/workspace/$vname/bin/activate
pip install -U pip --no-cache
pip install -r reqs.txt
cd ../..
pip install .
