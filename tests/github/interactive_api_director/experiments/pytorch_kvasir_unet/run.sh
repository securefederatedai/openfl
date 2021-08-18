cd director
bash start_director.sh &

sleep 3
cd ../envoy
python kvasir_shard_descriptor.py &

sleep 2
cd ../../../../../..
python -m tests.github.interactive_api_director.experiments.pytorch_kvasir_unet.experiment
