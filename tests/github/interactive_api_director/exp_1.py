from time import sleep
import getpass


from tests.github.interactive_api_director.experiment_runner import run_federation
from tests.github.interactive_api_director.experiment_runner import stop_federation
from tests.github.interactive_api_director.experiment_runner import Shard
from tests.github.interactive_api_director.experiment_runner import create_federation


col_names = ['one', 'two']
username = getpass.getuser()
director_path = f'/home/{username}/test/exp_1/director'

director_host = 'localhost'
director_port = 50051

shards = {
    f'/home/{username}/test/exp_1/{col_name}':
        Shard(
            shard_name=col_name,
            director_host=director_host,
            director_port=director_port,
            data_path=f'/home/{username}/test/data/{col_name}'
        )
    for col_name in col_names
}

create_federation(director_path, shards.keys())

processes = run_federation(shards, director_path)

# run experiments
sleep(5)

stop_federation(processes)
