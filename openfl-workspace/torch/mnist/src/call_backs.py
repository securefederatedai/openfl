from tictoc import bench_dict
from openfl.callbacks.callback import Callback


class my_callback(Callback):
    def on_round_begin(round_number):
        print("called callback")
        bench_dict["global"].gstep()

    def on_round_end(round_number):
        print("called callbackcalled callbackcalled callbackcalled callbackcalled callback")
        bench_dict["global"].gstop()

    def set_params(self, params):
        self.params = params
        print(self.params)
        