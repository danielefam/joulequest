# run_manager.py
import time
import argparse
import numpy as np
import importlib.util
import gc


class RunManager:
    def __init__(self, runner_cls, model_path, nb_run, sleep_time, backend):
        #self.runner = runner_cls(model_path)
        self.runner = runner_cls
        self.nb_run = nb_run
        self.sleep_time = sleep_time
        self.model_path = model_path
        self.backend = backend

    def execute(self):
        #input_data = np.random.rand(1, 2).astype(np.float32)
        runner = self.runner(self.model_path,self.backend)
        runner.generate_input()

        for i in range(self.nb_run):
            print(f"Run {i+1}/{self.nb_run}")
            
            runner.run_inference(count=100000, repeat=1)
            # del runner
            # gc.collect()
            if i < self.nb_run - 1:  # Don't sleep after the last run
                time.sleep(self.sleep_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tpu", "cpu","cuda"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--nb_run", type=int, default=5)
    parser.add_argument("--sleep_time", type=int, default=10)
    args = parser.parse_args()

    if args.backend == "tpu":
        from runner import TFLiteTPURunner
        runner_cls = TFLiteTPURunner 
    elif args.backend == "cpu" or args.backend == "cuda":
        from runner_old import TorchRunner
        runner_cls = TorchRunner
    else:
        raise ValueError("Selected backend is not available or supported.")
    manager = RunManager(runner_cls, args.model, args.nb_run, args.sleep_time,args.backend)
    manager.execute()


if __name__ == "__main__":
    main()
