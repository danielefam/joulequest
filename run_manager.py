# run_manager.py
import time
import argparse
import numpy as np
import importlib.util
import gc
import torch


class RunManager:
    def __init__(self, runner_cls, model_path, number_of_cycles, sleep_time, backend, inferences_per_cycle):
        #self.runner = runner_cls(model_path)
        self.runner = runner_cls
        self.number_of_cycles = number_of_cycles
        self.sleep_time = sleep_time
        self.model_path = model_path
        self.backend = backend
        self.inferences_per_cycle = inferences_per_cycle

    def execute(self):
        def execute_cycle(self):
            if hasattr(runner, "randomize_parameters"):
                runner.randomize_parameters()
            if hasattr(runner, "generate_input"):
                runner.generate_input()
            runner.run_inference(inferences_per_cycle=self.inferences_per_cycle)
        
        #input_data = np.random.rand(1, 2).astype(np.float32)
        runner = self.runner(self.model_path,self.backend)

        # discard the first
        execute_cycle(self)

        # cooling down
        time.sleep(self.sleep_time)
        
        # cycle inference time estimation
        if hasattr(runner, "randomize_parameters"):
            runner.randomize_parameters()
        if hasattr(runner, "generate_input"):
            runner.generate_input()
        runner.measure_cycle_inference_time(inferences_per_cycle=self.inferences_per_cycle)
        print(f"cycle inference time= "
                f"{runner.measure_cycle_inference_time(self.inferences_per_cycle)}ms")

        for i in range(self.number_of_cycles):
            print(f"Run {i+1}/{self.number_of_cycles}")
            execute_cycle(self)
            time.sleep(self.sleep_time)

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tpu", "cpu","cuda"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--number_of_cycles", type=int, default=5)
    parser.add_argument("--sleep_time", type=int, default=10)
    parser.add_argument("--inferences_per_cycle", type=int, default=110)
    args = parser.parse_args()

    if args.backend == "tpu":
        from runner import TFLiteTPURunner
        runner_cls = TFLiteTPURunner 
    elif args.backend == "cpu" or args.backend == "cuda":
        from runner import TorchRunner
        runner_cls = TorchRunner
    else:
        raise ValueError("Selected backend is not available or supported.")
    manager = RunManager(runner_cls, args.model, args.number_of_cycles, args.sleep_time,args.backend, args.inferences_per_cycle)
    manager.execute()


if __name__ == "__main__":
    main()
