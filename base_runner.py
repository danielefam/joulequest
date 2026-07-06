# base_runner.py
import time
from abc import ABC, abstractmethod


class InferenceRunner(ABC):
    """Abstract base class for inference runners."""

    def __init__(self, model_path,device=None):
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def _load_model(self):
        """Load the model from file."""
        pass

    @abstractmethod
    def run_inference(self, input_data):
        """Run inference on input data and return output."""
        pass

    # def benchmark(self, input_data, count=5, repeat=10):
    #     """Run inference multiple times for performance testing."""
    #     for r in range(repeat):
    #         for _ in range(count):
    #             self.run_inference(input_data)
    #         time.sleep()
    #         # print(f"[Repeat {r+1}/{repeat}] Total time: {end - start:.4f}s")
