# base_runner.py
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class BurstResult:
    requested_inferences: int
    executed_inferences: int
    elapsed_seconds: float

    @property
    def latency_seconds(self):
        """Return average latency per inference in a burst."""
        if self.executed_inferences <= 0:
            raise ValueError("executed_inferences must be positive")
        return self.elapsed_seconds / self.executed_inferences # (burst exec time) / (number of exec inferences)


class InferenceRunner(ABC):
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def _load_model(self):
        """Load the model from file."""
        raise NotImplementedError

    @abstractmethod
    def generate_input(self):
        """Create and store the input for the next inference burst."""
        raise NotImplementedError

    @abstractmethod
    def run_inference(self, inferences_per_cycle):
        """Execute exactly ``inferences_per_cycle`` forward passes."""
        raise NotImplementedError

    @property
    def input_batch_size(self):
        """Return the number of input samples processed by one forward pass."""
        return 1

    def randomize_parameters(self):
        """Create a new parameter state before the next burst."""

    def prepare(self):
        """Perform one-time setup before the first burst."""

    def prepare_burst(self):
        """Regenerate parameters and input before one timed burst.

        This method is not included in the elapsed time.
        Every burst the input and the parameters of the model are different
        """
        self.randomize_parameters()
        self.generate_input()

    def synchronize(self):
        """Wait for asynchronous backend work. In the case of cpu execution (raspberry pi) this
        method is empty. Working with cuda (jetson boards) this method will use torch.cuda.synchronize
        """

    @staticmethod
    def _validate_inference_count(inference_count):
        if not isinstance(inference_count, int) or inference_count <= 0:
            raise ValueError("inference_count must be a positive integer")

    def run_burst(self, inference_count):
        self._validate_inference_count(inference_count)
        self.prepare_burst() # the preparation is NOT included in elapsed_seconds
        return self.run_prepared_burst(inference_count)

    def run_prepared_burst(self, inference_count):
        self._validate_inference_count(inference_count)
        self.synchronize()
        start = time.perf_counter()
        self.run_inference(inferences_per_cycle=inference_count)
        self.synchronize()
        elapsed_seconds = time.perf_counter() - start

        if elapsed_seconds <= 0:
            raise RuntimeError("The measured burst duration must be positive")

        return BurstResult(
            requested_inferences=inference_count,
            executed_inferences=inference_count,
            elapsed_seconds=elapsed_seconds,
        )

    def close(self):
        """Release backend resources when a campaign ends."""
        return None
