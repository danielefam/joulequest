from abc import ABC, abstractmethod

class BaseModelBuilder(ABC):
    """Abstract base class for building TPU models."""
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
    
    @abstractmethod
    def build_and_compile(self):
        """Builds and compiles the model."""
        pass