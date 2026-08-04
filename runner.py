# runner.py
import platform
import numpy as np

from base_runner import InferenceRunner
import os


import importlib
from importlib.util import find_spec


# Define only the runners supported by the active target environment.
_tf_available = find_spec("tflite_runtime") is not None
_torch_available = find_spec("torch") is not None


if _tf_available:
    tflite = importlib.import_module("tflite_runtime.interpreter")
    EDGETPU_SHARED_LIB = {
        'Linux': 'libedgetpu.so.1',
        'Darwin': 'libedgetpu.1.dylib',
        'Windows': 'edgetpu.dll'
    }[platform.system()]

    class TFLiteTPURunner(InferenceRunner):
        """Inference runner for a compiled Edge TPU TFLite model.

        The TPU path is retained for compatibility, although current campaign
        development and validation prioritize the PyTorch CPU/CUDA path.
        """

        def __init__(self, model_path, device="tpu"):
            super().__init__(model_path, device=device)
            self._load_model()

        def _load_model(self):
            self.interpreter = tflite.Interpreter(
                model_path=self.model_path,
                experimental_delegates=[
                    tflite.load_delegate(EDGETPU_SHARED_LIB)
                ]
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

        def generate_input(self):
            """Generate and store the input for the next burst."""
            details = self.input_details[0]
            input_data = np.random.random_sample(tuple(details['shape'])).astype(np.float32)
            scale, zero_point = details['quantization']

            if scale > 0 and np.issubdtype(details['dtype'], np.integer):
                limits = np.iinfo(details['dtype'])
                input_data = np.clip(
                    np.rint(input_data / scale + zero_point),
                    limits.min,
                    limits.max,
                ).astype(details['dtype'])
            else:
                input_data = input_data.astype(details['dtype'])

            self.interpreter.set_tensor(details['index'], input_data)

        def run_inference(self, inferences_per_cycle):
            for _ in range(inferences_per_cycle):
                self.interpreter.invoke()

if _torch_available:
    torch = importlib.import_module("torch")
    nn = torch.nn

    class TorchRunner(InferenceRunner):
        """PyTorch runner"""

        def __init__(self, model_path, device="cpu", from_state_dict=False):
            super().__init__(model_path, device=device)
            self.device = torch.device(device)
            self.from_state_dict = from_state_dict
            self._load_model()

        def _load_model(self):
            # Layer dimensions currently come from the established file naming convention.
            # A typed layer manifest will replace this later.
            
            self.params = self._extract_layer_info()
            self.model = self._build_model()

            if self.from_state_dict:
                state_dict = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=True,
                )
                self.model.load_state_dict(state_dict)

            self.model.eval()
            print(f"Torch model prepared from specification: {self.model_path}")

        def randomize_parameters(self):
            self.model.apply(
                lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
            )
            self.model.eval()

        def run_inference(self, inferences_per_cycle):
            with torch.inference_mode():
                for _ in range(inferences_per_cycle):
                    self.model(self.input_data)

        def synchronize(self):
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

        def _extract_layer_info(self):
            """Extract and validate the current Linear/Conv filename schema."""
            filename = os.path.splitext(os.path.basename(self.model_path))[0]
            parts = filename.split("_")
            layer_type = parts[0].lower()
            try:
                params = [int(p) for p in parts[1:]]
            except ValueError as error:
                raise ValueError(
                    f"Invalid numeric layer parameters in model name: {filename}"
                ) from error

            if layer_type == "linear":
                if len(params) != 2:
                    raise ValueError(
                        "Linear model names must be Linear_<in_features>_<out_features>"
                    )
                return  {
                        "type": "linear",
                        "in_features": params[0],
                        "out_features": params[1],
                        }

            elif layer_type == "conv":
                if len(params) != 4:
                    raise ValueError(
                        "Conv model names must be "
                        "Conv_<in_channels>_<image_size>_<kernel_size>_<padding>"
                    )
                return  {
                        "type": "conv",
                        "in_channels": params[0],
                        "image_size": params[1],
                        "kernel_size": params[2],
                        "padding": params[3],
                        }

            elif layer_type == "lenet":
                return {
                    "type": "lenet"
                }

            raise ValueError(f"Unsupported layer type in model name: {layer_type}")

        def _build_model(self):
            """Build the layer represented by the validated model name."""
            if self.params["type"] == "linear":
                return torch.nn.Linear(
                    self.params["in_features"],
                    self.params["out_features"],
                ).to(self.device)

            if self.params["type"] == "conv":
                return torch.nn.Conv2d(
                    self.params["in_channels"],
                    1,
                    kernel_size=self.params["kernel_size"],
                    padding=self.params["padding"],
                ).to(self.device)

            if self.params["type"] == 'lenet':
                return nn.Sequential(
                        nn.Conv2d(1,8,5),
                        nn.MaxPool2d(2),
                        nn.ReLU(),
                        nn.Conv2d(8,32,5),
                        nn.MaxPool2d(2),
                        nn.AdaptiveMaxPool2d(4),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(512,128),
                        nn.ReLU(),
                        nn.Linear(128,64)
                    ).to(self.device)

            raise ValueError(f"Unsupported layer type: {self.params['type']}")

        def generate_input(self):
            """Generate a random input for the next burst."""
            with torch.inference_mode():
                if self.params["type"] == "linear":
                    shape = (
                            1,
                            self.params["in_features"],
                            )

                elif self.params["type"] == "conv":
                    shape = (
                            1,
                            self.params["in_channels"],
                            self.params["image_size"],
                            self.params["image_size"],
                            )
                elif self.params["type"] == "lenet":
                    shape = (1,1,32,32)

                else:
                    raise ValueError(f"Unsupported layer type: {self.params['type']}")
                
                self.input_data = torch.randn(
                                                *shape,
                                                dtype=torch.float32,
                                                device=self.device
                                            )
                