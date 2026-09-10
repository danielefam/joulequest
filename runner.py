# runner.py
import platform
import numpy as np

from base_runner import InferenceRunner
import os
from dataclasses import dataclass


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

        def __init__(self, model_path, device="tpu", batch_size=1):
            super().__init__(model_path, device=device)
            if isinstance(batch_size, bool) or not isinstance(batch_size, int):
                raise ValueError("batch_size must be a positive integer")
            if batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")
            self.batch_size = batch_size
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
            input_shape = self.input_details[0]["shape"].copy()
            if input_shape[0] != self.batch_size:
                input_shape[0] = self.batch_size
                self.interpreter.resize_tensor_input(
                    self.input_details[0]["index"], input_shape, strict=False
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

        @property
        def input_batch_size(self):
            return self.batch_size

if _torch_available:
    torch = importlib.import_module("torch")
    nn = torch.nn
    from layers.attention import RotarySelfAttention, SelfAttention
    from layers.resnet import (
        PrunedOrinResNet18,
        PrunedPi5ResNet18,
        PrunedResNet18,
        ResNet18,
        ResNet50,
        ResNetResidualAdd,
    )

    @dataclass(frozen=True)
    class TorchLayerDefinition:
        parse: object
        build: object
        input_shape: object

    def _require_parameter_count(filename, params, expected, notation):
        if len(params) not in expected:
            raise ValueError(f"{filename} names must be {notation}")

    def _parse_linear(params, filename):
        _require_parameter_count(
            "Linear",
            params,
            {2},
            "Linear_<in_features>_<out_features>",
        )
        return {
            "type": "linear",
            "in_features": params[0],
            "out_features": params[1],
        }

    def _parse_conv(params, filename):
        _require_parameter_count(
            "Conv",
            params,
            {4, 5},
            "Conv_<in_channels>_<image_size>_<kernel_size>_<padding>"
            "[_<out_channels>]",
        )
        return {
            "type": "conv",
            "in_channels": params[0],
            "image_size": params[1],
            "kernel_size": params[2],
            "padding": params[3],
            "out_channels": params[4] if len(params) == 5 else 1,
        }

    def _parse_lenet(params, filename):
        return {"type": "lenet"}

    def _parse_maxpool(params, filename):
        _require_parameter_count(
            "MaxPool",
            params,
            {3},
            "MaxPool_<in_channels>_<image_size>_<kernel_size>",
        )
        return {
            "type": "maxpool",
            "in_channels": params[0],
            "image_size": params[1],
            "kernel_size": params[2],
        }

    def _parse_adapool(params, filename):
        _require_parameter_count(
            "AdaPool",
            params,
            {3},
            "AdaPool_<in_channels>_<image_size>_<output_size>",
        )
        return {
            "type": "adapool",
            "in_channels": params[0],
            "image_size": params[1],
            "output_size": params[2],
        }

    def _parse_attention(params, filename):
        _require_parameter_count(
            "Attention",
            params,
            {3},
            "Attention_<input_token>_<embed_dim>_<num_heads>",
        )
        return {
            "type": "attention",
            "input_token": params[0],
            "embed_dim": params[1],
            "num_heads": params[2],
        }

    def _parse_rotary_attention(params, filename):
        parsed = _parse_attention(params, filename)
        head_dim = parsed["embed_dim"] // parsed["num_heads"]
        if parsed["embed_dim"] % parsed["num_heads"] != 0:
            raise ValueError("RotaryAttention embed_dim must be divisible by num_heads")
        if head_dim % 2 != 0:
            raise ValueError("RotaryAttention head dimension must be even")
        parsed["type"] = "rotaryattention"
        return parsed

    def _parse_resnet18(params, filename):
        _require_parameter_count(
            "ResNet18",
            params,
            {1, 2},
            "ResNet18_<image_size>[_<num_classes>]",
        )
        if params[0] <= 0 or (len(params) == 2 and params[1] <= 0):
            raise ValueError("ResNet18 image_size and num_classes must be positive")
        return {
            "type": "resnet18",
            "image_size": params[0],
            "num_classes": params[1] if len(params) == 2 else 1000,
        }

    def _parse_pruned_resnet18(params, filename):
        _require_parameter_count(
            "PrunedResNet18",
            params,
            {1, 2},
            "PrunedResNet18_<image_size>[_<num_classes>]",
        )
        if params[0] <= 0 or (len(params) == 2 and params[1] <= 0):
            raise ValueError(
                "PrunedResNet18 image_size and num_classes must be positive"
            )
        return {
            "type": "prunedresnet18",
            "image_size": params[0],
            "num_classes": params[1] if len(params) == 2 else 10,
        }

    def _parse_pruned_orin_resnet18(params, filename):
        _require_parameter_count(
            filename,
            params,
            {1, 2},
            f"{filename}_<image_size>[_<num_classes>]",
        )
        if params[0] <= 0 or (len(params) == 2 and params[1] <= 0):
            raise ValueError(
                f"{filename} image_size and num_classes must be positive"
            )
        return {
            "type": "prunedorinresnet18",
            "image_size": params[0],
            "num_classes": params[1] if len(params) == 2 else 10,
        }

    def _parse_pruned_pi5_resnet18(params, filename):
        _require_parameter_count(
            filename,
            params,
            {1, 2},
            f"{filename}_<image_size>[_<num_classes>]",
        )
        if params[0] <= 0 or (len(params) == 2 and params[1] <= 0):
            raise ValueError(
                f"{filename} image_size and num_classes must be positive"
            )
        return {
            "type": "prunedpi5resnet18",
            "image_size": params[0],
            "num_classes": params[1] if len(params) == 2 else 10,
        }

    def _parse_resnet50(params, filename):
        _require_parameter_count(
            "ResNet50",
            params,
            {1, 2},
            "ResNet50_<image_size>[_<num_classes>]",
        )
        if params[0] <= 0 or (len(params) == 2 and params[1] <= 0):
            raise ValueError("ResNet50 image_size and num_classes must be positive")
        return {
            "type": "resnet50",
            "image_size": params[0],
            "num_classes": params[1] if len(params) == 2 else 1000,
        }

    def _parse_resnet_conv(params, filename):
        _require_parameter_count(
            "ResNetConv",
            params,
            {6},
            "ResNetConv_<in_channels>_<out_channels>_<image_size>_"
            "<kernel_size>_<stride>_<padding>",
        )
        if any(value <= 0 for value in params[:5]) or params[5] < 0:
            raise ValueError("ResNetConv dimensions and stride must be positive")
        return {
            "type": "resnetconv",
            "in_channels": params[0],
            "out_channels": params[1],
            "image_size": params[2],
            "kernel_size": params[3],
            "stride": params[4],
            "padding": params[5],
        }

    def _parse_resnet_batch_norm(params, filename):
        _require_parameter_count(
            "ResNetBatchNorm",
            params,
            {2},
            "ResNetBatchNorm_<channels>_<image_size>",
        )
        if any(value <= 0 for value in params):
            raise ValueError("ResNetBatchNorm dimensions must be positive")
        return {
            "type": "resnetbatchnorm",
            "channels": params[0],
            "image_size": params[1],
        }

    def _parse_resnet_maxpool(params, filename):
        _require_parameter_count(
            "ResNetMaxPool",
            params,
            {5},
            "ResNetMaxPool_<channels>_<image_size>_<kernel_size>_<stride>_<padding>",
        )
        if any(value <= 0 for value in params[:4]) or params[4] < 0:
            raise ValueError("ResNetMaxPool dimensions and stride must be positive")
        return {
            "type": "resnetmaxpool",
            "channels": params[0],
            "image_size": params[1],
            "kernel_size": params[2],
            "stride": params[3],
            "padding": params[4],
        }

    def _parse_resnet_residual_add(params, filename):
        _require_parameter_count(
            "ResNetResidualAdd",
            params,
            {2},
            "ResNetResidualAdd_<channels>_<image_size>",
        )
        if any(value <= 0 for value in params):
            raise ValueError("ResNetResidualAdd dimensions must be positive")
        return {
            "type": "resnetresidualadd",
            "channels": params[0],
            "image_size": params[1],
        }

    def _parse_resnet_avgpool(params, filename):
        _require_parameter_count(
            "ResNetAvgPool",
            params,
            {3},
            "ResNetAvgPool_<channels>_<image_size>_<output_size>",
        )
        if any(value <= 0 for value in params):
            raise ValueError("ResNetAvgPool dimensions must be positive")
        return {
            "type": "resnetavgpool",
            "channels": params[0],
            "image_size": params[1],
            "output_size": params[2],
        }

    def _parse_activation(layer_type):
        def parse(params, filename):
            if not params or any(dimension <= 0 for dimension in params):
                raise ValueError(
                    f"{filename} names must contain positive input dimensions"
                )
            return {"type": layer_type, "input_shape": tuple(params)}

        return parse

    # To add a filename-defined layer, register its parser, module factory, and
    # input-shape function here. TorchRunner itself does not need another branch.
    TORCH_LAYER_DEFINITIONS = {
        "linear": TorchLayerDefinition(
            parse=_parse_linear,
            build=lambda params: nn.Linear(
                params["in_features"],
                params["out_features"],
            ),
            input_shape=lambda params: (1, params["in_features"]),
        ),
        "conv": TorchLayerDefinition(
            parse=_parse_conv,
            build=lambda params: nn.Conv2d(
                params["in_channels"],
                params["out_channels"],
                kernel_size=params["kernel_size"],
                padding=params["padding"],
            ),
            input_shape=lambda params: (
                1,
                params["in_channels"],
                params["image_size"],
                params["image_size"],
            ),
        ),
        "lenet": TorchLayerDefinition(
            parse=_parse_lenet,
            build=lambda params: nn.Sequential(
                nn.Conv2d(1, 8, 5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(8, 32, 5),
                nn.MaxPool2d(2),
                nn.AdaptiveMaxPool2d(4),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            ),
            input_shape=lambda params: (1, 1, 32, 32),
        ),
        "maxpool": TorchLayerDefinition(
            parse=_parse_maxpool,
            build=lambda params: nn.MaxPool2d(params["kernel_size"]),
            input_shape=lambda params: (
                1,
                params["in_channels"],
                params["image_size"],
                params["image_size"],
            ),
        ),
        "adapool": TorchLayerDefinition(
            parse=_parse_adapool,
            build=lambda params: nn.AdaptiveMaxPool2d(params["output_size"]),
            input_shape=lambda params: (
                1,
                params["in_channels"],
                params["image_size"],
                params["image_size"],
            ),
        ),
        "attention": TorchLayerDefinition(
            parse=_parse_attention,
            build=lambda params: SelfAttention(
                params["embed_dim"],
                params["num_heads"],
            ),
            input_shape=lambda params: (
                params["input_token"],
                1,
                params["embed_dim"],
            ),
        ),
        "rotaryattention": TorchLayerDefinition(
            parse=_parse_rotary_attention,
            build=lambda params: RotarySelfAttention(
                params["embed_dim"],
                params["num_heads"],
            ),
            input_shape=lambda params: (
                params["input_token"],
                1,
                params["embed_dim"],
            ),
        ),
        "resnet18": TorchLayerDefinition(
            parse=_parse_resnet18,
            build=lambda params: ResNet18(
                params["num_classes"],
                cifar_stem=(params["image_size"] <= 32),
            ),
            input_shape=lambda params: (1, 3, params["image_size"], params["image_size"]),
        ),
        "prunedresnet18": TorchLayerDefinition(
            parse=_parse_pruned_resnet18,
            build=lambda params: PrunedResNet18(
                num_classes=params["num_classes"],
                cifar_stem=(params["image_size"] <= 32),
            ),
            input_shape=lambda params: (1, 3, params["image_size"], params["image_size"]),
        ),
        "prunedorinresnet18": TorchLayerDefinition(
            parse=_parse_pruned_orin_resnet18,
            build=lambda params: PrunedOrinResNet18(
                num_classes=params["num_classes"],
                cifar_stem=(params["image_size"] <= 32),
            ),
            input_shape=lambda params: (1, 3, params["image_size"], params["image_size"]),
        ),
        "prunedresnet18orin": TorchLayerDefinition(
            parse=_parse_pruned_orin_resnet18,
            build=lambda params: PrunedOrinResNet18(
                num_classes=params["num_classes"],
                cifar_stem=(params["image_size"] <= 32),
            ),
            input_shape=lambda params: (1, 3, params["image_size"], params["image_size"]),
        ),
        "prunedpi5resnet18": TorchLayerDefinition(
            parse=_parse_pruned_pi5_resnet18,
            build=lambda params: PrunedPi5ResNet18(
                num_classes=params["num_classes"],
                cifar_stem=(params["image_size"] <= 32),
            ),
            input_shape=lambda params: (1, 3, params["image_size"], params["image_size"]),
        ),
        "prunedresnet18pi5": TorchLayerDefinition(
            parse=_parse_pruned_pi5_resnet18,
            build=lambda params: PrunedPi5ResNet18(
                num_classes=params["num_classes"],
                cifar_stem=(params["image_size"] <= 32),
            ),
            input_shape=lambda params: (1, 3, params["image_size"], params["image_size"]),
        ),
        "resnet50": TorchLayerDefinition(
            parse=_parse_resnet50,
            build=lambda params: ResNet50(params["num_classes"]),
            input_shape=lambda params: (1, 3, params["image_size"], params["image_size"]),
        ),
        "resnetconv": TorchLayerDefinition(
            parse=_parse_resnet_conv,
            build=lambda params: nn.Conv2d(
                params["in_channels"],
                params["out_channels"],
                kernel_size=params["kernel_size"],
                stride=params["stride"],
                padding=params["padding"],
                bias=False,
            ),
            input_shape=lambda params: (
                1,
                params["in_channels"],
                params["image_size"],
                params["image_size"],
            ),
        ),
        "resnetbatchnorm": TorchLayerDefinition(
            parse=_parse_resnet_batch_norm,
            build=lambda params: nn.BatchNorm2d(params["channels"]),
            input_shape=lambda params: (
                1,
                params["channels"],
                params["image_size"],
                params["image_size"],
            ),
        ),
        "resnetmaxpool": TorchLayerDefinition(
            parse=_parse_resnet_maxpool,
            build=lambda params: nn.MaxPool2d(
                kernel_size=params["kernel_size"],
                stride=params["stride"],
                padding=params["padding"],
            ),
            input_shape=lambda params: (
                1,
                params["channels"],
                params["image_size"],
                params["image_size"],
            ),
        ),
        "resnetresidualadd": TorchLayerDefinition(
            parse=_parse_resnet_residual_add,
            build=lambda params: ResNetResidualAdd(),
            input_shape=lambda params: (
                1,
                params["channels"],
                params["image_size"],
                params["image_size"],
            ),
        ),
        "resnetavgpool": TorchLayerDefinition(
            parse=_parse_resnet_avgpool,
            build=lambda params: nn.AdaptiveAvgPool2d(params["output_size"]),
            input_shape=lambda params: (
                1,
                params["channels"],
                params["image_size"],
                params["image_size"],
            ),
        ),
        "relu": TorchLayerDefinition(
            parse=_parse_activation("relu"),
            build=lambda params: nn.ReLU(),
            input_shape=lambda params: params["input_shape"],
        ),
        "gelu": TorchLayerDefinition(
            parse=_parse_activation("gelu"),
            build=lambda params: nn.GELU(),
            input_shape=lambda params: params["input_shape"],
        ),
        "silu": TorchLayerDefinition(
            parse=_parse_activation("silu"),
            build=lambda params: nn.SiLU(),
            input_shape=lambda params: params["input_shape"],
        ),
        "flatten": TorchLayerDefinition(
            parse=_parse_activation("flatten"),
            build=lambda params: nn.Flatten(),
            input_shape=lambda params: params["input_shape"],
        ),
    }

    class TorchRunner(InferenceRunner):
        """PyTorch runner"""

        def __init__(
            self,
            model_path,
            device="cpu",
            from_state_dict=False,
            batch_size=1,
        ):
            super().__init__(model_path, device=device)
            if isinstance(batch_size, bool) or not isinstance(batch_size, int):
                raise ValueError("batch_size must be a positive integer")
            if batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")
            self.device = torch.device(device)
            self.from_state_dict = from_state_dict
            self.batch_size = batch_size
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
            """Extract and validate a registered layer filename schema."""
            filename = os.path.splitext(os.path.basename(self.model_path))[0].lower()
            parts = filename.split("_")
            layer_type = parts[0]
            try:
                params = [int(p) for p in parts[1:]]
            except ValueError as error:
                raise ValueError(
                    f"Invalid numeric layer parameters in model name: {filename}"
                ) from error

            definition = TORCH_LAYER_DEFINITIONS.get(layer_type)
            if definition is None:
                raise ValueError(f"Unsupported layer type in model name: {layer_type}")
            return definition.parse(params, parts[0])

        def _build_model(self):
            """Build the layer represented by the validated model name."""
            definition = TORCH_LAYER_DEFINITIONS[self.params["type"]]
            return definition.build(self.params).to(self.device)

        def generate_input(self):
            """Generate a random input for the next burst."""
            with torch.inference_mode():
                definition = TORCH_LAYER_DEFINITIONS[self.params["type"]]
                shape = list(definition.input_shape(self.params))
                batch_axis = (
                    1
                    if self.params["type"] in {"attention", "rotaryattention"}
                    else 0
                )
                shape[batch_axis] = self.batch_size
                self.input_data = torch.randn(
                    *shape,
                    dtype=torch.float32,
                    device=self.device,
                )

        @property
        def input_batch_size(self):
            return self.batch_size
                