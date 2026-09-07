import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from runner import TorchRunner
    from layers.base_attention import MultiheadAttention


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchRunnerLegacyCompatibilityTests(unittest.TestCase):
    def build_runner(self, model_name):
        return TorchRunner(str(Path("models") / model_name), device="cpu")

    def test_legacy_layer_names_preserve_parameters_and_output_shapes(self):
        cases = [
            (
                "Linear_8_4.pt",
                {"type": "linear", "in_features": 8, "out_features": 4},
                (1, 8),
                (1, 4),
            ),
            (
                "Conv_3_16_3_1.pt",
                {
                    "type": "conv",
                    "in_channels": 3,
                    "image_size": 16,
                    "kernel_size": 3,
                    "padding": 1,
                    "out_channels": 1,
                },
                (1, 3, 16, 16),
                (1, 1, 16, 16),
            ),
            (
                "Conv_3_16_3_1_6.pt",
                {
                    "type": "conv",
                    "in_channels": 3,
                    "image_size": 16,
                    "kernel_size": 3,
                    "padding": 1,
                    "out_channels": 6,
                },
                (1, 3, 16, 16),
                (1, 6, 16, 16),
            ),
            (
                "MaxPool_4_16_2.pt",
                {
                    "type": "maxpool",
                    "in_channels": 4,
                    "image_size": 16,
                    "kernel_size": 2,
                },
                (1, 4, 16, 16),
                (1, 4, 8, 8),
            ),
            (
                "AdaPool_4_16_2.pt",
                {
                    "type": "adapool",
                    "in_channels": 4,
                    "image_size": 16,
                    "output_size": 2,
                },
                (1, 4, 16, 16),
                (1, 4, 2, 2),
            ),
            (
                "Attention_5_8_2.pt",
                {
                    "type": "attention",
                    "input_token": 5,
                    "embed_dim": 8,
                    "num_heads": 2,
                },
                (5, 1, 8),
                (5, 1, 8),
            ),
            (
                "Lenet.pt",
                {"type": "lenet"},
                (1, 1, 32, 32),
                (1, 64),
            ),
            (
                "ReLU_1_4_8_8.pt",
                {"type": "relu", "input_shape": (1, 4, 8, 8)},
                (1, 4, 8, 8),
                (1, 4, 8, 8),
            ),
            (
                "GELU_16_1_32.pt",
                {"type": "gelu", "input_shape": (16, 1, 32)},
                (1, 1, 32),
                (1, 1, 32),
            ),
            (
                "SiLU_1_8_8_8.pt",
                {"type": "silu", "input_shape": (1, 8, 8, 8)},
                (1, 8, 8, 8),
                (1, 8, 8, 8),
            ),
            (
                "Flatten_1_32_4_4.pt",
                {"type": "flatten", "input_shape": (1, 32, 4, 4)},
                (1, 32, 4, 4),
                (1, 512),
            ),
            (
                "RotaryAttention_5_8_2.pt",
                {
                    "type": "rotaryattention",
                    "input_token": 5,
                    "embed_dim": 8,
                    "num_heads": 2,
                },
                (5, 1, 8),
                (5, 1, 8),
            ),
            (
                "ResNet18_32_10.pt",
                {"type": "resnet18", "image_size": 32, "num_classes": 10},
                (1, 3, 32, 32),
                (1, 10),
            ),
            (
                "PrunedResNet18_32_10.pt",
                {"type": "prunedresnet18", "image_size": 32, "num_classes": 10},
                (1, 3, 32, 32),
                (1, 10),
            ),
            (
                "ResNet50_32_10.pt",
                {"type": "resnet50", "image_size": 32, "num_classes": 10},
                (1, 3, 32, 32),
                (1, 10),
            ),
            (
                "ResNetConv_3_64_32_7_2_3.pt",
                {
                    "type": "resnetconv",
                    "in_channels": 3,
                    "out_channels": 64,
                    "image_size": 32,
                    "kernel_size": 7,
                    "stride": 2,
                    "padding": 3,
                },
                (1, 3, 32, 32),
                (1, 64, 16, 16),
            ),
            (
                "ResNetBatchNorm_64_16.pt",
                {"type": "resnetbatchnorm", "channels": 64, "image_size": 16},
                (1, 64, 16, 16),
                (1, 64, 16, 16),
            ),
            (
                "ResNetMaxPool_64_16_3_2_1.pt",
                {
                    "type": "resnetmaxpool",
                    "channels": 64,
                    "image_size": 16,
                    "kernel_size": 3,
                    "stride": 2,
                    "padding": 1,
                },
                (1, 64, 16, 16),
                (1, 64, 8, 8),
            ),
            (
                "ResNetResidualAdd_64_8.pt",
                {"type": "resnetresidualadd", "channels": 64, "image_size": 8},
                (1, 64, 8, 8),
                (1, 64, 8, 8),
            ),
            (
                "ResNetAvgPool_512_1_1.pt",
                {"type": "resnetavgpool", "channels": 512, "image_size": 1, "output_size": 1},
                (1, 512, 1, 1),
                (1, 512, 1, 1),
            ),
        ]

        for model_name, expected_params, input_shape, output_shape in cases:
            with self.subTest(model_name=model_name):
                runner = self.build_runner(model_name)
                self.assertEqual(runner.params, expected_params)
                runner.generate_input()
                self.assertEqual(tuple(runner.input_data.shape), input_shape)
                expected_batch_size = (
                    input_shape[1]
                    if expected_params["type"] in {"attention", "rotaryattention"}
                    else input_shape[0]
                )
                self.assertEqual(runner.input_batch_size, expected_batch_size)
                with torch.inference_mode():
                    output = runner.model(runner.input_data)
                self.assertEqual(tuple(output.shape), output_shape)

    def test_invalid_legacy_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported layer type"):
            self.build_runner("Unknown_1.pt")

    def test_layer_names_are_case_insensitive(self):
        cases = [
            ("lInEaR_8_4.pt", "linear"),
            ("rElU_1_8.pt", "relu"),
            ("rOtArYaTtEnTiOn_5_8_2.pt", "rotaryattention"),
            ("rEsNeT18_32_10.pt", "resnet18"),
            ("rEsNeT50_32_10.pt", "resnet50"),
        ]

        for model_name, expected_type in cases:
            with self.subTest(model_name=model_name):
                self.assertEqual(self.build_runner(model_name).params["type"], expected_type)

    def test_activation_dimensions_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive input dimensions"):
            self.build_runner("ReLU_1_0_8_8.pt")

    def test_batch_size_replaces_the_model_input_batch_dimension(self):
        linear = TorchRunner("models/Linear_8_4.pt", batch_size=4)
        attention = TorchRunner("models/Attention_5_8_2.pt", batch_size=3)
        rotary_attention = TorchRunner(
            "models/RotaryAttention_5_8_2.pt", batch_size=3
        )

        linear.generate_input()
        attention.generate_input()
        rotary_attention.generate_input()

        self.assertEqual(tuple(linear.input_data.shape), (4, 8))
        self.assertEqual(tuple(attention.input_data.shape), (5, 3, 8))
        self.assertEqual(tuple(rotary_attention.input_data.shape), (5, 3, 8))
        self.assertEqual(linear.input_batch_size, 4)
        self.assertEqual(attention.input_batch_size, 3)
        self.assertEqual(rotary_attention.input_batch_size, 3)

    def test_rotary_attention_requires_compatible_head_dimension(self):
        with self.assertRaisesRegex(ValueError, "divisible by num_heads"):
            self.build_runner("RotaryAttention_5_10_3.pt")
        with self.assertRaisesRegex(ValueError, "head dimension must be even"):
            self.build_runner("RotaryAttention_5_6_2.pt")

    def test_attention_fallback_supports_pytorch_1_10(self):
        attention = MultiheadAttention(8, 2).eval()
        inputs = torch.randn(5, 3, 8)
        native_attention = getattr(torch.nn.functional, "scaled_dot_product_attention", None)

        try:
            delattr(torch.nn.functional, "scaled_dot_product_attention")
            with torch.inference_mode():
                outputs, weights = attention(inputs, inputs, inputs, need_weights=False)
        finally:
            if native_attention is not None:
                torch.nn.functional.scaled_dot_product_attention = native_attention

        self.assertEqual(tuple(outputs.shape), (5, 3, 8))
        self.assertIsNone(weights)


if __name__ == "__main__":
    unittest.main()
