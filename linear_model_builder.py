# import tensorflow as tf

import numpy as np
import os
import subprocess

from base_model_builder import BaseModelBuilder
import importlib.util

_tf_available = importlib.util.find_spec("tensorflow") is not None
_torch_available = importlib.util.find_spec("torch") is not None

if _tf_available:
    import tensorflow as tf
    class TFTPULinearModelBuilder(BaseModelBuilder):
        """Builds and compiles a quantized linear model for TPU inference. Uses TensorFlow."""
        def __init__(self, input_size: int, output_size: int, 
                    model_dir="Models/TPU/Linear", compiled_dir="Models/TPU/Compiled"):
            assert isinstance(input_size, int) and input_size > 0, "Invalid input size"
            assert isinstance(output_size, int) and output_size > 0, "Invalid output size"
            
            self.input_size = input_size
            self.output_size = output_size
            self.model_dir = model_dir
            self.compiled_dir = compiled_dir
            
            # Ensure directories exist
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(self.compiled_dir, exist_ok=True)
            
            self.model = None
            self.tflite_model_path = os.path.join(
                self.model_dir, f"Linear_{self.input_size}_{self.output_size}.tflite"
            )
        
        def build_model(self):
            self.model = tf.keras.Sequential([
                tf.keras.Input(shape=(self.input_size,)),
                tf.keras.layers.Dense(units=self.output_size, activation=None)
            ])
            print(f"Built model with input size {self.input_size} and output size {self.output_size}")
        
        def convert_to_tflite_quantized(self):
            if self.model is None:
                raise RuntimeError("Model not built yet. Call build_model() first.")
            
            def representative_data_gen():
                for _ in range(1000):
                    data = np.random.rand(1, self.input_size).astype(np.float32)
                    yield [data]
            
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            tflite_model = converter.convert()
            
            with open(self.tflite_model_path, "wb") as f:
                f.write(tflite_model)
            
            print(f"Saved quantized TFLite model at {self.tflite_model_path}")
        
        def compile_model_for_tpu(self):
            command = [
                "edgetpu_compiler",
                self.tflite_model_path,
                "--out_dir", self.compiled_dir
            ]
            try:
                subprocess.run(command, check=True)
                print(f"Compiled model for input size {self.input_size} and output size {self.output_size}")
            except subprocess.CalledProcessError as e:
                print(f"Compilation failed: {e}")
            except FileNotFoundError:
                print("edgetpu_compiler not found. Please ensure it is installed and in your PATH.")
            except Exception as e:
                print(f"Unexpected error during compilation: {e}")
        
        def build_and_compile(self):
            self.build_model()
            self.convert_to_tflite_quantized()
            self.compile_model_for_tpu()

if _torch_available:
    import torch
    import torch.nn as nn
    
    class TorchLinearModelBuilder(BaseModelBuilder):
        """Builds and saves a linear model for CPU/GPU inference. Uses PyTorch."""
        def __init__(self, input_size: int, output_size: int, 
                    model_dir="Models/CPU/Linear"):
            assert isinstance(input_size, int) and input_size > 0, "Invalid input size"
            assert isinstance(output_size, int) and output_size > 0, "Invalid output size"

            self.input_size = input_size
            self.output_size = output_size
            self.model_dir = model_dir

            os.makedirs(self.model_dir, exist_ok=True)
            
            self.model = None
            #self.quantized_model = None
            self.model_path = os.path.join(
                self.model_dir, f"Linear_{self.input_size}_{self.output_size}.pt"
            )
            # self.quantized_model_path = os.path.join(
            #     self.model_dir, f"Linear_{self.input_size}_{self.output_size}_quantized.pt"
            # )

        def build_model(self):
            self.model = nn.Linear(self.input_size, self.output_size)
            print(f"Built CPU model with input size {self.input_size} and output size {self.output_size}")

        def save_model(self):
            if self.model is None:
                raise RuntimeError("Model not built yet. Call build_model() first.")
            torch.save(self.model.state_dict(), self.model_path)
            print(f"Saved CPU model at {self.model_path}")

        # def quantize_model(self):
        #     if self.model is None:
        #         raise RuntimeError("Model not built yet. Call build_model() first.")

        #     # Prepare the model for static quantization
        #     self.model.eval()
        #     self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        #     torch.quantization.prepare(self.model, inplace=True)

        #     # Calibrate with random data
        #     for _ in range(1000):
        #         input_data = torch.rand(1, self.input_size)
        #         self.model(input_data)

        #     # Convert to quantized version
        #     torch.quantization.convert(self.model, inplace=True)
        #     self.quantized_model = self.model

        #     torch.save(self.quantized_model.state_dict(), self.quantized_model_path)
        #     print(f"Saved quantized CPU model at {self.quantized_model_path}")

        def build_and_compile(self, quantize=False):
            self.build_model()
            self.save_model()
            # if quantize:
            #     self.quantize_model()
