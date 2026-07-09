# tflite_tpu_runner.py
import platform
import numpy as np

from base_runner import InferenceRunner
import time
import os
from pprint import pprint 


# Detect pytorch env or tensorflow env
import importlib.util
_tf_available = importlib.util.find_spec("tflite_runtime") is not None
_torch_available = importlib.util.find_spec("torch") is not None


if _tf_available:
    import tflite_runtime.interpreter as tflite
    EDGETPU_SHARED_LIB = {
        'Linux': 'libedgetpu.so.1',
        'Darwin': 'libedgetpu.1.dylib',
        'Windows': 'edgetpu.dll'
    }[platform.system()]

    class TFLiteTPURunner(InferenceRunner):
        def __init__(self, model_path,device):
            super().__init__(model_path)
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

        def run_inference(self,count=5, repeat=10):

            input_shape = self.input_details[0]['shape']
            input_data = np.random.rand(1, input_shape[1]).astype(np.float32)

            # Quantization
            scale, zero_point = self.input_details[0]['quantization']
            input_uint8 = (input_data / scale + zero_point).astype(np.uint8)
            

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_uint8)
            #interpreter.set_tensor(input_details[0]['index'], input_data)        

            for _ in range(repeat):
                for _ in range(count):
                    self.interpreter.invoke()
                time.sleep(1)

            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            print("Input:")
            print(input_data)
            print("\nOutput:")
            print(output_data)

if _torch_available:
    import torch
    class TorchRunner(InferenceRunner):
        def __init__(self, model_path,device):
            super().__init__(model_path,device=device)
            self._load_model(device=device) 


        def _load_model(self,from_state_dict=False,device="cpu"):

            self.device = torch.device(device)

            #Extract model info from the file name
            self.params = self._extract_layer_info()
            pprint(self.params)
            #print(f"Model info - Layer: {self.layer_type}, Input size: {self.input_size}, Output size: {self.output_size}")
            #Build the model architecture
            self.model = self._build_model()

            if from_state_dict:
                
                #Load the state dict
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            
            self.model.eval()
            print(f" Torch Model loaded: {self.model_path}")

        def run_inference(self,count=5, repeat=10):
            for _ in range(repeat):
                for _ in range(count):
                    output_data = self.model(self.input_data)
                # torch.cuda.synchronize()
                time.sleep(1)

            print("Input:")
            print(self.input_data)
            print("\nOutput:")
            print(output_data)
        
        def _extract_layer_info(self):

            #String path treatment to extract model info
            filename = os.path.splitext(os.path.basename(self.model_path))[0]
            
            parts = filename.split("_")
            layer_type = parts[0].lower()
            params = [int(p) for p in parts[1:]]  # Convert all but the first part to integers
    
            if layer_type == "linear":
                return  {
                        "type": "linear",
                        "in_features": params[0],
                        "out_features": params[1],
                        }   

            elif layer_type == "conv":
                return  {
                        "type": "conv",
                        "in_channels": params[0],
                        "image_size": params[1],
                        "kernel_size": params[2],
                        "padding": params[3],
                        }

            else:
                return  {
                        "type": layer_type,
                        "params": params
                        }

        def _build_model(self):
            #Build the model architecture
            
            #Add here other layer types if needed (e.g., Conv2d, LSTM, etc.)
            if self.params["type"]=="linear":
                model = torch.nn.Linear(self.params["in_features"],self.params["out_features"]).to(self.device)
        
            elif self.params["type"]=="conv":
                model = torch.nn.Conv2d(self.params["in_channels"],1,kernel_size=self.params["kernel_size"],padding=self.params["padding"]).to(self.device)
            
            return model
            
        def generate_input(self):
            with torch.no_grad():
                if self.params["type"]=="linear":
                    shape = (
                            1,
                            self.params["in_features"], 
                            )
            
                elif self.params["type"]=="conv":
                    shape = (
                            1,
                            self.params["in_channels"],
                            self.params["image_size"],
                            self.params["image_size"], 
                            )

                else:
                    raise ValueError(f"Unsupported layer type: {self.params['type']}")
                
                self.input_data = torch.randn(
                                                *shape,
                                                dtype=torch.float32,
                                                device=self.device
                                            )
                