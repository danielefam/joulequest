from base_model_builder import BaseModelBuilder
from linear_model_builder import TFTPULinearModelBuilder
from typing import Type, List, Tuple

from itertools import product

# This set of functions is used to buid and compile all the needed models according to a list of given input and output sizes.
#
# def tpu_generate_linear_model_collection(all_layers):
#     """Builds all quantized linear models for TPU inference."""
#     for input_size, output_size in all_layers:
#         print(f"Building model with input size {input_size} and output size {output_size}")
        
#         # Build the model
#         #tpu_bclm.tpu_build_quantized_linear_model(input_size, output_size)
#         builder = tpu_lmf.TPULinearModelBuilder(input_size, output_size)
#         builder.build_and_compile()
#         time.sleep(1) 

# def tpu_compile_all_quantized_linear_models(all_layers):
#     for input_size, output_size in all_layers:

#         # Compile the model
#         tpu_blm.tpu_compile_quantized_linear_models(input_size, output_size)
#         time.sleep(1)


class ModelBuilderManager:
    def __init__(self, builder_class: Type[BaseModelBuilder], specs_list: List[Tuple[int, int]]):
        """Initializes the ModelFactoryManager with a builder class and a list of specifications.
        Args:
            builder_class: The class used to build models.
            specs_list: A list of tuples, where each tuple contains the specifications for a model.
        """
        if not issubclass(builder_class, BaseModelBuilder):
            raise TypeError(f"{builder_class.__name__} must be a subclass of TPUBaseModelBuilder")
        self.builder_class = builder_class
        self.specs_list = specs_list
    
    def build_all(self):
        for spec in self.specs_list:
            builder = self.builder_class(*spec)
            builder.build_and_compile()

if __name__ == "__main__":

    sizes = [64,128,256,512,1024,2048,4096,8192]
    specs = list(product(sizes, sizes))

    manager = ModelBuilderManager(TFTPULinearModelBuilder, specs)
    print('done')