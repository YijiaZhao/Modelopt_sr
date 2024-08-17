import os
import sys
import argparse

import numpy as np
import tensorrt as trt

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

# from image_batcher import ImageBatcher
# import torchvision.transforms.functional as F
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch

class TensorRTInfer:
    """
    Implements inference for the EfficientNet TensorRT engine.
    """

    def __init__(self, engine_path, shape_list):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = shape_list[i]
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.context.set_input_shape(name, shape)
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return [(self.outputs[i]["shape"], self.outputs[i]["dtype"]) for i in range(len(self.outputs))]

    def infer(self, sample, timestep, encoder_hidden_states, aes_embedding):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param top: The number of classes to return as top_predicitons, in descending order by their score. By default,
        setting to one will return the same as the maximum score class. Useful for Top-5 accuracy metrics in validation.
        :return: Three items, as numpy arrays for each batch image: The maximum score class, the corresponding maximum
        score, and a list of the top N classes and scores.
        """
        # Prepare the output data
        out_lists=self.output_spec()
        outputs_final = []
        for i in range(len(out_lists)):
            outputs_final.append(np.zeros(*out_lists[i]))

        # Process I/O and execute the network
        cuda.memcpy_htod(
            self.inputs[0]["allocation"], np.ascontiguousarray(sample)
        )
        cuda.memcpy_htod(
            self.inputs[1]["allocation"], np.ascontiguousarray(timestep)
        )
        cuda.memcpy_htod(
            self.inputs[2]["allocation"], np.ascontiguousarray(encoder_hidden_states)
        )
        cuda.memcpy_htod(
            self.inputs[3]["allocation"], np.ascontiguousarray(aes_embedding)
        )
                                
        self.context.execute_v2(self.allocations)
        # import pdb;pdb.set_trace()
        for i in range(len(out_lists)):
            cuda.memcpy_dtoh(outputs_final[i], self.outputs[i]["allocation"])

        return outputs_final
    
    
if __name__ == "__main__":

    engine_path = "backbone.plan"
    
    io_shapes = []
    
    sample = torch.randn([2, 8, 128, 128], dtype=torch.float32).half()
    
    timestep = torch.ones([1], dtype=torch.float32).half()
    
    encoder_hidden_states = torch.randn([2, 77, 768], dtype=torch.float32).half()
    
    timestep_cond = torch.randn([2, 896], dtype=torch.float32).half()


    # sample2=np.load("/dit/debug_inputs/sample.npy")
    # timestep2=np.load("/dit/debug_inputs/timestep.npy")
    # encoder_hidden_states2=np.load("/dit/debug_inputs/encoder_hidden_states.npy")
    # aes_embedding2=np.load("/dit/debug_inputs/aes_embedding.npy")
    
    shape_sample = sample.shape
    shape_timestep = timestep.shape
    shape_encoder_hidden_states = encoder_hidden_states.shape
    shape_timestep_cond = timestep_cond.shape
    
    io_shapes.append(shape_sample)
    io_shapes.append(shape_timestep)
    io_shapes.append(shape_encoder_hidden_states)
    io_shapes.append(shape_timestep_cond)
    io_shapes.append([2,5,18,128])
    
    trt_infer = TensorRTInfer(engine_path, io_shapes)
    for _ in range(2):
        # outputs_trt = trt_infer.infer(sample.half(), timestep.half(), encoder_hidden_states.half(), timestep_cond.half())
        outputs_trt = trt_infer.infer(sample, timestep, encoder_hidden_states, timestep_cond)

    for _ in range(3):
        outputs_trt = trt_infer.infer(sample, timestep, encoder_hidden_states, timestep_cond)
    # outputs_trt = trt_infer.infer(sample.numpy(), timestep.numpy(), encoder_hidden_states.numpy(), timestep_cond.numpy())
    print(outputs_trt)