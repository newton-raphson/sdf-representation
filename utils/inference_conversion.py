import torch
from model.networks import ImplicitNetCompatible
import pickle
import os
import argparse
import io
import torch.onnx
import onnxruntime
from utils.pickling import CPU_Unpickler
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static,quantize_dynamic
from executor.executor import Executor
import numpy as np
# https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
# There is a bug in torch.load that prevents loading of CUDA tensors on CPU.
# When they are wrapped using pickle
# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else: return super().find_class(module, name)
# Run this function from the command line to convert a model to libtorch 
# python inference_conversion.py --path <path to model> --num_hidden_layers <number of hidden layers> --hidden_dim <hidden layer dimension>
def save_as_libtorch(path_chk,hidden_dim,num_hidden_layers,save_path):
    dims = [hidden_dim for i in range(num_hidden_layers)]
    skip_in = [num_hidden_layers//2]
    test_input = torch.FloatTensor([0.5,0.5,0.5])
    model = ImplicitNetCompatible(d_in=3,dims=dims,skip_in=skip_in)
    print(f"Initially Model output is {model(test_input)}")
    try:
        with open(path_chk, 'rb') as resume_file:
            print(f"....Loading model from {path_chk}...")
            # Use torch.load with map_location to ensure compatibility
            saved_data = CPU_Unpickler(resume_file).load()
            # saved_data = torch.load(resume_file, map_location={'cuda:0': 'cpu'})
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    # Adjust the keys in the state_dict to match the model architecture
    mapping = {
        f"lin{i}.weight": f"layers.{i}.weight" for i in range(len(dims) + 1)
    }
    mapping.update({f"lin{i}.bias": f"layers.{i}.bias" for i in range(len(dims) + 1)})
    # Remove the "module." prefix
    new_state_dict = {k.replace('module.', ''): v for k, v in saved_data['model_state_dict'].items()}
    new_state_dict = {mapping[k]: v for k, v in new_state_dict.items()}
    # Load the modified state_dict into your model
    model.load_state_dict(new_state_dict)
    model.eval()
    
    sm = torch.jit.script(model)

    sm.save("/work/mech-ai-scratch/samundra/projects/sdf-representation/ops/conversion_test/implicit_model.pt")
    # testing the model with 0.5,0.5,0.5

    output = model(test_input)
    # create a random input tensor between -1 and 1 to test the model
    input_tensor = torch.randn(100, 3)

    # obtain the output from the model
    output = model(input_tensor)

    # save the output and input in a csv file
    np.savetxt('/work/mech-ai-scratch/samundra/projects/sdf-representation/ops/conversion_test/output.csv', output.detach().numpy(), delimiter=',')
    np.savetxt('/work/mech-ai-scratch/samundra/projects/sdf-representation/ops/conversion_test/input.csv', input_tensor.detach().numpy(), delimiter=',')


def save_as_onxx(path_chk,hidden_dim,num_hidden_layers,save_path):
    dims = [hidden_dim for i in range(num_hidden_layers)]
    skip_in = [num_hidden_layers//2]
    test_input = torch.FloatTensor([0.5,0.5,0.5])
    model = ImplicitNetCompatible(d_in=3,dims=dims,skip_in=skip_in)
    print(f"Initially Model output is {model(test_input)}")
    try:
        with open(path_chk, 'rb') as resume_file:
            print(f"....Loading model from {path_chk}...")
            # Use torch.load with map_location to ensure compatibility
            saved_data = CPU_Unpickler(resume_file).load()
            # saved_data = torch.load(resume_file, map_location={'cuda:0': 'cpu'})
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    # Adjust the keys in the state_dict to match the model architecture
    mapping = {
        f"lin{i}.weight": f"layers.{i}.weight" for i in range(len(dims) + 1)
    }
    mapping.update({f"lin{i}.bias": f"layers.{i}.bias" for i in range(len(dims) + 1)})
    # Remove the "module." prefix
    new_state_dict = {k.replace('module.', ''): v for k, v in saved_data['model_state_dict'].items()}
    new_state_dict = {mapping[k]: v for k, v in new_state_dict.items()}
    # Load the modified state_dict into your model
    model.load_state_dict(new_state_dict)
    model.eval()
    # Input to the model
    batch_size = 1
    x = torch.randn(batch_size, 3, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    os.path.join(save_path,"model.onnx"),   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
def quantize_save(input_path, output_path,):
    quantized_model = quantize_dynamic(input_path, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Change the model to libtorch")
    parser.add_argument("--path", type=str, help="Path to model")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="Number of hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden layer dimension")
    parser.add_argument("--save_path", type=str, default=".", help="Path to save the model")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    save_as_libtorch(args.path,args.hidden_dim,args.num_hidden_layers)
    save_as_onxx(args.path,args.hidden_dim,args.num_hidden_layers,args.save_path)
    # quantize_save("implicit_model.pt","quantized_model.pt")
    print("Model saved as libtorch and onnx")
