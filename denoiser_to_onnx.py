import numpy as np
import onnx
import onnxruntime
import torch

from denoiser.pretrained import dns64, dns48
from denoiser.utils import deserialize_model

# pretrained model
def convert_dns48():
    model = dns48()
    model.eval()
    input_tensor = torch.randn(1, 1, 1024)

    with torch.no_grad():
        # Export the model
        torch.onnx.export(model,  # model being run
                          input_tensor,  # model input (or a tuple for multiple inputs)
                          onnx_model_path,  # where to save the model (can be a file or file-like object)
                          opset_version=11,
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size', 1: 'channel', 2: 'sequence_length'},
                                        # variable length axes
                                        'output': {0: 'batch_size', 1: 'channel', 2: 'sequence_length'}}
                          )

# pretrained model
def convert_dns64():
    model = dns64()
    model.eval()
    input_tensor = torch.randn(1, 1, 1024)

    with torch.no_grad():
        # Export the model
        torch.onnx.export(model,  # model being run
                          input_tensor,  # model input (or a tuple for multiple inputs)
                          onnx_model_path,  # where to save the model (can be a file or file-like object)
                          opset_version=11,
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'channel', 1: 'sequence_length'},
                                        # variable length axes
                                        'output': {0: 'channel', 1: 'sequence_length'}})

    # this function initialize model from file.


def init_denoiser_model_from_file(model_file):
    print(f"Loading model from {model_file}")
    pkg = torch.load(model_file, 'cpu')
    if 'model' in pkg:
        if 'best_state' in pkg:
            pkg['model']['state'] = pkg['best_state']
        model = deserialize_model(pkg['model'])
    else:
        model = deserialize_model(pkg)

    return model


def convert_to_onnx_from_path(model_file, onnx_model_path):
    model = init_denoiser_model_from_file(model_file)
    model.eval()
    input_tensor = torch.randn(1, 1, 512)  # this is the correct level
    with torch.no_grad():
        # Export the model
        torch.onnx.export(model,  # model being run
                          input_tensor,  # model input (or a tuple for multiple inputs)
                          onnx_model_path,  # where to save the model (can be a file or file-like object)
                          opset_version=11,
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size', 1: 'channel', 2: 'sequence_length'},
                                        # variable length axes
                                        'output': {0: 'batch_size', 1: 'channel', 2: 'sequence_length'}})


def infer_audio_with_denoiser(model, audio_array):
    out = torch.from_numpy(audio_array.reshape(1, len(audio_array))).to('cpu')
    print(f'tensor shape:{out.shape}')
    with torch.no_grad():
        estimated_batch = model(out)
        enhanced = estimated_batch
    print(f'enhanced shape:{enhanced.shape}, type :{type(enhanced)}')
    enhanced = enhanced / max(enhanced.abs().max().item(), 1)
    np_enhanced = np.squeeze(enhanced.detach().squeeze(0).cpu().numpy())
    print(f'numpy enhanced shape:{np_enhanced.shape}, type :{type(np_enhanced)}')
    return np_enhanced


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def compare_model(onnx_path, model_path, input_array):
    model = init_denoiser_model_from_file(model_path)
    model.eval()

    # Load the ONNX model
    model_onnx = onnx.load(onnx_path)

    # Check that the model is well formed
    onnx.checker.check_model(model_onnx)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model_onnx.graph))

    ort_session = onnxruntime.InferenceSession(onnx_path)

    torch_out = model(input_array)[0]
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_array)}
    print(f"tensor shape: {torch_out.shape}, input shape:{to_numpy(input_array).shape}")
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0].shape, torch_out.shape)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    torch_model_file = 'D:/zeynep/data/noise-cancelling/denoiser/dns/hidden=36/best.th'
    onnx_model_path = 'D:/zeynep/data/noise-cancelling/denoiser/dns/hidden=36/dns36_depth=5.onnx'
    convert_to_onnx_from_path(torch_model_file, onnx_model_path)

    model = init_denoiser_model_from_file(torch_model_file)

    model.eval()

    input_array = torch.randn(1, 1, 480)
    # Load the ONNX model
    model_onnx = onnx.load(onnx_model_path)

    # Check that the model is well formed
    onnx.checker.check_model(model_onnx)
    # Print a human readable representation of the graph

    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    torch_out = model.forward(input_array)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_array)}
    print(f"tensor shape: {torch_out.shape}, input shape:{to_numpy(input_array).shape}")
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0].shape, torch_out.shape)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-06)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
