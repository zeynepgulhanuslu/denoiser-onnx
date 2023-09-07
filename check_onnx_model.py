import onnx
import torch

from denoiser_convert_stream_onnx import DemucsOnnxStreamerTT
from denoiser_inference import init_denoiser_model_from_file

if __name__ == '__main__':

    onnx_tt_model_path = 'D:/zeynep/data/noise-cancelling/denoiser/dns/hidden=48-depth=4/dns48_depth=4_stream.onnx'
    torch_model_path = 'D:/zeynep/data/noise-cancelling/denoiser/dns/hidden=48-depth=4/best.th'
    model = init_denoiser_model_from_file(torch_model_path)

    streamer = DemucsOnnxStreamerTT(model, dry=0)
    onnx_model = onnx.load(onnx_tt_model_path)

    # Get PyTorch model parameters
    pytorch_params = streamer.state_dict()

    # Get ONNX model parameters
    onnx_params = {param.name: param for param in onnx_model.graph.initializer}
    f = open('onnx-params.txt', 'w', encoding='utf-8')
    f_torch = open('torch-params.txt', 'w', encoding='utf-8')
    for key, val in onnx_params.items():
        f.write(str(key) + ' ---- ' + str(val) + '\n')
    # Compare parameters
    for name, param in pytorch_params.items():
        onnx_param = onnx_params.get(name)
        f_torch.write(str(name) + ' ---- ' + str(param) + '\n')
        if onnx_param is None:
            print(f"Parameter '{name}' not found in ONNX model.")
        else:
            pytorch_value = param.cpu().numpy()
            onnx_value = onnx_param.float_data if onnx_param.data_type == 1 else onnx_param.int32_data
            onnx_value = torch.Tensor(onnx_value).cpu().numpy()

            if pytorch_value.shape == onnx_value.shape and torch.allclose(torch.tensor(pytorch_value),
                                                                          torch.tensor(onnx_value)):
                print(f"Parameter '{name}' matches between PyTorch and ONNX.")
            else:
                print(f"Parameter '{name}' does not match between PyTorch and ONNX.")
