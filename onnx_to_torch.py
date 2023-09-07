import onnx
import torch
from onnx2torch import convert


def onnx_to_torch(onnx_model_path):
    """Converts an ONNX model to a PyTorch model.

    Args:
      onnx_model_path: The path to the ONNX model.

    Returns:
      A PyTorch model.
    """
    x = torch.randn(1, 1024)

    frame_num = torch.tensor([2])
    hidden = 48
    depth = 4
    if depth == 4:
        conv_state_sizes = [
            (1, hidden, 148),
            (1, hidden * 2, 36),
            (1, hidden * 4, 8),
            (1, hidden * 4, 4),
            (1, hidden * 2, 4),
            (1, hidden, 4),
            (1, 1, 4)
        ]
    else:
        conv_state_sizes = [
            (1, hidden, 596),
            (1, hidden * 2, 148),
            (1, hidden * 4, 36),
            (1, hidden * 8, 8),
            (1, hidden * 8, 4),
            (1, hidden * 4, 4),
            (1, hidden * 2, 4),
            (1, hidden, 4),
            (1, 1, 4)
        ]
    resample_buffer = 64
    conv_state_list = [torch.randn(size) for size in conv_state_sizes]
    conv_state = torch.cat([t.view(1, -1) for t in conv_state_list], dim=1)
    lstm_state_1 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    lstm_state_2 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    resample_input_frame = torch.randn(1, resample_buffer)
    resample_out_frame = torch.randn(1, resample_buffer)

    onnx_model = onnx.load(onnx_model_path)
    torch_model = torch.jit.trace(onnx_model, example_inputs=(x, frame_num, resample_input_frame, resample_out_frame,
                                                              conv_state, lstm_state_1, lstm_state_2))
    return torch_model


if __name__ == "__main__":
    onnx_model_path = "D:/zeynep/data/noise-cancelling/denoiser/dns/hidden=48-depth=4/dns48_depth=4_stream.onnx"
    #torch_model = onnx_to_torch(onnx_model_path)


    # Path to ONNX model
    # You can pass the path to the onnx model to convert it or...
    torch_model_1 = convert(onnx_model_path)
    print(torch_model_1)
    # Or you can load a regular onnx model and pass it to the converter
    onnx_model = onnx.load(onnx_model_path)
    torch_model_2 = convert(onnx_model)
    print(torch_model_2)
