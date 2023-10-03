import argparse
import math

import numpy as np
import onnx
import onnxruntime
import torch
from denoiser.demucs import fast_conv
from denoiser.pretrained import dns48, dns64
from denoiser.resample import downsample2, upsample2
from torch import nn

from demucs_streamer import DemucsOnnxStreamerTT
from denoiser_inference import init_denoiser_model_from_file


def convert_stream_model(onnx_tt_model_path, torch_model_path=None, use_dns_48=False, use_dns64=False,
                         opset_version=13):
    if use_dns_48:
        model = dns48()
    elif torch_model_path is not None:
        model = init_denoiser_model_from_file(torch_model_path)
    elif use_dns64:
        model = dns64()
    else:
        model = dns48()
    model.eval()
    streamer = DemucsOnnxStreamerTT(model, dry=0)
    depth = streamer.demucs.depth
    streamer.eval()
    print(f"total length: {streamer.total_length},"
          f"resample buffer: {streamer.resample_buffer} , "
          f"stride: {streamer.stride},"
          f"depth: {depth}",
          f"hidden: {streamer.demucs.hidden}")

    # x = torch.randn(1, streamer.total_length)
    x = torch.randn(1, 1024)

    frame_num = torch.tensor([2])
    variance_tensor = torch.tensor([0.0], dtype=torch.float32)
    hidden = streamer.demucs.hidden

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

    conv_state_list = [torch.randn(size) for size in conv_state_sizes]
    conv_state = torch.cat([t.view(1, -1) for t in conv_state_list], dim=1)
    lstm_state_1 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    lstm_state_2 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    resample_input_frame = torch.randn(1, streamer.resample_buffer)
    resample_out_frame = torch.randn(1, streamer.resample_buffer)
    print(streamer.resample_buffer)
    with torch.no_grad():
        input_names = ['input', 'frame_num', 'variance', 'resample_input_frame', 'resample_out_frame',
                       'conv_state', 'lstm_state_1', 'lstm_state_2']

        output_names = ['output', 'frame_num', 'variance', 'resample_input_frame',
                        'resample_out_frame', 'conv_state', 'lstm_state_1', 'lstm_state_2']

        torch.onnx.export(streamer,
                          (x, frame_num, variance_tensor, resample_input_frame, resample_out_frame,
                           conv_state, lstm_state_1, lstm_state_2),
                          onnx_tt_model_path,
                          verbose=True,
                          opset_version=opset_version,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes={'input': {0: 'channel', 1: 'sequence_length'},
                                        # variable length axes
                                        'output': {0: 'channel', 1: 'sequence_length'}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--torch_model', type=str, required=True, help='Torch model file')
    parser.add_argument('-o', '--onnx_model', type=str, required=True, help='Onnx model path')
    parser.add_argument('-dns48', type=bool, required=False, default=False,
                        help='True if you want to convert pre-trained dns48 model.')
    parser.add_argument('-dns64', type=bool, required=False, default=False,
                        help='True if you want to convert pre-trained dns64 model.')
    parser.add_argument('-opset', type=int, required=False, default=13,
                        help='onnx export opset version.')

    args = parser.parse_args()
    torch_model_path = args.torch_model
    onnx_tt_model_path = args.onnx_model
    use_dns48 = args.dns48
    use_dns64 = args.dns64
    opset_version = args.opset
    convert_stream_model(onnx_tt_model_path, torch_model_path, use_dns48, use_dns64, opset_version)
