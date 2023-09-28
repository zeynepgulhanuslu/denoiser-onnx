import argparse
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

from deepfilternet_onnx_stream import generate_onnx_features


def quantize_onnx_model(onnx_model_file, quantized_model_file):
    onnx_opt_model = onnx.load(onnx_model_file)
    quantize_dynamic(onnx_model_file,
                     quantized_model_file,
                     weight_type=QuantType.QInt8)

    q_model = onnx.load(quantized_model_file)

    onnx.checker.check_model(q_model)

    return q_model


def check_quantized_model(onnx_model_file, quantized_model_file):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    onnx_session = ort.InferenceSession(onnx_model_file, sess_options, providers=['CPUExecutionProvider'])
    quantized_session = ort.InferenceSession(quantized_model_file, sess_options, providers=['CPUExecutionProvider'])

    OUTPUT_NAMES = [
        'enhanced_audio_frame', 'out_states', 'lsnr'
    ]

    states_onnx = torch.zeros(45304, device='cpu')
    states_quantized = torch.zeros(45304, device='cpu')
    atten_lim_db = torch.tensor(0.0, device='cpu')
    frame_size = 480
    for i in range(30):
        input_frame = torch.randn(frame_size)

        # onnx
        output_onnx = onnx_session.run(
            OUTPUT_NAMES,
            generate_onnx_features([input_frame, states_onnx, atten_lim_db]),
        )

        output_quantized = quantized_session.run(
            OUTPUT_NAMES,
            generate_onnx_features([input_frame, states_quantized, atten_lim_db])
        )

        for (x, y, name) in zip(output_quantized, output_onnx, OUTPUT_NAMES):
            y_tensor = torch.from_numpy(y)
            assert torch.allclose(x, y_tensor,
                                  atol=1e-3), f"out {name} - {i}, {x.flatten()[-5:]}, {y_tensor.flatten()[-5:]}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--onnx_model', type=str, required=True, help='Onnx model file')
    parser.add_argument('-q', '--quantized_model', type=str, required=True, help='Quantized model file')

    args = parser.parse_args()

    onnx_model_file = args.onnx_model
    quantized_model_file = args.quantized_model

    quantize_onnx_model(onnx_model_file, quantized_model_file)
    check_quantized_model(onnx_model_file, quantized_model_file)
