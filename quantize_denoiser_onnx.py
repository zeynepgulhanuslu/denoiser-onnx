import argparse

import numpy as np
import onnx
import onnxruntime
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
from pathlib import Path

from denoiser_inference import to_numpy


def convert_quantize_from_onnx(onnx_model_path, quantized_onnx_model_path):
    quantize_dynamic(Path(onnx_model_path), Path(quantized_onnx_model_path), weight_type=QuantType.QUInt8)
    q_model = onnx.load(quantized_onnx_model_path)

    onnx.checker.check_model(q_model)

    # Load the model
    quantized_session = onnxruntime.InferenceSession(quantized_onnx_model_path)
    onnx_session = onnxruntime.InferenceSession(onnx_model_path)
    # Assume your model expects a single float32 tensor of shape (1, 3, 224, 224)
    # Here's how you might create a dummy tensor of zeroes of the right type and shape
    with torch.no_grad():
        dummy_input = np.zeros((1, 1, 480)).astype(np.float32)

        # The name 'input' is model-dependent. Check your model's input name by printing sess.get_inputs()
        input_name = quantized_session.get_inputs()[0].name
        output_name = quantized_session.get_outputs()[0].name
        quantized_result = quantized_session.run([output_name], {input_name: dummy_input})

        input_name_onnx = onnx_session.get_inputs()[0].name
        output_name_onnx = onnx_session.get_outputs()[0].name
        ort_outs = onnx_session.run([output_name_onnx], {input_name_onnx: dummy_input})

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(quantized_result[0], ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def convert_quantize_from_streamtt_onnx(onnx_model_path, quantized_onnx_model_path,
                                        hidden, depth, stride, frame_length, resample_buffer):
    quantize_dynamic(Path(onnx_model_path), Path(quantized_onnx_model_path), weight_type=QuantType.QUInt8)
    q_model = onnx.load(quantized_onnx_model_path)

    onnx.checker.check_model(q_model)

    # Load the model
    quantized_session = onnxruntime.InferenceSession(quantized_onnx_model_path)
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    # Assume your model expects a single float32 tensor of shape (1, 3, 224, 224)
    # Here's how you might create a dummy tensor of zeroes of the right type and shape
    variance_tensor = torch.tensor([0.0], dtype=torch.float32)
    input_name = ort_session.get_inputs()[0].name
    frame_num_name = ort_session.get_inputs()[1].name
    variance_input_name = ort_session.get_inputs()[2].name
    resample_input_frame_name = ort_session.get_inputs()[3].name
    resample_out_frame_name = ort_session.get_inputs()[4].name
    conv_state_name = ort_session.get_inputs()[5].name
    lstm_state_1_name = ort_session.get_inputs()[6].name
    lstm_state_2_name = ort_session.get_inputs()[7].name

    lstm_state_1 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    lstm_state_2 = torch.randn(2, 1, hidden * 2 ** (depth - 1))

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
    conv_state_list = [torch.zeros(size) for size in conv_state_sizes]
    conv_state = torch.cat([t.view(1, -1) for t in conv_state_list], dim=1)

    output_name = ort_session.get_outputs()[0].name
    out_frame = ort_session.get_outputs()[1].name
    out_variance_name = ort_session.get_outputs()[2].name
    out_resample_in_frame = ort_session.get_outputs()[3].name
    out_resample_frame = ort_session.get_outputs()[4].name
    out_conv = ort_session.get_outputs()[5].name
    out_lstm_1 = ort_session.get_outputs()[6].name
    out_lstm_2 = ort_session.get_outputs()[7].name

    frame_num = torch.tensor([1])

    noisy = torch.randn(1, frame_length * 2)

    resample_input_frame = torch.zeros(1, resample_buffer)
    resample_out_frame = torch.zeros(1, resample_buffer)
    with torch.no_grad():
        pending = noisy
        outs = []
        while pending.shape[1] >= frame_length:
            frame = pending[:, :frame_length]
            frame_np = to_numpy(frame)

            input_values = {
                input_name: frame_np,
                frame_num_name: to_numpy(frame_num),
                variance_input_name: to_numpy(variance_tensor),
                resample_input_frame_name: to_numpy(resample_input_frame),
                resample_out_frame_name: to_numpy(resample_out_frame),
                conv_state_name: to_numpy(conv_state),
                lstm_state_1_name: to_numpy(lstm_state_1),
                lstm_state_2_name: to_numpy(lstm_state_2)
            }

            print(f"frame number: {frame_num}, frame numpy shape: {frame_np.shape},"
                  f" dtype: {frame_np.dtype}")

            print(f"frame number: {frame_num}, frame tensor shape: {frame.shape}, dtype: {frame.dtype}")

            out = ort_session.run([output_name, out_frame, out_variance_name, out_resample_in_frame,
                                   out_resample_frame, out_conv, out_lstm_1, out_lstm_2],
                                  input_values)

            # onnx out#
            output_np = out[0]
            variance_input = out[2]
            resample_input_frame = out[3]
            resample_out_frame = out[4]
            out_conv_tensor = torch.from_numpy(out[5])

            print(f"out conv tensor shape {out_conv_tensor.shape}")
            print("---------------------------------------------------")
            conv_state = out_conv_tensor
            lstm_state_1 = out[6]
            lstm_state_2 = out[7]
            outs.append(torch.from_numpy(output_np))
            pending = pending[:, stride:]
            frame_num.add_(1)

        if pending.shape[1] > 0:
            last_frame = torch.cat([pending, torch.zeros_like(pending)[:, :frame_length - pending.shape[1]]],
                                   dim=1)

            input_values = {
                input_name: to_numpy(last_frame),
                frame_num_name: to_numpy(frame_num),
                variance_input_name: to_numpy(variance_input),
                resample_input_frame_name: to_numpy(resample_input_frame),
                resample_out_frame_name: to_numpy(resample_out_frame),
                conv_state_name: to_numpy(conv_state),
                lstm_state_1_name: to_numpy(lstm_state_1),
                lstm_state_2_name: to_numpy(lstm_state_2)
            }

            out = ort_session.run([output_name, out_frame, out_variance_name, out_resample_in_frame, out_resample_frame,
                                   out_conv, out_lstm_1, out_lstm_2], input_values)
            output_np = out[0]
            outs.append(torch.from_numpy(output_np))

        estimate = torch.cat(outs, 1)
    print(f"onnx out shape: {estimate.shape}, type: {type(estimate)}")

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model file')
    parser.add_argument('-q', '--quantized_model', type=str, required=True, help='Quantized model file')
    parser.add_argument('-s', '--stream', action='store_true',
                        help='True if onnx model streaming mode.')
    parser.add_argument('-hs', '--hidden_size', type=int, required=False, default=48,
                        help='Hidden size for dns model.')
    parser.add_argument('-d', '--depth', type=int, required=False, default=4,
                        help='Model depth')
    parser.add_argument('-f', '--frame_length', type=int, required=False, default=480, help='frame length value')
    parser.add_argument('-b', '--resample_buffer', type=int, required=False, default=64, help='resample buffer value')
    parser.add_argument('-stride', '--stride', type=int, required=False, default=64, help='Stride value')

    args = parser.parse_args()
    onnx_model_file = args.model
    quantized_model_file = args.quantized_model
    is_stream = args.stream
    hidden_size = args.hidden_size
    depth = args.depth
    frame_length = args.frame_length
    resample_buffer = args.resample_buffer
    stride = args.stride

    if is_stream:
        convert_quantize_from_streamtt_onnx(onnx_model_file, quantized_model_file, hidden_size,
                                            depth, stride, frame_length, resample_buffer)
    else:
        convert_quantize_from_onnx(onnx_model_file, quantized_model_file)
    # quantize onnx stream model

