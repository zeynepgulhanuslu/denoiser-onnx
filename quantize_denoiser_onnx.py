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


def convert_quantize_from_streamtt_onnx(hidden, depth):
    quantize_dynamic(Path(onnx_model_path), Path(quantized_onnx_model_path), weight_type=QuantType.QUInt8)
    q_model = onnx.load(quantized_onnx_model_path)

    onnx.checker.check_model(q_model)

    # Load the model
    quantized_session = onnxruntime.InferenceSession(quantized_onnx_model_path)
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    # Assume your model expects a single float32 tensor of shape (1, 3, 224, 224)
    # Here's how you might create a dummy tensor of zeroes of the right type and shape

    input_name = ort_session.get_inputs()[0].name
    frame_num_name = ort_session.get_inputs()[1].name
    resample_input_frame_name = ort_session.get_inputs()[2].name
    resample_out_frame_name = ort_session.get_inputs()[3].name
    conv_state_name = ort_session.get_inputs()[4].name
    lstm_state_1_name = ort_session.get_inputs()[5].name
    lstm_state_2_name = ort_session.get_inputs()[6].name

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
    out_resample_in_frame = ort_session.get_outputs()[2].name
    out_resample_frame = ort_session.get_outputs()[3].name
    out_conv = ort_session.get_outputs()[4].name
    out_lstm_1 = ort_session.get_outputs()[5].name
    out_lstm_2 = ort_session.get_outputs()[6].name

    frame_num = torch.tensor([1])
    frame_length = 661
    if depth == 4:

        stride = 64
        resample_buffer = 64
    else:
        stride = 256
        resample_buffer = 256
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
                resample_input_frame_name: to_numpy(resample_input_frame),
                resample_out_frame_name: to_numpy(resample_out_frame),
                conv_state_name: to_numpy(conv_state),
                lstm_state_1_name: to_numpy(lstm_state_1),
                lstm_state_2_name: to_numpy(lstm_state_2)
            }

            print(f"frame number: {frame_num}, frame numpy shape: {frame_np.shape},"
                  f" dtype: {frame_np.dtype}")

            print(f"frame number: {frame_num}, frame tensor shape: {frame.shape}, dtype: {frame.dtype}")

            out = ort_session.run([output_name, out_frame, out_resample_in_frame,
                                   out_resample_frame, out_conv, out_lstm_1, out_lstm_2],
                                  input_values)

            # onnx out#
            output_np = out[0]
            resample_input_frame = out[2]
            resample_out_frame = out[3]
            out_conv_tensor = torch.from_numpy(out[4])

            print(f"out conv tensor shape {out_conv_tensor.shape}")
            print("---------------------------------------------------")
            conv_state = out_conv_tensor
            lstm_state_1 = out[5]
            lstm_state_2 = out[6]
            outs.append(torch.from_numpy(output_np))
            pending = pending[:, stride:]
            frame_num.add_(1)

        if pending.shape[1] > 0:
            last_frame = torch.cat([pending, torch.zeros_like(pending)[:, :frame_length - pending.shape[1]]],
                                   dim=1)

            input_values = {
                input_name: to_numpy(last_frame),
                frame_num_name: to_numpy(frame_num),
                resample_input_frame_name: to_numpy(resample_input_frame),
                resample_out_frame_name: to_numpy(resample_out_frame),
                conv_state_name: to_numpy(conv_state),
                lstm_state_1_name: to_numpy(lstm_state_1),
                lstm_state_2_name: to_numpy(lstm_state_2)
            }

            out = ort_session.run([output_name, out_frame, out_resample_in_frame, out_resample_frame,
                                   out_conv, out_lstm_1, out_lstm_2], input_values)
            output_np = out[0]
            outs.append(torch.from_numpy(output_np))

        estimate = torch.cat(outs, 1)
    print(f"onnx out shape: {estimate.shape}, type: {type(estimate)}")

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    root = 'D:/zeynep/data/noise-cancelling/denoiser/dns/hidden=36/'
    onnx_model_path = root + "dns36_depth=5_stream.onnx"
    quantized_onnx_model_path = root + "dns36_depth=5_stream_quantized.onnx"
    # quantize onnx model
    #convert_quantize_from_onnx(onnx_model_path, quantized_onnx_model_path)
    # quantize onnx stream model
    convert_quantize_from_streamtt_onnx(36, 5)
