import argparse

import numpy as np
import onnx
import onnxruntime
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import numpy.testing as np_testing
from denoiser_inference import to_numpy

seed = 2036
torch.manual_seed(seed)
np.random.seed(seed)


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
    quantize_session = onnxruntime.InferenceSession(quantized_onnx_model_path)
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    estimate_onnx = test_denoise_with_onnx(depth, frame_length, hidden, ort_session, quantize_session, resample_buffer,
                                      stride)
    estimate_quantized = test_denoise_with_onnx(depth, frame_length, hidden, quantize_session, quantize_session, resample_buffer,
                                      stride)

    np.testing.assert_allclose(to_numpy(estimate_quantized), to_numpy(estimate_onnx), rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def test_denoise_with_onnx(depth, frame_length, hidden, ort_session, quantize_session, resample_buffer, stride):
    input_name = ort_session.get_inputs()[0].name
    frame_buffer_name = ort_session.get_inputs()[1].name
    frames_input_name = ort_session.get_inputs()[2].name
    variance_input_name = ort_session.get_inputs()[3].name
    resample_input_name = ort_session.get_inputs()[4].name
    resample_output_name = ort_session.get_inputs()[5].name
    h0_input_name = ort_session.get_inputs()[6].name
    c0_input_name = ort_session.get_inputs()[7].name
    conv_state_name = ort_session.get_inputs()[8].name
    output_name = ort_session.get_outputs()[0].name
    out_buffer_name = ort_session.get_outputs()[1].name
    frames_output_name = ort_session.get_outputs()[2].name
    variance_output_name = ort_session.get_outputs()[3].name
    out_resample_input_name = ort_session.get_outputs()[4].name
    out_resample_out_name = ort_session.get_outputs()[5].name
    h0_output_name = ort_session.get_outputs()[6].name
    c0_output_name = ort_session.get_outputs()[7].name
    out_conv_state_name = ort_session.get_outputs()[8].name
    output_names = [output_name,
                    out_buffer_name,
                    frames_output_name,
                    variance_output_name,
                    out_resample_input_name,
                    out_resample_out_name,
                    h0_output_name,
                    c0_output_name,
                    out_conv_state_name]
    frame_buffer = np.zeros((1, frame_length), dtype=np.float32)
    frames_input = torch.tensor([1])  # Wrap the scalar inside a tensor
    h0 = np.zeros((2, 1, hidden * 2 ** (depth - 1)), dtype=np.float32)
    c0 = np.zeros((2, 1, hidden * 2 ** (depth - 1)), dtype=np.float32)
    variance = torch.tensor([0.0], dtype=torch.float32)
    resample_input_frame = np.zeros((1, resample_buffer), dtype=np.float32)
    resample_out_frame = np.zeros((1, resample_buffer), dtype=np.float32)
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
    print("testing with onnx model")
    noisy = torch.randn(1, frame_length * 3, dtype=torch.float32)
    print(f"noisy shape: {noisy.shape}")
    with torch.no_grad():
        pending = noisy
        outs = []
        while pending.shape[1] >= stride:
            frame = pending[:, :stride].numpy()
            input_dict = {input_name: frame,
                          frame_buffer_name: frame_buffer,
                          frames_input_name: to_numpy(frames_input),
                          variance_input_name: to_numpy(variance),
                          resample_input_name: resample_input_frame,
                          resample_output_name: resample_out_frame,
                          h0_input_name: h0,
                          c0_input_name: c0,
                          conv_state_name: to_numpy(conv_state)}
            quantize_session
            out, out_frame_buffer, out_frame_num, out_variance, out_resample_input_frame, out_resample_out_frame, \
            out_h, out_c, out_conv_state = ort_session.run(
                output_names,
                input_dict)

            outs.append(torch.from_numpy(out))
            frame_buffer = out_frame_buffer
            variance = out_variance
            resample_input_frame = out_resample_input_frame
            resample_out_frame = out_resample_out_frame
            h0 = out_h
            c0 = out_c
            conv_state = out_conv_state
            pending = pending[:, stride:]
            print(f"second pending result {pending.shape}")
            frames_input.add_(1)
        estimate = torch.cat(outs, 1)
    return estimate


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
    parser.add_argument('-f', '--frame_length', type=int, required=False, default=362, help='frame length value')
    parser.add_argument('-b', '--resample_buffer', type=int, required=False, default=128, help='resample buffer value')
    parser.add_argument('-stride', '--stride', type=int, required=False, default=128, help='Stride value')

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
