import numpy as np
import onnx
import onnxruntime
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path


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


if __name__ == '__main__':
    onnx_model_path = "D:/zeynep/data/noise-cancelling/denoiser/dns/hidden=64-depth=4/dns64_depth=4_buffer=480.onnx"
    quantized_onnx_model_path = "D:/zeynep/data/noise-cancelling/denoiser/dns/hidden=64-depth=4/dns64_depth=4_buffer=480_quantized.onnx"
    convert_quantize_from_onnx(onnx_model_path, quantized_onnx_model_path)
