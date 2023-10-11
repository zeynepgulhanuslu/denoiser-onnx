import argparse
import io
import os
import time
import zipfile

import librosa
import numpy as np
import onnxruntime
import torch
import torchaudio

seed = 2036
torch.manual_seed(seed)


# torch-> numpy dönüşümü
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    else:
        return tensor


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping

    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def load_onnx_from_zip(onnx_tt_model_path):
    # .zip arşivini yükleyin (örnek olarak "model.zip" adını varsayalım)
    with open(onnx_tt_model_path, "rb") as file:
        zip_file = file.read()

    # Zip dosyasını bellekte açın
    zip_buffer = io.BytesIO(zip_file)

    # Zip arşivini açın
    with zipfile.ZipFile(zip_buffer, "r") as archive:
        # ONNX model dosyasını yükleyin
        model_bytes = archive.read("model.onnx")

    return model_bytes


def is_zip_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() == ".zip"


def denoise_audio(noisy, onnx_model_file, hidden, depth, stride, total_length, resample_buffer, out_file):
    ort_session = onnxruntime.InferenceSession(onnx_model_file)
    # onnx model input names:
    input_name = ort_session.get_inputs()[0].name
    frame_buffer_name = ort_session.get_inputs()[1].name
    frames_input_name = ort_session.get_inputs()[2].name
    variance_input_name = ort_session.get_inputs()[3].name
    resample_input_name = ort_session.get_inputs()[4].name
    resample_output_name = ort_session.get_inputs()[5].name
    h0_input_name = ort_session.get_inputs()[6].name
    c0_input_name = ort_session.get_inputs()[7].name
    conv_state_name = ort_session.get_inputs()[8].name

    # onnx model output names:
    output_name = ort_session.get_outputs()[0].name
    out_buffer_name = ort_session.get_outputs()[1].name
    frames_output_name = ort_session.get_outputs()[2].name
    variance_output_name = ort_session.get_outputs()[3].name
    out_resample_input_name = ort_session.get_outputs()[4].name
    out_resample_out_name = ort_session.get_outputs()[5].name
    h0_output_name = ort_session.get_outputs()[6].name
    c0_output_name = ort_session.get_outputs()[7].name
    out_conv_state_name = ort_session.get_outputs()[8].name
    # output names list
    output_names = [output_name,
                    out_buffer_name,
                    frames_output_name,
                    variance_output_name,
                    out_resample_input_name,
                    out_resample_out_name,
                    h0_output_name,
                    c0_output_name,
                    out_conv_state_name]
    # this will simulate overlap.
    frame_buffer = np.zeros((1, total_length), dtype=np.float32)
    frames_input = torch.tensor([1])
    h0 = np.zeros((2, 1, hidden * 2 ** (depth - 1)), dtype=np.float32)  # lstm first state
    c0 = np.zeros((2, 1, hidden * 2 ** (depth - 1)), dtype=np.float32)  # lstm second state
    variance = torch.tensor([0.0], dtype=torch.float32)  # variance value
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
    total_frame = 0
    total_inference_time = 0
    frame_in_ms = (total_length / 16000) * 1000  # her bir frame in ms cinsinden uzunluğu
    total_duration = (len(noisy) / 16000) * 1000

    with torch.no_grad():
        pending = noisy
        outs = []
        while pending.shape[1] >= stride:
            frame = pending[:, :stride].numpy()  # send frame stride length
            # onnx model input dict.
            input_dict = {input_name: frame,
                          frame_buffer_name: frame_buffer,
                          frames_input_name: to_numpy(frames_input),
                          variance_input_name: to_numpy(variance),
                          resample_input_name: resample_input_frame,
                          resample_output_name: resample_out_frame,
                          h0_input_name: h0,
                          c0_input_name: c0,
                          conv_state_name: to_numpy(conv_state)}
            start_time = time.time()
            out, out_frame_buffer, out_frame_num, out_variance, out_resample_input_frame, out_resample_out_frame, \
            out_h, out_c, out_conv_state = ort_session.run(
                output_names,
                input_dict)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            if inference_time > 0.1:
                total_frame += 1

            total_inference_time += inference_time
            rtf = inference_time / frame_in_ms
            print(f"inference time in ms for frame {total_frame + 1}, noisy frame in ms: {frame_in_ms}, "
                  f"{inference_time} ms. rtf: {rtf}")
            outs.append(torch.from_numpy(out))  # add out enhanced frame to list
            # update inputs with results
            frame_buffer = out_frame_buffer
            variance = out_variance
            resample_input_frame = out_resample_input_frame
            resample_out_frame = out_resample_out_frame
            h0 = out_h
            c0 = out_c
            conv_state = out_conv_state
            pending = pending[:, stride:]
            frames_input.add_(1)

        if pending.shape[1] > 0:
            # Expand the remaining audio with zeros
            zeros_needed = stride - pending.shape[1]
            zeros_to_add = torch.zeros(1, zeros_needed)
            last_frame = torch.cat([pending, zeros_to_add], dim=1)
            input_dict[input_name] = last_frame.numpy()
            start_time = time.time()
            out = ort_session.run(
                output_names,
                input_dict)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time
            rtf = inference_time / frame_in_ms
            print(f"inference time in ms for frame {total_frame + 1}, frame in ms: {frame_in_ms}, "
                  f"{inference_time} ms. rtf: {rtf}")
            outs.append(torch.from_numpy(out[0]))
            if inference_time > 0.1:
                total_frame += 1
    estimate = torch.cat(outs, 1)

    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / frame_in_ms:.6f}")
    write(estimate.to('cpu'), out_file, sr=16000)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model file')
    parser.add_argument('-n', '--noisy_path', type=str, required=True, help='Noisy audio file or a noisy directory.')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='Out enhanced file or out directory for enhanced files.')
    parser.add_argument('-hs', '--hidden_size', type=int, required=False, default=48,
                        help='Hidden size for dns model.')
    parser.add_argument('-d', '--depth', type=int, required=False, default=4,
                        help='Model depth')
    parser.add_argument('-l', '--total_length', type=int, required=False, default=362, help='Total frame length')
    parser.add_argument('-b', '--resample_buffer', type=int, required=False, default=128, help='resample buffer value')
    parser.add_argument('-s', '--stride', type=int, required=False, default=128, help='Stride value')
    parser.add_argument('-r', '--recurse', type=bool, required=False, default=True,
                        help='Recurse noisy audio directory.')

    args = parser.parse_args()
    onnx_model_file = args.model
    noisy_path = args.noisy_path
    out_path = args.out_path
    hidden = args.hidden_size  # don't change this
    depth = args.depth  # don't change this
    total_length = args.total_length  # don't change this
    resample_buffer = args.resample_buffer  # don't change this
    stride = args.stride  # don't change this
    recurse = args.recurse

    if os.path.isfile(noisy_path):
        noisy, sr = torchaudio.load(str(noisy_path))

        # noisy, sr = torchaudio.load(str(audio_file))
        print(f"inference starts for {noisy_path}")
        parent_out_dir = os.path.dirname(out_path)
        if not os.path.exists(parent_out_dir):
            os.mkdir(parent_out_dir)
        denoise_audio(noisy, onnx_model_file, hidden, depth, stride, total_length, resample_buffer, out_path)
    else:
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        noisy_files = librosa.util.find_files(noisy_path, ext='wav', recurse=recurse)
        for noisy_f in noisy_files:
            name = os.path.basename(noisy_f)
            out_file = os.path.join(out_path, name)
            noisy, sr = torchaudio.load(str(noisy_f))
            print(f"inference starts for {noisy_f}")

            denoise_audio(noisy, onnx_model_file, hidden, depth, stride, total_length, resample_buffer, out_file)
            print(f"inference done for {noisy_f}.")
