import io
import os
import time
import zipfile

import librosa
import numpy as np
import onnxruntime
import torch
import torchaudio
from scipy.io import wavfile
import torchaudio.transforms as transforms


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


def split_audio_into_frames(audio, block_len, block_shift):
    hop_size = block_len - block_shift

    print("blockSize:", block_len)
    print("overlapSize:", block_shift)
    print("hopSize:", hop_size)

    chunks = []
    i = 0
    data_len = audio.shape[1]
    count = 1
    while i < data_len:
        start = i
        end = min(i + block_len, data_len)
        chunk = audio[:, start:end]
        chunks.append(chunk)
        i += hop_size
        count += 1
    return chunks


def test_audio_denoising(noisy, onnx_tt_model_path, block_len, block_shift, sr, out_file):
    # Bu kısmın iyileştirme yapıp yapmadığına emin değilim.
    session_options = onnxruntime.SessionOptions()
    # session_options.intra_op_num_threads = 1
    # session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_profiling = False
    session_options.profile_file_prefix = "profile_onnx_dns48_buff50_quantized"
    # onnx runtime session oluşturulması
    if is_zip_file(onnx_tt_quantized_model_path):
        model_bytes = load_onnx_from_zip(onnx_tt_model_path)
        ort_session = onnxruntime.InferenceSession(model_bytes, session_options)
    else:
        ort_session = onnxruntime.InferenceSession(onnx_tt_model_path, session_options)

    # burada onnx modelinin giriş değerlerinin isimleri alınır.
    input_audio_frame_name = ort_session.get_inputs()[0].name  # ses dizisini ifade eder

    # onnx modelinin çıktısındaki değerlerin isimleri
    out_frame_name = ort_session.get_outputs()[0].name  # burası çıkış audio array idir.
    total_duration = (len(noisy) / sr) * 1000  # ses dosyasının ms cinsinden uzunluğu
    print(f"noisy shape: {noisy.shape}")
    chunks = split_audio_into_frames(noisy, block_len, block_shift)
    outs = []
    total_frame = 0
    total_inference_time = 0
    frame_in_ms = (block_len / sr) * 1000
    # output_audio = np.zeros(noisy.shape, dtype=np.float32)
    output_audio = np.zeros(noisy.shape, dtype=np.float32)

    for i, chunk in enumerate(chunks):
        print(f"frame number {i + 1}")
        # frame = chunk.expand(1, 1, block_len)
        start_time = time.time()
        # onnx modelinin çalışmsı
        ort_inputs = {input_audio_frame_name: to_numpy(chunk)}
        out = ort_session.run(None, ort_inputs)[0]
        print(f"input chunk shape {to_numpy(chunk).shape}")
        print(f"out shape {out.shape}, type {type(out)}")
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        total_inference_time += inference_time
        rtf = inference_time / frame_in_ms

        offset = i * block_len - i * block_shift  # calculate the offset of the current block, taking overlap into account
        end = min(offset + block_len, output_audio.shape[1])
        slice_length = end - offset
        print(f"offset {offset}, end {end}, slice length {slice_length}")

        if slice_length != block_len:
            out = out[:, :, :slice_length]
        reshaped_out = np.reshape(out, (1, slice_length))
        print(f"reshaped out {reshaped_out.shape}")
        output_audio[:, offset:end] = reshaped_out[:, :slice_length]

        total_frame += 1
        print(f"inference time in ms for frame {total_frame}, noisy frame in ms: {frame_in_ms}, "
              f"{inference_time} ms. rtf: {rtf}")
        print("**********")

    prof = ort_session.end_profiling()

    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / frame_in_ms:.6f}")
    write(torch.from_numpy(output_audio), out_file, sr=16000)  # wav dosyası olarak yazılır.


def test_audio_denoising_with_3sized_input(noisy, onnx_tt_model_path, block_len, block_shift, sr, out_file):
    # Bu kısmın iyileştirme yapıp yapmadığına emin değilim.
    session_options = onnxruntime.SessionOptions()
    # session_options.intra_op_num_threads = 1
    # session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_profiling = False
    session_options.profile_file_prefix = "profile_onnx_dns48_buff50_quantized"
    # onnx runtime session oluşturulması
    if is_zip_file(onnx_tt_quantized_model_path):
        model_bytes = load_onnx_from_zip(onnx_tt_model_path)
        ort_session = onnxruntime.InferenceSession(model_bytes, session_options)
    else:
        ort_session = onnxruntime.InferenceSession(onnx_tt_model_path, session_options)

    # burada onnx modelinin giriş değerlerinin isimleri alınır.
    input_audio_frame_name = ort_session.get_inputs()[0].name  # ses dizisini ifade eder

    # onnx modelinin çıktısındaki değerlerin isimleri
    out_frame_name = ort_session.get_outputs()[0].name  # burası çıkış audio array idir.
    total_duration = (len(noisy) / sr) * 1000  # ses dosyasının ms cinsinden uzunluğu
    print(f"noisy shape: {noisy.shape}")
    chunks = split_audio_into_frames(noisy, block_len, block_shift)
    outs = []
    total_frame = 0
    total_inference_time = 0
    frame_in_ms = (block_len / sr) * 1000
    # output_audio = np.zeros(noisy.shape, dtype=np.float32)
    output_audio = np.zeros(noisy.shape, dtype=np.float32)

    for i, chunk in enumerate(chunks):
        print(f"frame number {i + 1}")
        if chunk.shape[1] < block_len:
            frame = chunk.expand(1, 1, chunk.shape[1])
        else:
            frame = chunk.expand(1, 1, block_len)
        print(f"frame input size: {frame.shape}")
        start_time = time.time()
        # onnx modelinin çalışmsı
        ort_inputs = {input_audio_frame_name: to_numpy(frame)}
        out = ort_session.run(None, ort_inputs)[0]
        block_size = block_len
        if out.shape[2] < block_len:
            block_size = out.shape[2]

        print(f"block len: {block_size}")
        print(f"input chunk shape {to_numpy(frame).shape}")
        print(f"out shape {out.shape}, type {type(out)}")
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        total_inference_time += inference_time
        rtf = inference_time / frame_in_ms

        offset = i * block_size - i * block_shift  # calculate the offset of the current block, taking overlap into account
        end = min(offset + block_size, output_audio.shape[1])
        slice_length = end - offset
        print(f"offset {offset}, end {end}, slice length {slice_length}")

        if slice_length != block_size:
            out = out[:, :, :slice_length]

        reshaped_out = np.reshape(out, (1, slice_length))
        print(f"reshaped out {reshaped_out.shape}")
        output_audio[:, offset:end] = reshaped_out[:, :slice_length]

        total_frame += 1
        print(f"inference time in ms for frame {total_frame}, noisy frame in ms: {frame_in_ms}, "
              f"{inference_time} ms. rtf: {rtf}")
        print("**********")

    prof = ort_session.end_profiling()

    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / frame_in_ms:.6f}")
    write(torch.from_numpy(output_audio), out_file, sr=16000)  # wav dosyası olarak yazılır.
    print(f"output audio shape : {output_audio.shape}")


def test_audio_denoising_with_fix_size_input(noisy, onnx_tt_model_path, block_len, block_shift, sr, out_file):
    # Bu kısmın iyileştirme yapıp yapmadığına emin değilim.
    session_options = onnxruntime.SessionOptions()
    # session_options.intra_op_num_threads = 1
    # session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_profiling = False
    session_options.profile_file_prefix = "profile_onnx_dns48_buff50_quantized"
    # onnx runtime session oluşturulması
    if is_zip_file(onnx_tt_quantized_model_path):
        model_bytes = load_onnx_from_zip(onnx_tt_model_path)
        ort_session = onnxruntime.InferenceSession(model_bytes, session_options)
    else:
        ort_session = onnxruntime.InferenceSession(onnx_tt_model_path, session_options)

    # burada onnx modelinin giriş değerlerinin isimleri alınır.
    input_audio_frame_name = ort_session.get_inputs()[0].name  # ses dizisini ifade eder

    # onnx modelinin çıktısındaki değerlerin isimleri
    out_frame_name = ort_session.get_outputs()[0].name  # burası çıkış audio array idir.
    total_duration = (len(noisy) / sr) * 1000  # ses dosyasının ms cinsinden uzunluğu
    print(f"noisy shape: {noisy.shape}")
    chunks = split_audio_into_frames(noisy, block_len, block_shift)
    outs = []
    total_frame = 0
    total_inference_time = 0
    frame_in_ms = (block_len / sr) * 1000
    # output_audio = np.zeros(noisy.shape, dtype=np.float32)
    output_audio = np.zeros(noisy.shape, dtype=np.float32)

    for i, chunk in enumerate(chunks):
        print(f"frame number {i + 1}")
        if chunk.shape[1] < block_len:
            num_zeros = block_len - chunk.shape[1]
            # Create an array of zeros to be appended to the chunk.
            zeros_to_add = np.zeros((1, num_zeros), dtype=np.float32)
            # Append zeros to the chunk to expand it to block_len.
            chunk = np.hstack((chunk, zeros_to_add))
        frame = np.expand_dims(chunk, axis=0)

        # frame = chunk.expand(1, 1, block_len)
        print(f"frame input size : {frame.shape}")
        start_time = time.time()
        # onnx modelinin çalışmsı
        ort_inputs = {input_audio_frame_name: to_numpy(frame)}
        out = ort_session.run(None, ort_inputs)[0]
        block_size = block_len
        if out.shape[2] < block_len:
            block_size = out.shape[2]

        print(f"block len: {block_size}")
        print(f"input chunk shape {to_numpy(frame).shape}")
        print(f"out shape {out.shape}, type {type(out)}")
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        total_inference_time += inference_time
        rtf = inference_time / frame_in_ms

        offset = i * block_size - i * block_shift  # calculate the offset of the current block, taking overlap into account
        end = min(offset + block_size, output_audio.shape[1])
        slice_length = end - offset
        print(f"offset {offset}, end {end}, slice length {slice_length}")

        if slice_length != block_size:
            out = out[:, :, :slice_length]

        reshaped_out = np.reshape(out, (1, slice_length))
        print(f"reshaped out {reshaped_out.shape}")
        output_audio[:, offset:end] = reshaped_out[:, :slice_length]

        total_frame += 1
        print(f"inference time in ms for frame {total_frame}, noisy frame in ms: {frame_in_ms}, "
              f"{inference_time} ms. rtf: {rtf}")
        print("**********")

    prof = ort_session.end_profiling()

    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / frame_in_ms:.6f}")
    write(torch.from_numpy(output_audio), out_file, sr=16000)  # wav dosyası olarak yazılır.


if __name__ == '__main__':
    onnx_tt_quantized_model_path = 'D:/zeynep/data/noise-cancelling/denoiser/dns/hidden=36/dns36_buffer=480.onnx'
    # zip modeli kullanmak isterseniz.
    noisy_audio = 'D:/zeynep/data/noise-cancelling/romeda/records.30.05.2023-16k-test/noisy/'
    out_dir = 'D:/zeynep/data/noise-cancelling/romeda/records.30.05.2023-16k-test/dns36-onnx-buffer=480/'
    block_len = 480  # her bir frame uzunluğu
    block_shift = 160  # shift uzunluğu

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if os.path.isfile(noisy_audio):
        noisy, sr = torchaudio.load(str(noisy_audio))
        name = os.path.basename(noisy_audio)
        out_file = out_dir + name
        print(f"inference starts for {noisy_audio}")

        test_audio_denoising_with_3sized_input(noisy, onnx_tt_quantized_model_path, block_len,
                                               block_shift, sr, out_file)
    else:
        noisy_files = librosa.util.find_files(noisy_audio, ext='wav')
        for noisy_f in noisy_files:
            name = os.path.basename(noisy_f)
            out_file = out_dir + name
            noisy, sr = torchaudio.load(str(noisy_f))
            print(f"inference starts for {noisy_f}")

            test_audio_denoising_with_3sized_input(noisy, onnx_tt_quantized_model_path, block_len,
                                                   block_shift, sr, out_file)

            print(f"inference done for {noisy_f}.")