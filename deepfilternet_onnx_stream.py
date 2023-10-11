import argparse
import os
import time

import librosa
import onnxruntime as ort
import torch
import torchaudio


def split_audio_into_frames(audio, block_len):
    chunks = []
    i = 0
    data_len = audio.shape[1]
    while i < data_len:
        start = i
        end = min(i + block_len, data_len)
        chunk = audio[:, start:end]

        # Pad the last chunk with zeros if it is smaller than the frame size
        if end - start < block_len:
            chunk = torch.nn.functional.pad(chunk, (0, block_len - (end - start)), 'constant', value=0)

        chunks.append(chunk)
        i += block_len

    return chunks


def generate_onnx_features(input_features):
    return {
        x: y.detach().cpu().numpy()
        for x, y in zip(INPUT_NAMES, input_features)
    }


def denoise_audio_streaming(audio_file, frame_size, enhanced_file):
    noisy_audio, sr = torchaudio.load(audio_file, channels_first=True)
    frame_in_ms = (frame_size / sr) * 1000
    chunked_audio = split_audio_into_frames(noisy_audio, frame_size)

    output_frames = []
    total_inference_time = 0

    total_frame = 0
    states_onnx = torch.zeros(45304, device='cpu')
    atten_lim_db = torch.tensor(0.0, device='cpu')
    rtf_exceed_count = 0
    for i, chunk in enumerate(chunked_audio):
        start_time = time.time()
        frame = chunk.squeeze(0)
        print(f'frame shape: {frame.shape}')
        features = generate_onnx_features([frame, states_onnx, atten_lim_db])

        output_onnx = ort_session.run(
            OUTPUT_NAMES,
            features)

        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        total_inference_time += inference_time
        rtf = inference_time / frame_in_ms
        print(f"inference time in ms for frame {i + 1}, noisy frame in ms: {frame_in_ms}, "
              f"{inference_time} ms. rtf: {rtf}")
        if rtf >= 1:
            rtf_exceed_count += 1
            print(f'rtf exceed real time requirements: {rtf}, inference time: {inference_time} ms')

        out_enhanced_frame = output_onnx[0]
        out_states = output_onnx[1]
        states_onnx = torch.from_numpy(out_states)
        output_frames.append(torch.from_numpy(out_enhanced_frame))
        total_frame += 1
    average_inference_time = total_inference_time / total_frame

    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / frame_in_ms:.6f}")
    print(f'exceed rtf count: {rtf_exceed_count}')
    print(f'exceed rtf ratio: {rtf_exceed_count / total_frame :.2f}')
    enhanced_audio = torch.cat(output_frames).unsqueeze(0)
    estimate = enhanced_audio.detach().cpu()

    torchaudio.save(
        enhanced_file, estimate, sr,
        encoding="PCM_S", bits_per_sample=16
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model_file', type=str, required=True, help='Onnx model file')
    parser.add_argument('-n', '--noisy_path', type=str, required=True, help='Noisy audio file or a noisy directory.')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='Out enhanced file or out directory for enhanced files.')
    parser.add_argument('-b', '--block_len', type=int, default=480, help='block size in samples')

    args = parser.parse_args()
    onnx_model_file = args.model_file
    noisy_path = args.noisy_path
    out_path = args.out_path
    block_len = args.block_len

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = onnx_model_file
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    ort_session = ort.InferenceSession(onnx_model_file, sess_options, providers=['CPUExecutionProvider'])
    INPUT_NAMES = [
        'input_frame',
        'states',
        'atten_lim_db'
    ]
    OUTPUT_NAMES = [
        'enhanced_audio_frame', 'out_states', 'lsnr'
    ]

    if os.path.isdir(noisy_path):
        noisy_files = librosa.util.find_files(noisy_path, ext='wav')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        for noisy_file in noisy_files:
            name = os.path.basename(noisy_file)
            enhanced_file = os.path.join(out_path, name)
            denoise_audio_streaming(noisy_file, block_len, enhanced_file)

    else:
        if os.path.isdir(out_path):
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            name = os.path.basename(noisy_path)
            enhanced_file = os.path.join(out_path, name)
        else:
            parent_dir = os.path.dirname(out_path)
            if not os.path.exists(parent_dir):
                os.mkdir(parent_dir)
            enhanced_file = out_path

        denoise_audio_streaming(noisy_path, block_len, enhanced_file)
