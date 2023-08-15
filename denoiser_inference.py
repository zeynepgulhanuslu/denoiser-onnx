import argparse
import math
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from denoiser.utils import deserialize_model

import torch.autograd.profiler as profiler


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    else:
        return tensor

# this function initialize model from file.
def init_denoiser_model_from_file(model_file):
    print("Loading model from ", model_file)
    pkg = torch.load(model_file, 'cpu')
    if 'model' in pkg:
        if 'best_state' in pkg:
            pkg['model']['state'] = pkg['best_state']
        model = deserialize_model(pkg['model'])
    else:
        model = deserialize_model(pkg)
    model.eval()

    return model


def infer_audio_with_denoiser(model, audio_array):
    out = torch.from_numpy(audio_array.reshape(1, len(audio_array))).to('cpu')
    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            print(f"frame shape:{frame.shape}")
            start_time = time.time()
            estimated_batch = model(out)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            print(f"inference time {inference_time} for frame in ms {block_len}")
            enhanced = estimated_batch
    enhanced = enhanced / max(enhanced.abs().max().item(), 1)
    np_enhanced = np.squeeze(enhanced.detach().squeeze(0).cpu().numpy())
    print(prof.key_averages().table())  # Print layer execution times
    return np_enhanced, inference_time


def get_audio_block(data, sr, block_size, overlap_size):
    block_size = math.floor((sr / 1000.0) * block_size)
    overlap_size = math.floor((sr / 1000.0) * overlap_size)
    num_chunks = int(math.ceil(len(data) / (block_size - overlap_size)))  # Adjusted to round up

    chunks = []

    for i in range(num_chunks):
        start_index = i * (block_size - overlap_size)
        end_index = min((i + 1) * (block_size - overlap_size), len(data))
        chunk = data[start_index:end_index]
        chunks.append(chunk)

    return chunks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model_path', type=str, default="dns48.th",
        help='Model directory')

    parser.add_argument(
        '-a', '--audio_file', type=str, default="sample-small.wav", help='Noisy audio file')

    parser.add_argument(
        '-o', '--out_file', type=str, default="sample-small-dns48-enhanced.wav", help='Out file')

    parser.add_argument(
        '-b', '--block_len', type=int, default=40, help='block size in ms')

    parser.add_argument(
        '-s', '--block_shift', type=int, default=5, help='block shift in ms')

    parser.add_argument(
        '-sr', '--sample_rate', type=int, default=16000, help='sampling rate')

    args = parser.parse_args()
    model_path = args.model_path
    audio_file = args.audio_file
    out_file = args.out_file
    block_len = args.block_len
    block_shift = args.block_shift
    sr = args.sample_rate

    model = init_denoiser_model_from_file(model_path)

    noisy_array, _ = librosa.load(audio_file, sr=sr, mono=True, dtype='float32')
    chunks = get_audio_block(noisy_array, sr, block_len, block_shift)
    outs = []
    total_frame = 0
    total_inference_time = 0

    for chunk in chunks:
        frame = np.array(chunk)
        out, inference_time = infer_audio_with_denoiser(model, frame)
        total_inference_time += inference_time
        outs.append(out)
        total_frame += 1

    estimate = np.concatenate(outs)
    print(f"noisy shape:{noisy_array.shape}, enhanced shape: {estimate.shape}")
    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / block_len:.6f}")
    sf.write(out_file, estimate, samplerate=int(sr))
