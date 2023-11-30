import argparse
import os

import numpy as np
import torch
import torchaudio
from denoiser.demucs import DemucsStreamer, Demucs
from denoiser.resample import downsample2, upsample2

from demucs_streamer import DemucsOnnxStreamerTT
from denoiser_inference import init_denoiser_model_from_file, to_numpy
from denoiser_onnx_test import write


def test_audio_denoising_pytorch(model, audio_file, out_file):
    model.eval()
    streamer = DemucsOnnxStreamerTT(model, dry=0)
    print(f'stride: {streamer.demucs.total_stride}, frame length: {streamer.frame_length}, total frame length: {streamer.demucs.valid_length},'
          f'resample buffer {streamer.resample_buffer}')
    noisy, sr = torchaudio.load(str(audio_file))
    frame_num = 1
    h = None
    with torch.no_grad():
        pending = noisy
        outs = []
        while pending.shape[1] >= streamer.stride:
            frame = pending[:, :streamer.stride].numpy()
            out, h = streamer(torch.tensor(frame), frame_num, h)
            outs.append(out[0])
            pending = pending[:, streamer.stride:]
            frame_num += 1
        estimate = torch.cat(outs, 0)
    enhanced = estimate / max(estimate.abs().max().item(), 1)
    np_enhanced = np.squeeze(enhanced.detach().squeeze(0).cpu().numpy())
    print(f"out shape: {estimate.shape}, type:{type(estimate)}")

    write(torch.from_numpy(np_enhanced.reshape(1, len(np_enhanced))).to('cpu'), out_file, sr=sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model_path', type=str, default="dns48.th",
        help='Model directory')

    parser.add_argument(
        '-n', '--noisy_file', type=str, default="sample-small.wav", help='Noisy audio file')

    parser.add_argument(
        '-e', '--enhanced_file', type=str, default="sample-small-dns48-enhanced.wav", help='Out file')

    parser.add_argument(
        '-f', '--frame_length', type=int, default=480, help='frame length in ms')

    parser.add_argument(
        '-hs', '--hidden_size', type=int, default=48, help='Model hidden size')

    args = parser.parse_args()
    torch_model_path = args.model_path
    noisy_path = args.noisy_file
    enhanced_file = args.enhanced_file
    frame_length = args.frame_length
    hidden_size = args.hidden_size
    model = init_denoiser_model_from_file(torch_model_path)
    model.eval()

    test_audio_denoising_pytorch(model, noisy_path,  enhanced_file)
