import argparse
import os
import time

import librosa
import numpy as np
import torch
import torchaudio

from demucs_streamer import DemucsOnnxStreamerTT
from denoiser_inference import init_denoiser_model_from_file, to_numpy
from denoiser_onnx_test import write


def denoise_audio(model, noisy, out_file):
    streamer = DemucsOnnxStreamerTT(model, dry=0)
    frame_buffer = torch.zeros(1, streamer.total_length)
    frames_input = torch.tensor([1])
    hidden = streamer.demucs.hidden
    depth = streamer.demucs.depth

    variance = torch.tensor([0.0], dtype=torch.float32)
    h = torch.zeros(2, 1, hidden * 2 ** (depth - 1), dtype=torch.float32)
    c = torch.zeros(2, 1, hidden * 2 ** (depth - 1), dtype=torch.float32)
    resample_input_frame = torch.zeros(1, streamer.resample_buffer)
    resample_out_frame = torch.zeros(1, streamer.resample_buffer)

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

    with torch.no_grad():
        pending = noisy
        outs = []
        while pending.shape[1] >= streamer.stride:
            frame = pending[:, :streamer.stride]
            out, out_frame_buffer, out_frame_num, out_variance, out_resample_input_frame, out_resample_out_frame, \
            out_h, out_c, out_conv_state = streamer.forward(
                frame, frame_buffer,
                frames_input, variance,
                resample_input_frame,
                resample_out_frame, h,
                c, conv_state)

            outs.append(out)
            pending = pending[:, streamer.stride:]
            frames_input.add_(1)
            frame_buffer = out_frame_buffer
            variance = out_variance
            resample_input_frame = out_resample_input_frame
            resample_out_frame = out_resample_out_frame
            h = out_h
            c = out_c
            conv_state = out_conv_state

        if pending.shape[1] > 0:
            # Expand the remaining audio with zeros
            zeros_needed = streamer.stride - pending.shape[1]
            zeros_to_add = torch.zeros(1, zeros_needed)
            last_frame = torch.cat([pending, zeros_to_add], dim=1)

            out, out_frame_buffer, out_frame_num, out_variance, out_resample_input_frame, out_resample_out_frame, \
            out_h, out_c, out_conv_state = streamer.forward(
                last_frame, frame_buffer,
                frames_input, variance,
                resample_input_frame,
                resample_out_frame, h,
                c, conv_state)
            outs.append(out)

        estimate = torch.cat(outs, 1)

    write(estimate.to('cpu'), out_file, sr=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True, help='model file')
    parser.add_argument('-n', '--noisy_audio', type=str, required=True, help='noisy directory or file')
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='out directory or file')

    args = parser.parse_args()

    model_file = args.model
    noisy_audio = args.noisy_audio
    out_dir = args.out_dir

    model = init_denoiser_model_from_file(model_file)
    model.eval()
    if os.path.isfile(noisy_audio):
        noisy, sr = torchaudio.load(str(noisy_audio))

        name = os.path.basename(noisy_audio)
        if out_dir.endswith('.wav'):
            parent_dir = os.path.dirname(out_dir)
            if not os.path.exists(parent_dir):
                os.mkdir(parent_dir)
            out_file = out_dir
        else:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            out_file = os.path.join(out_dir, name)
        print(f"inference starts for {noisy_audio}")
        denoise_audio(model_file, noisy, out_file)

    else:
        noisy_files = librosa.util.find_files(noisy_audio, ext='wav')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for noisy_f in noisy_files:
            name = os.path.basename(noisy_f)
            out_file = os.path.join(out_dir, name)
            noisy, sr = torchaudio.load(str(noisy_f))

            print(f"inference starts for {noisy_f}")
            denoise_audio(model_file, noisy, out_file)

            print(f"inference done for {noisy_f}.")
