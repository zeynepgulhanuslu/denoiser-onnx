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


def denoise_audio(model, noisy, out_file, sr=16000):
    streamer = DemucsOnnxStreamerTT(demucs=model, dry=0)
    frame_buffer = torch.zeros(1, streamer.total_length)
    print(f'streamer buffer length: {streamer.total_length}')
    print(f'streamer frame length:{streamer.frame_length}, resample lookahead: {streamer.resample_lookahead}')
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

    frame_in_ms = (streamer.total_length / sr) * 1000  # her bir frame in ms cinsinden uzunluğu
    print(f'frame in ms:{frame_in_ms}')
    print(f'streamer stride: {streamer.stride}')
    rtf_exceed_count = 0
    total_frame = 0
    total_inference_time = 0
    with torch.no_grad():
        pending = noisy
        outs = []
        while pending.shape[1] >= streamer.stride:
            frame = pending[:, :streamer.stride]
            start_time = time.time()
            out, out_frame_buffer, out_frame_num, out_variance, out_resample_input_frame, out_resample_out_frame, \
            out_h, out_c, out_conv_state = streamer.forward(
                frame, frame_buffer,
                frames_input, variance,
                resample_input_frame,
                resample_out_frame, h,
                c, conv_state)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            if inference_time > 0.1:
                total_frame += 1
            total_inference_time += inference_time
            rtf = inference_time / frame_in_ms
            print(f"inference time in ms for frame {total_frame + 1}, noisy frame in ms: {frame_in_ms}, "
                  f"{inference_time} ms. rtf: {rtf}")
            if rtf > 1:
                rtf_exceed_count += 1
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
            start_time = time.time()
            out, out_frame_buffer, out_frame_num, out_variance, out_resample_input_frame, out_resample_out_frame, \
            out_h, out_c, out_conv_state = streamer.forward(
                last_frame, frame_buffer,
                frames_input, variance,
                resample_input_frame,
                resample_out_frame, h,
                c, conv_state)

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time
            rtf = inference_time / frame_in_ms
            print(f"inference time in ms for frame {total_frame + 1}, frame in ms: {frame_in_ms}, "
                  f"{inference_time} ms. rtf: {rtf}")
            if rtf > 1:
                rtf_exceed_count += 1
            if inference_time > 0.1:
                total_frame += 1
            outs.append(out)

        estimate = torch.cat(outs, 1)

    write(estimate.to('cpu'), out_file, sr=sr)
    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.4f}")
    avg_rtf = average_inference_time / frame_in_ms
    print(f"average rtf : {avg_rtf :.4f}")
    print(f"number of frames that exceed rtf: {rtf_exceed_count}")
    print(f"average exceed frames : {rtf_exceed_count / total_frame :.4f}")
    return average_inference_time, avg_rtf, rtf_exceed_count, total_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=True, help='model file')
    parser.add_argument('-n', '--noisy_audio', type=str, required=True, help='noisy directory or file')
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='out directory or file')
    parser.add_argument('-sr', '--sample-rate', type=int, required=False, default=16000, help='sample rate')

    args = parser.parse_args()

    model_file = args.model
    noisy_audio = args.noisy_audio
    out_dir = args.out_dir
    sr = args.sample_rate

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
        denoise_audio(model_file, noisy, out_file, sr)

    else:
        noisy_files = librosa.util.find_files(noisy_audio, ext='wav')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        audio_file_count = len(noisy_files)
        total_avg_rtf = 0
        total_duration = 0
        total_frame = 0
        rtf_exceed_total = 0
        total_avg_inference = 0
        for noisy_f in noisy_files:
            name = os.path.basename(noisy_f)
            out_file = os.path.join(out_dir, name)
            noisy, sr = torchaudio.load(str(noisy_f))

            duration = noisy.shape[1] / sr
            print(f"inference starts for {noisy_f}")
            total_duration += duration
            avg_inference_time, avg_rtf, rtf_exceed_count, frame_count = denoise_audio(model, noisy, out_file, sr)

            print(f"inference done for {noisy_f}.")
            total_avg_inference += avg_inference_time

            rtf_exceed_total += rtf_exceed_count
            total_frame += frame_count
            total_avg_rtf += avg_rtf

            print(f"inference done for {noisy_f}.")
        print(f"total duration in minutes : {total_duration / 60:.3f}")
        print(f"total duration in hours : {total_duration / 3600:.3f}")
        print(f"avg rtf for all files : {total_avg_rtf / audio_file_count:.3f}")
        print(f"when rtf exceed 1 --> frame count : {rtf_exceed_total}, total frame count: {total_frame},"
              f"ratio : {rtf_exceed_total / total_frame:.6f}")
        print(f"total audio files:{audio_file_count}")
        print(f"Audio duration avg: {total_duration / audio_file_count}")
        print(f"Avg inference time: {total_avg_inference / audio_file_count}")
