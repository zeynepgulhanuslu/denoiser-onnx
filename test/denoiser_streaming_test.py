import argparse

import numpy as np
import torch
import torchaudio
from denoiser.demucs import DemucsStreamer

from denoiser_convert_stream_onnx import DemucsOnnxStreamerTT
from denoiser_inference import init_denoiser_model_from_file, to_numpy
from denoiser_onnx_test import write


def test_denoiser_streamtt(out_file, frame_length, hidden=48):
    streamer = DemucsOnnxStreamerTT(demucs=model, dry=0)
    frames_input = torch.tensor([1])
    frame_num = torch.tensor([1])
    variance = torch.tensor([0.0], dtype=torch.float32)
    depth = streamer.demucs.depth
    lstm_state_1 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    lstm_state_2 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    print(f"lstm state : {lstm_state_1.shape}, {lstm_state_2.shape}")
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

    print(f"streamer stride: {streamer.stride}, total stride: {streamer.demucs.total_stride},"
          f" frame length: {streamer.frame_length}")
    print(f"streamer total length : {streamer.total_length}")
    conv_state_list = [torch.zeros(size) for size in conv_state_sizes]
    conv_state = torch.cat([t.view(1, -1) for t in conv_state_list], dim=1)
    with torch.no_grad():
        pending = noisy
        outs = []
        while pending.shape[1] >= frame_length:
            frame = pending[:, :frame_length]
            print(f'input frame shape:{frame.shape}')

            out = streamer.forward(frame, to_numpy(frames_input), to_numpy(variance),
                                   resample_input_frame, resample_out_frame,
                                   conv_state, lstm_state_1, lstm_state_2)
            outs.append(torch.from_numpy(out[0].numpy()))
            variance = out[2]
            resample_input_frame = out[3]
            resample_out_frame = out[4]
            conv_state = out[5]
            lstm_state_1 = out[6]
            lstm_state_2 = out[7]
            print(f'out shape:{out[0].shape}')
            print("--------------------------/n")
            pending = pending[:, streamer.stride:]

            frames_input.add_(1)
        if pending.shape[1] > 0:
            # Expand the remaining audio with zeros
            last_frame = torch.cat([pending, torch.zeros_like(pending)
            [:, :frame_length - pending.shape[1]]], dim=1)

            out = streamer.forward(last_frame, to_numpy(frames_input), to_numpy(variance),
                                   resample_input_frame, resample_out_frame,
                                   conv_state, lstm_state_1, lstm_state_2)
            outs.append(torch.from_numpy(out[0].numpy()))
        estimate = torch.cat(outs, 1)

        enhanced = estimate / max(estimate.abs().max().item(), 1)
        np_enhanced = np.squeeze(enhanced.detach().squeeze(0).cpu().numpy())
        write(torch.from_numpy(np_enhanced.reshape(1, len(np_enhanced))).to('cpu'), out_file, sr=16000)


def test_default_streaming_model(out_file):
    streamer = DemucsStreamer(demucs=model, dry=0)
    out_rt = []
    frame_size = 128
    with torch.no_grad():
        while noisy.shape[1] > 0:
            out_stream = streamer.feed(noisy[:, :frame_size])
            out_rt.append(out_stream)
            noisy = noisy[:, frame_size:]
            # frame_size = streamer.demucs.total_stride
    out_rt.append(streamer.flush())
    out_rt = torch.cat(out_rt, 1)

    enhanced = out_rt / max(out_rt.abs().max().item(), 1)
    np_enhanced = np.squeeze(enhanced.detach().squeeze(0).cpu().numpy())
    write(torch.from_numpy(np_enhanced.reshape(1, len(np_enhanced))).to('cpu'), out_file, sr=16000)


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
        '-h', '--hidden_size', type=int, default=48, help='Model hidden size')

    args = parser.parse_args()
    torch_model_path = args.model_path
    noisy_path = args.noisy_file
    enhanced_file = args.enhanced_file
    frame_length = args.frame_length
    hidden_size = args.hidden_size
    model = init_denoiser_model_from_file(torch_model_path)
    model.eval()

    noisy, sr = torchaudio.load(str(noisy_path))

