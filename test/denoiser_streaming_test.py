import numpy as np
import torch
import torchaudio

from denoiser_convert_stream_onnx import DemucsOnnxStreamerTT
from denoiser_inference import init_denoiser_model_from_file, to_numpy
from denoiser_onnx_test import write

if __name__ == "__main__":
    torch_model_path = 'D:/zeynep/data/noise-cancelling/denoiser/dns/exp_hidden=48_depth=4_stride=128_resample=2_kernel=4/best.th'

    model = init_denoiser_model_from_file(torch_model_path)
    # model = dns48()
    model.eval()
    hidden = 48
    noisy_path = '../sample-small.wav'
    out_file = '../sample-small-stream.wav'
    noisy, sr = torchaudio.load(str(noisy_path))
    streamer = DemucsOnnxStreamerTT(demucs=model, dry=0)
    streamer.demucs.valid_length(1)
    print(f'streamer stride:{streamer.stride}, {streamer.demucs.total_stride}')
    print(f'valid length: {streamer.demucs.valid_length(0)}, {streamer.demucs.valid_length(1)}')
    frames_input = torch.tensor([1])
    frame_num = torch.tensor([1])
    variance = torch.tensor([0.0], dtype=torch.float32)
    depth = streamer.demucs.depth
    lstm_state_1 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    lstm_state_2 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    print(f"lstm state : {lstm_state_1.shape}, {lstm_state_2.shape}")
    resample_input_frame = torch.zeros(1, streamer.resample_buffer)
    resample_out_frame = torch.zeros(1, streamer.resample_buffer)
    frame_length = 128
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
