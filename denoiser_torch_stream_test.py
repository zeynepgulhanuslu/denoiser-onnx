import argparse
import os
import time

import librosa
import numpy as np
import torch
import torchaudio

from denoiser_convert_stream_onnx import DemucsOnnxStreamerTT
from denoiser_inference import init_denoiser_model_from_file, to_numpy
from denoiser_onnx_test import write


def test_audio_denoising(torch_model_path, noisy, frame_length, hidden, depth=4):
    model = init_denoiser_model_from_file(torch_model_path)

    streamer = DemucsOnnxStreamerTT(model, dry=0)

    frames_input = torch.tensor([1])
    frame_num = torch.tensor([1])
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

    print(
        f"streamer stride: {streamer.stride}, total stride: {streamer.demucs.total_stride}, frame length: {streamer.frame_length}")
    print(f"streamer total length : {streamer.total_length}")
    conv_state_list = [torch.zeros(size) for size in conv_state_sizes]
    conv_state = torch.cat([t.view(1, -1) for t in conv_state_list], dim=1)
    frame_in_ms = (frame_length / 16000) * 1000  # her bir frame in ms cinsinden uzunluğu
    total_duration = (len(noisy) / 16000) * 1000  # ses dosyasının ms cinsinden uzunluğu
    with torch.no_grad():
        outs = []  # çıkışta oluşan audio tensor lerini tutacak
        total_frame = 0
        frames = noisy
        total_inference_time = 0

        # frame length sabit olacak
        while frames.shape[1] >= frame_length:
            frame = frames[:, :frame_length]  # burada ses dosyasının bir kısmı alınır. streaming simulasyonu.
            print(f"frame shape:{frame.shape}")

            start_time = time.time()
            # onnx modelinin çalışmsı
            out_frame, next_frame_num, next_resample_in, next_resample_out, next_conv_state, next_lstm_state_1, next_lstm_state_2 \
                = streamer.forward(frame, to_numpy(frame_num), resample_input_frame, resample_out_frame,
                                   conv_state, lstm_state_1, lstm_state_2)

            print(f'out shape:{out_frame.shape}, input frame shape: {frame.shape}')
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            if inference_time > 0.1:
                total_frame += 1

            total_inference_time += inference_time
            rtf = inference_time / frame_in_ms
            print(f"inference time in ms for frame {total_frame + 1}, noisy frame in ms: {frame_in_ms}, "
                  f"{inference_time} ms. rtf: {rtf}")

            # bundan sonraki çıktılar, bir sonraki frame için input olacak. dolayısıyla atama yapılıyor.
            resample_input_frame = next_resample_in
            resample_out_frame = next_resample_out
            conv_state = next_conv_state
            lstm_state_1 = next_lstm_state_1
            lstm_state_2 = next_lstm_state_2

            # temizlenmiş frame tensor olarak çıkış dizisine eklenir.
            outs.append(out_frame)
            frames = frames[:, stride:]  # bir sonraki frame e gidilir.
            frame_num.add_(1)  # frame sayısı arttırlır.

        # en sonda frame length den küçük kısım kaldıysa, orası da işlenir.
        if frames.shape[1] > 0:
            # Expand the remaining audio with zeros
            last_frame = torch.cat([frames, torch.zeros_like(frames)
            [:, :frame_length - frames.shape[1]]], dim=1)

            start_time = time.time()
            out_frame, next_frame_num, next_resample_in, next_resample_out, next_conv_state, next_lstm_state_1, next_lstm_state_2 \
                = streamer.forward(last_frame, to_numpy(frame_num), resample_input_frame, resample_out_frame,
                                   conv_state, lstm_state_1, lstm_state_2)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time
            rtf = inference_time / frame_in_ms
            print(f"inference time in ms for frame {total_frame + 1}, frame in ms: {frame_in_ms}, "
                  f"{inference_time} ms. rtf: {rtf}")
            outs.append(out_frame)

            if inference_time > 0.1:
                total_frame += 1

        estimate = torch.cat(outs, 1)  # burada çıkıştaki tensor ler bir torch tensor üne dönüştürülür.

    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / frame_in_ms:.6f}")
    enhanced = estimate / max(estimate.abs().max().item(), 1)
    np_enhanced = np.squeeze(enhanced.detach().squeeze(0).cpu().numpy())
    write(torch.from_numpy(np_enhanced.reshape(1, len(np_enhanced))).to('cpu'), out_file, sr=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True, help='model file')
    parser.add_argument('--noisy_audio', type=str, required=True, help='noisy directory')
    parser.add_argument('--out_dir', type=str, required=True, help='out directory')
    parser.add_argument('--stride', type=int, required=False, default=128, help='Stride value')
    parser.add_argument('--hidden', type=int, required=False, default=48, help='Hidden value')
    parser.add_argument('--depth', type=int, required=False, default=4, help='depth value')
    parser.add_argument('--frame_length', type=int, required=False, default=480, help='frame length value')
    parser.add_argument('--resample_buffer', type=int, required=False, default=64, help='resample buffer value')

    args = parser.parse_args()

    model_file = args.model
    noisy_audio = args.noisy_audio
    out_dir = args.out_dir
    hidden = args.hidden
    depth = args.depth
    frame_length = args.frame_length
    resample_buffer = args.resample_buffer
    stride = args.stride
    '''
    if depth == 4:
        frame_length = 480
        resample_buffer = 64
        stride = 256
    else:
        frame_length = 661  # depth = 5 -> 661
        resample_buffer = 256  # depth = 5 -> 256
        stride = 256   # depth = 5  -> 256
    '''
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
        # noisy, sr = torchaudio.load(str(audio_file))
        print(f"inference starts for {noisy_audio}")
        test_audio_denoising(model_file, noisy, frame_length, hidden, depth)

    else:
        noisy_files = librosa.util.find_files(noisy_audio, ext='wav')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for noisy_f in noisy_files:
            name = os.path.basename(noisy_f)
            out_file = os.path.join(out_dir, name)
            noisy, sr = torchaudio.load(str(noisy_f))
            print(f"inference starts for {noisy_f}")
            test_audio_denoising(model_file, noisy, frame_length, hidden, depth)
            print(f"inference done for {noisy_f}.")
