import argparse
import math

import numpy as np
import onnx
import onnxruntime
import torch
from denoiser.demucs import fast_conv
from denoiser.pretrained import dns48, dns64
from denoiser.resample import downsample2, upsample2
from torch import nn

from denoiser_inference import init_denoiser_model_from_file


class DemucsOnnxStreamerTT(nn.Module):
    """
    Streaming implementation for Demucs. It supports being fed with any amount
    of audio at a time. You will get back as much audio as possible at that
    point.

    Args:
        - demucs (Demucs): Demucs model.
        - dry (float): amount of dry (e.g. input) signal to keep. 0 is maximum
            noise removal, 1 just returns the input signal. Small values > 0
            allows to limit distortions.
        - num_frames (int): number of frames to process at once. Higher values
            will increase overall latency but improve the real time factor.
        - resample_lookahead (int): extra lookahead used for the resampling.
        - resample_buffer (int): size of the buffer of previous inputs/outputs
            kept for resampling.
    """

    def __init__(self, demucs, dry=0, num_frames=1, resample_lookahead=64, resample_buffer=256):
        super().__init__()
        self.demucs = demucs
        self.lstm_state = None
        self.conv_state = None
        self.dry = dry
        self.resample_lookahead = resample_lookahead
        resample_buffer = min(demucs.total_stride, resample_buffer)
        self.resample_buffer = resample_buffer
        self.frame_length = demucs.valid_length(1) + demucs.total_stride * (num_frames - 1)
        self.total_length = self.frame_length + self.resample_lookahead
        self.stride = demucs.total_stride * num_frames
        self.resample_in = torch.zeros(demucs.chin, resample_buffer)
        self.resample_out = torch.zeros(demucs.chin, resample_buffer)
        self.frames = 0
        self.total_time = 0
        self.variance = 0
        self.pending = torch.zeros(demucs.chin, 0)

        bias = demucs.decoder[0][2].bias
        weight = demucs.decoder[0][2].weight
        chin, chout, kernel = weight.shape
        self._bias = bias.view(-1, 1).repeat(1, kernel).view(-1, 1)
        self._weight = weight.permute(1, 2, 0).contiguous()

    def reset_time_per_frame(self):
        self.total_time = 0
        self.frames = 0

    @property
    def time_per_frame(self):
        return self.total_time / self.frames

    def flush(self):
        """
        Flush remaining audio by padding it with zero and initialize the previous
        status. Call this when you have no more input and want to get back the last
        chunk of audio.
        """
        self.lstm_state = None
        self.conv_state = None
        self.initial_frame = 0
        pending_length = self.pending.shape[1]
        padding = torch.zeros(self.demucs.chin, self.total_length)
        frame_num = self.frames if self.frames != 0 else torch.tensor([1])
        variance = self.variance if self.variance != 0 else torch.tensor([0.0])
        out, frame_num, variance_out, resample_in, resample_out, new_conv_state, new_lstm_state_1, new_lstm_state_2 = \
            self.forward(padding, frame_num, variance, self.resample_in, self.resample_out, None, None)

        return out[:, :pending_length]

    def forward(self, frame, frame_num, variance, resample_input_frame, resample_out_frame,
                conv_state=None, lstm_state_1=None, lstm_state_2=None):
        """
        Apply the model to mix using true real-time evaluation.
        Normalization is done online, as is the resampling.
        """
        self.frames = frame_num
        demucs = self.demucs
        resample_buffer = self.resample_buffer
        stride = self.stride
        resample = demucs.resample
        self.lstm_state = (lstm_state_1, lstm_state_2)
        self.conv_state = conv_state
        print(f'streamer total length : {self.total_length}, {self.total_length + self.stride}')
        print(f'streamer frame length : {self.frame_length}')
        dry_signal = frame[:, :stride]
        self.variance = variance
        if demucs.normalize:
            mono = frame.mean(0)
            variance_val = (mono ** 2).mean()
            print(f'local variance val: {variance_val}')
            print(f'self frames: {self.frames}')
            print(f'(1 - 1 / self.frames): {(1 - 1 / self.frames)}')
            print(f'(1 - 1 / self.frames) * self.variance: {(1 - 1 / self.frames) * self.variance}')
            self.variance = variance_val / self.frames + (1 - 1 / self.frames) * self.variance
            frame = frame / (demucs.floor + math.sqrt(self.variance))
        print(f"frame shape before padding : {frame.shape}")
        self.resample_in = resample_input_frame
        padded_frame = torch.cat([self.resample_in, frame], dim=-1)
        self.resample_in[:] = frame[:, stride - resample_buffer:stride]
        frame = padded_frame

        if resample == 4:
            frame = upsample2(upsample2(frame))
        elif resample == 2:
            frame = upsample2(frame)

        frame = frame[:, resample * resample_buffer:]  # remove pre-sampling buffer
        frame = frame[:, :resample * self.frame_length]  # remove extra samples after window
        print(f"frame shape : {frame.shape}")
        out, extra, new_conv_state, new_lstm_state_1, new_lstm_state_2 = self._separate_frame(frame,
                                                                                              conv_state,
                                                                                              lstm_state_1,
                                                                                              lstm_state_2)

        self.resample_out = resample_out_frame

        print(f"conv state shape: {self.conv_state.shape}")
        print(f"lstm state 1 shape: {lstm_state_1.shape}")
        print(f"lstm state 2 shape: {lstm_state_2.shape}")
        print(f"resample shape : {self.resample_out.shape}")
        padded_out = torch.cat([self.resample_out, out, extra], 1)
        print(f'padded out shape: {padded_out.shape}')
        self.resample_out[:] = out[:, -resample_buffer:]  # this will updated also.
        if resample == 4:
            out = downsample2(downsample2(padded_out))
        elif resample == 2:
            out = downsample2(padded_out)
        else:
            out = padded_out
        print(f'out with resample: {out.shape}')
        out = out[:, resample_buffer // resample:]
        out = out[:, :stride]
        print(f'out after stride: {out.shape}')

        if demucs.normalize:
            out *= math.sqrt(self.variance)

        out = self.dry * dry_signal + (1 - self.dry) * out
        self.conv_state = new_conv_state
        return out, frame_num, self.variance, self.resample_in, self.resample_out, \
            new_conv_state, new_lstm_state_1, new_lstm_state_2

    def _separate_frame(self, frame, conv_state=None, lstm_state_1=None, lstm_state_2=None):
        demucs = self.demucs
        skips = []
        next_state = []

        hidden = self.demucs.hidden
        depth = self.demucs.depth
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
        conv_state_sizes_cumsum = [0]
        for size in conv_state_sizes:
            cumsum = conv_state_sizes_cumsum[-1] + np.prod(size)
            conv_state_sizes_cumsum.append(cumsum)

        conv_state_list = [conv_state[..., conv_state_sizes_cumsum[i]:conv_state_sizes_cumsum[i + 1]].view(size) for
                           i, size in enumerate(conv_state_sizes)]

        first = True if self.frames == 0 else False
        print(f"first {first}")
        self.conv_state = conv_state
        stride = self.stride * demucs.resample
        x = frame[None]
        print(f"x shape 1:{x.shape}")
        for idx, encode in enumerate(demucs.encoder):
            stride //= demucs.stride
            length = x.shape[2]
            if idx == demucs.depth - 1:
                x = fast_conv(encode[0], x)
                x = encode[1](x)
                x = fast_conv(encode[2], x)
                x = encode[3](x)
            else:
                if not first:
                    prev = conv_state_list.pop(0)
                    prev = prev[..., stride:]
                    tgt = (length - demucs.kernel_size) // demucs.stride + 1
                    missing = tgt - prev.shape[-1]
                    offset = length - demucs.kernel_size - demucs.stride * (missing - 1)
                    x = x[..., offset:]
                print(f'dtype of x: {x.dtype}')
                x = encode[1](encode[0](x))
                x = fast_conv(encode[2], x)
                x = encode[3](x)
                if not first:
                    x = torch.cat([prev, x], -1)
                next_state.append(x)
            skips.append(x)

        x = x.permute(2, 0, 1)
        x, new_lstm_state = demucs.lstm(x, (lstm_state_1, lstm_state_2))
        print(f"new lstm state:{new_lstm_state[0].shape, new_lstm_state[1].shape}")
        x = x.permute(1, 2, 0)

        extra = None
        for idx, decode in enumerate(demucs.decoder):
            skip = skips.pop(-1)
            x += skip[..., :x.shape[-1]]
            x = fast_conv(decode[0], x)
            x = decode[1](x)

            if extra is not None:
                skip = skip[..., x.shape[-1]:]
                extra += skip[..., :extra.shape[-1]]
                extra = decode[2](decode[1](decode[0](extra)))
            x = decode[2](x)
            next_state.append(x[..., -demucs.stride:] - decode[2].bias.view(-1, 1))
            if extra is None:
                extra = x[..., -demucs.stride:]
            else:
                extra[..., :demucs.stride] += next_state[-1]
            x = x[..., :-demucs.stride]

            if not first:
                prev = conv_state_list.pop(0)

                x[..., :demucs.stride] += prev
            if idx != demucs.depth - 1:
                x = decode[3](x)
                extra = decode[3](extra)

        new_conv_state = torch.cat([t.view(1, -1) for t in next_state], dim=1)
        self.conv_state = new_conv_state
        for cs in self.conv_state:
            print(cs.shape)

        return x[0], extra[0], new_conv_state, new_lstm_state[0], new_lstm_state[1]


def convert_stream_model(onnx_tt_model_path, torch_model_path=None, use_dns_48=False, use_dns64=False,
                         opset_version=13):
    if use_dns_48:
        model = dns48()
    elif torch_model_path is not None:
        model = init_denoiser_model_from_file(torch_model_path)
    elif use_dns64:
        model = dns64()
    else:
        model = dns48()
    model.eval()
    streamer = DemucsOnnxStreamerTT(model, dry=0)
    depth = streamer.demucs.depth
    streamer.eval()
    print(f"total length: {streamer.total_length},"
          f"resample buffer: {streamer.resample_buffer} , "
          f"stride: {streamer.stride},"
          f"depth: {depth}",
          f"hidden: {streamer.demucs.hidden}")

    # x = torch.randn(1, streamer.total_length)
    x = torch.randn(1, 1024)

    frame_num = torch.tensor([2])
    variance_tensor = torch.tensor([0.0], dtype=torch.float32)
    hidden = streamer.demucs.hidden

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

    conv_state_list = [torch.randn(size) for size in conv_state_sizes]
    conv_state = torch.cat([t.view(1, -1) for t in conv_state_list], dim=1)
    lstm_state_1 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    lstm_state_2 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    resample_input_frame = torch.randn(1, streamer.resample_buffer)
    resample_out_frame = torch.randn(1, streamer.resample_buffer)
    print(streamer.resample_buffer)
    with torch.no_grad():
        input_names = ['input', 'frame_num', 'variance', 'resample_input_frame', 'resample_out_frame',
                       'conv_state', 'lstm_state_1', 'lstm_state_2']

        output_names = ['output', 'frame_num', 'variance', 'resample_input_frame',
                        'resample_out_frame', 'conv_state', 'lstm_state_1', 'lstm_state_2']

        torch.onnx.export(streamer,
                          (x, frame_num, variance_tensor, resample_input_frame, resample_out_frame,
                           conv_state, lstm_state_1, lstm_state_2),
                          onnx_tt_model_path,
                          verbose=True,
                          opset_version=opset_version,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes={'input': {0: 'channel', 1: 'sequence_length'},
                                        # variable length axes
                                        'output': {0: 'channel', 1: 'sequence_length'}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--torch_model', type=str, required=True, help='Torch model file')
    parser.add_argument('-o', '--onnx_model', type=str, required=True, help='Onnx model path')
    parser.add_argument('-dns48', type=bool, required=False, default=False,
                        help='True if you want to convert pre-trained dns48 model.')
    parser.add_argument('-dns64', type=bool, required=False, default=False,
                        help='True if you want to convert pre-trained dns64 model.')
    parser.add_argument('-opset', type=int, required=False, default=13,
                        help='onnx export opset version.')

    args = parser.parse_args()
    torch_model_path = args.torch_model
    onnx_tt_model_path = args.onnx_model
    use_dns48 = args.dns48
    use_dns64 = args.dns64
    opset_version = args.opset
    convert_stream_model(onnx_tt_model_path, torch_model_path, use_dns48, use_dns64, opset_version)
