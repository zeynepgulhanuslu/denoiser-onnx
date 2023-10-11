import numpy as np
import torch


def extend_input_frame_shape(input_frame, length):
    """Extends the input frame shape from torch.Size([1, 128]) to torch.Size([1, 480]).

    Args:
      input_frame: A PyTorch tensor of shape torch.Size([1, 128]).

    Returns:
      A PyTorch tensor of shape torch.Size([1, 480]).
    """

    # Create a zero tensor of shape 1, length (the difference between extended length and current length).
    zeros = torch.zeros(1, length)

    # Concatenate the two tensors along the first dimension.
    extended_input_frame = torch.cat((input_frame, zeros), dim=1)

    return extended_input_frame


if __name__ == '__main__':
    stride = 30
    total_length = 100
    in_frame = torch.randn(1, stride)

    print(in_frame)
    print('********************')

    frame_buffer = torch.zeros(1, total_length)

    frame_buffer = torch.cat((frame_buffer[:, stride:], in_frame), 1)
    print('frame buffer:')
    print(frame_buffer)
    print(frame_buffer.shape)
    print('********************')
    '''
    shifted_in_frame = torch.cat((in_frame[:, stride:], torch.zeros(1, stride)), 1)

    out_buffer = torch.cat((frame_buffer[:, stride:], shifted_in_frame), 1)
    print('out buffer:')
    print(out_buffer)
    print(out_buffer.shape)
    print('********************')
    '''
    next_in_frame = torch.randn(1, stride)

    print('next in frame')
    print(next_in_frame)

    print('********************')

    next_buffer = torch.cat((frame_buffer[:, stride:], next_in_frame), 1)
    print('next buffer:')
    print(next_buffer)
    print(next_buffer.shape)
