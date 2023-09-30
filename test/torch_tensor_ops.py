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
    tensor = torch.randn(1, 128)
    len = 480 - 128

    extended_input_frame = extend_input_frame_shape(tensor, len)
    print(f'extended_input_shape :{extended_input_frame.shape}')
    print(f'frame input :{tensor}')
    print(f'extended_input_shape :{extended_input_frame}')