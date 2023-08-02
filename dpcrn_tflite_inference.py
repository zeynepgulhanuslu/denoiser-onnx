import copy
import soundfile as sf
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import argparse
import os
import librosa

def mk_mask_mag(x):
    '''
    magnitude mask
    '''
    [noisy_real, noisy_imag, mag_mask] = x

    enh_mag_real = noisy_real * mag_mask
    enh_mag_imag = noisy_imag * mag_mask
    return enh_mag_real, enh_mag_imag


def mk_mask_pha(x):
    '''
    phase mask
    '''
    [enh_mag_real, enh_mag_imag, pha_cos, pha_sin] = x

    enh_real = enh_mag_real * pha_cos - enh_mag_imag * pha_sin
    enh_imag = enh_mag_real * pha_sin + enh_mag_imag * pha_cos

    return enh_real, enh_imag


def infer_audio_file(model_path, block_len, block_shift, noisy_file, out_file):
    start = time.time()
    # load models
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    #
    # # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # create states for the lstms
    inp = np.zeros([1, 1, 257, 3], dtype=np.float32)
    states = np.zeros(input_details[1]['shape'], dtype=np.float32)
    # load audio file at 16k fs (please change)
    win = np.sin(np.arange(.5, block_len - .5 + 1) / block_len * np.pi)
    #audio, fs = sf.read(noisy_file)
    audio, fs = librosa.load(noisy_file, sr=16000)
    print(f"sampling rate: {fs}")
    # check for sampling rate
    if fs != 16000:
        raise ValueError('This model only supports 16k sampling rate.')
    # preallocate output audio
    out_audio = np.zeros((len(audio)))

    # calculate number of blocks
    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
    time_array = []
    total_frame = 0
    frame_in_ms = (block_len / 16000) * 1000
    # iterate over the number of blcoks
    for idx in range(num_blocks):
        start_time = time.time()
        # shift values and write to buffer
        in_buffer = audio[idx * block_shift:(idx * block_shift) + block_len]
        # calculate fft of input block
        audio_buffer = in_buffer * win
        spec = np.fft.rfft(audio_buffer).astype('complex64')
        spec1 = copy.copy(spec)
        inp[0, 0, :, 0] = spec1.real
        inp[0, 0, :, 1] = spec1.imag
        inp[0, 0, :, 2] = 2 * np.log(abs(spec))

        # set tensors to the model
        interpreter.set_tensor(input_details[1]['index'], states)
        interpreter.set_tensor(input_details[0]['index'], inp)
        # run calculation
        interpreter.invoke()
        # get the output of the model
        output_mask = interpreter.get_tensor(output_details[0]['index'])
        output_cos = interpreter.get_tensor(output_details[1]['index'])
        output_sin = interpreter.get_tensor(output_details[2]['index'])
        states = interpreter.get_tensor(output_details[3]['index'])

        # calculate the ifft
        estimated_real, estimated_imag = mk_mask_mag([spec.real, spec.imag, output_mask])
        enh_real, enh_imag = mk_mask_pha([estimated_real, estimated_imag, output_cos, output_sin])
        estimated_complex = enh_real + 1j * enh_imag
        estimated_block = np.fft.irfft(estimated_complex)
        estimated_block = estimated_block * win
        # write block to output file
        out_audio[block_shift * idx: block_shift * idx + block_len] += np.squeeze(estimated_block)

        end_infer_time = time.time()
        inference_time = end_infer_time - start_time
        rtf = inference_time / frame_in_ms
        print(f"inference time in ms for frame {total_frame + 1}, noisy frame in ms: {frame_in_ms}, "
              f"{inference_time} ms. rtf: {rtf}")
        time_array.append(end_infer_time - start_time)
        total_frame += 1

    # write to .wav file
    sf.write(out_file, out_audio, fs)
    print('Processing Time [ms]:')
    total_inference_time = np.mean(np.stack(time_array)) * 1000
    print(total_inference_time)
    print(time.time() - start)
    print('Processing finished.')
    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / frame_in_ms:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model', type=str,default="dpcrn_quant.tflite", help='Model file as str')
    parser.add_argument('-noisy_path', '--noisy_path', type=str, default="sample.wav",
                        help='Noisy wav file or a directory that contain noisy audio files')
    parser.add_argument('-out_dir', '--out_dir', type=str, default="sample_dpcrn.wav",
                        help='Out directory for saving enhanced audio files.')

    args = parser.parse_args()
    dpcrn_model_file = args.model
    noisy_path = args.noisy_path
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    block_len = 512
    block_shift = 256
    if os.path.isdir(noisy_path):
        noisy_files = librosa.util.find_files(noisy_path, ext="wav")
        for noisy_file in noisy_files:
            name = os.path.basename(noisy_file)
            out_file = os.path.join(out_dir, name)
            infer_audio_file(dpcrn_model_file, block_len, block_shift, noisy_file, out_file)
    else:
        name = os.path.basename(noisy_path)
        out_file = os.path.join(out_dir, name)
        infer_audio_file(dpcrn_model_file, block_len, block_shift, noisy_path, out_file)

