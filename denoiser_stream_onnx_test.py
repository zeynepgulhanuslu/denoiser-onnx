import argparse
import io
import os
import time
import zipfile

import librosa
import numpy as np
import onnxruntime
import torch
import torchaudio


# torch-> numpy dönüşümü
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    else:
        return tensor


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping

    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def load_onnx_from_zip(onnx_tt_model_path):
    # .zip arşivini yükleyin (örnek olarak "model.zip" adını varsayalım)
    with open(onnx_tt_model_path, "rb") as file:
        zip_file = file.read()

    # Zip dosyasını bellekte açın
    zip_buffer = io.BytesIO(zip_file)

    # Zip arşivini açın
    with zipfile.ZipFile(zip_buffer, "r") as archive:
        # ONNX model dosyasını yükleyin
        model_bytes = archive.read("model.onnx")

    return model_bytes


def is_zip_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() == ".zip"


def test_audio_denoising_with_variance(noisy, onnx_tt_model_path, hidden, out_file, depth=4):
    # Bu kısmın iyileştirme yapıp yapmadığına emin değilim.
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_profiling = False
    session_options.profile_file_prefix = "profile_streamtt_1thread"
    # onnx runtime session oluşturulması
    if is_zip_file(onnx_tt_model_path):
        model_bytes = load_onnx_from_zip(onnx_tt_model_path)
        ort_session = onnxruntime.InferenceSession(model_bytes, session_options)
    else:
        ort_session = onnxruntime.InferenceSession(onnx_tt_model_path, session_options)

    # burada onnx modelinin giriş değerlerinin isimleri alınır.
    input_audio_frame_name = ort_session.get_inputs()[0].name  # ses dizisini ifade eder
    frame_num_name = ort_session.get_inputs()[1].name  # frame numarasıdır, her bir frame de artacak
    variance_input_name = ort_session.get_inputs()[2].name
    # modelin içerisinde güncellenen parametreler aşağıdaki gibidir.
    resample_input_frame_name = ort_session.get_inputs()[3].name
    resample_out_frame_name = ort_session.get_inputs()[4].name
    conv_state_name = ort_session.get_inputs()[5].name
    lstm_state_1_name = ort_session.get_inputs()[6].name
    lstm_state_2_name = ort_session.get_inputs()[7].name

    # conv state leri 9 tane tensor den oluşan bir array dir. Burada size ları belirtilmiştir.
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

    conv_state_list = [torch.zeros(size) for size in conv_state_sizes]  # içeriği sıfır olan torch array oluşturulur.
    conv_state = torch.cat([t.view(1, -1) for t in conv_state_list], dim=1)

    lstm_state_1 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    lstm_state_2 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    frame_num = torch.tensor([1])  # frame numarası
    variance_input = torch.tensor([0.0], dtype=torch.float32)
    resample_input_frame = torch.zeros(1, resample_buffer)
    resample_out_frame = torch.zeros(1, resample_buffer)

    # onnx modelinin çıktısındaki değerlerin isimleri
    out_frame_name = ort_session.get_outputs()[0].name  # burası çıkış audio array idir.
    out_num_frame_name = ort_session.get_outputs()[1].name

    # aşağıdakiler her adımda güncellenen değerlerdir. Bu çıkışlar bir sonraki adımda giriş olarak verilecektir.
    out_variance_name = ort_session.get_outputs()[2].name
    out_resample_in_frame = ort_session.get_outputs()[3].name
    out_resample_frame = ort_session.get_outputs()[4].name
    out_conv = ort_session.get_outputs()[5].name
    out_lstm_1 = ort_session.get_outputs()[6].name
    out_lstm_2 = ort_session.get_outputs()[7].name

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
            # onnx modelinin giriş değerleri.
            print(f'variance input value: {variance_input[0]:.2f}')
            input_values = {
                input_audio_frame_name: to_numpy(frame),
                frame_num_name: to_numpy(frame_num),
                variance_input_name: to_numpy(variance_input),
                resample_input_frame_name: to_numpy(resample_input_frame),
                resample_out_frame_name: to_numpy(resample_out_frame),
                conv_state_name: to_numpy(conv_state),
                lstm_state_1_name: to_numpy(lstm_state_1),
                lstm_state_2_name: to_numpy(lstm_state_2)
            }

            start_time = time.time()
            # onnx modelinin çalışmsı
            out = ort_session.run([out_frame_name, out_num_frame_name, out_variance_name, out_resample_in_frame,
                                   out_resample_frame, out_conv, out_lstm_1, out_lstm_2], input_values)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            if inference_time > 0.1:
                total_frame += 1

            total_inference_time += inference_time
            rtf = inference_time / frame_in_ms
            print(f"inference time in ms for frame {total_frame + 1}, noisy frame in ms: {frame_in_ms}, "
                  f"{inference_time} ms. rtf: {rtf}")

            output_np = out[0]  # enhanced out audio
            # bundan sonraki çıktılar, bir sonraki frame için input olacak. dolayısıyla atama yapılıyor.
            variance_input = out[2]
            resample_input_frame = out[3]
            resample_out_frame = out[4]
            conv_state = torch.from_numpy(out[5])
            lstm_state_1 = out[6]
            lstm_state_2 = out[7]
            # temizlenmiş frame tensor olarak çıkış dizisine eklenir.
            outs.append(torch.from_numpy(output_np))
            frames = frames[:, stride:]  # bir sonraki frame e gidilir.
            frame_num.add_(1)  # frame sayısı arttırlır.

            print(f'variance out length:{len(out[2])}, variance out value: {out[2][0]:.2f}')

        # en sonda frame length den küçük kısım kaldıysa, orası da işlenir.
        if frames.shape[1] > 0:
            # Expand the remaining audio with zeros
            last_frame = torch.cat([frames, torch.zeros_like(frames)
            [:, :frame_length - frames.shape[1]]], dim=1)

            input_values = {
                input_audio_frame_name: to_numpy(last_frame),
                frame_num_name: to_numpy(frame_num),
                variance_input_name: to_numpy(variance_input),
                resample_input_frame_name: to_numpy(resample_input_frame),
                resample_out_frame_name: to_numpy(resample_out_frame),
                conv_state_name: to_numpy(conv_state),
                lstm_state_1_name: to_numpy(lstm_state_1),
                lstm_state_2_name: to_numpy(lstm_state_2)
            }

            start_time = time.time()
            out = ort_session.run([out_frame_name, out_num_frame_name, out_variance_name, out_resample_in_frame,
                                   out_resample_frame, out_conv, out_lstm_1, out_lstm_2], input_values)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time
            rtf = inference_time / frame_in_ms
            print(f"inference time in ms for frame {total_frame + 1}, frame in ms: {frame_in_ms}, "
                  f"{inference_time} ms. rtf: {rtf}")
            outs.append(torch.from_numpy(out[0]))

            if inference_time > 0.1:
                total_frame += 1

        estimate = torch.cat(outs, 1)  # burada çıkıştaki tensor ler bir torch tensor üne dönüştürülür.
    # Run the model
    prof = ort_session.end_profiling()

    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / frame_in_ms:.6f}")
    enhanced = estimate / max(estimate.abs().max().item(), 1)
    np_enhanced = np.squeeze(enhanced.detach().squeeze(0).cpu().numpy())
    write(torch.from_numpy(np_enhanced.reshape(1, len(np_enhanced))).to('cpu'), out_file, sr=16000)


def test_audio_denoising(noisy, onnx_tt_model_path, hidden, out_file, depth=4):
    # Bu kısmın iyileştirme yapıp yapmadığına emin değilim.
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_profiling = False
    session_options.profile_file_prefix = "profile_streamtt_1thread"
    # onnx runtime session oluşturulması
    if is_zip_file(onnx_tt_model_path):
        model_bytes = load_onnx_from_zip(onnx_tt_model_path)
        ort_session = onnxruntime.InferenceSession(model_bytes, session_options)
    else:
        ort_session = onnxruntime.InferenceSession(onnx_tt_model_path, session_options)

    # burada onnx modelinin giriş değerlerinin isimleri alınır.
    input_audio_frame_name = ort_session.get_inputs()[0].name  # ses dizisini ifade eder
    frame_num_name = ort_session.get_inputs()[1].name  # frame numarasıdır, her bir frame de artacak

    # modelin içerisinde güncellenen parametreler aşağıdaki gibidir.
    resample_input_frame_name = ort_session.get_inputs()[2].name
    resample_out_frame_name = ort_session.get_inputs()[3].name
    conv_state_name = ort_session.get_inputs()[4].name
    lstm_state_1_name = ort_session.get_inputs()[5].name
    lstm_state_2_name = ort_session.get_inputs()[6].name

    # conv state leri 9 tane tensor den oluşan bir array dir. Burada size ları belirtilmiştir.
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

    conv_state_list = [torch.zeros(size) for size in conv_state_sizes]  # içeriği sıfır olan torch array oluşturulur.
    conv_state = torch.cat([t.view(1, -1) for t in conv_state_list], dim=1)

    lstm_state_1 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    lstm_state_2 = torch.randn(2, 1, hidden * 2 ** (depth - 1))
    frame_num = torch.tensor([1])  # frame numarası
    resample_input_frame = torch.zeros(1, resample_buffer)
    resample_out_frame = torch.zeros(1, resample_buffer)

    # onnx modelinin çıktısındaki değerlerin isimleri
    out_frame_name = ort_session.get_outputs()[0].name  # burası çıkış audio array idir.
    out_num_frame_name = ort_session.get_outputs()[1].name
    # aşağıdakiler her adımda güncellenen değerlerdir. Bu çıkışlar bir sonraki adımda giriş olarak verilecektir.
    out_resample_in_frame = ort_session.get_outputs()[2].name
    out_resample_frame = ort_session.get_outputs()[3].name
    out_conv = ort_session.get_outputs()[4].name
    out_lstm_1 = ort_session.get_outputs()[5].name
    out_lstm_2 = ort_session.get_outputs()[6].name

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
            # onnx modelinin giriş değerleri.
            input_values = {
                input_audio_frame_name: to_numpy(frame),
                frame_num_name: to_numpy(frame_num),
                resample_input_frame_name: to_numpy(resample_input_frame),
                resample_out_frame_name: to_numpy(resample_out_frame),
                conv_state_name: to_numpy(conv_state),
                lstm_state_1_name: to_numpy(lstm_state_1),
                lstm_state_2_name: to_numpy(lstm_state_2)
            }

            start_time = time.time()
            # onnx modelinin çalışmsı
            out = ort_session.run([out_frame_name, out_num_frame_name, out_resample_in_frame, out_resample_frame,
                                   out_conv, out_lstm_1, out_lstm_2], input_values)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            if inference_time > 0.1:
                total_frame += 1

            total_inference_time += inference_time
            rtf = inference_time / frame_in_ms
            print(f"inference time in ms for frame {total_frame + 1}, noisy frame in ms: {frame_in_ms}, "
                  f"{inference_time} ms. rtf: {rtf}")

            output_np = out[0]  # enhanced out audio
            # bundan sonraki çıktılar, bir sonraki frame için input olacak. dolayısıyla atama yapılıyor.
            resample_input_frame = out[2]
            resample_out_frame = out[3]
            conv_state = torch.from_numpy(out[4])
            lstm_state_1 = out[5]
            lstm_state_2 = out[6]
            # temizlenmiş frame tensor olarak çıkış dizisine eklenir.
            outs.append(torch.from_numpy(output_np))
            frames = frames[:, stride:]  # bir sonraki frame e gidilir.
            frame_num.add_(1)  # frame sayısı arttırlır.

        # en sonda frame length den küçük kısım kaldıysa, orası da işlenir.
        if frames.shape[1] > 0:
            # Expand the remaining audio with zeros
            last_frame = torch.cat([frames, torch.zeros_like(frames)
            [:, :frame_length - frames.shape[1]]], dim=1)

            input_values = {
                input_audio_frame_name: to_numpy(last_frame),
                frame_num_name: to_numpy(frame_num),
                resample_input_frame_name: to_numpy(resample_input_frame),
                resample_out_frame_name: to_numpy(resample_out_frame),
                conv_state_name: to_numpy(conv_state),
                lstm_state_1_name: to_numpy(lstm_state_1),
                lstm_state_2_name: to_numpy(lstm_state_2)
            }

            start_time = time.time()
            out = ort_session.run([out_frame_name, out_num_frame_name, out_resample_in_frame, out_resample_frame,
                                   out_conv, out_lstm_1, out_lstm_2], input_values)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time
            rtf = inference_time / frame_in_ms
            print(f"inference time in ms for frame {total_frame + 1}, frame in ms: {frame_in_ms}, "
                  f"{inference_time} ms. rtf: {rtf}")
            outs.append(torch.from_numpy(out[0]))

            if inference_time > 0.1:
                total_frame += 1

        estimate = torch.cat(outs, 1)  # burada çıkıştaki tensor ler bir torch tensor üne dönüştürülür.
    # Run the model
    prof = ort_session.end_profiling()

    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / frame_in_ms:.6f}")
    enhanced = estimate / max(estimate.abs().max().item(), 1)
    np_enhanced = np.squeeze(enhanced.detach().squeeze(0).cpu().numpy())
    write(torch.from_numpy(np_enhanced.reshape(1, len(np_enhanced))).to('cpu'), out_file, sr=16000)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help='Onnx model file')
    parser.add_argument('-n', '--noisy_path', type=str, required=True, help='Noisy audio file or a noisy directory.')
    parser.add_argument('-o', '--out_path', type=str, required=True,
                        help='Out enhanced file or out directory for enhanced files.')
    parser.add_argument('-hs', '--hidden_size', type=int, required=False, default=48,
                        help='Hidden size for dns model.')
    parser.add_argument('-d', '--depth', type=int, required=False, default=4,
                        help='Model depth')
    parser.add_argument('-f', '--frame_length', type=int, required=False, default=480, help='frame length value')
    parser.add_argument('-b', '--resample_buffer', type=int, required=False, default=64, help='resample buffer value')
    parser.add_argument('-s', '--stride', type=int, required=False, default=64, help='Stride value')
    parser.add_argument('-r', '--recurse', type=bool, required=False, default=True,
                        help='Recurse noisy audio directory.')
    parser.add_argument('-v', '--variance', action='store_true',
                        help='True if you test with a model that includes variance')

    args = parser.parse_args()
    onnx_model_file = args.model
    noisy_path = args.noisy_path
    out_path = args.out_path
    hidden = args.hidden_size
    depth = args.depth
    frame_length = args.frame_length
    resample_buffer = args.resample_buffer
    stride = args.stride
    recurse = args.recurse
    with_variance = args.variance

    '''
    if depth == 4:
        frame_length = 480
        resample_buffer = 64
        stride = 64
    else:
        frame_length = 661  # depth = 5
        resample_buffer = 256  # depth = 5
        stride = 256  # depth = 5
    '''

    if os.path.isfile(noisy_path):
        noisy, sr = torchaudio.load(str(noisy_path))

        # noisy, sr = torchaudio.load(str(audio_file))
        print(f"inference starts for {noisy_path}")
        parent_out_dir = os.path.dirname(out_path)
        if not os.path.exists(parent_out_dir):
            os.mkdir(parent_out_dir)
        if with_variance:
            test_audio_denoising_with_variance(noisy, onnx_model_file, hidden, out_path, depth)
        else:
            test_audio_denoising(noisy, onnx_model_file, hidden, out_path, depth)
    else:
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        noisy_files = librosa.util.find_files(noisy_path, ext='wav', recurse=recurse)
        for noisy_f in noisy_files:
            name = os.path.basename(noisy_f)
            out_file = os.path.join(out_path, name)
            noisy, sr = torchaudio.load(str(noisy_f))
            print(f"inference starts for {noisy_f}")

            if with_variance:
                test_audio_denoising_with_variance(noisy, onnx_model_file, hidden, out_file, depth)
            else:
                test_audio_denoising(noisy, onnx_model_file, hidden, out_file, depth)

            print(f"inference done for {noisy_f}.")
