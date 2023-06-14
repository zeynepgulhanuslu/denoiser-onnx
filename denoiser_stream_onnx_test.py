import io
import os
import time
import onnxruntime
import torch
import torchaudio
import zipfile
import json

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


def test_audio_denoising(noisy, onnx_tt_model_path, hidden):
    # Bu kısmın iyileştirme yapıp yapmadığına emin değilim.
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 5
    session_options.inter_op_num_threads = 5
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_profiling=True
    session_options.profile_file_prefix = "profile_streamtt_5thread"
    # onnx runtime session oluşturulması
    if is_zip_file(onnx_tt_quantized_model_path):
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
    lstm_state_1 = torch.randn(2, 1, hidden * 16)
    lstm_state_2 = torch.randn(2, 1, hidden * 16)
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
            total_frame += 1

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
            total_frame += 1

        estimate = torch.cat(outs, 1)  # burada çıkıştaki tensor ler bir torch tensor üne dönüştürülür.
    # Run the model
    prof = ort_session.end_profiling()



    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / frame_in_ms:.6f}")
    write(estimate, out_file, sr=16000)  # wav dosyası olarak yazılır.


if __name__ == '__main__':
    onnx_tt_quantized_model_path = 'dns48_streamtt_quantized.onnx'  # zip modeli kullanmak isterseniz.
    audio_file = 'sample.wav'
    out_file = 'sample_streamtt_48_quantized.wav'

    noisy, sr = torchaudio.load(str(audio_file))

    # aşağıdaki değerler değişmeyecek.
    frame_length = 661
    stride = 256
    resample_buffer = 256
    hidden = 48

    test_audio_denoising(noisy, onnx_tt_quantized_model_path, hidden)
