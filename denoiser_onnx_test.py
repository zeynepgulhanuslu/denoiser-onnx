import io
import os
import time
import onnxruntime
import torch
import torchaudio
import zipfile


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
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_profiling = True
    session_options.profile_file_prefix = "profile_onnx_1thread"
    # onnx runtime session oluşturulması
    if is_zip_file(onnx_tt_quantized_model_path):
        model_bytes = load_onnx_from_zip(onnx_tt_model_path)
        ort_session = onnxruntime.InferenceSession(model_bytes, session_options)
    else:
        ort_session = onnxruntime.InferenceSession(onnx_tt_model_path, session_options)

    # burada onnx modelinin giriş değerlerinin isimleri alınır.
    input_audio_frame_name = ort_session.get_inputs()[0].name  # ses dizisini ifade eder

    # onnx modelinin çıktısındaki değerlerin isimleri
    out_frame_name = ort_session.get_outputs()[0].name  # burası çıkış audio array idir.

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
            frame = frame.unsqueeze(0).expand(hidden, 1, frame_length)
            print(f"frame shape:{frame.shape}")
            start_time = time.time()
            # onnx modelinin çalışmsı
            ort_inputs = {input_audio_frame_name: to_numpy(frame)}
            out = ort_session.run(None, ort_inputs)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            total_inference_time += inference_time
            rtf = inference_time / frame_in_ms
            print(f"inference time in ms for frame {total_frame + 1}, noisy frame in ms: {frame_in_ms}, "
                  f"{inference_time} ms. rtf: {rtf}")

            output_np = out[0]  # enhanced out audio

            outs.append(torch.from_numpy(output_np))
            frames = frames[:, stride:]  # bir sonraki frame e gidilir.

            total_frame += 1

        # en sonda frame length den küçük kısım kaldıysa, orası da işlenir.
        if frames.shape[1] > 0:
            # Expand the remaining audio with zeros
            last_frame = torch.cat([frames, torch.zeros_like(frames)
            [:, :frame_length - frames.shape[1]]], dim=1)

            start_time = time.time()
            out = ort_session.run(None, {input_audio_frame_name: last_frame.detach().cpu().numpy()})[0]
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
    # Create an inference session

    # Run the model
    prof = ort_session.end_profiling()

    average_inference_time = total_inference_time / total_frame
    print(f"average inference time in ms: {average_inference_time:.6f}")
    print(f"average rtf : {average_inference_time / frame_in_ms:.6f}")
    write(estimate, out_file, sr=16000)  # wav dosyası olarak yazılır.


if __name__ == '__main__':
    onnx_tt_quantized_model_path = 'dns48_quantized.onnx'  # zip modeli kullanmak isterseniz.
    audio_file = 'sample.wav'
    out_file = 'sample_48_quantized.wav'

    noisy, sr = torchaudio.load(str(audio_file))

    # aşağıdaki değerler değişmeyecek.
    frame_length = 661
    stride = 5
    hidden = 48

    test_audio_denoising(noisy, onnx_tt_quantized_model_path, hidden)
