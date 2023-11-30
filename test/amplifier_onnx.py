import onnxruntime as ort
import numpy as np
import soundfile as sf


def run_onnx_inference(model_path, input_data):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_data})
    return result


def process_audio_file(model_path, input_file, output_file):
    # Ses dosyasını yükle
    data, samplerate = sf.read(input_file)

    # Ses dosyasını 480 örneklik parçalara böl
    chunks = [data[i:i + 480] for i in range(0, len(data), 480)]

    # Her parçayı modele gönder ve sonuçları birleştir
    processed_data = []
    for chunk in chunks:
        # Eğer parça 480 örnekten kısa ise, sonuna sıfır ekle
        if len(chunk) < 480:
            chunk = np.pad(chunk, (0, 480 - len(chunk)), 'constant')
        result = run_onnx_inference(model_path, chunk.astype(np.float32))
        processed_data.extend(result[0])

    # İşlenmiş veriyi bir ses dosyası olarak kaydet
    sf.write(output_file, np.array(processed_data), samplerate)


if __name__ == '__main__':
    # Model yolu ve dosya yollarını ayarla
    model_path = 'amplifier.onnx'
    input_file = 'sample.wav'  # Giriş ses dosyası yolu
    output_file = 'sample-amplify.wav'  # Çıkış ses dosyası yolu

    # İşlemi çalıştır
    process_audio_file(model_path, input_file, output_file)
