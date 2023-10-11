python versiyonu 3.7.16 ile test edilmiştir. 

Yeni bir conda environment oluşturularak gerekli paketleri yükleyebilirsiniz.

```bash
conda create --name denoiser-onnx python=3.7.16

conda activate denoiser-onnx

pip install -r requirements.txt
```

Aşağıdaki gibi çalıştırabilirsiniz.

```bash
python denoiser_stream_onnx_test.py -m ./dns48_depth=4_stride=128.onnx -n audio-wav-16k 
-o enhanced_dir 
```

zip modeli bir miktar daha ufak, ikisini de deneyebilirsiniz.


###Deepfilternet Onnx Modelinin Çalıştırılması

Deepfilternet onnx modeli 48 kHz ses dosyaları üzerinde çalışmaktadır. 
Onnx modelinin streaming simülasyon kodunu `deepfilternet_onnx_stream.py` 
scriptinde bulabilirsiniz. 

Bu model 480 sample işlemektedir. Dolayısıyla 10 ms alarak temizler.

Tek bir ses dosyası için çalıştırılması:

```bash
python deepfilternet_onnx_stream.py -m ./deepfilternet3.onnx -n sample.wav
-o ./sample-df-onnx.wav -b 480 
```

Output parametresi klasör olursa temizlenmiş ses dosyası 
bu dizinde aynı isimle oluşturulur.
```bash
python deepfilternet_onnx_stream.py -m ./deepfilternet3.onnx -n sample.wav
-o ./enhance_dir -b 480 
```

Bir dizindeki ses dosyalarının temizlenmesi için çalıştırılması:

```bash
python deepfilternet_onnx_stream.py -m ./deepfilternet3.onnx -n ./noisy_dir
-o ./enhance_dir -b 480 
```