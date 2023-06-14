python versiyonu 3.7.16 ile test edilmiştir. 

Yeni bir conda environment oluşturularak gerekli paketleri yükleyebilirsiniz.

```bash
conda create --name denoiser-onnx python=3.7.16

conda activate denoiser-onnx

pip install -r requirements.txt
```

Aşağıdaki gibi çalıştırabilirsiniz.

```bash
python denoiser_stream_onnx_test.py
```

zip modeli bir miktar daha ufak, ikisini de deneyebilirsiniz.
