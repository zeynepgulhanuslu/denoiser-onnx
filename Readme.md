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
-o enhanced_dir -b 128 -s 128 -f 480 -v
```

zip modeli bir miktar daha ufak, ikisini de deneyebilirsiniz.
