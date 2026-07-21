# TEST LIST

## GT:
1. Hardware:
Nvidia Titan:
2. Software:
* No overlay
* original way (batch size one + cpu decode)

3. stats:
- VRAM: 608MiB
- RAM: 930 MiB
- TIME: 
    * GPU forward: mean  13.60 ms/pair | median  13.39  p95  14.99  min  13.32  max  15.52
    * e2e (H2D+fwd+D2H): mean  14.56 ms/pair |  median  14.35  p95  15.94  min  14.21  max  16.97

## Test 1
### 1. Software:
* No overlay
* batch inference + cpu decode

### 2. stats:
1. batch size 4
- VRAM: 744MiB
- RAM: 1.132 GiB
- TIME: 
    * GPU forward: mean  46.00 ms/batch  |  per-image   5.75 ms  |  median  45.36  p95  48.75  min  45.13  max  50.67
    * e2e (H2D+fwd+D2H): mean  48.02 ms/batch  |  per-image   6.00 ms  |  median  47.37  p95  50.80  min  47.10  max  52.83

2. Batch size 32
- VRAM: 1658 mIb
- RAM: 952 mib
- Result:
Each forward = 1 batch = 32 stereo pairs = 64 images (batch 32x2x3x256x256)
       GPU forward: mean 300.62 ms/batch  |  per-image   4.70 ms  |  median 299.01  p95 306.70  min 295.78  max 307.05
 e2e (H2D+fwd+D2H): mean 322.41 ms/batch  |  per-image   5.04 ms  |  median 320.30  p95 328.17  min 319.06  max 328.68
        throughput:    3.3 batch/s  =   212.9 img/s  (GPU forward only)


3. Batch size: 256
- VRAM: 7524 mib
- RAM: 1.4gb
- Result:
Each forward = 1 batch = 256 stereo pairs = 512 images (batch 256x2x3x256x256)
       GPU forward: mean 2217.04 ms/batch  |  per-image   4.33 ms  |  median 2217.04  p95 2217.04  min 2217.04  max 2217.04
 e2e (H2D+fwd+D2H): mean 2408.92 ms/batch  |  per-image   4.70 ms  |  median 2408.92  p95 2408.92  min 2408.92  max 2408.92
        throughput:    0.5 batch/s  =   230.9 img/s  (GPU forward only)

4. Batch size: 512
- VRAM: 14460 MiB
- RAM:2.67 GiB
- Result:
Each forward = 1 batch = 512 stereo pairs = 1024 images (batch 512x2x3x256x256)
       GPU forward: mean 4437.86 ms/batch  |  per-image   4.33 ms  |  median 4437.86  p95 4437.86  min 4437.86  max 4437.86
 e2e (H2D+fwd+D2H): mean 4819.70 ms/batch  |  per-image   4.71 ms  |  median 4819.70  p95 4819.70  min 4819.70  max 4819.70
        throughput:    0.2 batch/s  =   230.7 img/s  (GPU forward only)



## Test 2
### 1. Software:
- GPU decode + batch inference

1. Batch 32
vram: 12814 MiB
ram: 1.6 GiB

=== Inference timing (NVIDIA TITAN RTX, warmup 10 excluded, n=6) ===
Each forward = 1 batch = 32 stereo pairs = 64 images (batch 32x2x3x256x256)
       GPU forward: mean 271.76 ms/batch  |  per-image   4.25 ms  |  median 272.23  p95 273.33  min 269.33  max 273.50
 e2e (H2D+fwd+D2H): mean 284.86 ms/batch  |  per-image   4.45 ms  |  median 285.32  p95 286.90  min 282.39  max 287.24
        throughput:    3.7 batch/s  =   235.5 img/s  (GPU forward only)