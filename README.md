# hash bench

Simple script to benchmark file hashing algorithms in python. The files are stable diffusion models.

## Setup

Requires `blake3` and `tqdm`. The rest is python builtins.

```sh
pip install -r requirements.txt
```

## Algorithms

- `SHA1_fast`
  - SHA1 with a block size of `2**16`. This produced incorrect hashes because the block size adds padding.
- `SHA1_correct`
  - SHA1 without adding a block size. Much slower, but correct.
- `MD5`
- `BLAKE3`
  - Uses memory mapping and threads for best performance.
- `SHA256`
- `SHA512`
  - I expected this to be [faster than SHA256](https://crypto.stackexchange.com/questions/26336/sha-512-faster-than-sha-256) but it's not. Dunno why.

### Adding an Algorithm

To add an algorithm, create a function that hashes a file, returning the hash as a string. Then add it to the `hash_functions` dict and it will be benchmarked.

## Results

Here are my results. The files are on an external SSD.

```
File: /media/rhino/invokeai/models/sd-1/embedding/easynegative.safetensors (24.08 KB)
  SHA1_fast    :  0.14 ms (SD 0.0001)
  SHA1_correct :   0.1 ms (SD 0.0)
  MD5          :   0.2 ms (SD 0.0)
  BLAKE3       :  0.97 ms (SD 0.0013)
  SHA256       :  0.19 ms (SD 0.0001)
  SHA512       :  0.24 ms (SD 0.0)

File: /media/rhino/invokeai/models/sdxl/main/stable-diffusion-xl-base-1-0/vae/diffusion_pytorch_model.fp16.safetensors (159.58 MB)
  SHA1_fast    :    0.16 ms (SD 0.0002)
  SHA1_correct :  150.39 ms (SD 0.0663)
  MD5          :  208.99 ms (SD 0.0)
  BLAKE3       :    8.01 ms (SD 0.0005)
  SHA256       :  220.27 ms (SD 0.0002)
  SHA512       :  277.61 ms (SD 0.1295)

File: /media/rhino/invokeai/models/sd-1/main/stable-diffusion-v1-5-inpainting/safety_checker/model.fp16.safetensors (579.85 MB)
  SHA1_fast    :    0.07 ms (SD 0.0)
  SHA1_correct :  393.38 ms (SD 0.002)
  MD5          :  822.72 ms (SD 0.0654)
  BLAKE3       :   23.18 ms (SD 0.0002)
  SHA256       :  507.35 ms (SD 0.0017)
  SHA512       :  730.58 ms (SD 0.0011)

File: /media/rhino/invokeai/models/core/convert/stable-diffusion-2-clip/text_encoder/model.safetensors (1.27 GB)
  SHA1_fast    :    0.08 ms (SD 0.0)
  SHA1_correct :  865.37 ms (SD 0.0012)
  MD5          :     1.67 s (SD 0.0188)
  BLAKE3       :   47.93 ms (SD 0.0025)
  SHA256       :  970.45 ms (SD 0.0383)
  SHA512       :     1.66 s (SD 0.0691)

File: /media/rhino/invokeai/models/sdxl/main/dreamshaperXL_v21TurboDPMSDE.safetensors (6.46 GB)
  SHA1_fast    :    0.08 ms (SD 0.0)
  SHA1_correct :     4.38 s (SD 0.0074)
  MD5          :     8.39 s (SD 0.0079)
  BLAKE3       :  237.96 ms (SD 0.0037)
  SHA256       :     4.73 s (SD 0.0617)
  SHA512       :      8.2 s (SD 0.0094)
```
