# hash bench

Simple script to benchmark file hashing algorithms in python. The files are stable diffusion models.

## Setup

Requires `blake3` and `tqdm`. The rest is python builtins.

```sh
pip install -r requirements.txt
```

## Algorithms

You can easily add any `hashlib` algorithm in a naive (read the whole file into memory at once) or optimized (via memory view) implementations.

Update `hash_functions` to test others:

```py
hash_functions = {
    # Any `hashlib` algorithm works here
    "SHA1_naive": get_hashlib_naive("sha1"),
    "SHA1_mv": get_hashlib_mv("sha1"),
    "MD5_naive": get_hashlib_naive("md5"),
    "MD5_mv": get_hashlib_mv("md5"),
    # BLAKE3 is not in `hashlib`, must use the provided functions
    "BLAKE3_mmap": blake3_mmap,
    "BLAKE3_mv": blake3_mv,
}
```
