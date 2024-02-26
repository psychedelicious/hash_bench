from dataclasses import dataclass
import hashlib
import os
import statistics
import time
from typing import Any
from blake3 import blake3
from tqdm import tqdm

test_cases = [
    "/media/rhino/invokeai/models/sd-1/embedding/easynegative.safetensors",  # 24.08 KB
    "/media/rhino/invokeai/models/sdxl/main/stable-diffusion-xl-base-1-0/vae/diffusion_pytorch_model.fp16.safetensors",  # 159.58 MB
    "/media/rhino/invokeai/models/sd-1/main/stable-diffusion-v1-5-inpainting/safety_checker/model.fp16.safetensors",  # 579.85 MB
    "/media/rhino/invokeai/models/core/convert/stable-diffusion-2-clip/text_encoder/model.safetensors",  # 1.27 GB
    "/media/rhino/invokeai/models/sdxl/main/dreamshaperXL_v21TurboDPMSDE.safetensors",  # 6.46 GB
]


@dataclass
class Stats:
    file_path: str
    filesize_bytes: int
    avg_sha1_fast: float
    std_dev_sha1_fast: float
    avg_sha1_correct: float
    std_dev_sha1_correct: float
    avg_md5: float
    std_dev_md5: float
    avg_b3: float
    std_dev_b3: float

    def __repr__(self):
        repr = f"File: {self.file_path}\n"
        repr += f"  File Size: {pretty_file_size(self.filesize_bytes)}\n"
        repr += f"  SHA1 (Fast): {pretty_time(self.avg_sha1_fast)}, std dev {round(self.std_dev_sha1_fast, 4)}\n"
        repr += f"  SHA1 (Correct): {pretty_time(self.avg_sha1_correct)}, std dev {round(self.std_dev_sha1_correct, 4)}\n"
        repr += f"  MD5: {pretty_time(self.avg_md5)}, std dev {round(self.std_dev_md5, 4)}\n"
        repr += f"  BLAKE3: {pretty_time(self.avg_b3)}, std dev {round(self.std_dev_b3, 4)}\n"
        return repr


def pretty_file_size(bytes: int) -> str:
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{round(bytes / 1024, 2)} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{round(bytes / 1024 / 1024, 2)} MB"
    else:
        return f"{round(bytes / 1024 / 1024 / 1024, 2)} GB"


def pretty_time(seconds: float) -> str:
    if seconds < 1:
        return f"{round(seconds * 1000, 2)} ms"
    elif seconds < 60:
        return f"{round(seconds, 2)} s"
    elif seconds < 3600:
        return f"{round(seconds / 60, 2)} m"
    else:
        return f"{round(seconds / 3600, 2)} h"


def calculate_std_dev(numbers: list[int | float]) -> float:
    return statistics.stdev(numbers)


# Fast SHA1 - usage of block size means the resultant hashes are incorrect due to padding!
def get_hash_sha1_fast(file_path: str) -> str:
    BLOCK_SIZE = 2**16
    file_hash = hashlib.sha1()
    with open(file_path, "rb") as f:
        data = f.read(BLOCK_SIZE)
        file_hash.update(data)
    sha1_hash = file_hash.hexdigest()
    return sha1_hash


# Correct SHA1 - python 3.11 has a better method to do this: https://docs.python.org/3/library/hashlib.html#file-hashing
def get_hash_sha1_correct(file_path: str) -> str:
    file_hash = hashlib.sha1()
    with open(file_path, "rb") as f:
        file_hash.update(f.read())
    sha1_hash = file_hash.hexdigest()
    return sha1_hash


# MD5
def get_hash_md5(file_path: str) -> str:
    file_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        file_hash.update(f.read())
    sha1_hash = file_hash.hexdigest()
    return sha1_hash


# BLAKE3 go brrr
def get_hash_b3(file_path: str) -> str:
    file_hasher = blake3(max_threads=blake3.AUTO)
    file_hasher.update_mmap(file_path)
    b3_hash = file_hasher.hexdigest()
    return b3_hash


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


results_sha1_fast: list[list[float]] = []
results_sha1_correct: list[list[float]] = []
results_md5: list[list[float]] = []
results_b3: list[list[float]] = []
file_sizes: list[int] = []

iterations = 10

for file_path in tqdm(test_cases, desc="Overall Progress"):
    sha1_fast_times: list[float] = []
    sha1_correct_times: list[float] = []
    md5_times: list[float] = []
    b3_times: list[float] = []

    file_size = os.path.getsize(file_path)
    file_sizes.append(file_size)

    for i in tqdm(
        range(iterations + 1),  #  we'll throw away the first
        desc=f"Hashing {file_path} ({pretty_file_size(file_size)})",
        leave=False,
    ):
        with Timer() as sha1_fast_timer:
            get_hash_sha1_fast(file_path)

        with Timer() as sha1_correct_timer:
            get_hash_sha1_correct(file_path)

        with Timer() as md5_timer:
            get_hash_md5(file_path)

        with Timer() as b3_timer:
            get_hash_b3(file_path)

        # Skip the first result as it is usually an outlier
        if i != 1:
            sha1_fast_times.append(sha1_fast_timer.interval)
            sha1_correct_times.append(sha1_correct_timer.interval)
            md5_times.append(md5_timer.interval)
            b3_times.append(b3_timer.interval)

    results_sha1_fast.append(sha1_fast_times)
    results_sha1_correct.append(sha1_correct_times)
    results_md5.append(md5_times)
    results_b3.append(b3_times)

avg_sha1_fast = [sum(times) / len(times) for times in results_sha1_fast]
avg_sha1_correct = [sum(times) / len(times) for times in results_sha1_correct]
avg_md5 = [sum(times) / len(times) for times in results_md5]
avg_b3 = [sum(times) / len(times) for times in results_b3]


stats = [
    Stats(
        file_path=test_cases[i],
        filesize_bytes=file_sizes[i],
        avg_sha1_fast=avg_sha1_fast[i],
        std_dev_sha1_fast=calculate_std_dev(results_sha1_fast[i]),
        avg_sha1_correct=avg_sha1_correct[i],
        std_dev_sha1_correct=calculate_std_dev(results_sha1_correct[i]),
        avg_md5=avg_md5[i],
        std_dev_md5=calculate_std_dev(results_md5[i]),
        avg_b3=avg_b3[i],
        std_dev_b3=calculate_std_dev(results_b3[i]),
    )
    for i in range(len(test_cases))
]

for s in stats:
    print(s)
