from dataclasses import dataclass
import hashlib
import os
import statistics
import time
from typing import Any
from blake3 import blake3
from tqdm import tqdm

# Test cases

test_cases = [
    "/media/rhino/invokeai/models/sd-1/embedding/easynegative.safetensors",  # 24.08 KB
    "/media/rhino/invokeai/models/sdxl/main/stable-diffusion-xl-base-1-0/vae/diffusion_pytorch_model.fp16.safetensors",  # 159.58 MB
    "/media/rhino/invokeai/models/sd-1/main/stable-diffusion-v1-5-inpainting/safety_checker/model.fp16.safetensors",  # 579.85 MB
    "/media/rhino/invokeai/models/core/convert/stable-diffusion-2-clip/text_encoder/model.safetensors",  # 1.27 GB
    "/media/rhino/invokeai/models/sdxl/main/dreamshaperXL_v21TurboDPMSDE.safetensors",  # 6.46 GB
]

iterations = 5

# Algorithms

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


# SHA256
def get_hash_sha256(file_path: str) -> str:
    file_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        file_hash.update(f.read())
    sha256_hash = file_hash.hexdigest()
    return sha256_hash


# SHA512
def get_hash_sha512(file_path: str) -> str:
    file_hash = hashlib.sha512()
    with open(file_path, "rb") as f:
        file_hash.update(f.read())
    sha512_hash = file_hash.hexdigest()
    return sha512_hash


hash_functions = {
    "SHA1_fast": get_hash_sha1_fast,
    "SHA1_correct": get_hash_sha1_correct,
    "MD5": get_hash_md5,
    "BLAKE3": get_hash_b3,
    "SHA256": get_hash_sha256,
    "SHA512": get_hash_sha512,
}

# Helpers

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


@dataclass
class AlgoStats:
    name: str
    avg: float
    std_dev: float

    def __repr__(self):
        return f"{self.name}: {pretty_time(self.avg)}, std dev {round(self.std_dev, 4)}"


@dataclass
class FileStats:
    file_path: str
    filesize_bytes: int
    stats: dict[str, AlgoStats]

    def __repr__(self):
        max_name_length = max(len(algo_stat.name) for algo_stat in self.stats.values())
        max_avg_length = max(
            len(pretty_time(algo_stat.avg)) for algo_stat in self.stats.values()
        )
        repr = f"File: {self.file_path} ({pretty_file_size(self.filesize_bytes)})\n"
        for algo_stat in self.stats.values():
            repr += f"  {algo_stat.name.ljust(max_name_length + 1)}: {pretty_time(algo_stat.avg).rjust(max_avg_length + 1)} (SD {round(algo_stat.std_dev, 4)})\n"
        return repr


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any):
        self.end = time.perf_counter()
        self.interval = self.end - self.start

# Test!

test_cases_with_filesize: list[tuple[str, int]] = [
    (file, os.path.getsize(file)) for file in test_cases
]

# Dict of hash function names to list of times per file
results: dict[str, list[list[float]]] = {name: [] for name in hash_functions.keys()}

for test_case in tqdm(test_cases_with_filesize, desc="Overall Progress"):
    file_path, file_size = test_case
    times: dict[str, list[float]] = {name: [] for name in hash_functions.keys()}

    for i in tqdm(
        range(iterations + 1),  #  we'll throw away the first
        desc=f"Hashing {file_path} ({pretty_file_size(file_size)})",
        leave=False,
    ):
        for name, func in hash_functions.items():
            with Timer() as timer:
                func(file_path)

            # Skip the first result as it is usually an outlier
            if i != 1:
                times[name].append(timer.interval)

    for name in hash_functions.keys():
        results[name].append(times[name])

# Calculate averages and standard deviations
averages = {
    name: [sum(times) / len(times) for times in results[name]]
    for name in hash_functions.keys()
}
std_devs = {
    name: [calculate_std_dev(times) for times in results[name]]
    for name in hash_functions.keys()
}


# Generate stats
stats = [
    FileStats(
        file_path=test_cases_with_filesize[i][0],
        filesize_bytes=test_cases_with_filesize[i][1],
        stats={
            name: AlgoStats(
                name=name,
                avg=averages[name][i],
                std_dev=std_devs[name][i],
            )
            for name in hash_functions.keys()
        },
    )
    for i in range(len(test_cases_with_filesize))
]

for s in stats:
    print(s)
