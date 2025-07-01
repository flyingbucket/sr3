import os
import argparse
from io import BytesIO
from pathlib import Path
from multiprocessing import Process, Lock, RawValue
from PIL import Image
from tqdm import tqdm

import numpy as np
import lmdb
import pywt
import torch
import torchvision.transforms.functional as TF


def resize(img, size, resample):
    if img.size[0] != size:
        img = TF.resize(img, size, interpolation=resample)
        img = TF.center_crop(img, size)
    return img


def wavelet_decompose(img: Image.Image):
    """Apply 2D Haar wavelet transform to grayscale image."""
    arr = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
    coeffs2 = pywt.dwt2(arr, 'haar')
    LL, (LH, HL, HH) = coeffs2
    # Stack to 4-channel (H, W, 4)
    wave = np.stack([LL, LH, HL, HH], axis=-1)
    return wave.astype(np.float32)


def np_array_to_bytes(arr: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, arr)
    return buffer.getvalue()


def pil_to_bytes(img: Image.Image) -> bytes:
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def resize_and_transform(img: Image.Image, sizes):
    img_hr = resize(img, sizes[1], Image.BICUBIC)  # HR
    img_lr = resize(img, sizes[0], Image.BICUBIC)  # LR
    sr_wave = wavelet_decompose(img_hr)           # Wavelet condition
    return img_lr, img_hr, sr_wave


class WorkerContext:
    def __init__(self, sizes, env, out_path):
        self.sizes = sizes
        self.env = env
        self.out_path = out_path
        self.counter = RawValue('i', 0)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.lock:
            return self.counter.value


def process_file(file, ctx: WorkerContext):
    img = Image.open(file).convert('L')  # Grayscale
    lr_img, hr_img, sr_wave = resize_and_transform(img, ctx.sizes)

    key = file.stem.zfill(5)

    with ctx.env.begin(write=True) as txn:
        txn.put(f'lr_{ctx.sizes[0]}_{key}'.encode(), pil_to_bytes(lr_img))
        txn.put(f'hr_{ctx.sizes[1]}_{key}'.encode(), pil_to_bytes(hr_img))
        txn.put(f'sr_{ctx.sizes[0]}_{ctx.sizes[1]}_{key}'.encode(), np_array_to_bytes(sr_wave))

        count = ctx.increment()
        txn.put(b'length', str(count).encode())


def worker_process(files, ctx: WorkerContext):
    for file in files:
        process_file(file, ctx)


def main(img_dir, out_path, sizes, n_worker=4):
    files = sorted([p for p in Path(img_dir).rglob('*') if p.suffix.lower() in ['.jpg', '.png']])
    env = lmdb.open(out_path, map_size=1024**4, readahead=False, writemap=True)

    ctx = WorkerContext(sizes, env, out_path)

    file_chunks = np.array_split(files, n_worker)
    workers = []
    for chunk in file_chunks:
        p = Process(target=worker_process, args=(chunk, ctx))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    print(f"\n✅ Finished writing {ctx.value()} samples to LMDB at {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True, help='Input image folder')
    parser.add_argument('--out', '-o', type=str, required=True, help='Output LMDB path')
    parser.add_argument('--size', type=str, default='64,512', help='Low,High resolution sizes')
    parser.add_argument('--n_worker', type=int, default=4)
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.size.split(',')]
    main(args.path, args.out, sizes, args.n_worker)
