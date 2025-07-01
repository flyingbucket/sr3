import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import pywt
import torchvision.transforms.functional as TF


def resize(img, size, resample):
    if img.size[0] != size:
        img = TF.resize(img, size, interpolation=resample)
        img = TF.center_crop(img, size)
    return img


def wavelet_decompose(img: Image.Image):
    """Apply 2D Haar wavelet transform to grayscale image."""
    arr = np.array(img).astype(np.float32) / 255.0
    coeffs2 = pywt.dwt2(arr, 'haar')
    LL, (LH, HL, HH) = coeffs2
    wave = np.stack([LL, LH, HL, HH], axis=-1)
    return wave.astype(np.float32)  # shape (H/2, W/2, 4)


def prepare_and_save(file, sizes, out_path):
    key = file.stem.zfill(5)
    img = Image.open(file).convert('L')  # 单通道灰度图

    # Resize
    img_hr = resize(img, sizes[1], Image.BICUBIC)
    img_lr = resize(img, sizes[0], Image.BICUBIC)

    # Wavelet
    wave = wavelet_decompose(img_hr)

    # Save
    img_lr.save(f'{out_path}/lr_{sizes[0]}/{key}.png')
    img_hr.save(f'{out_path}/hr_{sizes[1]}/{key}.png')
    np.save(f'{out_path}/sr_{sizes[0]}_{sizes[1]}/{key}.npy', wave)


def main(img_dir, out_path, sizes):
    img_dir = Path(img_dir)
    out_path = Path(out_path)
    files = sorted([p for p in img_dir.rglob('*') if p.suffix.lower() in ['.jpg', '.png']])

    # Make output folders
    (out_path / f'lr_{sizes[0]}').mkdir(parents=True, exist_ok=True)
    (out_path / f'hr_{sizes[1]}').mkdir(parents=True, exist_ok=True)
    (out_path / f'sr_{sizes[0]}_{sizes[1]}').mkdir(parents=True, exist_ok=True)

    for file in tqdm(files, desc="Processing"):
        prepare_and_save(file, sizes, out_path)

    print(f"\n✅ Finished saving {len(files)} images to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True, help='Input image folder')
    parser.add_argument('--out', '-o', type=str, required=True, help='Output base folder')
    parser.add_argument('--size', type=str, default='64,512', help='Low,High resolution sizes (e.g. 64,512)')
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.size.split(',')]
    main(args.path, args.out, sizes)
