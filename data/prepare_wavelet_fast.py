import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import pywt
import torchvision.transforms.functional as TF
from concurrent.futures import ProcessPoolExecutor, as_completed

def resize(img, size, resample):
    if img.size[0] != size:
        img = TF.resize(img, size, interpolation=resample)
        img = TF.center_crop(img, size)
    return img

def wavelet_decompose(img: Image.Image, add_ori=False):
    arr = np.array(img).astype(np.float32) / 255.0
    coeffs2 = pywt.swt2(arr,'haar',level=1)
    LL, (LH, HL, HH) = coeffs2[0]
    if not add_ori:
        wave = np.stack([LL, LH, HL, HH], axis=-1)
    else:
        wave = np.stack([arr, LL, LH, HL, HH], axis=-1)
    return wave.astype(np.float32)

def prepare_and_save(args):
    file, sizes, out_path, add_ori = args
    key = file.stem.zfill(5)
    img = Image.open(file).convert('L')

    # Resize
    img_hr = resize(img, sizes[1], Image.BICUBIC)
    img_lr = resize(img, sizes[0], Image.BICUBIC)
    img_sr = resize(img_lr, sizes[1], Image.BICUBIC)

    # Wavelet
    wave_target = wavelet_decompose(img_hr, add_ori=add_ori)
    wave_in = wavelet_decompose(img_sr, add_ori=add_ori)

    # Save
    (out_path / f'lr_{sizes[0]}').mkdir(parents=True, exist_ok=True)
    (out_path / f'hr_{sizes[1]}').mkdir(parents=True, exist_ok=True)
    (out_path / f'sr_{sizes[0]}_{sizes[1]}').mkdir(parents=True, exist_ok=True)

    img_lr.save(out_path / f'lr_{sizes[0]}' / f'{key}.png')
    np.save(out_path / f'sr_{sizes[0]}_{sizes[1]}' / f'{key}.npy', wave_in, allow_pickle=False)
    np.save(out_path / f'hr_{sizes[1]}' / f'{key}.npy', wave_target, allow_pickle=False)

def main(img_dir, out_path, add_ori, sizes, max_workers=4):
    img_dir = Path(img_dir)
    out_path = Path(out_path)
    files = sorted([p for p in img_dir.rglob('*') if p.suffix.lower() in ['.jpg', '.png']])

    args_list = [(file, sizes, out_path, add_ori) for file in files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(prepare_and_save, args) for args in args_list]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            pass

    print(f"\n✅ Finished saving {len(files)} images to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True, help='Input image folder')
    parser.add_argument('--out', '-o', type=str, required=True, help='Output base folder')
    parser.add_argument('--size', type=str, default='64,512', help='Low,High resolution sizes (e.g. 64,512)')
    parser.add_argument('--add_ori', type=bool, required=True, help='Whether to add original image to wavelet decomposition')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.size.split(',')]
    main(args.path, args.out, args.add_ori, sizes, max_workers=args.workers)
