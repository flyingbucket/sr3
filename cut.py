import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def normalize_image(img):
    """将输入图像归一化为 [0, 1] 的 float32"""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    elif img.dtype in [np.float32, np.float64]:
        return np.clip(img, 0, 1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image dtype: {img.dtype}")


def save_png_tile(tile, out_path):
    """将 tile 归一化为 uint8 并保存为 PNG 图像"""
    tile_uint8 = (tile * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(out_path, tile_uint8)


def process_single(input_path, out_folder, row, col, size, step):
    picname = os.path.splitext(os.path.basename(input_path))[0]
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read {input_path}")
        return

    height, width = img.shape[:2]
    x_offset = col * step
    y_offset = row * step

    if x_offset + size > width or y_offset + size > height:
        return  # Skip tile if out of bounds

    tile = img[y_offset : y_offset + size, x_offset : x_offset + size]

    # If image is grayscale (2D), keep it as is. Else, keep only first channel.
    if tile.ndim == 2:
        pass
    elif tile.ndim == 3:
        tile = tile[:, :, 0]  # Assume single channel is enough
    else:
        print(f"Unexpected tile shape: {tile.shape}")
        return

    out_path = os.path.join(out_folder, f"{picname}_{row}_{col}.png")
    save_png_tile(tile, out_path)


def process_all(input_folder, out_folder, size, overlap):
    os.makedirs(out_folder, exist_ok=True)
    step = size - overlap
    tasks = []

    with ProcessPoolExecutor() as executor:
        for filename in tqdm(os.listdir(input_folder), desc="Reading input folder"):
            if not filename.lower().endswith(
                (".tif", ".tiff", ".png", ".jpg", ".jpeg")
            ):
                continue

            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Failed to read {input_path}")
                continue

            img = normalize_image(img)
            height, width = img.shape[:2]
            rows = (height - overlap) // step
            cols = (width - overlap) // step

            for row in range(rows):
                for col in range(cols):
                    tasks.append(
                        executor.submit(
                            process_single,
                            input_path,
                            out_folder,
                            row,
                            col,
                            size,
                            step,
                        )
                    )

        # 等待所有任务完成
        for task in tqdm(tasks, desc="Processing tiles"):
            task.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()

    process_all(args.input_folder, args.out_folder, args.size, args.overlap)
