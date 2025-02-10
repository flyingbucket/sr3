import os
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

def process_single(input_path, out_folder, row, col, counter):
    size = 512
    with rasterio.open(input_path) as src:
        x_offset = col * size
        y_offset = row * size

        if x_offset + size > src.width or y_offset + size > src.height:
            raise ValueError("The image size is not sufficient for the specified grid.")

        window = Window(x_offset, y_offset, size, size)
        cropped_image = src.read(window=window)
        meta = src.meta.copy()
        meta.update({
            "width": size,
            "height": size,
            "transform": src.window_transform(window)
        })
        
        # 将裁剪后的图像转换为 PNG 格式并保存
        out_path = os.path.join(out_folder, f"{counter:04d}.png")
        cropped_image = np.moveaxis(cropped_image, 0, -1)  # 将波段轴移到最后
        if cropped_image.shape[2] == 1:  # 如果是单波段图像，转换为三通道
            cropped_image = np.repeat(cropped_image, 3, axis=2)
        cropped_image = cropped_image.astype(np.uint8)  # 确保数据类型为 uint8
        Image.fromarray(cropped_image).save(out_path)

def process_all(input_folder, out_folder):
    counter = 1
    rows, cols = 5, 6
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            for row in range(rows):
                for col in range(cols):
                    process_single(input_path, out_folder, row, col, counter)
                    counter += 1

if __name__ == "__main__":
    input_folder = "dataset/WHU"
<<<<<<< HEAD
    out_folder = "dataset/WHU_512_full"
=======
    out_folder = "dataset/WHU_512"
>>>>>>> origin/sar_v1
    process_all(input_folder, out_folder)